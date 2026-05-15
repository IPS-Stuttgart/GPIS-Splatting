from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gpis_splatting.gpis import GPISPrediction, rbf_kernel, surface_band_probability
from gpis_splatting.gpis_backends import GPISBackend, InducingPointGPISBackend, solve_lower_triangular_features

Tensor = torch.Tensor


@dataclass(frozen=True)
class LargeScaleGPISScoreConfig:
    """Configuration for streaming GPIS scoring over large query-center arrays."""

    epsilon: float = 0.08
    batch_size: int | None = None
    memory_budget_mib: int | None = 512
    prediction_device: str = "auto"
    prediction_dtype: str = "float32"
    output_device: str = "cpu"
    include_prediction: bool = True


@dataclass(frozen=True)
class LargeScaleGPISScores:
    """Large-scale GPIS scores for query centers."""

    gate: Tensor
    mean: Tensor | None = None
    variance: Tensor | None = None
    gradient: Tensor | None = None
    distance: Tensor | None = None
    distance_std: Tensor | None = None

    @property
    def num_points(self) -> int:
        return int(self.gate.shape[0])


@dataclass(frozen=True)
class LargeScaleGPISScoreResult:
    """Scores plus runtime metadata."""

    scores: LargeScaleGPISScores
    stats: dict[str, object]


def score_large_scale_gpis(backend: GPISBackend, x_query: Tensor, *, config: LargeScaleGPISScoreConfig | None = None) -> LargeScaleGPISScoreResult:
    """Score many query centers with bounded accelerator memory.

    Inducing-point GPIS models are the intended million-query path. They are moved
    to the requested prediction device/dtype and evaluated in streaming chunks. The
    inducing gradient is computed with matrix products, avoiding a
    ``batch x num_inducing x dims`` allocation. Other backend types are accepted via
    their ordinary ``predict`` method, but are not recommended for million-scale
    scoring.
    """

    config = config or LargeScaleGPISScoreConfig()
    validate_large_scale_config(config)
    if x_query.ndim != 2:
        raise ValueError("x_query must have shape (n_query, n_dims).")

    start_time = time.perf_counter()
    prediction_backend = prepare_prediction_backend(backend, config)
    batch_size = resolve_large_scale_batch_size(prediction_backend, config)
    output_device = resolve_output_device(config.output_device)
    x_query = x_query.detach()
    n_query = int(x_query.shape[0])
    dims = int(x_query.shape[1])
    output_dtype = prediction_backend.dtype

    gate = torch.empty((n_query,), dtype=output_dtype, device=output_device)
    mean = torch.empty((n_query,), dtype=output_dtype, device=output_device) if config.include_prediction else None
    variance = torch.empty((n_query,), dtype=output_dtype, device=output_device) if config.include_prediction else None
    gradient = torch.empty((n_query, dims), dtype=output_dtype, device=output_device) if config.include_prediction else None
    distance = torch.empty((n_query,), dtype=output_dtype, device=output_device) if config.include_prediction else None
    distance_std = torch.empty((n_query,), dtype=output_dtype, device=output_device) if config.include_prediction else None

    for chunk_start in range(0, n_query, batch_size):
        chunk_end = min(chunk_start + batch_size, n_query)
        prediction = predict_large_scale_chunk(prediction_backend, x_query[chunk_start:chunk_end], batch_size=batch_size)
        chunk_gate = surface_band_probability(prediction, config.epsilon)
        gate[chunk_start:chunk_end] = chunk_gate.to(device=output_device, non_blocking=True)
        if config.include_prediction:
            assert mean is not None and variance is not None and gradient is not None and distance is not None and distance_std is not None
            mean[chunk_start:chunk_end] = prediction.mean.to(device=output_device, non_blocking=True)
            variance[chunk_start:chunk_end] = prediction.variance.to(device=output_device, non_blocking=True)
            gradient[chunk_start:chunk_end] = prediction.gradient.to(device=output_device, non_blocking=True)
            distance[chunk_start:chunk_end] = prediction.distance.to(device=output_device, non_blocking=True)
            distance_std[chunk_start:chunk_end] = prediction.distance_std.to(device=output_device, non_blocking=True)

    if prediction_backend.device.type == "cuda":
        torch.cuda.synchronize(prediction_backend.device)
    elapsed = time.perf_counter() - start_time
    stats: dict[str, object] = {
        "num_points": n_query,
        "dims": dims,
        "backend": prediction_backend.backend_name,
        "prediction_device": str(prediction_backend.device),
        "prediction_dtype": str(prediction_backend.dtype).removeprefix("torch."),
        "output_device": str(output_device),
        "batch_size": batch_size,
        "memory_budget_mib": config.memory_budget_mib,
        "elapsed_sec": elapsed,
        "points_per_sec": n_query / elapsed if elapsed > 0.0 else math.inf,
        "epsilon": config.epsilon,
        "include_prediction": config.include_prediction,
        "gate_min": float(torch.min(gate).item()) if n_query else 0.0,
        "gate_mean": float(torch.mean(gate.to(dtype=torch.float64)).item()) if n_query else 0.0,
        "gate_max": float(torch.max(gate).item()) if n_query else 0.0,
    }
    return LargeScaleGPISScoreResult(
        scores=LargeScaleGPISScores(gate=gate, mean=mean, variance=variance, gradient=gradient, distance=distance, distance_std=distance_std),
        stats=stats,
    )


def prepare_prediction_backend(backend: GPISBackend, config: LargeScaleGPISScoreConfig) -> GPISBackend:
    if isinstance(backend, InducingPointGPISBackend):
        return move_inducing_backend(backend, device=resolve_prediction_device(config.prediction_device), dtype=resolve_prediction_dtype(config.prediction_dtype))
    if config.prediction_device not in ("cpu", "auto"):
        raise ValueError("Only InducingPointGPISBackend supports moving large-scale prediction to CUDA.")
    return backend


def move_inducing_backend(backend: InducingPointGPISBackend, *, device: torch.device, dtype: torch.dtype) -> InducingPointGPISBackend:
    return replace(
        backend,
        inducing_points=backend.inducing_points.to(device=device, dtype=dtype),
        weight_mean=backend.weight_mean.to(device=device, dtype=dtype),
        weight_cov=backend.weight_cov.to(device=device, dtype=dtype),
        chol_uu=backend.chol_uu.to(device=device, dtype=dtype),
    )


def predict_large_scale_chunk(backend: GPISBackend, x_query: Tensor, *, batch_size: int) -> GPISPrediction:
    if isinstance(backend, InducingPointGPISBackend):
        return predict_inducing_large_scale(backend, x_query)
    return backend.predict(x_query, batch_size=batch_size)


def predict_inducing_large_scale(backend: InducingPointGPISBackend, x_query: Tensor) -> GPISPrediction:
    x_query = x_query.detach().to(dtype=backend.dtype, device=backend.device)
    if x_query.numel() == 0:
        return GPISPrediction(
            mean=torch.empty((0,), dtype=backend.dtype, device=backend.device),
            variance=torch.empty((0,), dtype=backend.dtype, device=backend.device),
            gradient=torch.empty((0, backend.inducing_points.shape[1]), dtype=backend.dtype, device=backend.device),
        )

    k_xu = rbf_kernel(x_query, backend.inducing_points, backend.lengthscale, backend.variance)
    features = solve_lower_triangular_features(k_xu, backend.chol_uu)
    mean = backend.mean_constant + features @ backend.weight_mean
    prior_variance = torch.as_tensor(backend.variance, dtype=backend.dtype, device=backend.device)
    projected_prior_variance = torch.sum(features * features, dim=1)
    posterior_projected_variance = torch.sum((features @ backend.weight_cov) * features, dim=1)
    residual_variance = torch.clamp(prior_variance - projected_prior_variance, min=0.0)
    variance = torch.clamp(residual_variance + posterior_projected_variance, min=1e-12)

    kernel_alpha = torch.linalg.solve_triangular(backend.chol_uu.T, backend.weight_mean[:, None], upper=True).reshape(-1)
    weighted_kernel = k_xu * kernel_alpha[None, :]
    gradient = (weighted_kernel @ backend.inducing_points - x_query * torch.sum(weighted_kernel, dim=1, keepdim=True)) / (backend.lengthscale**2)
    return GPISPrediction(mean=mean, variance=variance, gradient=gradient)


def resolve_large_scale_batch_size(backend: GPISBackend, config: LargeScaleGPISScoreConfig) -> int:
    if config.batch_size is not None:
        return int(config.batch_size)
    if isinstance(backend, InducingPointGPISBackend):
        return estimate_inducing_query_batch_size(
            num_inducing=backend.num_inducing,
            dtype=backend.dtype,
            memory_budget_mib=config.memory_budget_mib or 512,
        )
    return 8192


def estimate_inducing_query_batch_size(*, num_inducing: int, dtype: str | torch.dtype = torch.float32, memory_budget_mib: int = 512, scratch_matrices: int = 5) -> int:
    """Estimate a safe query batch size for inducing-point prediction."""

    if num_inducing < 1:
        raise ValueError("num_inducing must be positive.")
    if memory_budget_mib < 1:
        raise ValueError("memory_budget_mib must be positive.")
    if scratch_matrices < 1:
        raise ValueError("scratch_matrices must be positive.")
    resolved_dtype = resolve_prediction_dtype(dtype)
    dtype_bytes = torch.empty((), dtype=resolved_dtype).element_size()
    budget_bytes = int(memory_budget_mib * 1024 * 1024)
    per_query_bytes = int(num_inducing * dtype_bytes * scratch_matrices)
    return max(1, budget_bytes // max(per_query_bytes, 1))


def write_large_scale_scores_npz(path: str | Path, result: LargeScaleGPISScoreResult) -> None:
    data: dict[str, np.ndarray] = {"gate": result.scores.gate.detach().cpu().numpy()}
    for key in ("mean", "variance", "gradient", "distance", "distance_std"):
        value = getattr(result.scores, key)
        if value is not None:
            data[key] = value.detach().cpu().numpy()
    np.savez_compressed(path, **data)


def write_large_scale_stats_json(path: str | Path, result: LargeScaleGPISScoreResult, *, config: LargeScaleGPISScoreConfig) -> None:
    payload: dict[str, Any] = {"config": asdict(config), "stats": result.stats}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_prediction_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for large-scale GPIS scoring, but torch.cuda.is_available() is False.")
    return torch.device(device)


def resolve_prediction_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError("prediction dtype must be 'float32' or 'float64'.")


def resolve_output_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    return resolve_prediction_device(device)


def validate_large_scale_config(config: LargeScaleGPISScoreConfig) -> None:
    if config.epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if config.batch_size is not None and config.batch_size < 1:
        raise ValueError("batch_size must be positive when provided.")
    if config.memory_budget_mib is not None and config.memory_budget_mib < 1:
        raise ValueError("memory_budget_mib must be positive when provided.")
    if config.output_device not in ("cpu", "auto", "cuda"):
        raise ValueError("output_device must be 'cpu', 'auto', or 'cuda'.")
