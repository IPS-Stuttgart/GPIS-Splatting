from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from gpis_splatting.gpis import GPISPrediction, rbf_kernel
from gpis_splatting.gpis_backends import (
    cholesky_with_jitter,
    normalize_observation_noise,
    select_inducing_indices,
    solve_lower_triangular_features,
    validate_training_inputs,
)

Tensor = torch.Tensor
NeighborBackendName = Literal["auto", "cdist", "chunked", "scipy"]
ScalableBackendName = Literal["local_exact_scalable", "inducing_points_scalable", "gpu_inducing_points"]
ScalableInducingSelectionName = Literal["farthest", "uniform", "first", "grid"]


def resolve_compute_device(compute_device: str = "auto") -> torch.device:
    """Resolve a user-facing compute-device specifier.

    ``auto`` and ``gpu_inducing_points`` style callers use CUDA when it is present,
    but every path remains usable on CPU-only CI.
    """

    if compute_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compute_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compute_device.startswith("cuda:"):
        return torch.device(compute_device if torch.cuda.is_available() else "cpu")
    if compute_device == "cpu":
        return torch.device("cpu")
    raise ValueError("compute_device must be one of: cpu, cuda, auto, cuda:N.")


def nearest_training_indices_scalable(
    x_query: Tensor,
    x_train: Tensor,
    *,
    num_neighbors: int,
    neighbor_backend: NeighborBackendName = "auto",
    train_chunk_size: int = 65536,
) -> Tensor:
    """Return exact k-nearest training indices without requiring a full distance matrix.

    ``cdist`` preserves the original dense behavior. ``chunked`` scans training
    points in bounded chunks and is therefore suitable for many Gaussian centers.
    ``scipy`` uses cKDTree when SciPy is installed. ``auto`` prefers cKDTree for CPU
    tensors and otherwise falls back to the exact chunked scanner.
    """

    if num_neighbors < 1:
        raise ValueError("num_neighbors must be positive.")
    if train_chunk_size < 1:
        raise ValueError("train_chunk_size must be positive.")
    if x_train.shape[0] == 0:
        raise ValueError("At least one training observation is required.")
    k = min(int(num_neighbors), int(x_train.shape[0]))
    if neighbor_backend == "cdist":
        distances = torch.cdist(x_query, x_train)
        return torch.topk(distances, k=k, dim=1, largest=False).indices
    if neighbor_backend == "scipy" or (neighbor_backend == "auto" and x_query.device.type == "cpu" and x_train.device.type == "cpu"):
        try:
            from scipy.spatial import cKDTree  # type: ignore

            tree = cKDTree(x_train.detach().cpu().numpy())
            _, indices = tree.query(x_query.detach().cpu().numpy(), k=k)
            indices = np.asarray(indices)
            if indices.ndim == 1:
                indices = indices[:, None]
            return torch.from_numpy(indices).to(dtype=torch.long, device=x_query.device)
        except Exception:
            if neighbor_backend == "scipy":
                raise
    if neighbor_backend not in ("auto", "chunked"):
        raise ValueError("neighbor_backend must be one of: auto, cdist, chunked, scipy.")
    return nearest_training_indices_chunked(x_query, x_train, num_neighbors=k, train_chunk_size=train_chunk_size)


def nearest_training_indices_chunked(x_query: Tensor, x_train: Tensor, *, num_neighbors: int, train_chunk_size: int) -> Tensor:
    """Exact kNN scan over training chunks with bounded temporary storage."""

    k = min(int(num_neighbors), int(x_train.shape[0]))
    best_distances = torch.full((x_query.shape[0], k), float("inf"), dtype=x_query.dtype, device=x_query.device)
    best_indices = torch.zeros((x_query.shape[0], k), dtype=torch.long, device=x_query.device)
    for start in range(0, x_train.shape[0], train_chunk_size):
        chunk = x_train[start : start + train_chunk_size].to(device=x_query.device, dtype=x_query.dtype)
        distances = torch.cdist(x_query, chunk)
        candidate_distances = torch.cat((best_distances, distances), dim=1)
        candidate_indices = torch.cat(
            (
                best_indices,
                torch.arange(start, start + chunk.shape[0], dtype=torch.long, device=x_query.device)[None, :].expand(x_query.shape[0], -1),
            ),
            dim=1,
        )
        best_distances, order = torch.topk(candidate_distances, k=k, dim=1, largest=False)
        best_indices = torch.gather(candidate_indices, dim=1, index=order)
    return best_indices


@dataclass(frozen=True)
class ScalableLocalExactGPISBackend:
    """Local exact GPIS with scalable neighbor lookup and batched local solves."""

    x_train: Tensor
    y_train: Tensor
    lengthscale: float = 0.34
    variance: float = 1.0
    noise_std: float = 0.035
    mean_constant: float = 0.0
    jitter: float = 1e-6
    observation_noise_std: Tensor | None = None
    num_neighbors: int = 64
    neighbor_backend: NeighborBackendName = "auto"
    neighbor_train_chunk_size: int = 65536
    local_solve_batch_size: int = 256
    backend_name: str = "local_exact_scalable"

    @property
    def dtype(self) -> torch.dtype:
        return self.x_train.dtype

    @property
    def device(self) -> torch.device:
        return self.x_train.device

    @classmethod
    def fit(
        cls,
        x_train: Tensor,
        y_train: Tensor,
        *,
        lengthscale: float = 0.34,
        variance: float = 1.0,
        noise_std: float = 0.035,
        observation_noise_std: Tensor | None = None,
        mean_constant: float | None = None,
        jitter: float = 1e-6,
        num_neighbors: int = 64,
        neighbor_backend: NeighborBackendName = "auto",
        neighbor_train_chunk_size: int = 65536,
        local_solve_batch_size: int = 256,
    ) -> "ScalableLocalExactGPISBackend":
        x_train = x_train.detach().to(dtype=torch.float64, device="cpu")
        y_train = y_train.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        validate_training_inputs(x_train, y_train)
        if mean_constant is None:
            mean_constant = float(y_train.mean())
        return cls(
            x_train=x_train,
            y_train=y_train,
            lengthscale=float(lengthscale),
            variance=float(variance),
            noise_std=float(noise_std),
            mean_constant=float(mean_constant),
            jitter=float(jitter),
            observation_noise_std=normalize_observation_noise(observation_noise_std, y_train),
            num_neighbors=int(num_neighbors),
            neighbor_backend=neighbor_backend,
            neighbor_train_chunk_size=int(neighbor_train_chunk_size),
            local_solve_batch_size=int(local_solve_batch_size),
        )

    def predict(self, x_query: Tensor, *, batch_size: int = 8192) -> GPISPrediction:
        x_query = x_query.detach().to(dtype=self.dtype, device=self.device)
        if x_query.numel() == 0:
            return GPISPrediction(
                mean=torch.empty((0,), dtype=self.dtype, device=self.device),
                variance=torch.empty((0,), dtype=self.dtype, device=self.device),
                gradient=torch.empty((0, self.x_train.shape[1]), dtype=self.dtype, device=self.device),
            )
        means: list[Tensor] = []
        variances: list[Tensor] = []
        gradients: list[Tensor] = []
        for start in range(0, x_query.shape[0], batch_size):
            batch = x_query[start : start + batch_size]
            neighbor_indices = nearest_training_indices_scalable(
                batch,
                self.x_train,
                num_neighbors=min(self.num_neighbors, int(self.x_train.shape[0])),
                neighbor_backend=self.neighbor_backend,
                train_chunk_size=self.neighbor_train_chunk_size,
            )
            for local_start in range(0, batch.shape[0], self.local_solve_batch_size):
                local_query = batch[local_start : local_start + self.local_solve_batch_size]
                local_indices = neighbor_indices[local_start : local_start + self.local_solve_batch_size]
                mean, variance, gradient = self._predict_local_batch(local_query, local_indices)
                means.append(mean)
                variances.append(variance)
                gradients.append(gradient)
        return GPISPrediction(mean=torch.cat(means), variance=torch.cat(variances), gradient=torch.cat(gradients))

    def _predict_local_batch(self, queries: Tensor, indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_local = self.x_train[indices]
        y_local = self.y_train[indices]
        if self.observation_noise_std is None:
            noise_variance = torch.full_like(y_local, self.noise_std**2)
        else:
            noise_variance = torch.clamp(self.observation_noise_std[indices], min=1e-8).pow(2)
        centered = y_local - self.mean_constant
        b, k, dim = x_local.shape
        flat_x = x_local.reshape(b * k, dim)
        kernel = rbf_kernel(flat_x, flat_x, self.lengthscale, self.variance).reshape(b, k, b, k)
        local_kernel = kernel[torch.arange(b), :, torch.arange(b), :]
        system = local_kernel + torch.diag_embed(noise_variance)
        eye = torch.eye(k, dtype=self.dtype, device=self.device)[None, :, :]
        chol, info = torch.linalg.cholesky_ex(system + self.jitter * eye)
        if torch.any(info != 0):
            raise RuntimeError("Local batched Cholesky factorization failed.")
        alpha = torch.cholesky_solve(centered[:, :, None], chol).squeeze(-1)
        k_x = self.variance * torch.exp(-0.5 * torch.sum(((queries[:, None, :] - x_local) / self.lengthscale).pow(2), dim=2))
        mean = self.mean_constant + torch.sum(k_x * alpha, dim=1)
        solved = torch.cholesky_solve(k_x[:, :, None], chol).squeeze(-1)
        variance = torch.clamp(torch.as_tensor(self.variance, dtype=self.dtype, device=self.device) - torch.sum(k_x * solved, dim=1), min=1e-12)
        gradient = torch.sum((k_x * alpha)[:, :, None] * (x_local - queries[:, None, :]), dim=1) / (self.lengthscale**2)
        return mean, variance, gradient

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None:
        data: dict[str, object] = {
            "backend_name": np.array(self.backend_name),
            "x_train": self.x_train.detach().cpu().numpy(),
            "y_train": self.y_train.detach().cpu().numpy(),
            "lengthscale": np.array(self.lengthscale),
            "variance": np.array(self.variance),
            "noise_std": np.array(self.noise_std),
            "mean_constant": np.array(self.mean_constant),
            "jitter": np.array(self.jitter),
            "num_neighbors": np.array(self.num_neighbors, dtype=np.int64),
            "neighbor_backend": np.array(self.neighbor_backend),
            "neighbor_train_chunk_size": np.array(self.neighbor_train_chunk_size, dtype=np.int64),
            "local_solve_batch_size": np.array(self.local_solve_batch_size, dtype=np.int64),
        }
        if self.observation_noise_std is not None:
            data["observation_noise_std"] = self.observation_noise_std.detach().cpu().numpy()
        if metadata:
            for key, value in metadata.items():
                data[f"meta_{key}"] = np.array(value)
        np.savez_compressed(path, **data)


@dataclass(frozen=True)
class ScalableInducingPointGPISBackend:
    """Sparse inducing-point GPIS with optional CUDA execution and grid inducing points."""

    inducing_points: Tensor
    weight_mean: Tensor
    weight_cov: Tensor
    chol_uu: Tensor
    lengthscale: float = 0.34
    variance: float = 1.0
    noise_std: float = 0.035
    mean_constant: float = 0.0
    jitter: float = 1e-6
    training_count: int = 0
    inducing_selection: ScalableInducingSelectionName = "farthest"
    compute_device: str = "cpu"
    backend_name: str = "inducing_points_scalable"

    @property
    def dtype(self) -> torch.dtype:
        return self.inducing_points.dtype

    @property
    def device(self) -> torch.device:
        return self.inducing_points.device

    @property
    def num_inducing(self) -> int:
        return int(self.inducing_points.shape[0])

    @classmethod
    def fit(
        cls,
        x_train: Tensor,
        y_train: Tensor,
        *,
        lengthscale: float = 0.34,
        variance: float = 1.0,
        noise_std: float = 0.035,
        observation_noise_std: Tensor | None = None,
        mean_constant: float | None = None,
        jitter: float = 1e-6,
        num_inducing: int = 512,
        inducing_selection: ScalableInducingSelectionName = "farthest",
        fit_batch_size: int = 8192,
        compute_device: str = "auto",
        backend_name: str = "inducing_points_scalable",
    ) -> "ScalableInducingPointGPISBackend":
        device = resolve_compute_device(compute_device)
        x_cpu = x_train.detach().to(dtype=torch.float64, device="cpu")
        y_cpu = y_train.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        validate_training_inputs(x_cpu, y_cpu)
        observation_noise = normalize_observation_noise(observation_noise_std, y_cpu)
        if mean_constant is None:
            mean_constant = float(y_cpu.mean())
        if inducing_selection == "grid":
            inducing_cpu = make_grid_inducing_points(x_cpu, num_inducing=num_inducing)
        else:
            inducing_cpu = x_cpu[select_inducing_indices(x_cpu, num_inducing=num_inducing, method=inducing_selection)]
        inducing_points = inducing_cpu.to(device=device)
        x_train_dev = x_cpu.to(device=device)
        y_train_dev = y_cpu.to(device=device)
        observation_noise_dev = observation_noise.to(device=device) if observation_noise is not None else None
        m = int(inducing_points.shape[0])
        k_uu = rbf_kernel(inducing_points, inducing_points, lengthscale, variance)
        chol_uu = cholesky_with_jitter(k_uu, jitter=jitter, error_prefix="Scalable inducing K_uu")
        precision = torch.eye(m, dtype=torch.float64, device=device)
        rhs = torch.zeros((m,), dtype=torch.float64, device=device)
        for start in range(0, x_train_dev.shape[0], fit_batch_size):
            batch_x = x_train_dev[start : start + fit_batch_size]
            batch_y = y_train_dev[start : start + fit_batch_size] - float(mean_constant)
            k_xu = rbf_kernel(batch_x, inducing_points, lengthscale, variance)
            features = solve_lower_triangular_features(k_xu, chol_uu)
            projected_prior_variance = torch.sum(features * features, dim=1)
            residual_variance = torch.clamp(torch.as_tensor(variance, dtype=torch.float64, device=device) - projected_prior_variance, min=0.0)
            if observation_noise_dev is None:
                noise_variance = torch.full_like(batch_y, noise_std**2)
            else:
                noise_variance = torch.clamp(observation_noise_dev[start : start + fit_batch_size], min=1e-8).pow(2)
            effective_noise = torch.clamp(noise_variance + residual_variance, min=1e-12)
            weighted_features = features / effective_noise[:, None]
            precision = precision + features.T @ weighted_features
            rhs = rhs + weighted_features.T @ batch_y
        posterior_chol = cholesky_with_jitter(precision, jitter=jitter, error_prefix="Scalable inducing posterior")
        weight_mean = torch.cholesky_solve(rhs[:, None], posterior_chol).reshape(-1)
        weight_cov = torch.cholesky_inverse(posterior_chol)
        return cls(
            inducing_points=inducing_points,
            weight_mean=weight_mean,
            weight_cov=weight_cov,
            chol_uu=chol_uu,
            lengthscale=float(lengthscale),
            variance=float(variance),
            noise_std=float(noise_std),
            mean_constant=float(mean_constant),
            jitter=float(jitter),
            training_count=int(x_cpu.shape[0]),
            inducing_selection=inducing_selection,
            compute_device=str(device),
            backend_name=backend_name,
        )

    def predict(self, x_query: Tensor, *, batch_size: int = 8192) -> GPISPrediction:
        x_query = x_query.detach().to(dtype=self.dtype, device=self.device)
        means: list[Tensor] = []
        variances: list[Tensor] = []
        gradients: list[Tensor] = []
        kernel_alpha = torch.linalg.solve_triangular(self.chol_uu.T, self.weight_mean[:, None], upper=True).reshape(-1)
        prior_variance = torch.as_tensor(self.variance, dtype=self.dtype, device=self.device)
        for start in range(0, x_query.shape[0], batch_size):
            batch = x_query[start : start + batch_size]
            k_xu = rbf_kernel(batch, self.inducing_points, self.lengthscale, self.variance)
            features = solve_lower_triangular_features(k_xu, self.chol_uu)
            mean = self.mean_constant + features @ self.weight_mean
            projected_prior_variance = torch.sum(features * features, dim=1)
            posterior_projected_variance = torch.sum((features @ self.weight_cov) * features, dim=1)
            residual_variance = torch.clamp(prior_variance - projected_prior_variance, min=0.0)
            variance = torch.clamp(residual_variance + posterior_projected_variance, min=1e-12)
            diff = self.inducing_points[None, :, :] - batch[:, None, :]
            gradient = torch.sum((k_xu * kernel_alpha[None, :])[:, :, None] * diff, dim=1) / (self.lengthscale**2)
            means.append(mean.detach().cpu())
            variances.append(variance.detach().cpu())
            gradients.append(gradient.detach().cpu())
        return GPISPrediction(mean=torch.cat(means), variance=torch.cat(variances), gradient=torch.cat(gradients))

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None:
        data: dict[str, object] = {
            "backend_name": np.array(self.backend_name),
            "inducing_points": self.inducing_points.detach().cpu().numpy(),
            "weight_mean": self.weight_mean.detach().cpu().numpy(),
            "weight_cov": self.weight_cov.detach().cpu().numpy(),
            "chol_uu": self.chol_uu.detach().cpu().numpy(),
            "lengthscale": np.array(self.lengthscale),
            "variance": np.array(self.variance),
            "noise_std": np.array(self.noise_std),
            "mean_constant": np.array(self.mean_constant),
            "jitter": np.array(self.jitter),
            "training_count": np.array(self.training_count, dtype=np.int64),
            "inducing_selection": np.array(self.inducing_selection),
            "compute_device": np.array(self.compute_device),
        }
        if metadata:
            for key, value in metadata.items():
                data[f"meta_{key}"] = np.array(value)
        np.savez_compressed(path, **data)


def make_grid_inducing_points(x_train: Tensor, *, num_inducing: int) -> Tensor:
    """Create deterministic scene-bounds lattice inducing points."""

    if num_inducing < 1:
        raise ValueError("num_inducing must be positive.")
    dim = int(x_train.shape[1])
    per_axis = max(1, int(np.ceil(num_inducing ** (1.0 / dim))))
    lower = torch.min(x_train, dim=0).values
    upper = torch.max(x_train, dim=0).values
    axes = [torch.linspace(float(lower[d]), float(upper[d]), steps=per_axis, dtype=x_train.dtype, device=x_train.device) for d in range(dim)]
    mesh = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack([axis.reshape(-1) for axis in mesh], dim=1)
    if grid.shape[0] > num_inducing:
        select = torch.linspace(0, grid.shape[0] - 1, steps=num_inducing, device=x_train.device).round().to(dtype=torch.long)
        grid = grid[select]
    return grid.contiguous()


def fit_scalable_gpis_backend(
    backend: ScalableBackendName,
    x_train: Tensor,
    y_train: Tensor,
    **kwargs: object,
) -> ScalableLocalExactGPISBackend | ScalableInducingPointGPISBackend:
    """Factory for scalable backends that can be used without changing legacy code."""

    if backend == "local_exact_scalable":
        return ScalableLocalExactGPISBackend.fit(x_train, y_train, **kwargs)
    if backend == "inducing_points_scalable":
        return ScalableInducingPointGPISBackend.fit(x_train, y_train, **kwargs)
    if backend == "gpu_inducing_points":
        kwargs.setdefault("compute_device", "auto")
        return ScalableInducingPointGPISBackend.fit(x_train, y_train, backend_name="gpu_inducing_points", **kwargs)
    raise ValueError("Unknown scalable GPIS backend.")
