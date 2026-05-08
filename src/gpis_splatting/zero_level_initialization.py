from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gpis_splatting.gpis import GPISPrediction, surface_band_probability
from gpis_splatting.gpis_backends import GPISBackend, load_gpis_backend
from gpis_splatting.serialization import write_json
from gpis_splatting.splats import SplatCloud, load_splats, save_splats

Tensor = torch.Tensor
SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class ZeroLevelInitializationConfig:
    num_candidates: int = 40000
    target_count: int = 5000
    seed: int = 13
    projection_iterations: int = 4
    max_projection_step: float = 0.08
    surface_band: float = 0.03
    min_confidence: float = 0.05
    max_distance_std: float | None = None
    min_gradient_norm: float = 1e-4
    nms_radius: float = 0.015
    bounds_margin_fraction: float = 0.05
    surface_seed_fraction: float = 0.75
    seed_jitter_scale: float | None = None
    tangent_scale: float = 0.025
    normal_scale: float = 0.006
    normal_uncertainty_scale: float = 0.0
    min_scale: float = 1e-4
    max_scale: float | None = None
    tau: float = 0.45
    color_source_max_points: int = 25000
    batch_size: int = 8192


@dataclass(frozen=True)
class ZeroLevelInitializationResult:
    splats: SplatCloud
    report: dict[str, Any]


def initialize_zero_level_splats(
    backend: GPISBackend,
    *,
    config: ZeroLevelInitializationConfig | None = None,
    seed_points: Tensor | np.ndarray | None = None,
    seed_colors: Tensor | np.ndarray | None = None,
    bounds: tuple[Tensor | np.ndarray, Tensor | np.ndarray] | None = None,
) -> ZeroLevelInitializationResult:
    cfg = config or ZeroLevelInitializationConfig()
    validate_config(cfg)
    x_train, y_train = backend_training_data(backend)
    seeds = resolve_seed_points(seed_points, x_train=x_train, y_train=y_train, dtype=backend.dtype, device=backend.device)
    colors = resolve_seed_colors(seed_colors, expected_count=seeds.shape[0] if seed_points is not None else None, dtype=backend.dtype)
    lo, hi = infer_bounds(bounds=bounds, seed_points=seeds, x_train=x_train, y_train=y_train, config=cfg, dtype=backend.dtype, device=backend.device)

    candidates = sample_candidates(backend, seed_points=seeds, bounds_min=lo, bounds_max=hi, config=cfg)
    projected = project_to_zero_level(backend, candidates, bounds_min=lo, bounds_max=hi, config=cfg)
    prediction = backend.predict(projected, batch_size=cfg.batch_size)
    confidence = surface_band_probability(prediction, cfg.surface_band)
    accepted = torch.isfinite(projected).all(dim=1)
    accepted &= torch.isfinite(prediction.distance) & torch.isfinite(prediction.distance_std) & torch.isfinite(confidence)
    accepted &= prediction.distance.abs() <= cfg.surface_band
    accepted &= confidence >= cfg.min_confidence
    accepted &= prediction.grad_norm >= cfg.min_gradient_norm
    if cfg.max_distance_std is not None:
        accepted &= prediction.distance_std <= cfg.max_distance_std
    accepted_indices = torch.nonzero(accepted, as_tuple=False).reshape(-1)
    if accepted_indices.numel() == 0:
        raise ValueError("No GPIS zero-level candidates survived filtering.")

    points = projected[accepted_indices]
    pred = backend.predict(points, batch_size=cfg.batch_size)
    conf = surface_band_probability(pred, cfg.surface_band)
    score = conf / (1.0 + torch.clamp(pred.distance_std, min=0.0))
    selected = select_diverse(points, score, target_count=cfg.target_count, nms_radius=cfg.nms_radius, bounds_min=lo)
    if selected.numel() == 0:
        raise ValueError("No GPIS zero-level candidates remained after diversity selection.")

    points = points[selected]
    pred = GPISPrediction(mean=pred.mean[selected], variance=pred.variance[selected], gradient=pred.gradient[selected])
    conf = conf[selected]
    normals = normalize(pred.gradient)
    rotations = tangent_frame_quaternions(normals)
    scales = make_scales(pred.distance_std, config=cfg)
    covariances = covariance_from_scales_and_rotations(scales, rotations)
    rgb = assign_colors(points, seed_points=seeds if seed_points is not None else None, seed_colors=colors, bounds_min=lo, bounds_max=hi, config=cfg)
    tau = torch.full((points.shape[0],), cfg.tau, dtype=backend.dtype, device=backend.device)
    sigma = scales[:, :2].mean(dim=1)
    splats = SplatCloud(
        centers=points.detach().cpu(),
        colors=rgb.detach().cpu(),
        tau=tau.detach().cpu(),
        sigma=sigma.detach().cpu(),
        is_surface=torch.ones((points.shape[0],), dtype=torch.bool),
        normals=normals.detach().cpu(),
        scales=scales.detach().cpu(),
        rotations=rotations.detach().cpu(),
        covariances=covariances.detach().cpu(),
        confidence=conf.detach().cpu(),
    )
    report = make_report(
        backend=backend,
        config=cfg,
        bounds_min=lo,
        bounds_max=hi,
        candidate_count=int(candidates.shape[0]),
        accepted_count=int(accepted_indices.shape[0]),
        splats=splats,
        prediction=pred,
        used_reference_colors=colors is not None and seed_points is not None,
    )
    return ZeroLevelInitializationResult(splats=splats, report=report)


def initialize_zero_level_splats_from_files(
    *,
    model_path: str | Path,
    output_splats_path: str | Path,
    seed_splats_path: str | Path | None = None,
    output_gate_path: str | Path | None = None,
    output_report_path: str | Path | None = None,
    output_ply_path: str | Path | None = None,
    config: ZeroLevelInitializationConfig | None = None,
) -> dict[str, Any]:
    backend, metadata = load_gpis_backend(model_path)
    seed_points = None
    seed_colors = None
    if seed_splats_path is not None:
        seed_splats = load_splats(str(seed_splats_path))
        seed_points = seed_splats.centers
        seed_colors = seed_splats.colors
    result = initialize_zero_level_splats(backend, config=config, seed_points=seed_points, seed_colors=seed_colors)

    splats_path = Path(output_splats_path)
    splats_path.parent.mkdir(parents=True, exist_ok=True)
    save_splats(str(splats_path), result.splats)

    gate_path = Path(output_gate_path) if output_gate_path is not None else splats_path.with_name(f"{splats_path.stem}_confidence_gate.npz")
    confidence = result.splats.confidence.detach().cpu().numpy() if result.splats.confidence is not None else np.ones(result.splats.centers.shape[0])
    np.savez_compressed(
        gate_path,
        gate=confidence,
        raw_gate=confidence,
        model_path=np.array(str(Path(model_path))),
        splats_path=np.array(str(splats_path)),
        source=np.array("gpis_zero_level_initialization"),
    )

    ply_path = None
    if output_ply_path is not None:
        ply_path = Path(output_ply_path)
        write_zero_level_3dgs_ply(ply_path, result.splats)

    report_path = Path(output_report_path) if output_report_path is not None else splats_path.with_name(f"{splats_path.stem}_report.json")
    report = {
        **result.report,
        "model_path": str(Path(model_path)),
        "seed_splats_path": str(Path(seed_splats_path)) if seed_splats_path is not None else None,
        "output_splats_path": str(splats_path),
        "output_gate_path": str(gate_path),
        "output_ply_path": str(ply_path) if ply_path is not None else None,
        "model_metadata": {key: json_safe(value) for key, value in metadata.items()},
    }
    write_json(report_path, report)
    return {"splats_path": splats_path, "gate_path": gate_path, "report_path": report_path, "ply_path": ply_path, "report": report}


def validate_config(config: ZeroLevelInitializationConfig) -> None:
    if config.num_candidates < 1 or config.target_count < 1:
        raise ValueError("num_candidates and target_count must be positive.")
    if config.projection_iterations < 0:
        raise ValueError("projection_iterations must be non-negative.")
    if config.max_projection_step <= 0.0 or config.surface_band <= 0.0:
        raise ValueError("max_projection_step and surface_band must be positive.")
    if not 0.0 <= config.min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1].")
    if config.max_distance_std is not None and config.max_distance_std <= 0.0:
        raise ValueError("max_distance_std must be positive when set.")
    if config.nms_radius < 0.0 or not 0.0 <= config.surface_seed_fraction <= 1.0:
        raise ValueError("nms_radius must be non-negative and surface_seed_fraction must be in [0, 1].")
    if config.tangent_scale <= 0.0 or config.normal_scale <= 0.0 or config.min_scale <= 0.0:
        raise ValueError("scales must be positive.")
    if config.max_scale is not None and config.max_scale < config.min_scale:
        raise ValueError("max_scale must be >= min_scale.")
    if config.batch_size < 1 or config.tau < 0.0:
        raise ValueError("batch_size must be positive and tau must be non-negative.")


def backend_training_data(backend: GPISBackend) -> tuple[Tensor, Tensor | None]:
    if hasattr(backend, "x_train"):
        x_train = getattr(backend, "x_train")
        y_train = getattr(backend, "y_train", None)
        return x_train.detach().to(dtype=backend.dtype, device=backend.device), to_backend_tensor(y_train, backend) if y_train is not None else None
    model = getattr(backend, "model", None)
    if model is not None and hasattr(model, "x_train"):
        y_train = getattr(model, "y_train", None)
        return model.x_train.detach().to(dtype=backend.dtype, device=backend.device), to_backend_tensor(y_train, backend) if y_train is not None else None
    raise ValueError("The GPIS backend does not expose training points.")


def to_backend_tensor(value: Tensor, backend: GPISBackend) -> Tensor:
    return value.detach().to(dtype=backend.dtype, device=backend.device)


def resolve_seed_points(seed_points: Tensor | np.ndarray | None, *, x_train: Tensor, y_train: Tensor | None, dtype: torch.dtype, device: torch.device) -> Tensor:
    if seed_points is not None:
        resolved = torch.as_tensor(seed_points, dtype=dtype, device=device)
        if resolved.ndim != 2 or resolved.shape[1] != 3 or resolved.shape[0] == 0:
            raise ValueError("seed_points must have non-empty shape (N, 3).")
        return resolved
    if y_train is not None:
        threshold = max(1e-8, float(torch.median(torch.abs(y_train))) * 0.05)
        mask = torch.abs(y_train) <= threshold
        if int(mask.sum()) >= 3:
            return x_train[mask]
    return x_train


def resolve_seed_colors(seed_colors: Tensor | np.ndarray | None, *, expected_count: int | None, dtype: torch.dtype) -> Tensor | None:
    if seed_colors is None:
        return None
    colors = torch.as_tensor(seed_colors, dtype=dtype, device="cpu")
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("seed_colors must have shape (N, 3).")
    if expected_count is not None and colors.shape[0] != expected_count:
        raise ValueError("seed_colors must contain one RGB triplet per seed point.")
    return torch.clamp(colors, 0.0, 1.0)


def infer_bounds(
    *,
    bounds: tuple[Tensor | np.ndarray, Tensor | np.ndarray] | None,
    seed_points: Tensor,
    x_train: Tensor,
    y_train: Tensor | None,
    config: ZeroLevelInitializationConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if bounds is not None:
        lo = torch.as_tensor(bounds[0], dtype=dtype, device=device).reshape(3)
        hi = torch.as_tensor(bounds[1], dtype=dtype, device=device).reshape(3)
    else:
        source = seed_points
        if y_train is not None:
            threshold = max(1e-8, float(torch.median(torch.abs(y_train))) * 0.05)
            mask = torch.abs(y_train) <= threshold
            if int(mask.sum()) >= 3:
                source = x_train[mask]
        lo = torch.min(source, dim=0).values
        hi = torch.max(source, dim=0).values
    min_margin = max(config.surface_band, config.nms_radius, 1e-6)
    span = torch.clamp(hi - lo, min=min_margin)
    margin = torch.clamp(span * config.bounds_margin_fraction, min=min_margin)
    lo = lo - margin
    hi = hi + margin
    if not bool(torch.all(hi > lo)):
        raise ValueError("Invalid sampling bounds.")
    return lo, hi


def sample_candidates(backend: GPISBackend, *, seed_points: Tensor, bounds_min: Tensor, bounds_max: Tensor, config: ZeroLevelInitializationConfig) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    seed_count = min(int(round(config.num_candidates * config.surface_seed_fraction)), config.num_candidates)
    uniform_count = config.num_candidates - seed_count
    parts = []
    if seed_count > 0 and seed_points.numel() > 0:
        source_indices = torch.randint(seed_points.shape[0], (seed_count,), generator=generator)
        jitter_scale = config.seed_jitter_scale if config.seed_jitter_scale is not None else max(config.surface_band, config.nms_radius, 1e-6)
        jitter = torch.randn((seed_count, 3), generator=generator, dtype=backend.dtype) * float(jitter_scale)
        parts.append(seed_points.detach().cpu()[source_indices].to(dtype=backend.dtype) + jitter)
    if uniform_count > 0:
        random = torch.rand((uniform_count, 3), generator=generator, dtype=backend.dtype)
        parts.append(bounds_min.detach().cpu() + random * (bounds_max.detach().cpu() - bounds_min.detach().cpu()))
    candidates = torch.cat(parts, dim=0).to(dtype=backend.dtype, device=backend.device)
    return clamp_points(candidates, bounds_min=bounds_min, bounds_max=bounds_max)


def project_to_zero_level(backend: GPISBackend, points: Tensor, *, bounds_min: Tensor, bounds_max: Tensor, config: ZeroLevelInitializationConfig) -> Tensor:
    projected = points.detach().to(dtype=backend.dtype, device=backend.device)
    for _ in range(config.projection_iterations):
        pred = backend.predict(projected, batch_size=config.batch_size)
        denom = torch.clamp(torch.sum(pred.gradient * pred.gradient, dim=1), min=1e-12)
        step = (pred.mean / denom)[:, None] * pred.gradient
        step_norm = torch.linalg.norm(step, dim=1, keepdim=True)
        scale = torch.clamp(config.max_projection_step / torch.clamp(step_norm, min=1e-12), max=1.0)
        projected = clamp_points(projected - step * scale, bounds_min=bounds_min, bounds_max=bounds_max)
    return projected


def clamp_points(points: Tensor, *, bounds_min: Tensor, bounds_max: Tensor) -> Tensor:
    return torch.minimum(torch.maximum(points, bounds_min.reshape(1, 3)), bounds_max.reshape(1, 3))


def select_diverse(points: Tensor, scores: Tensor, *, target_count: int, nms_radius: float, bounds_min: Tensor) -> Tensor:
    order = torch.argsort(scores, descending=True)
    if nms_radius <= 0.0:
        return order[:target_count]
    points_np = points.detach().cpu().numpy()
    origin = bounds_min.detach().cpu().numpy()
    radius2 = nms_radius * nms_radius
    selected: list[int] = []
    grid: dict[tuple[int, int, int], list[int]] = {}
    for candidate in order.detach().cpu().numpy().tolist():
        point = points_np[int(candidate)]
        cell = tuple(np.floor((point - origin) / nms_radius).astype(np.int64).tolist())
        too_close = False
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    for existing in grid.get((cell[0] + ox, cell[1] + oy, cell[2] + oz), []):
                        delta = point - points_np[existing]
                        if float(np.dot(delta, delta)) < radius2:
                            too_close = True
                            break
                    if too_close:
                        break
                if too_close:
                    break
            if too_close:
                break
        if too_close:
            continue
        selected.append(int(candidate))
        grid.setdefault(cell, []).append(int(candidate))
        if len(selected) >= target_count:
            break
    return torch.as_tensor(selected, dtype=torch.int64, device=points.device)


def normalize(vectors: Tensor, eps: float = 1e-12) -> Tensor:
    return vectors / torch.clamp(torch.linalg.norm(vectors, dim=1, keepdim=True), min=eps)


def tangent_frame_quaternions(normals: Tensor) -> Tensor:
    normals = normalize(normals)
    helper_x = torch.tensor([1.0, 0.0, 0.0], dtype=normals.dtype, device=normals.device).expand_as(normals)
    helper_y = torch.tensor([0.0, 1.0, 0.0], dtype=normals.dtype, device=normals.device).expand_as(normals)
    helper = torch.where(normals[:, :1].abs() < 0.9, helper_x, helper_y)
    tangent_1 = normalize(torch.cross(helper, normals, dim=1))
    tangent_2 = normalize(torch.cross(normals, tangent_1, dim=1))
    rotations = torch.stack((tangent_1, tangent_2, normals), dim=2)
    return rotation_matrices_to_quaternions(rotations)


def rotation_matrices_to_quaternions(rotations: Tensor) -> Tensor:
    matrices = rotations.detach().cpu().numpy().astype(np.float64)
    quaternions = np.empty((matrices.shape[0], 4), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quat = np.array([0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale])
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-12)) * 2.0
                quat = np.array([(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale])
            elif axis == 1:
                scale = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-12)) * 2.0
                quat = np.array([(matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale])
            else:
                scale = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-12)) * 2.0
                quat = np.array([(matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale])
        quaternions[index] = quat / max(np.linalg.norm(quat), 1e-12)
    return torch.from_numpy(quaternions).to(dtype=rotations.dtype, device=rotations.device)


def quaternion_to_rotation_matrices(quaternions: Tensor) -> Tensor:
    q = quaternions / torch.clamp(torch.linalg.norm(quaternions, dim=1, keepdim=True), min=1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        (
            torch.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)), dim=1),
            torch.stack((2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)), dim=1),
            torch.stack((2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)), dim=1),
        ),
        dim=1,
    )


def make_scales(distance_std: Tensor, *, config: ZeroLevelInitializationConfig) -> Tensor:
    tangent = torch.full_like(distance_std, config.tangent_scale)
    normal = torch.full_like(distance_std, config.normal_scale)
    if config.normal_uncertainty_scale > 0.0:
        normal = normal + torch.clamp(distance_std, min=0.0) * config.normal_uncertainty_scale
    tangent = torch.clamp(tangent, min=config.min_scale)
    normal = torch.clamp(normal, min=config.min_scale)
    if config.max_scale is not None:
        tangent = torch.clamp(tangent, max=config.max_scale)
        normal = torch.clamp(normal, max=config.max_scale)
    return torch.stack((tangent, tangent, normal), dim=1)


def covariance_from_scales_and_rotations(scales: Tensor, rotations: Tensor) -> Tensor:
    matrices = quaternion_to_rotation_matrices(rotations)
    return matrices @ torch.diag_embed(scales.pow(2)) @ matrices.transpose(1, 2)


def assign_colors(
    centers: Tensor,
    *,
    seed_points: Tensor | None,
    seed_colors: Tensor | None,
    bounds_min: Tensor,
    bounds_max: Tensor,
    config: ZeroLevelInitializationConfig,
) -> Tensor:
    if seed_points is not None and seed_colors is not None and seed_points.shape[0] == seed_colors.shape[0]:
        return nearest_neighbor_colors(centers, seed_points=seed_points, seed_colors=seed_colors, max_points=config.color_source_max_points, seed=config.seed)
    normalized = torch.clamp((centers - bounds_min.reshape(1, 3)) / torch.clamp(bounds_max - bounds_min, min=1e-9).reshape(1, 3), 0.0, 1.0)
    return torch.stack((normalized[:, 0], 0.35 + 0.5 * normalized[:, 1], 1.0 - 0.65 * normalized[:, 2]), dim=1).clamp(0.0, 1.0)


def nearest_neighbor_colors(centers: Tensor, *, seed_points: Tensor, seed_colors: Tensor, max_points: int, seed: int) -> Tensor:
    source_points = seed_points.detach().to(dtype=centers.dtype, device=centers.device)
    source_colors = seed_colors.detach().to(dtype=centers.dtype, device=centers.device)
    if max_points > 0 and source_points.shape[0] > max_points:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        indices = torch.randperm(source_points.shape[0], generator=generator)[:max_points].to(device=centers.device)
        source_points = source_points[indices]
        source_colors = source_colors[indices]
    assigned = []
    for start in range(0, centers.shape[0], 4096):
        nearest = torch.argmin(torch.cdist(centers[start : start + 4096], source_points), dim=1)
        assigned.append(source_colors[nearest])
    return torch.cat(assigned, dim=0)


def make_report(
    *,
    backend: GPISBackend,
    config: ZeroLevelInitializationConfig,
    bounds_min: Tensor,
    bounds_max: Tensor,
    candidate_count: int,
    accepted_count: int,
    splats: SplatCloud,
    prediction: GPISPrediction,
    used_reference_colors: bool,
) -> dict[str, Any]:
    confidence = splats.confidence.detach().cpu().numpy() if splats.confidence is not None else np.asarray([])
    scales = splats.scales.detach().cpu().numpy() if splats.scales is not None else np.asarray([])
    distance = prediction.distance.detach().cpu().numpy()
    distance_std = prediction.distance_std.detach().cpu().numpy()
    grad_norm = prediction.grad_norm.detach().cpu().numpy()
    return {
        "schema_version": 1,
        "initializer": "gpis_zero_level",
        "backend": getattr(backend, "backend_name", type(backend).__name__),
        "candidate_count": candidate_count,
        "accepted_before_nms_count": accepted_count,
        "selected_splat_count": int(splats.centers.shape[0]),
        "target_count": int(config.target_count),
        "seed": int(config.seed),
        "surface_band": float(config.surface_band),
        "min_confidence": float(config.min_confidence),
        "max_distance_std": None if config.max_distance_std is None else float(config.max_distance_std),
        "min_gradient_norm": float(config.min_gradient_norm),
        "nms_radius": float(config.nms_radius),
        "projection_iterations": int(config.projection_iterations),
        "max_projection_step": float(config.max_projection_step),
        "bounds_min": bounds_min.detach().cpu().numpy().tolist(),
        "bounds_max": bounds_max.detach().cpu().numpy().tolist(),
        "used_reference_colors": bool(used_reference_colors),
        "confidence_min": float(confidence.min()) if confidence.size else None,
        "confidence_max": float(confidence.max()) if confidence.size else None,
        "confidence_mean": float(confidence.mean()) if confidence.size else None,
        "abs_distance_mean": float(np.mean(np.abs(distance))) if distance.size else None,
        "abs_distance_max": float(np.max(np.abs(distance))) if distance.size else None,
        "distance_std_mean": float(distance_std.mean()) if distance_std.size else None,
        "distance_std_max": float(distance_std.max()) if distance_std.size else None,
        "grad_norm_min": float(grad_norm.min()) if grad_norm.size else None,
        "grad_norm_mean": float(grad_norm.mean()) if grad_norm.size else None,
        "tangent_scale": float(config.tangent_scale),
        "normal_scale": float(config.normal_scale),
        "normal_uncertainty_scale": float(config.normal_uncertainty_scale),
        "scale_min": float(scales.min()) if scales.size else None,
        "scale_max": float(scales.max()) if scales.size else None,
        "tau": float(config.tau),
    }


def write_zero_level_3dgs_ply(path: str | Path, splats: SplatCloud, *, sh_degree: int = 3) -> None:
    if splats.normals is None or splats.scales is None or splats.rotations is None:
        raise ValueError("Zero-level 3DGS PLY export requires splat normals, scales, and rotations.")
    centers = splats.centers.detach().cpu().numpy().astype(np.float32)
    normals = splats.normals.detach().cpu().numpy().astype(np.float32)
    colors = np.clip(splats.colors.detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)
    tau = splats.tau.detach().cpu().numpy().astype(np.float64)
    scales = np.clip(splats.scales.detach().cpu().numpy(), 1e-8, None).astype(np.float64)
    rotations = splats.rotations.detach().cpu().numpy().astype(np.float64)
    rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-12)
    rest_count = 3 * (((sh_degree + 1) ** 2) - 1)
    fields = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("f_dc_0", "<f4"),
        ("f_dc_1", "<f4"),
        ("f_dc_2", "<f4"),
        *[(f"f_rest_{index}", "<f4") for index in range(rest_count)],
        ("opacity", "<f4"),
        ("scale_0", "<f4"),
        ("scale_1", "<f4"),
        ("scale_2", "<f4"),
        ("rot_0", "<f4"),
        ("rot_1", "<f4"),
        ("rot_2", "<f4"),
        ("rot_3", "<f4"),
    ]
    vertices = np.zeros((centers.shape[0],), dtype=np.dtype(fields))
    vertices["x"], vertices["y"], vertices["z"] = centers[:, 0], centers[:, 1], centers[:, 2]
    vertices["nx"], vertices["ny"], vertices["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    dc = (colors.astype(np.float64) - 0.5) / SH_C0
    vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = dc[:, 0].astype(np.float32), dc[:, 1].astype(np.float32), dc[:, 2].astype(np.float32)
    alpha = np.clip(1.0 - np.exp(-np.clip(tau, 0.0, None)), 1e-6, 1.0 - 1e-6)
    vertices["opacity"] = np.log(alpha / (1.0 - alpha)).astype(np.float32)
    log_scales = np.log(scales).astype(np.float32)
    vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = log_scales[:, 0], log_scales[:, 1], log_scales[:, 2]
    vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"] = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {vertices.shape[0]}"]
    header.extend(f"property float {name}" for name, _field_type in fields)
    header.extend(["end_header", ""])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes("\n".join(header).encode("ascii") + vertices.tobytes())


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
