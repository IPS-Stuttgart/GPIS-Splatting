from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gpis_splatting.confidence import format_threshold_label, read_calibrated_confidence_bundle
from gpis_splatting.gpis import GPISModel, GPISPrediction, predict_gpis, surface_band_probability
from gpis_splatting.gpis_backends import GPISBackend, load_gpis_backend
from gpis_splatting.real_scene import load_prepared_scene
from gpis_splatting.serialization import write_json
from gpis_splatting.splats import SplatCloud, load_splats, save_splats

SH_C0 = 0.28209479177387814
EPS = 1e-12


@dataclass(frozen=True)
class GPISAwareInitializationConfig:
    target_count: int | None = None
    proposals_per_seed: int = 3
    include_seed_points: bool = True
    jitter_std: float | None = None
    projection_iterations: int = 4
    max_projection_step: float | None = None
    epsilon: float = 0.08
    max_abs_distance: float | None = None
    max_distance_std: float | None = None
    min_surface_probability: float = 0.0
    min_grad_norm: float = 1e-5
    min_view_count: int = 0
    min_separation: float | None = None
    normal_scale: float | None = None
    tangent_scale: float | None = None
    normal_scale_factor: float = 0.20
    tangent_scale_factor: float = 0.80
    scale_from_uncertainty: bool = True
    max_uncertainty_scale_multiplier: float = 3.0
    opacity: float = 0.55
    opacity_confidence_power: float = 1.0
    min_opacity: float = 0.02
    max_opacity: float = 0.95
    batch_size: int = 4096
    seed: int = 13


@dataclass(frozen=True)
class GPISAwareInitialization:
    centers: np.ndarray
    colors: np.ndarray
    opacity: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    normals: np.ndarray
    confidence: np.ndarray
    surface_probability: np.ndarray
    distance_std: np.ndarray
    view_count: np.ndarray
    source_index: np.ndarray
    selected_candidate_index: np.ndarray
    field_scores: pd.DataFrame = field(repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.centers.shape[0])

    def to_splat_cloud(self) -> SplatCloud:
        return SplatCloud(
            centers=torch.from_numpy(self.centers).to(dtype=torch.float64),
            colors=torch.from_numpy(self.colors).to(dtype=torch.float64),
            tau=torch.from_numpy(self.opacity).to(dtype=torch.float64),
            sigma=torch.from_numpy(np.mean(self.scales, axis=1)).to(dtype=torch.float64),
            is_surface=torch.ones((self.count,), dtype=torch.bool),
        )


def initialize_gpis_aware_gaussians(
    model_or_backend: GPISModel | GPISBackend,
    seed_points: np.ndarray | torch.Tensor,
    *,
    seed_colors: np.ndarray | torch.Tensor | None = None,
    frames: list[dict[str, Any]] | None = None,
    scene_meta: dict[str, Any] | None = None,
    projection_convention: str = "auto",
    near_plane: float = 1e-4,
    confidence_model_path: str | Path | None = None,
    confidence_threshold: float | None = None,
    config: GPISAwareInitializationConfig | None = None,
) -> GPISAwareInitialization:
    cfg = config or GPISAwareInitializationConfig()
    validate_config(cfg, near_plane=near_plane)
    seeds = points_array(seed_points, "seed_points")
    colors = color_array(seed_colors, seeds.shape[0])
    spacing = point_spacing(seeds, cfg.seed)
    proposals, source_index, proposal_kind = proposals_from_seeds(seeds, spacing, cfg)
    projected, prediction = project_to_zero_level(model_or_backend, proposals, cfg)
    band = surface_band_probability(prediction, cfg.epsilon).detach().cpu().numpy()
    views = view_counts(projected, frames or [], scene_meta=scene_meta, convention=projection_convention, near_plane=near_plane)
    table = field_table(projected, prediction, band, source_index, proposal_kind, views, cfg.epsilon)
    confidence, confidence_source, threshold, predictions = candidate_confidence(table, confidence_model_path, confidence_threshold, cfg.epsilon)
    table["confidence"] = confidence
    if predictions is not None:
        for column in predictions.columns:
            if column not in table.columns:
                table[column] = predictions[column].to_numpy()
    selected = select_candidates(table, confidence, cfg)
    if selected.size == 0:
        raise ValueError("No GPIS-aware initialization candidates survived filtering.")

    selected_prediction = take_prediction(prediction, selected)
    normals = normalize_rows(selected_prediction.gradient.detach().cpu().numpy())
    distance_std = table.loc[selected, "distance_std"].to_numpy(dtype=np.float64)
    selected_confidence = np.clip(confidence[selected], 0.0, 1.0)
    selected_table = table.assign(selected=False, selection_rank=-1)
    selected_table.loc[selected, "selected"] = True
    selected_table.loc[selected, "selection_rank"] = np.arange(selected.shape[0], dtype=np.int64)
    return GPISAwareInitialization(
        centers=projected[selected].astype(np.float64),
        colors=np.clip(colors[source_index[selected]], 0.0, 1.0).astype(np.float64),
        opacity=confidence_to_opacity(selected_confidence, cfg).astype(np.float64),
        scales=scales_from_uncertainty(distance_std, spacing, cfg).astype(np.float64),
        rotations=quaternions_from_normals(normals).astype(np.float64),
        normals=normals.astype(np.float64),
        confidence=selected_confidence.astype(np.float64),
        surface_probability=band[selected].astype(np.float64),
        distance_std=distance_std.astype(np.float64),
        view_count=views[selected].astype(np.int64),
        source_index=source_index[selected].astype(np.int64),
        selected_candidate_index=selected.astype(np.int64),
        field_scores=selected_table,
        metadata={
            "schema_version": 1,
            "input_seed_count": int(seeds.shape[0]),
            "proposal_count": int(proposals.shape[0]),
            "selected_count": int(selected.shape[0]),
            "estimated_seed_spacing": float(spacing),
            "confidence_source": confidence_source,
            "confidence_threshold": threshold,
            "config": json_ready(asdict(cfg)),
        },
    )


def run_gpis_aware_initialization(
    *,
    scene_dir: str | Path,
    model_path: str | Path | None = None,
    splats_path: str | Path | None = None,
    output_prefix: str = "gpis_init",
    output_dir: str | Path | None = None,
    confidence_model_path: str | Path | None = None,
    confidence_threshold: float | None = None,
    projection_convention: str = "auto",
    near_plane: float = 1e-4,
    sh_degree: int = 3,
    config: GPISAwareInitializationConfig | None = None,
) -> dict[str, Any]:
    scene_root = Path(scene_dir)
    scene_meta, frames, splits = load_prepared_scene(scene_root)
    backend, metadata = load_gpis_backend(resolve_scene_file(scene_root, model_path, "real_gpis_model.npz"))
    splats_file = resolve_scene_file(scene_root, splats_path, "real_splats.npz")
    splats = load_splats(str(splats_file))
    train = [frames[int(i)] for i in splits.get("train", []) if 0 <= int(i) < len(frames)] or frames
    init = initialize_gpis_aware_gaussians(
        backend,
        splats.centers.detach().cpu().numpy(),
        seed_colors=splats.colors.detach().cpu().numpy(),
        frames=train,
        scene_meta=scene_meta,
        projection_convention=projection_convention,
        near_plane=near_plane,
        confidence_model_path=confidence_model_path,
        confidence_threshold=confidence_threshold,
        config=config,
    )
    out_dir = Path(output_dir) if output_dir is not None else scene_root
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = out_dir / f"{output_prefix}_gaussians.npz"
    splats_path_out = out_dir / f"{output_prefix}_splats.npz"
    ply_path = out_dir / f"{output_prefix}_3dgs.ply"
    field_scores_path = out_dir / f"{output_prefix}_field_scores.csv"
    status_path = out_dir / f"{output_prefix}_status.json"
    report_path = out_dir / f"{output_prefix}_report.md"
    save_gpis_aware_initialization(arrays_path, init)
    save_splats(str(splats_path_out), init.to_splat_cloud())
    write_3dgs_initialization_ply(ply_path, init, sh_degree=sh_degree)
    init.field_scores.to_csv(field_scores_path, index=False)
    status = {
        **init.metadata,
        "scene": scene_meta.get("scene"),
        "scene_dir": str(scene_root),
        "model_metadata": json_ready(metadata),
        "output_prefix": output_prefix,
        "output_dir": str(out_dir),
        "train_frame_count": int(len(train)),
        "projection_convention": resolve_projection_convention(scene_meta, projection_convention),
        "arrays_path": str(arrays_path),
        "splats_path_out": str(splats_path_out),
        "ply_path": str(ply_path),
        "field_scores_path": str(field_scores_path),
        "report_path": str(report_path),
    }
    write_json(status_path, status)
    report_path.write_text(format_initialization_report(status), encoding="utf-8")
    return {
        "arrays_path": arrays_path,
        "splats_path": splats_path_out,
        "ply_path": ply_path,
        "field_scores_path": field_scores_path,
        "status_path": status_path,
        "report_path": report_path,
        "initialization": init,
        "status": status,
    }


def validate_config(config: GPISAwareInitializationConfig, *, near_plane: float) -> None:
    if near_plane <= 0.0 or config.epsilon <= 0.0 or config.batch_size < 1:
        raise ValueError("near_plane, epsilon, and batch_size must be positive.")
    if config.target_count is not None and config.target_count < 1:
        raise ValueError("target_count must be positive when provided.")
    if config.proposals_per_seed < 0 or config.projection_iterations < 0:
        raise ValueError("proposal and projection counts must be non-negative.")
    if not config.include_seed_points and config.proposals_per_seed == 0:
        raise ValueError("At least one proposal source is required.")
    if config.min_view_count < 0 or config.min_grad_norm < 0.0 or not 0.0 <= config.min_surface_probability <= 1.0:
        raise ValueError("minimum filters are invalid.")
    if not 0.0 < config.opacity <= 1.0 or config.opacity_confidence_power <= 0.0:
        raise ValueError("opacity parameters are invalid.")
    if not 0.0 <= config.min_opacity <= config.max_opacity <= 1.0:
        raise ValueError("opacity bounds are invalid.")


def points_array(points: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    array = points.detach().cpu().numpy() if isinstance(points, torch.Tensor) else np.asarray(points)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape (N, 3) and N > 0.")
    return array


def color_array(colors: np.ndarray | torch.Tensor | None, count: int) -> np.ndarray:
    if colors is None:
        return np.full((count, 3), 0.7, dtype=np.float64)
    array = colors.detach().cpu().numpy() if isinstance(colors, torch.Tensor) else np.asarray(colors)
    if array.shape != (count, 3):
        raise ValueError("seed_colors must have shape (N, 3).")
    return np.clip(np.asarray(array, dtype=np.float64), 0.0, 1.0)


def point_spacing(points: np.ndarray, seed: int, max_points: int = 1024) -> float:
    if points.shape[0] < 2:
        return 0.025
    rng = np.random.default_rng(seed)
    sample = points[np.sort(rng.choice(points.shape[0], size=max_points, replace=False))] if points.shape[0] > max_points else points
    distances = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    finite = np.min(distances, axis=1)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    return float(np.median(finite)) if finite.size else 0.025


def proposals_from_seeds(points: np.ndarray, spacing: float, config: GPISAwareInitializationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    proposals: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    kinds: list[np.ndarray] = []
    if config.include_seed_points:
        proposals.append(points.copy())
        sources.append(np.arange(points.shape[0], dtype=np.int64))
        kinds.append(np.full((points.shape[0],), "seed", dtype=object))
    jitter_std = float(config.jitter_std if config.jitter_std is not None else max(spacing, EPS) * 0.5)
    for index in range(config.proposals_per_seed):
        proposals.append(points + rng.normal(scale=jitter_std, size=points.shape))
        sources.append(np.arange(points.shape[0], dtype=np.int64))
        kinds.append(np.full((points.shape[0],), f"jitter_{index}", dtype=object))
    return np.vstack(proposals), np.concatenate(sources), np.concatenate(kinds)


def project_to_zero_level(model_or_backend: GPISModel | GPISBackend, proposals: np.ndarray, config: GPISAwareInitializationConfig) -> tuple[np.ndarray, GPISPrediction]:
    points = proposals.astype(np.float64, copy=True)
    prediction = predict_any(model_or_backend, points, config.batch_size)
    for _ in range(config.projection_iterations):
        gradient = prediction.gradient.detach().cpu().numpy()
        mean = prediction.mean.detach().cpu().numpy()
        step = (mean / (np.sum(gradient * gradient, axis=1) + EPS))[:, None] * gradient
        if config.max_projection_step is not None:
            step *= np.minimum(1.0, float(config.max_projection_step) / np.maximum(np.linalg.norm(step, axis=1), EPS))[:, None]
        points -= step
        prediction = predict_any(model_or_backend, points, config.batch_size)
    return points, prediction


def predict_any(model_or_backend: GPISModel | GPISBackend, points: np.ndarray, batch_size: int) -> GPISPrediction:
    query = torch.from_numpy(np.asarray(points, dtype=np.float64))
    return predict_gpis(model_or_backend, query, batch_size=batch_size) if isinstance(model_or_backend, GPISModel) else model_or_backend.predict(query, batch_size=batch_size)


def field_table(points: np.ndarray, prediction: GPISPrediction, band: np.ndarray, source: np.ndarray, kind: np.ndarray, views: np.ndarray, epsilon: float) -> pd.DataFrame:
    mean = prediction.mean.detach().cpu().numpy()
    variance = prediction.variance.detach().cpu().numpy()
    sigma = prediction.std.detach().cpu().numpy()
    grad_norm = prediction.grad_norm.detach().cpu().numpy()
    distance = prediction.distance.detach().cpu().numpy()
    distance_std = prediction.distance_std.detach().cpu().numpy()
    abs_distance = np.abs(distance)
    safe_epsilon = max(float(epsilon), EPS)
    return pd.DataFrame(
        {
            "splat_index": np.arange(points.shape[0], dtype=np.int64),
            "candidate_index": np.arange(points.shape[0], dtype=np.int64),
            "source_splat_index": source.astype(np.int64),
            "candidate_type": kind.astype(str),
            "query_x": points[:, 0],
            "query_y": points[:, 1],
            "query_z": points[:, 2],
            "eval_x": points[:, 0],
            "eval_y": points[:, 1],
            "eval_z": points[:, 2],
            "mu": mean,
            "variance": variance,
            "sigma": sigma,
            "grad_norm": grad_norm,
            "signed_distance": distance,
            "abs_signed_distance": abs_distance,
            "distance_std": distance_std,
            "view_count": views.astype(np.int64),
            "score_current_gate": band,
            "score_raw_surface_band": band,
            "score_negative_abs_distance": -abs_distance,
            "score_negative_distance_std": -distance_std,
            "score_variance_penalized_band": band / (1.0 + distance_std / safe_epsilon),
            "score_variance_penalized_exp": np.exp(-abs_distance / safe_epsilon) / (1.0 + distance_std / safe_epsilon),
            "score_negative_abs_mu": -np.abs(mean),
            "score_view_count": views.astype(np.float64),
        }
    )


def candidate_confidence(table: pd.DataFrame, model_path: str | Path | None, threshold: float | None, epsilon: float) -> tuple[np.ndarray, str, float | None, pd.DataFrame | None]:
    if model_path is None:
        return np.clip(table["score_variance_penalized_band"].to_numpy(dtype=np.float64), 0.0, 1.0), "variance_penalized_gpis_band", None, None
    from gpis_splatting.calibrated_confidence_api import ConfidenceFeatureConfig, ConfidenceFeatureExtractor

    bundle = read_calibrated_confidence_bundle(model_path)
    chosen = threshold if threshold is not None else min(bundle.thresholds, key=lambda value: abs(value - epsilon))
    extracted = ConfidenceFeatureExtractor(ConfidenceFeatureConfig()).fit_transform(table)
    predictions = bundle.predict(extracted.table, threshold=float(chosen))
    column = f"confidence_{format_threshold_label(float(chosen))}"
    return np.clip(predictions[column].to_numpy(dtype=np.float64), 0.0, 1.0), "calibrated_confidence", float(chosen), predictions


def select_candidates(table: pd.DataFrame, confidence: np.ndarray, config: GPISAwareInitializationConfig) -> np.ndarray:
    abs_distance = table["abs_signed_distance"].to_numpy(dtype=np.float64)
    distance_std = table["distance_std"].to_numpy(dtype=np.float64)
    grad_norm = table["grad_norm"].to_numpy(dtype=np.float64)
    views = table["view_count"].to_numpy(dtype=np.int64)
    max_abs_distance = float(config.max_abs_distance if config.max_abs_distance is not None else config.epsilon)
    mask = (abs_distance <= max_abs_distance) & (grad_norm >= config.min_grad_norm) & (views >= config.min_view_count)
    mask &= table["score_raw_surface_band"].to_numpy(dtype=np.float64) >= config.min_surface_probability
    if config.max_distance_std is not None:
        mask &= distance_std <= config.max_distance_std
    candidates = np.flatnonzero(mask & np.isfinite(confidence))
    ordered = candidates[np.lexsort((distance_std[candidates], -views[candidates], -confidence[candidates]))]
    return spatial_subsample(table[["query_x", "query_y", "query_z"]].to_numpy(dtype=np.float64), ordered, config.min_separation, config.target_count)


def spatial_subsample(points: np.ndarray, ordered: np.ndarray, min_separation: float | None, target_count: int | None) -> np.ndarray:
    if min_separation is None:
        return ordered if target_count is None else ordered[:target_count]
    chosen: list[int] = []
    for index in ordered:
        if all(np.linalg.norm(points[int(index)] - points[old]) >= min_separation for old in chosen):
            chosen.append(int(index))
            if target_count is not None and len(chosen) >= target_count:
                break
    return np.asarray(chosen, dtype=np.int64)


def take_prediction(prediction: GPISPrediction, indices: np.ndarray) -> GPISPrediction:
    idx = torch.as_tensor(indices, dtype=torch.long, device=prediction.mean.device)
    return GPISPrediction(mean=prediction.mean[idx], variance=prediction.variance[idx], gradient=prediction.gradient[idx])


def scales_from_uncertainty(distance_std: np.ndarray, spacing: float, config: GPISAwareInitializationConfig) -> np.ndarray:
    normal = float(config.normal_scale if config.normal_scale is not None else max(spacing * config.normal_scale_factor, EPS))
    tangent = float(config.tangent_scale if config.tangent_scale is not None else max(spacing * config.tangent_scale_factor, EPS))
    multiplier = np.ones_like(distance_std)
    if config.scale_from_uncertainty:
        multiplier += np.clip(distance_std / max(config.epsilon, EPS), 0.0, config.max_uncertainty_scale_multiplier)
    return np.column_stack((normal * (1.0 + 0.25 * (multiplier - 1.0)), tangent * (1.0 + 0.5 * (multiplier - 1.0)), tangent * (1.0 + 0.5 * (multiplier - 1.0))))


def confidence_to_opacity(confidence: np.ndarray, config: GPISAwareInitializationConfig) -> np.ndarray:
    opacity = float(config.opacity) * np.power(np.clip(confidence, 0.0, 1.0), config.opacity_confidence_power)
    return np.clip(opacity, config.min_opacity, config.max_opacity)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    fallback = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64), (values.shape[0], 1))
    return np.where(norm > EPS, values / np.maximum(norm, EPS), fallback)


def quaternions_from_normals(normals: np.ndarray) -> np.ndarray:
    rows = []
    for normal in normalize_rows(normals):
        reference = np.asarray([0.0, 0.0, 1.0]) if abs(float(normal[2])) <= 0.9 else np.asarray([0.0, 1.0, 0.0])
        tangent_1 = np.cross(reference, normal)
        tangent_1 /= max(float(np.linalg.norm(tangent_1)), EPS)
        tangent_2 = np.cross(normal, tangent_1)
        matrix = np.column_stack((normal, tangent_1, tangent_2))
        rows.append(quaternion_from_matrix(matrix))
    quats = np.asarray(rows, dtype=np.float64)
    return quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), EPS)


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.asarray([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.asarray([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.asarray([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.asarray([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def view_counts(points: np.ndarray, frames: list[dict[str, Any]], *, scene_meta: dict[str, Any] | None, convention: str, near_plane: float) -> np.ndarray:
    counts = np.zeros((points.shape[0],), dtype=np.int64)
    if not frames:
        return counts
    convention = resolve_projection_convention(scene_meta, convention)
    homogeneous = np.column_stack((points, np.ones((points.shape[0],), dtype=np.float64)))
    for frame in frames:
        if frame.get("world_to_camera") is None:
            continue
        intrinsics = frame.get("intrinsics") or {}
        width, height = frame.get("width") or intrinsics.get("width"), frame.get("height") or intrinsics.get("height")
        fx, fy, cx, cy = (intrinsics.get(key) for key in ("fx", "fy", "cx", "cy"))
        if width is None or height is None or any(value is None for value in (fx, fy, cx, cy)):
            continue
        camera = (homogeneous @ np.asarray(frame["world_to_camera"], dtype=np.float64).T)[:, :3]
        with np.errstate(divide="ignore", invalid="ignore"):
            if convention == "opencv":
                depth = camera[:, 2]
                u = float(fx) * camera[:, 0] / depth + float(cx)
                v = float(fy) * camera[:, 1] / depth + float(cy)
            else:
                depth = -camera[:, 2]
                u = float(fx) * camera[:, 0] / depth + float(cx)
                v = float(cy) - float(fy) * camera[:, 1] / depth
        visible = (depth > near_plane) & np.isfinite(u) & np.isfinite(v) & (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))
        counts += visible.astype(np.int64)
    return counts


def resolve_projection_convention(scene_meta: dict[str, Any] | None, requested: str) -> str:
    if requested != "auto":
        return requested
    return "opengl" if scene_meta is not None and scene_meta.get("source_format") == "transforms" else "opencv"


def save_gpis_aware_initialization(path: str | Path, initialization: GPISAwareInitialization) -> None:
    np.savez_compressed(
        path,
        centers=initialization.centers,
        colors=initialization.colors,
        opacity=initialization.opacity,
        tau=initialization.opacity,
        sigma=np.mean(initialization.scales, axis=1),
        scales=initialization.scales,
        rotations=initialization.rotations,
        normals=initialization.normals,
        confidence=initialization.confidence,
        surface_probability=initialization.surface_probability,
        distance_std=initialization.distance_std,
        view_count=initialization.view_count,
        source_index=initialization.source_index,
        selected_candidate_index=initialization.selected_candidate_index,
        schema_version=np.asarray(1, dtype=np.int64),
    )


def write_3dgs_initialization_ply(path: str | Path, initialization: GPISAwareInitialization, *, sh_degree: int = 3) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rest_count = 3 * ((int(sh_degree) + 1) ** 2 - 1)
    headers = ["ply", "format ascii 1.0", f"element vertex {initialization.count}"]
    headers += [f"property float {name}" for name in ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2")]
    headers += [f"property float f_rest_{index}" for index in range(rest_count)]
    headers += [f"property float {name}" for name in ("opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")]
    headers.append("end_header")
    log_opacity = alpha_to_logit(initialization.opacity)
    log_scales = np.log(np.clip(initialization.scales, 1e-8, None))
    rows = []
    for index in range(initialization.count):
        dc = (np.clip(initialization.colors[index], 0.0, 1.0) - 0.5) / SH_C0
        values = [
            *initialization.centers[index],
            *initialization.normals[index],
            *dc,
            *([0.0] * rest_count),
            log_opacity[index],
            *log_scales[index],
            *initialization.rotations[index],
        ]
        rows.append(" ".join(f"{float(value):.9g}" for value in values))
    output.write_text("\n".join([*headers, *rows]) + "\n", encoding="ascii")


def alpha_to_logit(alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(alpha / (1.0 - alpha))


def format_initialization_report(status: dict[str, Any]) -> str:
    lines = [
        "# GPIS-Aware Gaussian Initialization",
        "",
        f"- Scene: `{status.get('scene')}`",
        f"- Input seeds: `{status['input_seed_count']}`",
        f"- Proposals: `{status['proposal_count']}`",
        f"- Selected Gaussians: `{status['selected_count']}`",
        f"- Confidence source: `{status['confidence_source']}`",
        f"- Estimated seed spacing: `{status['estimated_seed_spacing']:.6g}`",
        f"- 3DGS PLY: `{status['ply_path']}`",
        f"- Internal splats: `{status['splats_path_out']}`",
        f"- Field scores: `{status['field_scores_path']}`",
        "",
        "The PLY stores GPIS normals, anisotropic log-scales, and scalar-first quaternions.",
    ]
    return "\n".join(lines) + "\n"


def resolve_scene_file(scene_root: Path, path: str | Path | None, default_name: str) -> Path:
    if path is None:
        return scene_root / default_name
    resolved = Path(path)
    return scene_root / resolved if not resolved.is_absolute() else resolved


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
