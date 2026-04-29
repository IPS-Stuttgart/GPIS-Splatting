from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gpis_splatting.real_field_scores import run_tanks_temples_gpis_field_score_diagnostics
from gpis_splatting.real_geometry import (
    crop_mask,
    deterministic_subsample,
    load_alignment_matrix,
    resolve_optional_scene_file,
    resolve_scene_file,
    transform_points,
)
from gpis_splatting.real_scene import load_prepared_scene
from gpis_splatting.real_score_calibration import run_gpis_splat_score_calibration
from gpis_splatting.serialization import read_json, write_json
from gpis_splatting.splats import SplatCloud, load_splats, save_splats


@dataclass(frozen=True)
class HardNegativeOutputs:
    splats_path: Path
    candidates_path: Path
    status_path: Path
    report_path: Path
    candidate_table: pd.DataFrame
    status: dict[str, Any]


def run_tanks_temples_hard_negative_calibration(
    *,
    scene_dir: str | Path,
    model_path: str | Path,
    splats_path: str | Path | None = None,
    method_name: str = "hard_negative",
    output_dir: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
    crop_path: str | Path | None = None,
    max_source_splats: int | None = 5000,
    seed: int = 13,
    include_source: bool = True,
    jitter_copies: int = 1,
    ray_copies: int = 1,
    behind_copies: int = 1,
    random_count: int | None = None,
    jitter_std: float = 0.03,
    ray_shift_min: float = 0.03,
    ray_shift_max: float = 0.15,
    behind_shift_min: float = 0.08,
    behind_shift_max: float = 0.25,
    random_bounds_scale: float = 1.25,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.1),
    topk_fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
    calibration_validation_fraction: float = 0.35,
    max_pred_points: int | None = 100_000,
    max_gt_points: int | None = 100_000,
    apply_alignment: bool | None = None,
    invert_alignment: bool = False,
    use_crop: bool = True,
    epsilon: float = 0.24,
    gate_floor: float = 0.0,
    batch_size: int = 4096,
    distance_chunk_size: int = 256,
) -> dict[str, Any]:
    validate_hard_negative_config(
        jitter_copies=jitter_copies,
        ray_copies=ray_copies,
        behind_copies=behind_copies,
        random_count=random_count,
        jitter_std=jitter_std,
        ray_shift_min=ray_shift_min,
        ray_shift_max=ray_shift_max,
        behind_shift_min=behind_shift_min,
        behind_shift_max=behind_shift_max,
        random_bounds_scale=random_bounds_scale,
    )
    scene_root = Path(scene_dir)
    out_dir = Path(output_dir) if output_dir is not None else scene_root / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = generate_tanks_temples_hard_negative_splats(
        scene_dir=scene_root,
        splats_path=splats_path,
        output_splats_path=out_dir / f"{method_name}_hard_negative_splats.npz",
        output_metadata_path=out_dir / f"{method_name}_hard_negative_candidates.csv",
        output_status_path=out_dir / f"{method_name}_hard_negative_generation_status.json",
        output_report_path=out_dir / f"{method_name}_hard_negative_generation_report.md",
        alignment_path=alignment_path,
        crop_path=crop_path,
        max_source_splats=max_source_splats,
        seed=seed,
        include_source=include_source,
        jitter_copies=jitter_copies,
        ray_copies=ray_copies,
        behind_copies=behind_copies,
        random_count=random_count,
        jitter_std=jitter_std,
        ray_shift_min=ray_shift_min,
        ray_shift_max=ray_shift_max,
        behind_shift_min=behind_shift_min,
        behind_shift_max=behind_shift_max,
        random_bounds_scale=random_bounds_scale,
        apply_alignment=apply_alignment,
        invert_alignment=invert_alignment,
        use_crop=use_crop,
    )
    field_method = f"{method_name}_hard_negative"
    field_result = run_tanks_temples_gpis_field_score_diagnostics(
        scene_dir=scene_root,
        splats_path=generated.splats_path,
        model_path=model_path,
        ground_truth_path=ground_truth_path,
        alignment_path=alignment_path,
        crop_path=crop_path,
        output_dir=out_dir,
        method_name=field_method,
        thresholds=thresholds,
        topk_fractions=topk_fractions,
        max_pred_points=max_pred_points,
        max_gt_points=max_gt_points,
        seed=seed,
        apply_alignment=apply_alignment,
        invert_alignment=invert_alignment,
        use_crop=use_crop,
        epsilon=epsilon,
        gate_floor=gate_floor,
        batch_size=batch_size,
        distance_chunk_size=distance_chunk_size,
    )
    calibration_method = f"{method_name}_hard_negative_calibrated"
    calibration_result = run_gpis_splat_score_calibration(
        field_scores_path=field_result["field_scores_path"],
        output_dir=out_dir,
        method_name=calibration_method,
        thresholds=thresholds,
        topk_fractions=topk_fractions,
        validation_fraction=calibration_validation_fraction,
        seed=seed,
    )
    status_path = out_dir / f"{method_name}_hard_negative_workflow_status.json"
    report_path = out_dir / f"{method_name}_hard_negative_workflow_report.md"
    status = {
        "schema_version": 1,
        "scene_dir": str(scene_root),
        "method": method_name,
        "model_path": str(model_path),
        "generated_splats_path": str(generated.splats_path),
        "candidate_metadata_path": str(generated.candidates_path),
        "generation_status_path": str(generated.status_path),
        "field_scores_path": str(field_result["field_scores_path"]),
        "field_summary_path": str(field_result["score_summary_path"]),
        "calibration_summary_path": str(calibration_result["summary_path"]),
        "calibrated_scores_path": str(calibration_result["predictions_path"]),
        "calibrated_confidence_path": str(calibration_result["confidence_path"]),
        "thresholds": list(thresholds),
        "topk_fractions": list(topk_fractions),
        "best_calibrators": calibration_result["status"]["best_by_threshold"],
        "candidate_counts": generated.status["candidate_counts"],
    }
    write_json(status_path, status)
    report_path.write_text(format_hard_negative_workflow_report(status, calibration_result["summary"]), encoding="utf-8")
    return {
        "generated_splats_path": generated.splats_path,
        "candidate_metadata_path": generated.candidates_path,
        "generation_status_path": generated.status_path,
        "field_scores_path": field_result["field_scores_path"],
        "field_summary_path": field_result["score_summary_path"],
        "calibration_summary_path": calibration_result["summary_path"],
        "calibrated_scores_path": calibration_result["predictions_path"],
        "calibrated_confidence_path": calibration_result["confidence_path"],
        "status_path": status_path,
        "report_path": report_path,
        "status": status,
        "calibration_summary": calibration_result["summary"],
    }


def generate_tanks_temples_hard_negative_splats(
    *,
    scene_dir: str | Path,
    splats_path: str | Path | None,
    output_splats_path: str | Path,
    output_metadata_path: str | Path,
    output_status_path: str | Path,
    output_report_path: str | Path,
    alignment_path: str | Path | None = None,
    crop_path: str | Path | None = None,
    max_source_splats: int | None = 5000,
    seed: int = 13,
    include_source: bool = True,
    jitter_copies: int = 1,
    ray_copies: int = 1,
    behind_copies: int = 1,
    random_count: int | None = None,
    jitter_std: float = 0.03,
    ray_shift_min: float = 0.03,
    ray_shift_max: float = 0.15,
    behind_shift_min: float = 0.08,
    behind_shift_max: float = 0.25,
    random_bounds_scale: float = 1.25,
    apply_alignment: bool | None = None,
    invert_alignment: bool = False,
    use_crop: bool = True,
) -> HardNegativeOutputs:
    scene_root = Path(scene_dir)
    scene_meta, frames, _ = load_prepared_scene(scene_root)
    tanks_temples_meta = scene_meta.get("tanks_temples") or {}
    resolved_splats = resolve_scene_file(scene_root, splats_path, "real_splats.npz")
    resolved_alignment = resolve_optional_scene_file(scene_root, alignment_path, tanks_temples_meta.get("alignment_path"))
    resolved_crop = resolve_optional_scene_file(scene_root, crop_path, tanks_temples_meta.get("crop_path"))
    alignment_applied = bool(resolved_alignment is not None) if apply_alignment is None else bool(apply_alignment)
    alignment_matrix = None
    if alignment_applied:
        if resolved_alignment is None:
            raise FileNotFoundError("Alignment was requested but no alignment file was resolved.")
        alignment_matrix = load_alignment_matrix(resolved_alignment)
        if invert_alignment:
            alignment_matrix = np.linalg.inv(alignment_matrix)

    rng = np.random.default_rng(seed)
    splats = load_splats(str(resolved_splats))
    source_centers_all = splats.centers.detach().cpu().numpy().astype(np.float64)
    source_centers, source_indices = deterministic_subsample(source_centers_all, max_points=max_source_splats, seed=seed)
    colors = splats.colors.detach().cpu().numpy().astype(np.float64)[source_indices]
    tau = splats.tau.detach().cpu().numpy().astype(np.float64)[source_indices]
    sigma = splats.sigma.detach().cpu().numpy().astype(np.float64)[source_indices]
    if source_centers.shape[0] == 0:
        raise ValueError("No source splats are available for hard-negative generation.")

    raw_blocks: list[np.ndarray] = []
    color_blocks: list[np.ndarray] = []
    tau_blocks: list[np.ndarray] = []
    sigma_blocks: list[np.ndarray] = []
    surface_blocks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    def append_block(points: np.ndarray, block_colors: np.ndarray, block_tau: np.ndarray, block_sigma: np.ndarray, *, candidate_type: str, source_ids: np.ndarray, is_surface: bool) -> None:
        start = sum(block.shape[0] for block in raw_blocks)
        raw_blocks.append(points)
        color_blocks.append(block_colors)
        tau_blocks.append(block_tau)
        sigma_blocks.append(block_sigma)
        surface_blocks.append(np.full((points.shape[0],), is_surface, dtype=bool))
        for offset, (point, source_id) in enumerate(zip(points, source_ids, strict=False)):
            rows.append(
                {
                    "candidate_index": start + offset,
                    "candidate_type": candidate_type,
                    "source_splat_index": int(source_id),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "is_generated_hard_negative": candidate_type != "source",
                }
            )

    if include_source:
        append_block(source_centers, colors, tau, sigma, candidate_type="source", source_ids=source_indices, is_surface=True)
    if jitter_copies > 0:
        for copy_index in range(jitter_copies):
            jitter = rng.normal(loc=0.0, scale=jitter_std, size=source_centers.shape)
            append_block(
                source_centers + jitter,
                colors,
                tau,
                sigma,
                candidate_type=f"surface_jitter_{copy_index}",
                source_ids=source_indices,
                is_surface=False,
            )
    camera_centers = camera_centers_from_frames(frames)
    ray_dirs = source_ray_directions(source_centers, camera_centers, rng=rng)
    if ray_copies > 0:
        for copy_index in range(ray_copies):
            shifts = rng.uniform(ray_shift_min, ray_shift_max, size=(source_centers.shape[0], 1))
            append_block(
                source_centers - ray_dirs * shifts,
                colors,
                tau,
                sigma,
                candidate_type=f"camera_ray_freespace_{copy_index}",
                source_ids=source_indices,
                is_surface=False,
            )
    if behind_copies > 0:
        for copy_index in range(behind_copies):
            shifts = rng.uniform(behind_shift_min, behind_shift_max, size=(source_centers.shape[0], 1))
            append_block(
                source_centers + ray_dirs * shifts,
                colors,
                tau,
                sigma,
                candidate_type=f"camera_ray_behind_{copy_index}",
                source_ids=source_indices,
                is_surface=False,
            )
    resolved_random_count = source_centers.shape[0] if random_count is None else random_count
    if resolved_random_count > 0:
        random_points = random_candidate_points(
            source_centers=source_centers,
            count=resolved_random_count,
            rng=rng,
            random_bounds_scale=random_bounds_scale,
            crop_path=resolved_crop,
            alignment_matrix=alignment_matrix,
            use_crop=use_crop,
        )
        random_color = np.full((resolved_random_count, 3), 0.5, dtype=np.float64)
        random_tau = np.full((resolved_random_count,), float(np.median(tau)), dtype=np.float64)
        random_sigma = np.full((resolved_random_count,), float(np.median(sigma)), dtype=np.float64)
        append_block(
            random_points,
            random_color,
            random_tau,
            random_sigma,
            candidate_type="crop_random",
            source_ids=np.full((resolved_random_count,), -1, dtype=np.int64),
            is_surface=False,
        )

    if not raw_blocks:
        raise ValueError("Hard-negative generation produced no candidate splats. Enable source splats or at least one generated candidate family.")
    centers = np.concatenate(raw_blocks, axis=0)
    splat_out = SplatCloud(
        centers=torch.from_numpy(centers).to(dtype=torch.float64),
        colors=torch.from_numpy(np.concatenate(color_blocks, axis=0)).to(dtype=torch.float64),
        tau=torch.from_numpy(np.concatenate(tau_blocks, axis=0)).to(dtype=torch.float64),
        sigma=torch.from_numpy(np.concatenate(sigma_blocks, axis=0)).to(dtype=torch.float64),
        is_surface=torch.from_numpy(np.concatenate(surface_blocks, axis=0)).to(dtype=torch.bool),
    )
    output_splats = Path(output_splats_path).resolve()
    output_splats.parent.mkdir(parents=True, exist_ok=True)
    save_splats(str(output_splats), splat_out)
    candidate_table = pd.DataFrame(rows)
    candidates_path = Path(output_metadata_path).resolve()
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_table.to_csv(candidates_path, index=False)
    status_path = Path(output_status_path).resolve()
    report_path = Path(output_report_path).resolve()
    candidate_counts = candidate_table["candidate_type"].value_counts().sort_index().to_dict()
    status = {
        "schema_version": 1,
        "scene": scene_meta["scene"],
        "dataset": scene_meta.get("dataset"),
        "scene_dir": str(scene_root),
        "input_splats_path": str(resolved_splats),
        "output_splats_path": str(output_splats),
        "candidate_metadata_path": str(candidates_path),
        "source_splat_count_input": int(source_centers_all.shape[0]),
        "source_splat_count_used": int(source_centers.shape[0]),
        "candidate_count": int(centers.shape[0]),
        "candidate_counts": {str(key): int(value) for key, value in candidate_counts.items()},
        "alignment_path": str(resolved_alignment) if resolved_alignment is not None else None,
        "alignment_applied": alignment_applied,
        "crop_path": str(resolved_crop) if resolved_crop is not None else None,
        "use_crop": use_crop,
        "seed": seed,
        "parameters": {
            "include_source": include_source,
            "jitter_copies": jitter_copies,
            "ray_copies": ray_copies,
            "behind_copies": behind_copies,
            "random_count": resolved_random_count,
            "jitter_std": jitter_std,
            "ray_shift_min": ray_shift_min,
            "ray_shift_max": ray_shift_max,
            "behind_shift_min": behind_shift_min,
            "behind_shift_max": behind_shift_max,
            "random_bounds_scale": random_bounds_scale,
        },
    }
    write_json(status_path, status)
    report_path.write_text(format_hard_negative_generation_report(status), encoding="utf-8")
    return HardNegativeOutputs(
        splats_path=output_splats,
        candidates_path=candidates_path,
        status_path=status_path,
        report_path=report_path,
        candidate_table=candidate_table,
        status=status,
    )


def validate_hard_negative_config(
    *,
    jitter_copies: int,
    ray_copies: int,
    behind_copies: int,
    random_count: int | None,
    jitter_std: float,
    ray_shift_min: float,
    ray_shift_max: float,
    behind_shift_min: float,
    behind_shift_max: float,
    random_bounds_scale: float,
) -> None:
    if any(value < 0 for value in (jitter_copies, ray_copies, behind_copies)):
        raise ValueError("candidate copy counts must be non-negative.")
    if random_count is not None and random_count < 0:
        raise ValueError("random_count must be non-negative.")
    if jitter_std < 0.0:
        raise ValueError("jitter_std must be non-negative.")
    if ray_shift_min < 0.0 or ray_shift_max < ray_shift_min:
        raise ValueError("ray shift range must be non-negative and ordered.")
    if behind_shift_min < 0.0 or behind_shift_max < behind_shift_min:
        raise ValueError("behind shift range must be non-negative and ordered.")
    if random_bounds_scale <= 0.0:
        raise ValueError("random_bounds_scale must be positive.")


def camera_centers_from_frames(frames: list[dict[str, Any]]) -> np.ndarray:
    centers = []
    for frame in frames:
        camera_to_world = frame.get("camera_to_world")
        if camera_to_world is None:
            continue
        centers.append(np.asarray(camera_to_world, dtype=np.float64)[:3, 3])
    if not centers:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(centers, axis=0)


def source_ray_directions(source_centers: np.ndarray, camera_centers: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    if camera_centers.size == 0:
        reference = source_centers.mean(axis=0, keepdims=True)
        vectors = source_centers - reference
    else:
        distances = np.sum((source_centers[:, None, :] - camera_centers[None, :, :]) ** 2, axis=2)
        nearest = camera_centers[np.argmin(distances, axis=1)]
        vectors = source_centers - nearest
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    bad = lengths[:, 0] < 1e-9
    if np.any(bad):
        vectors[bad] = random_unit_vectors(int(bad.sum()), rng)
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-12)


def random_unit_vectors(count: int, rng: np.random.Generator) -> np.ndarray:
    vectors = rng.normal(size=(count, 3))
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-12)


def random_candidate_points(
    *,
    source_centers: np.ndarray,
    count: int,
    rng: np.random.Generator,
    random_bounds_scale: float,
    crop_path: Path | None,
    alignment_matrix: np.ndarray | None,
    use_crop: bool,
) -> np.ndarray:
    if use_crop and crop_path is not None:
        crop = read_json(crop_path)
        eval_points = sample_crop_points(crop, count=count, rng=rng)
        if alignment_matrix is not None:
            return transform_points(eval_points, np.linalg.inv(alignment_matrix))
        return eval_points
    lo = source_centers.min(axis=0)
    hi = source_centers.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * np.maximum(hi - lo, 1e-3) * random_bounds_scale
    return rng.uniform(center - half, center + half, size=(count, 3))


def sample_crop_points(crop: dict[str, Any], *, count: int, rng: np.random.Generator) -> np.ndarray:
    if "min" in crop and "max" in crop:
        lo = np.asarray(crop["min"], dtype=np.float64)
        hi = np.asarray(crop["max"], dtype=np.float64)
        return rng.uniform(lo, hi, size=(count, 3))
    if {"axis_min", "axis_max", "bounding_polygon", "orthogonal_axis"}.issubset(crop):
        polygon = np.asarray(crop["bounding_polygon"], dtype=np.float64)
        axis_min = float(crop["axis_min"])
        axis_max = float(crop["axis_max"])
        lo = np.minimum(polygon.min(axis=0), polygon.max(axis=0))
        hi = np.maximum(polygon.min(axis=0), polygon.max(axis=0))
        axis = {"X": 0, "Y": 1, "Z": 2}[str(crop["orthogonal_axis"]).upper()]
        lo[axis], hi[axis] = sorted((axis_min, axis_max))
        points: list[np.ndarray] = []
        attempts = 0
        while sum(chunk.shape[0] for chunk in points) < count and attempts < 50:
            attempts += 1
            candidates = rng.uniform(lo, hi, size=(max(count, 256), 3))
            points.append(candidates[crop_mask(candidates, crop)])
        if points:
            stacked = np.concatenate(points, axis=0)
            if stacked.shape[0] >= count:
                return stacked[:count]
        return rng.uniform(lo, hi, size=(count, 3))
    raise ValueError("Unsupported crop format. Expected min/max bounds or Tanks and Temples SelectionPolygonVolume fields.")


def format_hard_negative_generation_report(status: dict[str, Any]) -> str:
    lines = [
        "# Hard-Negative Splat Candidates",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Input splats: `{status['input_splats_path']}`",
        f"- Output splats: `{status['output_splats_path']}`",
        f"- Candidate metadata: `{status['candidate_metadata_path']}`",
        f"- Source splats used: `{status['source_splat_count_used']}`",
        f"- Candidate count: `{status['candidate_count']}`",
        "",
        "## Candidate Counts",
        "",
    ]
    for name, count in status["candidate_counts"].items():
        lines.append(f"- `{name}`: `{count}`")
    return "\n".join(lines) + "\n"


def format_hard_negative_workflow_report(status: dict[str, Any], calibration_summary: pd.DataFrame) -> str:
    lines = [
        "# Hard-Negative GPIS Calibration Workflow",
        "",
        f"- Method: `{status['method']}`",
        f"- Generated splats: `{status['generated_splats_path']}`",
        f"- Candidate metadata: `{status['candidate_metadata_path']}`",
        f"- Field scores: `{status['field_scores_path']}`",
        f"- Calibration summary: `{status['calibration_summary_path']}`",
        f"- Calibrated confidence: `{status['calibrated_confidence_path']}`",
        "",
        "## Candidate Counts",
        "",
    ]
    for name, count in status["candidate_counts"].items():
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(["", "## Best Calibrators", ""])
    for row in status["best_calibrators"]:
        auc = "n/a" if row["auc"] is None else f"{row['auc']:.6g}"
        average_precision = "n/a" if row.get("average_precision") is None else f"{row['average_precision']:.6g}"
        lines.append(
            f"- threshold `{row['geometry_threshold']:.6g}`: `{row['method_name']}`, "
            f"Brier `{row['brier']:.6g}`, AUC `{auc}`, AP `{average_precision}`, best F-score `{row['best_f_score']:.6g}`"
        )
    if not calibration_summary.empty:
        lines.extend(["", "## Calibration Metrics", "", format_calibration_metric_table(calibration_summary)])
    return "\n".join(lines) + "\n"


def format_calibration_metric_table(summary: pd.DataFrame) -> str:
    columns = ["geometry_threshold", "method_name", "method_family", "brier", "average_precision", "auc", "best_topk_fraction", "best_f_score"]
    available = [column for column in columns if column in summary.columns]
    lines = [
        "| threshold | method | family | brier | ap | auc | top_k | best_f |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    sorted_summary = summary[available].sort_values(["geometry_threshold", "average_precision", "auc"], ascending=[True, False, False], na_position="last")
    for row in sorted_summary.itertuples(index=False):
        average_precision = getattr(row, "average_precision", None)
        auc = getattr(row, "auc", None)
        lines.append(
            f"| {row.geometry_threshold:.6g} | `{row.method_name}` | `{row.method_family}` | {row.brier:.6g} | "
            f"{format_optional_metric(average_precision)} | {format_optional_metric(auc)} | {row.best_topk_fraction:.6g} | {row.best_f_score:.6g} |"
        )
    return "\n".join(lines)


def format_optional_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.6g}"
