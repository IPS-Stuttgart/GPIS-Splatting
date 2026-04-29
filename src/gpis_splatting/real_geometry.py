from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gpis_splatting.gpis import load_model
from gpis_splatting.real_bootstrap import load_ply_point_cloud
from gpis_splatting.real_scene import load_prepared_scene
from gpis_splatting.serialization import read_json, write_json
from gpis_splatting.splats import SplatCloud, gpis_gate_for_splats, load_splats


def evaluate_tanks_temples_geometry(
    *,
    scene_dir: str | Path,
    splats_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
    crop_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    method_name: str | None = None,
    thresholds: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1),
    max_pred_points: int | None = 100_000,
    max_gt_points: int | None = 100_000,
    seed: int = 13,
    apply_alignment: bool | None = None,
    invert_alignment: bool = False,
    use_crop: bool = True,
    gate_path: str | Path | None = None,
    model_path: str | Path | None = None,
    epsilon: float = 0.24,
    gate_floor: float = 0.0,
    gate_thresholds: tuple[float, ...] = (0.5,),
    gate_batch_size: int = 4096,
    distance_chunk_size: int = 256,
) -> dict[str, Any]:
    if not thresholds:
        raise ValueError("At least one distance threshold is required.")
    if any(threshold <= 0.0 for threshold in thresholds):
        raise ValueError("Distance thresholds must be positive.")
    if distance_chunk_size < 1:
        raise ValueError("distance_chunk_size must be positive.")
    if not 0.0 <= gate_floor <= 1.0:
        raise ValueError("gate_floor must be in [0, 1].")

    scene_root = Path(scene_dir)
    scene_meta, _, _ = load_prepared_scene(scene_root)
    tanks_temples_meta = scene_meta.get("tanks_temples") or {}
    resolved_splats = resolve_scene_file(scene_root, splats_path, "real_splats.npz")
    resolved_gt = resolve_optional_scene_file(scene_root, ground_truth_path, tanks_temples_meta.get("ground_truth_path"))
    if resolved_gt is None:
        raise FileNotFoundError("Could not resolve ground-truth geometry. Pass --ground-truth-path or prepare a Tanks and Temples scene with ground truth.")
    resolved_alignment = resolve_optional_scene_file(scene_root, alignment_path, tanks_temples_meta.get("alignment_path"))
    resolved_crop = resolve_optional_scene_file(scene_root, crop_path, tanks_temples_meta.get("crop_path"))
    resolved_method = method_name or resolved_splats.stem
    out_dir = Path(output_dir) if output_dir is not None else scene_root / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)

    splats = load_splats(str(resolved_splats))
    pred_points_raw = splats.centers.detach().cpu().numpy().astype(np.float64)
    pred_count_input = int(pred_points_raw.shape[0])
    gates = resolve_splat_gates(
        scene_root=scene_root,
        splats=splats,
        gate_path=gate_path,
        model_path=model_path,
        epsilon=epsilon,
        gate_floor=gate_floor,
        gate_batch_size=gate_batch_size,
    )
    pred_points_raw, pred_indices = deterministic_subsample(pred_points_raw, max_points=max_pred_points, seed=seed)
    gates = gates[pred_indices] if gates is not None else None

    alignment_applied = bool(resolved_alignment is not None) if apply_alignment is None else bool(apply_alignment)
    pred_points = pred_points_raw
    alignment_matrix = None
    if alignment_applied:
        if resolved_alignment is None:
            raise FileNotFoundError("Alignment was requested but no alignment file was resolved.")
        alignment_matrix = load_alignment_matrix(resolved_alignment)
        if invert_alignment:
            alignment_matrix = np.linalg.inv(alignment_matrix)
        pred_points = transform_points(pred_points, alignment_matrix)

    gt_cloud = load_ply_point_cloud(resolved_gt)
    gt_points = gt_cloud.points.astype(np.float64)
    gt_count_input = int(gt_points.shape[0])

    crop_summary = {"enabled": False}
    if use_crop and resolved_crop is not None:
        crop = read_json(resolved_crop)
        pred_crop_mask = crop_mask(pred_points, crop)
        gt_crop_mask = crop_mask(gt_points, crop)
        crop_summary = {
            "enabled": True,
            "path": str(resolved_crop),
            "pred_kept": int(pred_crop_mask.sum()),
            "pred_total": int(pred_crop_mask.shape[0]),
            "gt_kept": int(gt_crop_mask.sum()),
            "gt_total": int(gt_crop_mask.shape[0]),
        }
        pred_points = pred_points[pred_crop_mask]
        gates = gates[pred_crop_mask] if gates is not None else None
        gt_points = gt_points[gt_crop_mask]

    if pred_points.size == 0:
        raise ValueError("No predicted splat centers remain after subsampling/cropping.")
    if gt_points.size == 0:
        raise ValueError("No ground-truth points remain after cropping.")

    gt_points, gt_indices = deterministic_subsample(gt_points, max_points=max_gt_points, seed=seed + 1)
    gt_count_evaluated = int(gt_points.shape[0])
    pred_count_evaluated = int(pred_points.shape[0])

    groups = [GeometryGroup(name="all", points=pred_points, gates=gates)]
    if gates is not None:
        groups.extend(gate_groups(pred_points, gates, gate_thresholds=gate_thresholds))

    summary_rows = []
    threshold_rows = []
    for group in groups:
        if group.points.shape[0] == 0:
            continue
        summary, threshold_metrics = evaluate_geometry_group(
            group.points,
            gt_points,
            thresholds=thresholds,
            distance_chunk_size=distance_chunk_size,
        )
        summary_rows.append(
            {
                "scene": scene_meta["scene"],
                "dataset": scene_meta.get("dataset"),
                "method": resolved_method,
                "group": group.name,
                "pred_point_count": int(group.points.shape[0]),
                "gt_point_count": gt_count_evaluated,
                "gate_min": optional_stat(group.gates, "min"),
                "gate_max": optional_stat(group.gates, "max"),
                "gate_mean": optional_stat(group.gates, "mean"),
                **summary,
            }
        )
        for row in threshold_metrics:
            threshold_rows.append(
                {
                    "scene": scene_meta["scene"],
                    "dataset": scene_meta.get("dataset"),
                    "method": resolved_method,
                    "group": group.name,
                    "pred_point_count": int(group.points.shape[0]),
                    "gt_point_count": gt_count_evaluated,
                    **row,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    summary_path = out_dir / f"{resolved_method}_geometry_summary.csv"
    threshold_path = out_dir / f"{resolved_method}_geometry_thresholds.csv"
    status_path = out_dir / f"{resolved_method}_geometry_status.json"
    report_path = out_dir / f"{resolved_method}_geometry_report.md"
    summary_df.to_csv(summary_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    status = {
        "schema_version": 1,
        "scene": scene_meta["scene"],
        "dataset": scene_meta.get("dataset"),
        "method": resolved_method,
        "scene_dir": str(scene_root),
        "splats_path": str(resolved_splats),
        "ground_truth_path": str(resolved_gt),
        "alignment_path": str(resolved_alignment) if resolved_alignment is not None else None,
        "alignment_applied": alignment_applied,
        "invert_alignment": invert_alignment,
        "crop_path": str(resolved_crop) if resolved_crop is not None else None,
        "crop": crop_summary,
        "thresholds": list(thresholds),
        "max_pred_points": max_pred_points,
        "max_gt_points": max_gt_points,
        "pred_count_input": pred_count_input,
        "pred_count_sampled": int(pred_points_raw.shape[0]),
        "pred_count_evaluated": pred_count_evaluated,
        "gt_count_input": gt_count_input,
        "gt_count_after_crop": int(crop_summary.get("gt_kept", gt_count_input)),
        "gt_count_evaluated": gt_count_evaluated,
        "gt_sample_indices_count": int(gt_indices.shape[0]),
        "gate_available": gates is not None,
        "summary_path": str(summary_path),
        "threshold_metrics_path": str(threshold_path),
        "report_path": str(report_path),
        "summary": summary_rows,
    }
    write_json(status_path, status)
    report_path.write_text(format_geometry_report(status, summary_df, threshold_df), encoding="utf-8")
    return {
        "summary_path": summary_path,
        "threshold_metrics_path": threshold_path,
        "status_path": status_path,
        "report_path": report_path,
        "summary": summary_rows,
        "status": status,
    }


class GeometryGroup:
    def __init__(self, *, name: str, points: np.ndarray, gates: np.ndarray | None) -> None:
        self.name = name
        self.points = points
        self.gates = gates


def evaluate_geometry_group(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    *,
    thresholds: tuple[float, ...],
    distance_chunk_size: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    pred_to_gt = nearest_neighbor_distances(pred_points, gt_points, query_chunk_size=distance_chunk_size)
    gt_to_pred = nearest_neighbor_distances(gt_points, pred_points, query_chunk_size=distance_chunk_size)
    summary = {
        "accuracy_mean": float(pred_to_gt.mean()),
        "accuracy_median": float(np.median(pred_to_gt)),
        "accuracy_rmse": float(np.sqrt(np.mean(pred_to_gt**2))),
        "completion_mean": float(gt_to_pred.mean()),
        "completion_median": float(np.median(gt_to_pred)),
        "completion_rmse": float(np.sqrt(np.mean(gt_to_pred**2))),
        "chamfer_l1": float(pred_to_gt.mean() + gt_to_pred.mean()),
        "chamfer_l2": float(np.mean(pred_to_gt**2) + np.mean(gt_to_pred**2)),
    }
    threshold_metrics = []
    for threshold in thresholds:
        precision = float(np.mean(pred_to_gt <= threshold))
        recall = float(np.mean(gt_to_pred <= threshold))
        f_score = 0.0 if precision + recall <= 0.0 else float(2.0 * precision * recall / (precision + recall))
        threshold_metrics.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
            }
        )
    return summary, threshold_metrics


def nearest_neighbor_distances(query: np.ndarray, reference: np.ndarray, *, query_chunk_size: int) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("query points must have shape (N, 3).")
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference points must have shape (N, 3).")
    if query.shape[0] == 0 or reference.shape[0] == 0:
        raise ValueError("nearest-neighbor distances require non-empty point sets.")
    ref_t = reference.T
    ref_norm = np.sum(reference**2, axis=1)
    distances = np.empty((query.shape[0],), dtype=np.float64)
    for start in range(0, query.shape[0], query_chunk_size):
        chunk = query[start : start + query_chunk_size]
        chunk_norm = np.sum(chunk**2, axis=1, keepdims=True)
        squared = np.maximum(chunk_norm + ref_norm[None, :] - 2.0 * (chunk @ ref_t), 0.0)
        distances[start : start + chunk.shape[0]] = np.sqrt(np.min(squared, axis=1))
    return distances


def resolve_splat_gates(
    *,
    scene_root: Path,
    splats: SplatCloud,
    gate_path: str | Path | None,
    model_path: str | Path | None,
    epsilon: float,
    gate_floor: float,
    gate_batch_size: int,
) -> np.ndarray | None:
    if gate_path is not None:
        resolved_gate = resolve_scene_file(scene_root, gate_path, "real_splat_gates.npz")
        with np.load(resolved_gate, allow_pickle=False) as data:
            key = "gate" if "gate" in data.files else "raw_gate"
            gates = np.asarray(data[key], dtype=np.float64).reshape(-1)
    elif model_path is not None:
        resolved_model = resolve_scene_file(scene_root, model_path, "real_gpis_model.npz")
        model, _ = load_model(str(resolved_model))
        raw_gate = gpis_gate_for_splats(splats, model, epsilon, batch_size=gate_batch_size)
        gates = torch.clamp(gate_floor + (1.0 - gate_floor) * raw_gate, min=0.0, max=1.0).detach().cpu().numpy()
    else:
        return None
    if gates.shape[0] != splats.centers.shape[0]:
        raise ValueError(f"Gate count {gates.shape[0]} does not match splat count {splats.centers.shape[0]}.")
    return np.clip(gates, 0.0, 1.0)


def gate_groups(points: np.ndarray, gates: np.ndarray, *, gate_thresholds: tuple[float, ...]) -> list[GeometryGroup]:
    groups = []
    for threshold in gate_thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("gate thresholds must be in [0, 1].")
        high = gates >= threshold
        low = gates < threshold
        groups.append(GeometryGroup(name=f"gate_ge_{format_threshold_label(threshold)}", points=points[high], gates=gates[high]))
        groups.append(GeometryGroup(name=f"gate_lt_{format_threshold_label(threshold)}", points=points[low], gates=gates[low]))
    return groups


def crop_mask(points: np.ndarray, crop: dict[str, Any]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if "min" in crop and "max" in crop:
        lo = np.asarray(crop["min"], dtype=np.float64)
        hi = np.asarray(crop["max"], dtype=np.float64)
        return np.all((points >= lo[None, :]) & (points <= hi[None, :]), axis=1)
    if {"axis_min", "axis_max", "bounding_polygon", "orthogonal_axis"}.issubset(crop):
        axis = axis_index(str(crop["orthogonal_axis"]))
        other_axes = [index for index in range(3) if index != axis]
        polygon = np.asarray(crop["bounding_polygon"], dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 3 or polygon.shape[0] < 3:
            raise ValueError("Tanks and Temples crop bounding_polygon must have shape (N, 3), N >= 3.")
        axis_min = float(crop["axis_min"])
        axis_max = float(crop["axis_max"])
        axis_lo, axis_hi = sorted((axis_min, axis_max))
        in_axis = (points[:, axis] >= axis_lo) & (points[:, axis] <= axis_hi)
        in_polygon = points_in_polygon_2d(points[:, other_axes], polygon[:, other_axes])
        return in_axis & in_polygon
    raise ValueError("Unsupported crop format. Expected min/max bounds or Tanks and Temples SelectionPolygonVolume fields.")


def points_in_polygon_2d(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    xj, yj = polygon[-1, 0], polygon[-1, 1]
    for xi, yi in polygon:
        crosses = (yi > y) != (yj > y)
        x_intersection = (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
        inside ^= crosses & (x < x_intersection)
        xj, yj = xi, yi
    return inside


def axis_index(axis: str) -> int:
    mapping = {"X": 0, "Y": 1, "Z": 2}
    key = axis.upper()
    if key not in mapping:
        raise ValueError(f"Unsupported crop orthogonal axis {axis!r}. Expected X, Y, or Z.")
    return mapping[key]


def load_alignment_matrix(path: str | Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Alignment matrix {path} must have shape (4, 4).")
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points, np.ones((points.shape[0], 1), dtype=np.float64)), axis=1)
    return (homogeneous @ matrix.T)[:, :3]


def deterministic_subsample(points: np.ndarray, *, max_points: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        indices = np.arange(points.shape[0], dtype=np.int64)
        return points, indices
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(points.shape[0], size=max_points, replace=False)).astype(np.int64)
    return points[indices], indices


def optional_stat(values: np.ndarray | None, name: str) -> float | None:
    if values is None or values.size == 0:
        return None
    if name == "min":
        return float(values.min())
    if name == "max":
        return float(values.max())
    if name == "mean":
        return float(values.mean())
    raise ValueError(f"Unsupported optional statistic {name!r}.")


def resolve_scene_file(scene_root: Path, path: str | Path | None, default_name: str) -> Path:
    resolved = Path(default_name) if path is None else Path(path)
    if not resolved.is_absolute():
        resolved = scene_root / resolved
    return resolved


def resolve_optional_scene_file(scene_root: Path, requested: str | Path | None, fallback: str | None) -> Path | None:
    value = requested if requested is not None else fallback
    if value is None:
        return None
    resolved = Path(value)
    if not resolved.is_absolute():
        resolved = scene_root / resolved
    return resolved


def format_threshold_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def format_geometry_report(status: dict[str, Any], summary: pd.DataFrame, thresholds: pd.DataFrame) -> str:
    lines = [
        "# Real Geometry Evaluation",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Method: `{status['method']}`",
        f"- Dataset: `{status.get('dataset')}`",
        f"- Splat points evaluated: `{status['pred_count_evaluated']}`",
        f"- Ground-truth points evaluated: `{status['gt_count_evaluated']}`",
        f"- Alignment applied: `{status['alignment_applied']}`",
        f"- Crop enabled: `{status['crop'].get('enabled', False)}`",
        f"- Summary CSV: `{status['summary_path']}`",
        f"- Threshold CSV: `{status['threshold_metrics_path']}`",
    ]
    if not summary.empty:
        all_row = summary[summary["group"] == "all"].iloc[0]
        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- Chamfer L1: `{all_row['chamfer_l1']:.6g}`",
                f"- Chamfer L2: `{all_row['chamfer_l2']:.6g}`",
                f"- Accuracy mean: `{all_row['accuracy_mean']:.6g}`",
                f"- Completion mean: `{all_row['completion_mean']:.6g}`",
            ]
        )
    if not thresholds.empty:
        lines.extend(["", "## Threshold Metrics", "", format_threshold_table(thresholds)])
    return "\n".join(lines) + "\n"


def format_threshold_table(thresholds: pd.DataFrame) -> str:
    columns = ["group", "threshold", "precision", "recall", "f_score"]
    lines = [
        "| group | threshold | precision | recall | f_score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in thresholds[columns].itertuples(index=False):
        lines.append(f"| `{row.group}` | {row.threshold:.6g} | {row.precision:.6g} | {row.recall:.6g} | {row.f_score:.6g} |")
    return "\n".join(lines)
