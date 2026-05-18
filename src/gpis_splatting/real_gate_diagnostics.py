from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.real_bootstrap import load_ply_point_cloud
from gpis_splatting.real_geometry import (
    crop_mask,
    deterministic_subsample,
    format_threshold_label,
    load_alignment_matrix,
    nearest_neighbor_distances,
    resolve_optional_scene_file,
    resolve_scene_file,
    resolve_splat_gates,
    transform_points,
)
from gpis_splatting.real_scene import load_prepared_scene
from gpis_splatting.serialization import read_json, write_json
from gpis_splatting.splats import load_splats


@dataclass(frozen=True)
class GateDiagnosticInputs:
    scene_meta: dict[str, Any]
    scene_dir: Path
    splats_path: Path
    ground_truth_path: Path
    alignment_path: Path | None
    crop_path: Path | None
    alignment_applied: bool
    crop_summary: dict[str, Any]
    method: str
    pred_points: np.ndarray
    gt_points: np.ndarray
    gates: np.ndarray
    splat_indices: np.ndarray
    pred_count_input: int
    pred_count_sampled: int
    pred_count_evaluated: int
    gt_count_input: int
    gt_count_evaluated: int


def run_tanks_temples_gate_diagnostics(
    *,
    scene_dir: str | Path,
    splats_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
    crop_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    method_name: str | None = None,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.1),
    topk_fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
    num_bins: int = 10,
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
    gate_batch_size: int = 4096,
    distance_chunk_size: int = 256,
) -> dict[str, Any]:
    validate_diagnostic_config(thresholds=thresholds, topk_fractions=topk_fractions, num_bins=num_bins, distance_chunk_size=distance_chunk_size)
    inputs = prepare_gate_diagnostic_inputs(
        scene_dir=scene_dir,
        splats_path=splats_path,
        ground_truth_path=ground_truth_path,
        alignment_path=alignment_path,
        crop_path=crop_path,
        method_name=method_name,
        max_pred_points=max_pred_points,
        max_gt_points=max_gt_points,
        seed=seed,
        apply_alignment=apply_alignment,
        invert_alignment=invert_alignment,
        use_crop=use_crop,
        gate_path=gate_path,
        model_path=model_path,
        epsilon=epsilon,
        gate_floor=gate_floor,
        gate_batch_size=gate_batch_size,
    )
    pred_to_gt = nearest_neighbor_distances(inputs.pred_points, inputs.gt_points, query_chunk_size=distance_chunk_size)
    splat_table = build_splat_quality_table(
        inputs,
        pred_to_gt,
        thresholds=thresholds,
    )
    ranked = build_ranked_gate_table(
        gates=inputs.gates,
        pred_to_gt=pred_to_gt,
        gt_points=inputs.gt_points,
        pred_points=inputs.pred_points,
        thresholds=thresholds,
        topk_fractions=topk_fractions,
        distance_chunk_size=distance_chunk_size,
    )
    bins = build_gate_bin_table(
        gates=inputs.gates,
        pred_to_gt=pred_to_gt,
        thresholds=thresholds,
        num_bins=num_bins,
    )
    correlations = gate_error_correlations(inputs.gates, pred_to_gt)
    best_topk = best_topk_rows(ranked)

    out_dir = Path(output_dir) if output_dir is not None else inputs.scene_dir / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = inputs.method
    splat_table_path = out_dir / f"{prefix}_gate_quality_splats.csv"
    ranked_path = out_dir / f"{prefix}_gate_quality_ranked.csv"
    bins_path = out_dir / f"{prefix}_gate_quality_bins.csv"
    status_path = out_dir / f"{prefix}_gate_quality_status.json"
    report_path = out_dir / f"{prefix}_gate_quality_report.md"
    splat_table.to_csv(splat_table_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    bins.to_csv(bins_path, index=False)
    status = {
        "schema_version": 1,
        "scene": inputs.scene_meta["scene"],
        "dataset": inputs.scene_meta.get("dataset"),
        "method": inputs.method,
        "scene_dir": str(inputs.scene_dir),
        "splats_path": str(inputs.splats_path),
        "ground_truth_path": str(inputs.ground_truth_path),
        "alignment_path": str(inputs.alignment_path) if inputs.alignment_path is not None else None,
        "alignment_applied": inputs.alignment_applied,
        "crop_path": str(inputs.crop_path) if inputs.crop_path is not None else None,
        "crop": inputs.crop_summary,
        "thresholds": list(thresholds),
        "topk_fractions": list(topk_fractions),
        "num_bins": num_bins,
        "max_pred_points": max_pred_points,
        "max_gt_points": max_gt_points,
        "pred_count_input": inputs.pred_count_input,
        "pred_count_sampled": inputs.pred_count_sampled,
        "pred_count_evaluated": inputs.pred_count_evaluated,
        "gt_count_input": inputs.gt_count_input,
        "gt_count_evaluated": inputs.gt_count_evaluated,
        "correlations": correlations,
        "best_topk_by_f_score": best_topk,
        "splat_quality_path": str(splat_table_path),
        "ranked_quality_path": str(ranked_path),
        "gate_bin_path": str(bins_path),
        "report_path": str(report_path),
    }
    write_json(status_path, status)
    report_path.write_text(format_gate_quality_report(status, ranked, bins), encoding="utf-8")
    return {
        "splat_quality_path": splat_table_path,
        "ranked_quality_path": ranked_path,
        "gate_bin_path": bins_path,
        "status_path": status_path,
        "report_path": report_path,
        "splat_quality": splat_table,
        "ranked_quality": ranked,
        "gate_bins": bins,
        "status": status,
    }


def validate_diagnostic_config(
    *,
    thresholds: tuple[float, ...],
    topk_fractions: tuple[float, ...],
    num_bins: int,
    distance_chunk_size: int,
) -> None:
    if not thresholds:
        raise ValueError("At least one distance threshold is required.")
    if any(threshold <= 0.0 for threshold in thresholds):
        raise ValueError("Distance thresholds must be positive.")
    if not topk_fractions:
        raise ValueError("At least one top-k fraction is required.")
    if any(fraction <= 0.0 or fraction > 1.0 for fraction in topk_fractions):
        raise ValueError("Top-k fractions must be in (0, 1].")
    if num_bins < 1:
        raise ValueError("num_bins must be positive.")
    if distance_chunk_size < 1:
        raise ValueError("distance_chunk_size must be positive.")


def prepare_gate_diagnostic_inputs(
    *,
    scene_dir: str | Path,
    splats_path: str | Path | None,
    ground_truth_path: str | Path | None,
    alignment_path: str | Path | None,
    crop_path: str | Path | None,
    method_name: str | None,
    max_pred_points: int | None,
    max_gt_points: int | None,
    seed: int,
    apply_alignment: bool | None,
    invert_alignment: bool,
    use_crop: bool,
    gate_path: str | Path | None,
    model_path: str | Path | None,
    epsilon: float,
    gate_floor: float,
    gate_batch_size: int,
) -> GateDiagnosticInputs:
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
    if gates is None:
        raise ValueError("Gate diagnostics require --gate-path or --model-path so splat gates can be evaluated.")

    alignment_applied = bool(resolved_alignment is not None) if apply_alignment is None else bool(apply_alignment)
    pred_points = pred_points_raw
    splat_indices = np.arange(pred_count_input, dtype=np.int64)
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
        pred_crop = crop_mask(pred_points, crop)
        gt_crop = crop_mask(gt_points, crop)
        crop_summary = {
            "enabled": True,
            "path": str(resolved_crop),
            "pred_kept": int(pred_crop.sum()),
            "pred_total": int(pred_crop.shape[0]),
            "gt_kept": int(gt_crop.sum()),
            "gt_total": int(gt_crop.shape[0]),
        }
        pred_points = pred_points[pred_crop]
        gates = gates[pred_crop]
        splat_indices = splat_indices[pred_crop]
        gt_points = gt_points[gt_crop]

    if pred_points.size == 0:
        raise ValueError("No predicted splat centers remain after alignment/cropping.")
    if gt_points.size == 0:
        raise ValueError("No ground-truth points remain after cropping.")

    pred_points, pred_sample_indices = deterministic_subsample(pred_points, max_points=max_pred_points, seed=seed)
    gates = gates[pred_sample_indices]
    splat_indices = splat_indices[pred_sample_indices]
    gt_points, _ = deterministic_subsample(gt_points, max_points=max_gt_points, seed=seed + 1)
    return GateDiagnosticInputs(
        scene_meta=scene_meta,
        scene_dir=scene_root,
        splats_path=resolved_splats,
        ground_truth_path=resolved_gt,
        alignment_path=resolved_alignment,
        crop_path=resolved_crop,
        alignment_applied=alignment_applied,
        crop_summary=crop_summary,
        method=resolved_method,
        pred_points=pred_points,
        gt_points=gt_points,
        gates=gates,
        splat_indices=splat_indices,
        pred_count_input=pred_count_input,
        pred_count_sampled=int(pred_points.shape[0]),
        pred_count_evaluated=int(pred_points.shape[0]),
        gt_count_input=gt_count_input,
        gt_count_evaluated=int(gt_points.shape[0]),
    )


def build_splat_quality_table(inputs: GateDiagnosticInputs, pred_to_gt: np.ndarray, *, thresholds: tuple[float, ...]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "splat_index": inputs.splat_indices,
            "x": inputs.pred_points[:, 0],
            "y": inputs.pred_points[:, 1],
            "z": inputs.pred_points[:, 2],
            "gate": inputs.gates,
            "nearest_gt_distance": pred_to_gt,
        }
    )
    for threshold in thresholds:
        table[f"within_{format_threshold_label(threshold)}"] = pred_to_gt <= threshold
    return table


def build_ranked_gate_table(
    *,
    gates: np.ndarray,
    pred_to_gt: np.ndarray,
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    thresholds: tuple[float, ...],
    topk_fractions: tuple[float, ...],
    distance_chunk_size: int,
) -> pd.DataFrame:
    order = np.argsort(-gates, kind="mergesort")
    n_points = int(gates.shape[0])
    rows = []
    for fraction in sorted(set(topk_fractions)):
        count = max(1, min(n_points, int(np.ceil(fraction * n_points))))
        selected = order[:count]
        selected_distances = pred_to_gt[selected]
        selected_points = pred_points[selected]
        completion_distances = nearest_neighbor_distances(gt_points, selected_points, query_chunk_size=distance_chunk_size)
        for threshold in thresholds:
            precision = float(np.mean(selected_distances <= threshold))
            recall = float(np.mean(completion_distances <= threshold))
            f_score = 0.0 if precision + recall <= 0.0 else float(2.0 * precision * recall / (precision + recall))
            rows.append(
                {
                    "topk_fraction": float(fraction),
                    "selected_pred_point_count": count,
                    "retention_fraction": count / n_points,
                    "gate_min": float(gates[selected].min()),
                    "gate_mean": float(gates[selected].mean()),
                    "accuracy_mean": float(selected_distances.mean()),
                    "geometry_threshold": float(threshold),
                    "precision": precision,
                    "recall": recall,
                    "f_score": f_score,
                }
            )
    return pd.DataFrame(rows)


def build_gate_bin_table(*, gates: np.ndarray, pred_to_gt: np.ndarray, thresholds: tuple[float, ...], num_bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    rows = []
    for index in range(num_bins):
        lo = float(edges[index])
        hi = float(edges[index + 1])
        if index == num_bins - 1:
            mask = (gates >= lo) & (gates <= hi)
        else:
            mask = (gates >= lo) & (gates < hi)
        if not np.any(mask):
            continue
        bin_gates = gates[mask]
        bin_distances = pred_to_gt[mask]
        for threshold in thresholds:
            within = float(np.mean(bin_distances <= threshold))
            rows.append(
                {
                    "bin_index": index,
                    "gate_bin_min": lo,
                    "gate_bin_max": hi,
                    "splat_count": int(mask.sum()),
                    "mean_gate": float(bin_gates.mean()),
                    "mean_nearest_gt_distance": float(bin_distances.mean()),
                    "geometry_threshold": float(threshold),
                    "within_threshold_fraction": within,
                    "gate_minus_within_fraction": float(bin_gates.mean() - within),
                    "abs_gate_calibration_error": float(abs(bin_gates.mean() - within)),
                }
            )
    return pd.DataFrame(rows)


def gate_error_correlations(gates: np.ndarray, pred_to_gt: np.ndarray) -> dict[str, float | None]:
    return {
        "spearman_gate_vs_negative_distance": finite_or_none(correlation(rankdata_average(gates), rankdata_average(-pred_to_gt))),
        "spearman_gate_vs_distance": finite_or_none(correlation(rankdata_average(gates), rankdata_average(pred_to_gt))),
        "pearson_gate_vs_negative_distance": finite_or_none(correlation(gates, -pred_to_gt)),
        "pearson_gate_vs_distance": finite_or_none(correlation(gates, pred_to_gt)),
    }


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape[0] < 2:
        return float("nan")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denom)


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def best_topk_rows(ranked: pd.DataFrame) -> list[dict[str, Any]]:
    best = []
    for geometry_threshold, group in ranked.groupby("geometry_threshold"):
        row = group.sort_values(["f_score", "retention_fraction"], ascending=[False, True]).iloc[0]
        best.append(
            {
                "geometry_threshold": float(geometry_threshold),
                "topk_fraction": float(row["topk_fraction"]),
                "retention_fraction": float(row["retention_fraction"]),
                "selected_pred_point_count": int(row["selected_pred_point_count"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f_score": float(row["f_score"]),
            }
        )
    return best


def format_gate_quality_report(status: dict[str, Any], ranked: pd.DataFrame, bins: pd.DataFrame) -> str:
    corr = status["correlations"]
    lines = [
        "# Tanks and Temples Gate Quality Diagnostics",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Method: `{status['method']}`",
        f"- Splat points evaluated: `{status['pred_count_evaluated']}`",
        f"- Ground-truth points evaluated: `{status['gt_count_evaluated']}`",
        f"- Spearman(gate, -distance): `{format_optional_float(corr['spearman_gate_vs_negative_distance'])}`",
        f"- Pearson(gate, -distance): `{format_optional_float(corr['pearson_gate_vs_negative_distance'])}`",
        f"- Per-splat CSV: `{status['splat_quality_path']}`",
        f"- Ranked CSV: `{status['ranked_quality_path']}`",
        f"- Gate-bin CSV: `{status['gate_bin_path']}`",
        "",
        "## Best Top-K By F-Score",
        "",
    ]
    for row in status["best_topk_by_f_score"]:
        lines.append(
            f"- `{row['geometry_threshold']:.6g}`: top `{row['topk_fraction']:.6g}`, "
            f"retention `{row['retention_fraction']:.6g}`, F-score `{row['f_score']:.6g}`"
        )
    lines.extend(["", "## Ranked Retention Curve", "", format_ranked_table(ranked), "", "## Gate Bins", "", format_bin_table(bins)])
    return "\n".join(lines) + "\n"


def format_ranked_table(ranked: pd.DataFrame) -> str:
    columns = ["topk_fraction", "geometry_threshold", "retention_fraction", "precision", "recall", "f_score", "accuracy_mean", "gate_mean"]
    lines = [
        "| top_k | threshold | retention | precision | recall | f_score | accuracy_mean | gate_mean |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked[columns].itertuples(index=False):
        lines.append(
            f"| {row.topk_fraction:.6g} | {row.geometry_threshold:.6g} | {row.retention_fraction:.6g} | {row.precision:.6g} | "
            f"{row.recall:.6g} | {row.f_score:.6g} | {row.accuracy_mean:.6g} | {row.gate_mean:.6g} |"
        )
    return "\n".join(lines)


def format_bin_table(bins: pd.DataFrame) -> str:
    columns = ["bin_index", "geometry_threshold", "splat_count", "mean_gate", "within_threshold_fraction", "abs_gate_calibration_error"]
    lines = [
        "| bin | threshold | count | mean_gate | within_threshold | abs_error |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in bins[columns].itertuples(index=False):
        lines.append(
            f"| {row.bin_index} | {row.geometry_threshold:.6g} | {row.splat_count} | {row.mean_gate:.6g} | "
            f"{row.within_threshold_fraction:.6g} | {row.abs_gate_calibration_error:.6g} |"
        )
    return "\n".join(lines)


def format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def default_gate_quality_method_name(splats_path: str | Path | None) -> str:
    if splats_path is None:
        return "real_splats"
    return Path(splats_path).stem


def default_topk_fractions() -> tuple[float, ...]:
    return tuple(float(value) for value in (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0))
