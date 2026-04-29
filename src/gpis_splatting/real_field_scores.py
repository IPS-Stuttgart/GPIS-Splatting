from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gpis_splatting.gpis import load_model, predict_gpis, surface_band_probability
from gpis_splatting.real_gate_diagnostics import (
    best_topk_rows,
    correlation,
    finite_or_none,
    prepare_gate_diagnostic_inputs,
    rankdata_average,
    validate_diagnostic_config,
)
from gpis_splatting.real_geometry import nearest_neighbor_distances, resolve_scene_file
from gpis_splatting.serialization import write_json
from gpis_splatting.splats import load_splats


def run_tanks_temples_gpis_field_score_diagnostics(
    *,
    scene_dir: str | Path,
    model_path: str | Path,
    splats_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
    crop_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    method_name: str | None = None,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.1),
    topk_fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
    score_lambdas: tuple[float, ...] = (0.25, 0.5, 1.0),
    max_pred_points: int | None = 100_000,
    max_gt_points: int | None = 100_000,
    seed: int = 13,
    apply_alignment: bool | None = None,
    invert_alignment: bool = False,
    use_crop: bool = True,
    epsilon: float = 0.24,
    gate_floor: float = 0.0,
    batch_size: int = 4096,
    distance_chunk_size: int = 256,
) -> dict[str, Any]:
    validate_field_score_config(
        thresholds=thresholds,
        topk_fractions=topk_fractions,
        score_lambdas=score_lambdas,
        epsilon=epsilon,
        gate_floor=gate_floor,
        batch_size=batch_size,
        distance_chunk_size=distance_chunk_size,
    )
    scene_root = Path(scene_dir)
    resolved_model = resolve_scene_file(scene_root, model_path, "real_gpis_model.npz")
    inputs = prepare_gate_diagnostic_inputs(
        scene_dir=scene_root,
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
        gate_path=None,
        model_path=resolved_model,
        epsilon=epsilon,
        gate_floor=gate_floor,
        gate_batch_size=batch_size,
    )
    splats = load_splats(str(inputs.splats_path))
    query_points = splats.centers.detach().cpu().numpy().astype(np.float64)[inputs.splat_indices]
    model, model_metadata = load_model(str(resolved_model))
    prediction = predict_gpis(model, torch.from_numpy(query_points), batch_size=batch_size)
    field_table = build_field_table(
        inputs=inputs,
        query_points=query_points,
        prediction=prediction,
        current_gate=inputs.gates,
        epsilon=epsilon,
        gate_floor=gate_floor,
        score_lambdas=score_lambdas,
    )
    pred_to_gt = nearest_neighbor_distances(inputs.pred_points, inputs.gt_points, query_chunk_size=distance_chunk_size)
    field_table["nearest_gt_distance"] = pred_to_gt
    for threshold in thresholds:
        field_table[f"within_{format_threshold_label(threshold)}"] = pred_to_gt <= threshold

    score_names = score_columns(field_table)
    ranked = build_score_ranked_table(
        field_table=field_table,
        score_names=score_names,
        pred_to_gt=pred_to_gt,
        gt_points=inputs.gt_points,
        pred_points=inputs.pred_points,
        thresholds=thresholds,
        topk_fractions=topk_fractions,
        distance_chunk_size=distance_chunk_size,
    )
    summary = build_score_summary(field_table=field_table, ranked=ranked, score_names=score_names, pred_to_gt=pred_to_gt)
    out_dir = Path(output_dir) if output_dir is not None else inputs.scene_dir / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = inputs.method
    field_path = out_dir / f"{prefix}_gpis_field_scores.csv"
    summary_path = out_dir / f"{prefix}_gpis_field_score_summary.csv"
    ranked_path = out_dir / f"{prefix}_gpis_field_score_ranked.csv"
    status_path = out_dir / f"{prefix}_gpis_field_score_status.json"
    report_path = out_dir / f"{prefix}_gpis_field_score_report.md"
    field_table.to_csv(field_path, index=False)
    summary.to_csv(summary_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    status = {
        "schema_version": 1,
        "scene": inputs.scene_meta["scene"],
        "dataset": inputs.scene_meta.get("dataset"),
        "method": inputs.method,
        "scene_dir": str(inputs.scene_dir),
        "splats_path": str(inputs.splats_path),
        "model_path": str(resolved_model),
        "model_metadata": stringify_metadata(model_metadata),
        "ground_truth_path": str(inputs.ground_truth_path),
        "alignment_path": str(inputs.alignment_path) if inputs.alignment_path is not None else None,
        "alignment_applied": inputs.alignment_applied,
        "crop_path": str(inputs.crop_path) if inputs.crop_path is not None else None,
        "crop": inputs.crop_summary,
        "thresholds": list(thresholds),
        "topk_fractions": list(topk_fractions),
        "score_lambdas": list(score_lambdas),
        "epsilon": epsilon,
        "gate_floor": gate_floor,
        "max_pred_points": max_pred_points,
        "max_gt_points": max_gt_points,
        "pred_count_input": inputs.pred_count_input,
        "pred_count_sampled": inputs.pred_count_sampled,
        "pred_count_evaluated": inputs.pred_count_evaluated,
        "gt_count_input": inputs.gt_count_input,
        "gt_count_evaluated": inputs.gt_count_evaluated,
        "field_scores_path": str(field_path),
        "score_summary_path": str(summary_path),
        "score_ranked_path": str(ranked_path),
        "report_path": str(report_path),
        "best_by_spearman": best_summary_rows(summary, metric="spearman_score_vs_negative_distance"),
        "best_by_delta_f_score": best_summary_rows(summary, metric="delta_best_f_score_vs_full"),
    }
    write_json(status_path, status)
    report_path.write_text(format_field_score_report(status, summary), encoding="utf-8")
    return {
        "field_scores_path": field_path,
        "score_summary_path": summary_path,
        "score_ranked_path": ranked_path,
        "status_path": status_path,
        "report_path": report_path,
        "field_scores": field_table,
        "score_summary": summary,
        "score_ranked": ranked,
        "status": status,
    }


def validate_field_score_config(
    *,
    thresholds: tuple[float, ...],
    topk_fractions: tuple[float, ...],
    score_lambdas: tuple[float, ...],
    epsilon: float,
    gate_floor: float,
    batch_size: int,
    distance_chunk_size: int,
) -> None:
    validate_diagnostic_config(thresholds=thresholds, topk_fractions=topk_fractions, num_bins=1, distance_chunk_size=distance_chunk_size)
    if not score_lambdas:
        raise ValueError("At least one score lambda is required.")
    if any(value < 0.0 for value in score_lambdas):
        raise ValueError("Score lambdas must be non-negative.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if not 0.0 <= gate_floor <= 1.0:
        raise ValueError("gate_floor must be in [0, 1].")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")


def build_field_table(
    *,
    inputs: Any,
    query_points: np.ndarray,
    prediction: Any,
    current_gate: np.ndarray,
    epsilon: float,
    gate_floor: float,
    score_lambdas: tuple[float, ...],
) -> pd.DataFrame:
    mean = prediction.mean.detach().cpu().numpy()
    variance = prediction.variance.detach().cpu().numpy()
    std = prediction.std.detach().cpu().numpy()
    grad_norm = prediction.grad_norm.detach().cpu().numpy()
    distance = prediction.distance.detach().cpu().numpy()
    distance_std = prediction.distance_std.detach().cpu().numpy()
    raw_band = surface_band_probability(prediction, epsilon).detach().cpu().numpy()
    abs_distance = np.abs(distance)
    safe_epsilon = max(float(epsilon), 1e-12)
    table = pd.DataFrame(
        {
            "splat_index": inputs.splat_indices,
            "query_x": query_points[:, 0],
            "query_y": query_points[:, 1],
            "query_z": query_points[:, 2],
            "eval_x": inputs.pred_points[:, 0],
            "eval_y": inputs.pred_points[:, 1],
            "eval_z": inputs.pred_points[:, 2],
            "mu": mean,
            "variance": variance,
            "sigma": std,
            "grad_norm": grad_norm,
            "signed_distance": distance,
            "abs_signed_distance": abs_distance,
            "distance_std": distance_std,
            "score_current_gate": current_gate,
            "score_raw_surface_band": raw_band,
            "score_exp_neg_abs_distance": np.exp(-abs_distance / safe_epsilon),
            "score_negative_abs_distance": -abs_distance,
            "score_negative_distance_std": -distance_std,
            "score_variance_penalized_band": raw_band / (1.0 + distance_std / safe_epsilon),
            "score_variance_penalized_exp": np.exp(-abs_distance / safe_epsilon) / (1.0 + distance_std / safe_epsilon),
            "score_negative_abs_mu": -np.abs(mean),
        }
    )
    table["score_gate_floor_applied_raw_band"] = np.clip(gate_floor + (1.0 - gate_floor) * raw_band, 0.0, 1.0)
    for value in score_lambdas:
        label = format_label(value)
        table[f"score_combined_distance_uncertainty_l{label}"] = -abs_distance - float(value) * distance_std
    return table


def build_score_ranked_table(
    *,
    field_table: pd.DataFrame,
    score_names: list[str],
    pred_to_gt: np.ndarray,
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    thresholds: tuple[float, ...],
    topk_fractions: tuple[float, ...],
    distance_chunk_size: int,
) -> pd.DataFrame:
    rows = []
    n_points = int(pred_points.shape[0])
    for score_name in score_names:
        scores = sanitize_scores(field_table[score_name].to_numpy(dtype=np.float64))
        order = np.argsort(-scores, kind="mergesort")
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
                        "score_name": score_name,
                        "topk_fraction": float(fraction),
                        "selected_pred_point_count": count,
                        "retention_fraction": count / n_points,
                        "score_min": float(scores[selected].min()),
                        "score_mean": float(scores[selected].mean()),
                        "accuracy_mean": float(selected_distances.mean()),
                        "geometry_threshold": float(threshold),
                        "precision": precision,
                        "recall": recall,
                        "f_score": f_score,
                    }
                )
    return pd.DataFrame(rows)


def build_score_summary(*, field_table: pd.DataFrame, ranked: pd.DataFrame, score_names: list[str], pred_to_gt: np.ndarray) -> pd.DataFrame:
    rows = []
    for score_name in score_names:
        scores = sanitize_scores(field_table[score_name].to_numpy(dtype=np.float64))
        correlations = score_error_correlations(scores, pred_to_gt)
        score_ranked = ranked[ranked["score_name"] == score_name]
        full_rows = score_ranked[score_ranked["topk_fraction"] == 1.0]
        best_rows = best_topk_rows(score_ranked.drop(columns=["score_name"]))
        for best in best_rows:
            threshold = float(best["geometry_threshold"])
            full = full_rows[full_rows["geometry_threshold"] == threshold].iloc[0]
            rows.append(
                {
                    "score_name": score_name,
                    "geometry_threshold": threshold,
                    **correlations,
                    "best_topk_fraction": float(best["topk_fraction"]),
                    "best_retention_fraction": float(best["retention_fraction"]),
                    "best_selected_pred_point_count": int(best["selected_pred_point_count"]),
                    "best_precision": float(best["precision"]),
                    "best_recall": float(best["recall"]),
                    "best_f_score": float(best["f_score"]),
                    "full_precision": float(full["precision"]),
                    "full_recall": float(full["recall"]),
                    "full_f_score": float(full["f_score"]),
                    "delta_best_f_score_vs_full": float(best["f_score"] - full["f_score"]),
                    "score_min": float(np.min(scores)),
                    "score_max": float(np.max(scores)),
                    "score_mean": float(np.mean(scores)),
                }
            )
    return pd.DataFrame(rows)


def score_error_correlations(scores: np.ndarray, pred_to_gt: np.ndarray) -> dict[str, float | None]:
    return {
        "spearman_score_vs_negative_distance": finite_or_none(correlation(rankdata_average(scores), rankdata_average(-pred_to_gt))),
        "spearman_score_vs_distance": finite_or_none(correlation(rankdata_average(scores), rankdata_average(pred_to_gt))),
        "pearson_score_vs_negative_distance": finite_or_none(correlation(scores, -pred_to_gt)),
        "pearson_score_vs_distance": finite_or_none(correlation(scores, pred_to_gt)),
    }


def score_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns if column.startswith("score_")]


def sanitize_scores(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    fill = float(finite.min() - 1.0) if finite.size else 0.0
    return np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)


def best_summary_rows(summary: pd.DataFrame, *, metric: str) -> list[dict[str, Any]]:
    best = []
    for threshold, group in summary.groupby("geometry_threshold"):
        row = group.sort_values([metric, "best_f_score"], ascending=[False, False]).iloc[0]
        best.append(
            {
                "geometry_threshold": float(threshold),
                "score_name": row["score_name"],
                metric: maybe_float(row[metric]),
                "best_topk_fraction": float(row["best_topk_fraction"]),
                "best_f_score": float(row["best_f_score"]),
                "delta_best_f_score_vs_full": float(row["delta_best_f_score_vs_full"]),
            }
        )
    return best


def format_field_score_report(status: dict[str, Any], summary: pd.DataFrame) -> str:
    lines = [
        "# GPIS Field Score Diagnostics",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Method: `{status['method']}`",
        f"- Splat points evaluated: `{status['pred_count_evaluated']}`",
        f"- Ground-truth points evaluated: `{status['gt_count_evaluated']}`",
        f"- Field CSV: `{status['field_scores_path']}`",
        f"- Summary CSV: `{status['score_summary_path']}`",
        f"- Ranked CSV: `{status['score_ranked_path']}`",
        "",
        "## Best By Spearman(score, -distance)",
        "",
    ]
    for row in status["best_by_spearman"]:
        lines.append(format_best_row(row, metric="spearman_score_vs_negative_distance"))
    lines.extend(["", "## Best By F-Score Gain Over Full Retention", ""])
    for row in status["best_by_delta_f_score"]:
        lines.append(format_best_row(row, metric="delta_best_f_score_vs_full"))
    if not summary.empty:
        lines.extend(["", "## Summary Table", "", format_summary_table(summary)])
    return "\n".join(lines) + "\n"


def format_best_row(row: dict[str, Any], *, metric: str) -> str:
    return (
        f"- threshold `{row['geometry_threshold']:.6g}`: `{row['score_name']}`, "
        f"{metric} `{format_optional(row[metric])}`, top `{row['best_topk_fraction']:.6g}`, "
        f"best F-score `{row['best_f_score']:.6g}`, delta `{row['delta_best_f_score_vs_full']:.6g}`"
    )


def format_summary_table(summary: pd.DataFrame) -> str:
    columns = [
        "score_name",
        "geometry_threshold",
        "spearman_score_vs_negative_distance",
        "best_topk_fraction",
        "best_f_score",
        "delta_best_f_score_vs_full",
    ]
    lines = [
        "| score | threshold | spearman | top_k | best_f | delta_f |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    sorted_summary = summary[columns].sort_values(["geometry_threshold", "spearman_score_vs_negative_distance"], ascending=[True, False])
    for row in sorted_summary.itertuples(index=False):
        lines.append(
            f"| `{row.score_name}` | {row.geometry_threshold:.6g} | {format_optional(row.spearman_score_vs_negative_distance)} | "
            f"{row.best_topk_fraction:.6g} | {row.best_f_score:.6g} | {row.delta_best_f_score_vs_full:.6g} |"
        )
    return "\n".join(lines)


def stringify_metadata(metadata: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def format_optional(value: Any) -> str:
    parsed = maybe_float(value)
    return "n/a" if parsed is None else f"{parsed:.6g}"


def format_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def format_threshold_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def default_score_lambdas() -> tuple[float, ...]:
    return tuple(float(value) for value in (0.25, 0.5, 1.0))
