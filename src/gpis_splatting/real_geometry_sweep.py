from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.real_geometry import evaluate_tanks_temples_geometry
from gpis_splatting.serialization import write_json


def run_tanks_temples_gate_sweep(
    *,
    scene_dir: str | Path,
    splats_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
    crop_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    method_name: str | None = None,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.1),
    gate_thresholds: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5),
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
    if not gate_thresholds:
        raise ValueError("At least one gate threshold is required.")
    if any(threshold < 0.0 or threshold > 1.0 for threshold in gate_thresholds):
        raise ValueError("Gate thresholds must be in [0, 1].")
    scene_root = Path(scene_dir)
    out_dir = Path(output_dir) if output_dir is not None else scene_root / "evaluations"
    resolved_method = method_name or (Path(splats_path).stem if splats_path is not None else "real_splats")

    geometry_status = evaluate_tanks_temples_geometry(
        scene_dir=scene_root,
        splats_path=splats_path,
        ground_truth_path=ground_truth_path,
        alignment_path=alignment_path,
        crop_path=crop_path,
        output_dir=out_dir,
        method_name=resolved_method,
        thresholds=thresholds,
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
        gate_thresholds=gate_thresholds,
        gate_batch_size=gate_batch_size,
        distance_chunk_size=distance_chunk_size,
    )
    if not geometry_status["status"]["gate_available"]:
        raise ValueError("Gate sweep requires --gate-path or --model-path so splat gates can be evaluated.")

    summary = pd.read_csv(geometry_status["summary_path"])
    threshold_metrics = pd.read_csv(geometry_status["threshold_metrics_path"])
    sweep = build_gate_sweep_table(summary, threshold_metrics)
    sweep_path = out_dir / f"{resolved_method}_gate_sweep.csv"
    status_path = out_dir / f"{resolved_method}_gate_sweep_status.json"
    report_path = out_dir / f"{resolved_method}_gate_sweep_report.md"
    sweep.to_csv(sweep_path, index=False)
    status = {
        "schema_version": 1,
        "scene": geometry_status["status"]["scene"],
        "method": resolved_method,
        "gate_thresholds": list(gate_thresholds),
        "geometry_thresholds": list(thresholds),
        "geometry_status_path": str(geometry_status["status_path"]),
        "geometry_summary_path": str(geometry_status["summary_path"]),
        "geometry_threshold_metrics_path": str(geometry_status["threshold_metrics_path"]),
        "sweep_path": str(sweep_path),
        "report_path": str(report_path),
        "best_by_f_score": best_gate_rows(sweep, metric="f_score"),
        "best_by_chamfer_l1": best_gate_rows(sweep, metric="chamfer_l1", ascending=True),
    }
    write_json(status_path, status)
    report_path.write_text(format_gate_sweep_report(status, sweep), encoding="utf-8")
    return {
        "sweep_path": sweep_path,
        "status_path": status_path,
        "report_path": report_path,
        "geometry_status": geometry_status,
        "sweep": sweep,
        "status": status,
    }


def build_gate_sweep_table(summary: pd.DataFrame, threshold_metrics: pd.DataFrame) -> pd.DataFrame:
    if "all" not in set(summary["group"]):
        raise ValueError("Geometry summary is missing the all group.")
    all_summary = summary[summary["group"] == "all"].iloc[0]
    all_count = max(float(all_summary["pred_point_count"]), 1.0)
    rows = []
    ordered_groups = ["all"] + sorted(
        [group for group in summary["group"].unique() if str(group).startswith("gate_ge_")],
        key=parse_gate_group_threshold,
    )
    for group in ordered_groups:
        summary_row = summary[summary["group"] == group].iloc[0]
        threshold_rows = threshold_metrics[threshold_metrics["group"] == group]
        gate_threshold = np.nan if group == "all" else parse_gate_group_threshold(group)
        selection = "all" if group == "all" else "gate_ge"
        for threshold_row in threshold_rows.itertuples(index=False):
            baseline = threshold_metrics[(threshold_metrics["group"] == "all") & (threshold_metrics["threshold"] == threshold_row.threshold)].iloc[0]
            rows.append(
                {
                    "scene": summary_row["scene"],
                    "dataset": summary_row["dataset"],
                    "method": summary_row["method"],
                    "selection": selection,
                    "group": group,
                    "gate_threshold": gate_threshold,
                    "geometry_threshold": float(threshold_row.threshold),
                    "selected_pred_point_count": int(summary_row["pred_point_count"]),
                    "retention_fraction": float(summary_row["pred_point_count"]) / all_count,
                    "gate_mean": summary_row["gate_mean"],
                    "accuracy_mean": float(summary_row["accuracy_mean"]),
                    "completion_mean": float(summary_row["completion_mean"]),
                    "chamfer_l1": float(summary_row["chamfer_l1"]),
                    "precision": float(threshold_row.precision),
                    "recall": float(threshold_row.recall),
                    "f_score": float(threshold_row.f_score),
                    "delta_precision_vs_all": float(threshold_row.precision - baseline["precision"]),
                    "delta_recall_vs_all": float(threshold_row.recall - baseline["recall"]),
                    "delta_f_score_vs_all": float(threshold_row.f_score - baseline["f_score"]),
                    "delta_chamfer_l1_vs_all": float(summary_row["chamfer_l1"] - all_summary["chamfer_l1"]),
                }
            )
    return pd.DataFrame(rows)


def parse_gate_group_threshold(group: str) -> float:
    if group == "all":
        return -1.0
    prefix = "gate_ge_"
    if not group.startswith(prefix):
        raise ValueError(f"Unsupported gate group {group!r}.")
    return float(group[len(prefix) :].replace("p", "."))


def best_gate_rows(sweep: pd.DataFrame, *, metric: str, ascending: bool = False) -> list[dict[str, Any]]:
    gate_rows = sweep[sweep["selection"] == "gate_ge"]
    best = []
    for geometry_threshold, group in gate_rows.groupby("geometry_threshold"):
        row = group.sort_values(metric, ascending=ascending).iloc[0]
        best.append(
            {
                "geometry_threshold": float(geometry_threshold),
                "gate_threshold": float(row["gate_threshold"]),
                metric: float(row[metric]),
                "retention_fraction": float(row["retention_fraction"]),
                "selected_pred_point_count": int(row["selected_pred_point_count"]),
            }
        )
    return best


def format_gate_sweep_report(status: dict[str, Any], sweep: pd.DataFrame) -> str:
    lines = [
        "# Tanks and Temples Gate Sweep",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Method: `{status['method']}`",
        f"- Sweep CSV: `{status['sweep_path']}`",
        f"- Geometry summary CSV: `{status['geometry_summary_path']}`",
        f"- Geometry threshold CSV: `{status['geometry_threshold_metrics_path']}`",
        "",
        "## Best F-Score By Geometry Threshold",
        "",
    ]
    for row in status["best_by_f_score"]:
        lines.append(
            f"- `{row['geometry_threshold']:.6g}`: gate >= `{row['gate_threshold']:.6g}`, "
            f"F-score `{row['f_score']:.6g}`, retention `{row['retention_fraction']:.6g}`"
        )
    lines.extend(["", "## Sweep Table", "", format_sweep_table(sweep)])
    return "\n".join(lines) + "\n"


def format_sweep_table(sweep: pd.DataFrame) -> str:
    columns = ["selection", "gate_threshold", "geometry_threshold", "retention_fraction", "precision", "recall", "f_score", "chamfer_l1"]
    lines = [
        "| selection | gate_threshold | geometry_threshold | retention | precision | recall | f_score | chamfer_l1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sweep[columns].itertuples(index=False):
        gate_threshold = "all" if pd.isna(row.gate_threshold) else f"{row.gate_threshold:.6g}"
        lines.append(
            f"| `{row.selection}` | {gate_threshold} | {row.geometry_threshold:.6g} | {row.retention_fraction:.6g} | "
            f"{row.precision:.6g} | {row.recall:.6g} | {row.f_score:.6g} | {row.chamfer_l1:.6g} |"
        )
    return "\n".join(lines)


def default_gate_sweep_method_name(splats_path: str | Path | None) -> str:
    if splats_path is None:
        return "real_splats"
    return Path(splats_path).stem


def default_gate_thresholds() -> tuple[float, ...]:
    return tuple(float(value) for value in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5))
