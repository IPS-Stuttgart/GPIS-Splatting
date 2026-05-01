from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gpis_splatting.real_geometry import format_threshold_label
from gpis_splatting.real_hard_negatives import run_tanks_temples_hard_negative_calibration
from gpis_splatting.real_splat_filtering import run_tanks_temples_calibrated_splat_filtering
from gpis_splatting.serialization import write_json


def run_tanks_temples_calibrated_confidence(
    *,
    scene_dir: str | Path,
    splats_path: str | Path | None = None,
    model_path: str | Path,
    method_name: str = "calibrated_confidence",
    output_dir: str | Path | None = None,
    calibration_threshold: float = 0.05,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.1),
    gate_thresholds: tuple[float, ...] = (0.25, 0.5, 0.75),
    max_source_splats: int | None = 5000,
    seed: int = 13,
    ground_truth_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
    crop_path: str | Path | None = None,
    max_pred_points: int | None = 100_000,
    max_gt_points: int | None = 100_000,
    apply_alignment: bool | None = None,
    invert_alignment: bool = False,
    use_crop: bool = True,
    epsilon: float = 0.24,
    gate_floor: float = 0.0,
    batch_size: int = 4096,
    distance_chunk_size: int = 256,
    render_split: str = "test",
    render_max_frames: int = 0,
    evaluate_render_metrics: bool = True,
    benchmark_target: str | Path | None = None,
) -> dict[str, Any]:
    """Run the calibrated GPIS-field confidence path as the primary splat-confidence workflow.

    The raw analytic GPIS zero-band gate is deliberately treated as a diagnostic baseline elsewhere.
    This routine generates hard-negative splat candidates, calibrates GPIS posterior field features to
    probabilities of geometric correctness, exports a gate-compatible confidence NPZ, then evaluates
    compacted and tau-scaled splat variants against geometry and optional render metrics.
    """
    validate_calibrated_confidence_config(calibration_threshold=calibration_threshold, thresholds=thresholds, render_max_frames=render_max_frames)
    scene_root = Path(scene_dir)
    out_dir = Path(output_dir) if output_dir is not None else scene_root / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)

    hard_negative_method = f"{method_name}_hard_negative"
    hard_negative = run_tanks_temples_hard_negative_calibration(
        scene_dir=scene_root,
        model_path=model_path,
        splats_path=splats_path,
        method_name=hard_negative_method,
        output_dir=out_dir,
        ground_truth_path=ground_truth_path,
        alignment_path=alignment_path,
        crop_path=crop_path,
        max_source_splats=max_source_splats,
        seed=seed,
        thresholds=thresholds,
        max_pred_points=max_pred_points,
        max_gt_points=max_gt_points,
        apply_alignment=apply_alignment,
        invert_alignment=invert_alignment,
        use_crop=use_crop,
        epsilon=epsilon,
        gate_floor=gate_floor,
        batch_size=batch_size,
        distance_chunk_size=distance_chunk_size,
    )
    calibrated_gate = select_calibrated_gate_path(hard_negative["status"], calibration_threshold=calibration_threshold)

    filtering_method = f"{method_name}_filtering"
    filtering = run_tanks_temples_calibrated_splat_filtering(
        scene_dir=scene_root,
        splats_path=hard_negative["generated_splats_path"],
        gate_path=calibrated_gate,
        method_name=filtering_method,
        output_dir=out_dir,
        gate_thresholds=gate_thresholds,
        ground_truth_path=ground_truth_path,
        alignment_path=alignment_path,
        crop_path=crop_path,
        thresholds=thresholds,
        max_pred_points=max_pred_points,
        max_gt_points=max_gt_points,
        seed=seed,
        apply_alignment=apply_alignment,
        invert_alignment=invert_alignment,
        use_crop=use_crop,
        distance_chunk_size=distance_chunk_size,
        render_split=render_split,
        render_max_frames=render_max_frames,
        evaluate_render_metrics=evaluate_render_metrics,
        benchmark_target=benchmark_target,
    )

    calibration_row = select_best_calibrator(hard_negative["status"], calibration_threshold=calibration_threshold)
    filtering_row = select_best_filtering_variant(filtering["comparison"], calibration_threshold=calibration_threshold)
    status_path = out_dir / f"{method_name}_calibrated_confidence_status.json"
    report_path = out_dir / f"{method_name}_calibrated_confidence_report.md"
    status = {
        "schema_version": 1,
        "method": method_name,
        "scene_dir": str(scene_root),
        "input_splats_path": str(splats_path) if splats_path is not None else None,
        "model_path": str(model_path),
        "calibration_threshold": calibration_threshold,
        "thresholds": list(thresholds),
        "gate_thresholds": list(gate_thresholds),
        "hard_negative_method": hard_negative_method,
        "filtering_method": filtering_method,
        "generated_splats_path": str(hard_negative["generated_splats_path"]),
        "calibrated_gate_path": str(calibrated_gate),
        "hard_negative_status_path": str(hard_negative["status_path"]),
        "hard_negative_report_path": str(hard_negative["report_path"]),
        "filtering_status_path": str(filtering["status_path"]),
        "filtering_report_path": str(filtering["report_path"]),
        "best_calibrator": calibration_row,
        "best_filtering_variant": filtering_row,
        "status_path": str(status_path),
        "report_path": str(report_path),
    }
    write_json(status_path, status)
    report_path.write_text(format_calibrated_confidence_report(status), encoding="utf-8")
    return {
        "status_path": status_path,
        "report_path": report_path,
        "status": status,
        "hard_negative": hard_negative,
        "filtering": filtering,
    }


def validate_calibrated_confidence_config(*, calibration_threshold: float, thresholds: tuple[float, ...], render_max_frames: int) -> None:
    if calibration_threshold <= 0.0:
        raise ValueError("calibration_threshold must be positive.")
    if not thresholds:
        raise ValueError("At least one geometry threshold is required.")
    if all(abs(threshold - calibration_threshold) > 1e-12 for threshold in thresholds):
        raise ValueError("calibration_threshold must also be listed in thresholds so a calibrated gate can be selected.")
    if render_max_frames < 0:
        raise ValueError("render_max_frames must be non-negative.")


def select_calibrated_gate_path(status: dict[str, Any], *, calibration_threshold: float) -> Path:
    label = format_threshold_label(calibration_threshold)
    gate_paths = status.get("calibrated_gate_paths") or {}
    if label in gate_paths:
        return Path(gate_paths[label])
    if str(calibration_threshold) in gate_paths:
        return Path(gate_paths[str(calibration_threshold)])
    available = ", ".join(sorted(gate_paths)) or "none"
    raise FileNotFoundError(f"No calibrated gate for threshold {calibration_threshold:g}. Available calibrated gates: {available}.")


def select_best_calibrator(status: dict[str, Any], *, calibration_threshold: float) -> dict[str, Any] | None:
    candidates = [row for row in status.get("best_calibrators", []) if abs(float(row.get("geometry_threshold", -1.0)) - calibration_threshold) <= 1e-12]
    if candidates:
        return dict(candidates[0])
    return None


def select_best_filtering_variant(comparison: pd.DataFrame, *, calibration_threshold: float) -> dict[str, Any] | None:
    if comparison.empty:
        return None
    table = comparison.copy()
    if "geometry_threshold" in table.columns:
        table = table[(table["geometry_threshold"].astype(float) - calibration_threshold).abs() <= 1e-12]
    if table.empty:
        return None
    ranked = table.sort_values(["f_score", "chamfer_l1", "retention_fraction"], ascending=[False, True, True], na_position="last")
    row = ranked.iloc[0].to_dict()
    return {key: serializable_value(value) for key, value in row.items()}


def serializable_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def format_calibrated_confidence_report(status: dict[str, Any]) -> str:
    best_calibrator = status.get("best_calibrator") or {}
    best_variant = status.get("best_filtering_variant") or {}
    lines = [
        "# Calibrated GPIS Confidence Evidence",
        "",
        "This report treats the analytic GPIS zero-band gate as a diagnostic baseline and uses calibrated GPIS posterior-field features as the primary splat-confidence interface.",
        "",
        "## Inputs",
        "",
        f"- Method: `{status['method']}`",
        f"- Scene directory: `{status['scene_dir']}`",
        f"- GPIS model: `{status['model_path']}`",
        f"- Calibration threshold: `{status['calibration_threshold']}`",
        f"- Calibrated gate: `{status['calibrated_gate_path']}`",
        f"- Generated hard-negative splats: `{status['generated_splats_path']}`",
        "",
        "## Best Calibrator",
        "",
    ]
    if best_calibrator:
        lines.extend(
            [
                f"- Method: `{best_calibrator.get('method_name')}`",
                f"- Family: `{best_calibrator.get('method_family')}`",
                f"- Feature set: `{best_calibrator.get('feature_set')}`",
                f"- Brier: `{format_metric(best_calibrator.get('brier'))}`",
                f"- NLL: `{format_metric(best_calibrator.get('nll'))}`",
                f"- ECE: `{format_metric(best_calibrator.get('ece'))}`",
                f"- AUC: `{format_metric(best_calibrator.get('auc'))}`",
                f"- AP: `{format_metric(best_calibrator.get('average_precision'))}`",
            ]
        )
    else:
        lines.append("- No matching calibrated-confidence row was found.")
    lines.extend(["", "## Best Filtering Variant", ""])
    if best_variant:
        lines.extend(
            [
                f"- Variant: `{best_variant.get('variant')}`",
                f"- Variant kind: `{best_variant.get('variant_kind')}`",
                f"- Retention: `{format_metric(best_variant.get('retention_fraction'))}`",
                f"- Precision: `{format_metric(best_variant.get('precision'))}`",
                f"- Recall: `{format_metric(best_variant.get('recall'))}`",
                f"- F-score: `{format_metric(best_variant.get('f_score'))}`",
                f"- Chamfer L1: `{format_metric(best_variant.get('chamfer_l1'))}`",
                f"- Mean PSNR: `{format_metric(best_variant.get('mean_psnr'))}`",
                f"- Mean SSIM: `{format_metric(best_variant.get('mean_ssim'))}`",
            ]
        )
    else:
        lines.append("- No filtering variant could be selected.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Hard-negative workflow report: `{status['hard_negative_report_path']}`",
            f"- Filtering report: `{status['filtering_report_path']}`",
            f"- Status JSON: `{status['status_path']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)
