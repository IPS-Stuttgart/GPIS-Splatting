from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.real_geometry import format_threshold_label
from gpis_splatting.real_splat_filtering import run_tanks_temples_calibrated_splat_filtering
from gpis_splatting.serialization import write_json
from gpis_splatting.trained_3dgs_evaluation import TRAINED_3DGS_RENDERERS, run_trained_3dgs_gpis_experiment


@dataclass(frozen=True)
class ActualTrained3DGSInput:
    name: str
    trained_ply_path: Path
    calibrated_rendered_predictions_root: Path | None = None
    raw_rendered_predictions_root: Path | None = None


@dataclass(frozen=True)
class ActualTrained3DGSAFConfig:
    scene_dir: Path
    gpis_model_path: Path
    baseline: ActualTrained3DGSInput
    regularized: ActualTrained3DGSInput
    output_dir: Path
    matrix_name: str = "trained_3dgs_af_matrix"
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.1)
    primary_geometry_threshold: float = 0.05
    calibration_threshold: float = 0.05
    gate_thresholds: tuple[float, ...] = (0.25, 0.5, 0.75)
    max_pred_points: int | None = None
    max_gt_points: int | None = 150_000
    seed: int = 13
    missing_gate_value: float = 1.0
    iteration: int = 30_000
    opacity_mode: str = "logit"
    opacity_scale_floor: float = 0.0
    renderer: str = "none"
    render_command_template: str | None = None
    prediction_subdir: str = ""
    render_split: str = "test"
    compute_lpips: bool = False
    require_all_images: bool = True
    require_all_variants: bool = True
    require_render_metrics: bool = True
    require_full_matrix: bool = True
    benchmark_target: Path | None = None
    gsplat_device: str = "auto"
    gsplat_max_frames: int | None = None
    gsplat_max_gaussians: int | None = None


@dataclass(frozen=True)
class ActualModelEvidence:
    input: ActualTrained3DGSInput
    calibrated_result: dict[str, Any]
    raw_result: dict[str, Any]
    raw_gate_path: Path
    raw_filtering_path: Path
    calibrated_filtering_path: Path


def run_actual_trained_3dgs_af_matrix(config: ActualTrained3DGSAFConfig) -> dict[str, Any]:
    validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_model_evidence(config, config.baseline, role="baseline")
    regularized = run_model_evidence(config, config.regularized, role="regularized")
    results = build_af_results(config, baseline, regularized)
    results = add_baseline_deltas(results, baseline_case="A")
    checks = build_checks(results, require_render_metrics=config.require_render_metrics)
    passed = bool(checks["passed"].all()) if not checks.empty else True
    paths = write_outputs(config, results, checks, passed)
    if config.require_full_matrix and not passed:
        failing = ", ".join(checks.loc[~checks["passed"], "case_id"].astype(str).tolist())
        raise ValueError(f"Actual trained 3DGS A-F matrix is incomplete for cases: {failing}")
    return {"baseline": baseline, "regularized": regularized, "results": results, "checks": checks, "passed": passed, **paths}


def validate_config(config: ActualTrained3DGSAFConfig) -> None:
    if not config.scene_dir.is_dir():
        raise FileNotFoundError(f"Prepared scene directory does not exist: {config.scene_dir}")
    if not config.gpis_model_path.exists():
        raise FileNotFoundError(f"GPIS model does not exist: {config.gpis_model_path}")
    for item in (config.baseline, config.regularized):
        if not item.trained_ply_path.is_file():
            raise FileNotFoundError(f"Trained 3DGS PLY does not exist for {item.name}: {item.trained_ply_path}")
    if config.renderer not in TRAINED_3DGS_RENDERERS:
        raise ValueError(f"renderer must be one of {TRAINED_3DGS_RENDERERS}.")
    if all(abs(float(threshold) - float(config.calibration_threshold)) > 1e-12 for threshold in config.thresholds):
        raise ValueError("calibration_threshold must be included in thresholds.")
    if not 0.0 <= config.missing_gate_value <= 1.0:
        raise ValueError("missing_gate_value must be in [0, 1].")


def run_model_evidence(config: ActualTrained3DGSAFConfig, model: ActualTrained3DGSInput, *, role: str) -> ActualModelEvidence:
    model_dir = config.output_dir / role
    model_dir.mkdir(parents=True, exist_ok=True)
    calibrated_method = f"{model.name}_calibrated"
    raw_method = f"{model.name}_raw"
    calibrated_result = run_single_model_experiment(
        config,
        model,
        method_name=calibrated_method,
        output_root=model_dir / "calibrated",
        gate_path=None,
        rendered_predictions_root=model.calibrated_rendered_predictions_root,
    )
    calibration = calibrated_result.get("calibration")
    if calibration is None:
        raise ValueError(f"No calibration output was produced for {model.name}.")
    scoring = calibrated_result.get("scoring")
    if scoring is None:
        raise ValueError(f"No GPIS field-score output was produced for {model.name}.")
    gaussian_count = int(calibrated_result["status"]["gaussian_count"])
    raw_gate_path = export_raw_surface_band_gate(
        field_scores_path=Path(scoring["field_scores_path"]),
        gate_count=gaussian_count,
        output_path=model_dir / f"{raw_method}_gate.npz",
        missing_gate_value=config.missing_gate_value,
    )
    raw_result = run_single_model_experiment(
        config,
        model,
        method_name=raw_method,
        output_root=model_dir / "raw",
        gate_path=raw_gate_path,
        rendered_predictions_root=model.raw_rendered_predictions_root,
    )
    raw_filtering = run_gate_filtering(config, model_dir, method_name=raw_method, splats_path=Path(calibrated_result["status"]["splats_path"]), gate_path=raw_gate_path)
    calibrated_filtering = run_gate_filtering(
        config,
        model_dir,
        method_name=calibrated_method,
        splats_path=Path(calibrated_result["status"]["splats_path"]),
        gate_path=Path(calibrated_result["status"]["gate_path"]),
    )
    return ActualModelEvidence(
        input=model,
        calibrated_result=calibrated_result,
        raw_result=raw_result,
        raw_gate_path=raw_gate_path,
        raw_filtering_path=Path(raw_filtering["comparison_path"]),
        calibrated_filtering_path=Path(calibrated_filtering["comparison_path"]),
    )


def run_single_model_experiment(
    config: ActualTrained3DGSAFConfig,
    model: ActualTrained3DGSInput,
    *,
    method_name: str,
    output_root: Path,
    gate_path: Path | None,
    rendered_predictions_root: Path | None,
) -> dict[str, Any]:
    renderer = config.renderer
    if renderer == "precomputed" and rendered_predictions_root is None:
        renderer = "none"
    return run_trained_3dgs_gpis_experiment(
        scene_dir=config.scene_dir,
        trained_ply_path=model.trained_ply_path,
        method_name=method_name,
        gpis_model_path=None if gate_path is not None else config.gpis_model_path,
        gate_path=gate_path,
        evaluations_dir=output_root / "evaluations",
        variants_dir=output_root / "variants",
        render_output_root=output_root / "renders",
        thresholds=config.thresholds,
        calibration_threshold=config.calibration_threshold,
        gate_thresholds=config.gate_thresholds,
        max_pred_points=config.max_pred_points,
        max_gt_points=config.max_gt_points,
        seed=config.seed,
        missing_gate_value=config.missing_gate_value,
        iteration=config.iteration,
        opacity_mode=config.opacity_mode,
        opacity_scale_floor=config.opacity_scale_floor,
        renderer=renderer,
        render_command_template=config.render_command_template,
        rendered_predictions_root=rendered_predictions_root,
        prediction_subdir=config.prediction_subdir,
        render_split=config.render_split,
        compute_lpips=config.compute_lpips,
        require_all_images=config.require_all_images,
        require_all_variants=config.require_all_variants,
        benchmark_target=config.benchmark_target,
        gsplat_device=config.gsplat_device,
        gsplat_max_frames=config.gsplat_max_frames,
        gsplat_max_gaussians=config.gsplat_max_gaussians,
    )


def run_gate_filtering(config: ActualTrained3DGSAFConfig, output_root: Path, *, method_name: str, splats_path: Path, gate_path: Path) -> dict[str, Any]:
    return run_tanks_temples_calibrated_splat_filtering(
        scene_dir=config.scene_dir,
        splats_path=splats_path,
        gate_path=gate_path,
        method_name=method_name,
        output_dir=output_root / "filtering",
        gate_thresholds=config.gate_thresholds,
        include_baseline=True,
        write_scaled=True,
        write_filtered=True,
        thresholds=config.thresholds,
        max_pred_points=config.max_pred_points,
        max_gt_points=config.max_gt_points,
        seed=config.seed,
        distance_chunk_size=256,
        render_split=config.render_split,
        render_max_frames=0,
        evaluate_render_metrics=False,
        benchmark_target=config.benchmark_target,
    )


def export_raw_surface_band_gate(*, field_scores_path: Path, gate_count: int, output_path: Path, missing_gate_value: float) -> Path:
    scores = pd.read_csv(field_scores_path)
    if "splat_index" not in scores.columns or "score_raw_surface_band" not in scores.columns:
        raise ValueError("Field-score table must contain splat_index and score_raw_surface_band columns.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_index = scores["splat_index"].to_numpy(dtype=np.int64)
    gate = np.full((gate_count,), float(missing_gate_value), dtype=np.float64)
    gate[scored_index] = np.clip(scores["score_raw_surface_band"].to_numpy(dtype=np.float64), 0.0, 1.0)
    scored_mask = np.zeros((gate_count,), dtype=bool)
    scored_mask[scored_index] = True
    np.savez_compressed(
        output_path,
        gate=gate,
        raw_gate=gate,
        splat_index=np.arange(gate_count, dtype=np.int64),
        scored_splat_index=scored_index,
        scored_mask=scored_mask,
        missing_gate_value=np.asarray(float(missing_gate_value), dtype=np.float64),
    )
    return output_path


def build_af_results(config: ActualTrained3DGSAFConfig, baseline: ActualModelEvidence, regularized: ActualModelEvidence) -> pd.DataFrame:
    rows = [
        build_case(config, "A", "plain_3dgs", "Plain trained 3DGS", baseline, filtering_path=baseline.calibrated_filtering_path, filtering_kind="baseline", render_result=baseline.calibrated_result, render_variant="baseline"),
        build_case(config, "B", "raw_gpis_gate", "Raw GPIS gate post-hoc", baseline, filtering_path=baseline.raw_filtering_path, exclude_filtering_kinds={"baseline", "random_same_retention"}, render_result=baseline.raw_result),
        build_case(config, "C", "calibrated_confidence", "Calibrated GPIS confidence post-hoc", baseline, filtering_path=baseline.calibrated_filtering_path, filtering_kind="gate_scaled", render_result=baseline.calibrated_result, render_variant="gate_scaled", calibration_result=baseline.calibrated_result),
        build_case(config, "D", "calibrated_pruning_refinement", "Calibrated pruning/refinement", baseline, filtering_path=baseline.calibrated_filtering_path, exclude_filtering_kinds={"baseline", "gate_scaled", "random_same_retention"}, render_result=baseline.calibrated_result),
        build_case(config, "E", "gpis_training_regularized_3dgs", "GPIS training-time regularizer", regularized, filtering_path=regularized.calibrated_filtering_path, filtering_kind="baseline", render_result=regularized.calibrated_result, render_variant="baseline"),
        build_case(config, "F", "regularized_plus_calibrated_confidence", "Regularizer plus calibrated confidence", regularized, filtering_path=regularized.calibrated_filtering_path, exclude_filtering_kinds={"baseline", "random_same_retention"}, render_result=regularized.calibrated_result, calibration_result=regularized.calibrated_result),
    ]
    return pd.DataFrame(rows)


def build_case(
    config: ActualTrained3DGSAFConfig,
    case_id: str,
    short_name: str,
    method_name: str,
    evidence: ActualModelEvidence,
    *,
    filtering_path: Path,
    filtering_kind: str | None = None,
    exclude_filtering_kinds: set[str] | None = None,
    render_result: dict[str, Any] | None = None,
    render_variant: str | None = None,
    calibration_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "short_name": short_name,
        "method_name": method_name,
        "trained_ply_path": str(evidence.input.trained_ply_path),
        "gaussian_count": evidence.calibrated_result["status"]["gaussian_count"],
        "variant": None,
        "variant_kind": None,
        "retained_count": np.nan,
        "retention_fraction": np.nan,
        "gate_threshold": np.nan,
        "geometry_threshold": config.primary_geometry_threshold,
        "precision": np.nan,
        "recall": np.nan,
        "f_score": np.nan,
        "chamfer_l1": np.nan,
        "chamfer_l2": np.nan,
        "mean_psnr": np.nan,
        "mean_ssim": np.nan,
        "mean_lpips_vgg": np.nan,
        "image_count": np.nan,
        "calibration_auc": np.nan,
        "calibration_auprc": np.nan,
        "calibration_brier": np.nan,
        "calibration_ece": np.nan,
        "source_artifacts": "",
    }
    sources = [str(filtering_path)]
    row.update(load_filtering_metrics(filtering_path, config.primary_geometry_threshold, preferred_kind=filtering_kind, exclude_kinds=exclude_filtering_kinds))
    if render_result is not None and render_result.get("render_evaluation") is not None:
        render_path = Path(render_result["render_evaluation"]["comparison_path"])
        row.update(load_render_metrics(render_path, preferred_variant=render_variant, fallback_variant=row.get("variant")))
        sources.append(str(render_path))
    if calibration_result is not None and calibration_result.get("calibration") is not None:
        summary_path = Path(calibration_result["calibration"]["summary_path"])
        row.update(load_calibration_metrics(summary_path, config.calibration_threshold))
        sources.append(str(summary_path))
    row["source_artifacts"] = ";".join(dict.fromkeys(sources))
    return row


def load_filtering_metrics(path: Path, threshold: float, *, preferred_kind: str | None, exclude_kinds: set[str] | None) -> dict[str, Any]:
    table = select_geometry_threshold(pd.read_csv(path), threshold)
    if preferred_kind and "variant_kind" in table.columns and preferred_kind in set(table["variant_kind"].astype(str)):
        table = table[table["variant_kind"].astype(str) == preferred_kind]
    if exclude_kinds and "variant_kind" in table.columns:
        filtered = table[~table["variant_kind"].astype(str).isin(exclude_kinds)]
        if not filtered.empty:
            table = filtered
    row = sort_geometry_rows(table).iloc[0]
    return {
        "variant": value(row, "variant"),
        "variant_kind": value(row, "variant_kind"),
        "geometry_threshold": number(row, "geometry_threshold"),
        "gate_threshold": number(row, "gate_threshold"),
        "retained_count": number(row, "retained_count"),
        "retention_fraction": number(row, "retention_fraction"),
        "precision": number(row, "precision"),
        "recall": number(row, "recall"),
        "f_score": number(row, "f_score"),
        "chamfer_l1": number(row, "chamfer_l1"),
        "chamfer_l2": number(row, "chamfer_l2"),
    }


def load_render_metrics(path: Path, *, preferred_variant: str | None, fallback_variant: Any) -> dict[str, Any]:
    table = pd.read_csv(path)
    candidates = table
    for variant in (preferred_variant, fallback_variant):
        if variant is not None and "variant" in table.columns and str(variant) in set(table["variant"].astype(str)):
            candidates = table[table["variant"].astype(str) == str(variant)]
            break
    row = candidates.sort_values([column for column in ("mean_psnr", "mean_ssim") if column in candidates.columns], ascending=False).iloc[0]
    return {
        "variant": value(row, "variant"),
        "variant_kind": value(row, "variant_kind"),
        "retained_count": number(row, "retained_count"),
        "retention_fraction": number(row, "retention_fraction"),
        "gate_threshold": number(row, "gate_threshold"),
        "mean_psnr": number(row, "mean_psnr"),
        "mean_ssim": number(row, "mean_ssim"),
        "mean_lpips_vgg": number(row, "mean_lpips_vgg"),
        "image_count": number(row, "image_count"),
    }


def load_calibration_metrics(path: Path, threshold: float) -> dict[str, Any]:
    table = select_geometry_threshold(pd.read_csv(path), threshold)
    row = table.sort_values([column for column in ("brier", "ece") if column in table.columns], ascending=True).iloc[0]
    return {
        "calibration_auc": number(row, "auc"),
        "calibration_auprc": number(row, "average_precision"),
        "calibration_brier": number(row, "brier"),
        "calibration_ece": number(row, "ece"),
    }


def select_geometry_threshold(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    column = "geometry_threshold" if "geometry_threshold" in table.columns else "threshold" if "threshold" in table.columns else None
    if column is None or table.empty:
        return table
    distances = (table[column].astype(float) - float(threshold)).abs()
    nearest = float(table.loc[distances.idxmin(), column])
    return table[table[column].astype(float) == nearest]


def sort_geometry_rows(table: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ("f_score", "precision", "chamfer_l1") if column in table.columns]
    ascending = [False if column != "chamfer_l1" else True for column in columns]
    return table.sort_values(columns, ascending=ascending) if columns else table


def value(row: pd.Series, column: str) -> Any:
    if column not in row or pd.isna(row[column]):
        return None
    return row[column]


def number(row: pd.Series, column: str) -> float:
    if column not in row or pd.isna(row[column]):
        return np.nan
    return float(row[column])


def add_baseline_deltas(results: pd.DataFrame, *, baseline_case: str) -> pd.DataFrame:
    output = results.copy()
    baseline = output[output["case_id"] == baseline_case]
    higher = {"precision", "recall", "f_score", "mean_psnr", "mean_ssim", "calibration_auc", "calibration_auprc"}
    lower = {"chamfer_l1", "chamfer_l2", "mean_lpips_vgg", "calibration_brier", "calibration_ece"}
    for metric in sorted(higher | lower):
        column = f"delta_{metric}_vs_{baseline_case}"
        output[column] = np.nan
        if baseline.empty or metric not in output.columns or pd.isna(baseline.iloc[0][metric]):
            continue
        base_value = float(baseline.iloc[0][metric])
        output[column] = output[metric].astype(float) - base_value if metric in higher else base_value - output[metric].astype(float)
    return output


def build_checks(results: pd.DataFrame, *, require_render_metrics: bool) -> pd.DataFrame:
    rows = []
    for row in results.to_dict(orient="records"):
        has_geometry = any(not pd.isna(row.get(metric)) for metric in ("f_score", "chamfer_l1", "chamfer_l2"))
        has_render = any(not pd.isna(row.get(metric)) for metric in ("mean_psnr", "mean_ssim", "mean_lpips_vgg"))
        has_source = bool(row.get("source_artifacts"))
        passed = has_source and has_geometry and (has_render or not require_render_metrics)
        rows.append({"case_id": row["case_id"], "short_name": row["short_name"], "passed": bool(passed), "has_geometry": bool(has_geometry), "has_render": bool(has_render), "has_source_artifact": bool(has_source), "details": "actual trained evidence loaded" if passed else "missing geometry/render/source evidence"})
    return pd.DataFrame(rows)


def write_outputs(config: ActualTrained3DGSAFConfig, results: pd.DataFrame, checks: pd.DataFrame, passed: bool) -> dict[str, Path]:
    prefix = config.matrix_name
    results_path = config.output_dir / f"{prefix}_actual_results.csv"
    checks_path = config.output_dir / f"{prefix}_actual_checks.csv"
    status_path = config.output_dir / f"{prefix}_actual_status.json"
    report_path = config.output_dir / f"{prefix}_actual_report.md"
    results.to_csv(results_path, index=False)
    checks.to_csv(checks_path, index=False)
    status = {
        "schema_version": 1,
        "matrix_name": prefix,
        "passed": passed,
        "scene_dir": str(config.scene_dir),
        "results_path": str(results_path),
        "checks_path": str(checks_path),
        "report_path": str(report_path),
        "case_count": int(len(results)),
        "passed_case_count": int(checks["passed"].sum()) if not checks.empty else 0,
    }
    write_json(status_path, status)
    report_path.write_text(format_report(status, results, checks), encoding="utf-8")
    return {"results_path": results_path, "checks_path": checks_path, "status_path": status_path, "report_path": report_path}


def format_report(status: dict[str, Any], results: pd.DataFrame, checks: pd.DataFrame) -> str:
    columns = [
        "case_id",
        "short_name",
        "variant",
        "variant_kind",
        "retention_fraction",
        "gate_threshold",
        "geometry_threshold",
        "precision",
        "recall",
        "f_score",
        "chamfer_l1",
        "chamfer_l2",
        "mean_psnr",
        "mean_ssim",
        "mean_lpips_vgg",
        "calibration_auc",
        "calibration_brier",
        "delta_f_score_vs_A",
        "delta_chamfer_l1_vs_A",
        "delta_mean_psnr_vs_A",
    ]
    return "\n".join(
        [
            "# Actual Trained 3DGS A-F Results",
            "",
            f"- Matrix: `{status['matrix_name']}`",
            f"- Scene: `{status['scene_dir']}`",
            f"- Passed: `{status['passed']}`",
            "",
            "## Results",
            "",
            markdown_table(results, columns),
            "",
            "## Checks",
            "",
            markdown_table(checks, list(checks.columns)),
            "",
        ]
    )


def markdown_table(data: pd.DataFrame, columns: list[str]) -> str:
    present = [column for column in columns if column in data.columns]
    if not present:
        return "No rows."
    lines = ["| " + " | ".join(present) + " |", "| " + " | ".join(["---"] * len(present)) + " |"]
    for _, row in data[present].iterrows():
        lines.append("| " + " | ".join(format_cell(row[column]) for column in present) + " |")
    return "\n".join(lines)


def format_cell(item: Any) -> str:
    try:
        if pd.isna(item):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(item, float):
        return f"{item:.6g}"
    return str(item).replace("|", "\\|").replace("\n", " ")


def gate_label(threshold: float) -> str:
    return format_threshold_label(threshold)
