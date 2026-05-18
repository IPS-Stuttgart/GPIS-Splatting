from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.serialization import write_json


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    short_name: str
    method_name: str
    hypothesis: str
    recommended_artifacts: tuple[str, ...]
    command_hint: str


@dataclass(frozen=True)
class ExperimentMatrixConfig:
    output_dir: Path
    matrix_name: str = "gpis_3dgs_matrix"
    scene_dir: Path | None = None
    primary_geometry_threshold: float = 0.05
    baseline_case: str = "A"
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    fail_on_missing: bool = False


KNOWN_ARTIFACT_ROLES: dict[str, str] = {
    "trained_3dgs_render_comparison": "CSV from evaluate_3dgs_variant_renders for the plain trained-3DGS baseline.",
    "trained_3dgs_geometry_summary": "Optional geometry summary CSV for the trained-3DGS baseline.",
    "raw_gate_sweep": "CSV from run_tanks_temples_gate_sweep for the raw GPIS gate.",
    "calibrated_gate_sweep": "Optional gate-sweep CSV using a calibrated confidence gate.",
    "calibrated_confidence_metrics": "Optional calibrator validation metrics CSV.",
    "calibrated_filtering_comparison": "CSV from run_tanks_temples_calibrated_splat_filtering.",
    "regularized_3dgs_render_comparison": "Render comparison CSV for 3DGS trained with the GPIS regularizer.",
    "regularized_geometry_summary": "Optional geometry summary CSV for the regularized 3DGS run.",
    "regularized_calibrated_render_comparison": "Render comparison CSV for regularized 3DGS plus calibrated confidence.",
    "regularized_calibrated_filtering_comparison": "Filtering comparison CSV for regularized 3DGS plus calibrated confidence.",
}

METRIC_COLUMNS = (
    "retained_count",
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
    "image_count",
    "calibration_auc",
    "calibration_auprc",
    "calibration_brier",
    "calibration_ece",
)
HIGHER_IS_BETTER = {"precision", "recall", "f_score", "mean_psnr", "mean_ssim", "calibration_auc", "calibration_auprc"}
LOWER_IS_BETTER = {"chamfer_l1", "chamfer_l2", "mean_lpips_vgg", "calibration_brier", "calibration_ece"}
DELTA_METRICS = tuple(sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER))
PROVENANCE_COLUMNS = (*METRIC_COLUMNS, "variant", "variant_kind")
PROVENANCE_COLUMN_SET = set(PROVENANCE_COLUMNS)
SOURCE_METADATA_KEYS = {
    "notes",
    "source_roles",
    "source_artifact",
    "metric_source_roles",
    "metric_source_artifacts",
    "metric_conflicts",
}


def default_matrix_cases() -> tuple[MatrixCase, ...]:
    return (
        MatrixCase(
            "A",
            "plain_3dgs",
            "Plain trained 3DGS",
            "Reference trained 3DGS without GPIS gating, calibration, or GPIS training regularization.",
            ("trained_3dgs_render_comparison", "trained_3dgs_geometry_summary"),
            "export_prepared_scene_to_colmap_3dgs -> train 3DGS -> evaluate_3dgs_variant_renders",
        ),
        MatrixCase(
            "B",
            "raw_gpis_gate",
            "Raw GPIS gate post-hoc",
            "Analytic GPIS zero-band gates select geometrically better splats than the ungated set.",
            ("raw_gate_sweep",),
            "run_tanks_temples_gate_sweep --model-path <gpis_model.npz>",
        ),
        MatrixCase(
            "C",
            "calibrated_confidence",
            "Calibrated GPIS confidence post-hoc",
            "Calibrated GPIS posterior-field confidence gives a stronger post-hoc gate than the raw surface-band probability.",
            ("calibrated_gate_sweep", "calibrated_confidence_metrics", "calibrated_filtering_comparison"),
            "run_tanks_temples_calibrated_confidence or calibrate_gpis_splat_scores + run_tanks_temples_gate_sweep --gate-path",
        ),
        MatrixCase(
            "D",
            "calibrated_pruning_refinement",
            "Calibrated pruning/refinement",
            "Confidence-thresholded or tau-scaled splat variants improve geometry/rendering at useful retention levels.",
            ("calibrated_filtering_comparison",),
            "run_tanks_temples_calibrated_splat_filtering --gate-path <calibrated_gate.npz>",
        ),
        MatrixCase(
            "E",
            "gpis_training_regularized_3dgs",
            "GPIS training-time regularizer",
            "Adding GPIS surface/opacity/normal losses during 3DGS training improves final geometry or render quality.",
            ("regularized_3dgs_render_comparison", "regularized_geometry_summary"),
            "Train with GPIS3DGSTrainingRegularizer and evaluate the trained output.",
        ),
        MatrixCase(
            "F",
            "regularized_plus_calibrated_confidence",
            "Regularizer plus calibrated confidence",
            "Training-time GPIS regularization plus calibrated post-hoc confidence improves the combined trade-off.",
            ("regularized_calibrated_render_comparison", "regularized_calibrated_filtering_comparison"),
            "Regularized 3DGS -> score/calibrate centers -> export/evaluate GPIS variants.",
        ),
    )


def run_experiment_matrix(config: ExperimentMatrixConfig) -> dict[str, Any]:
    validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = default_matrix_cases()
    manifest = build_matrix_manifest(cases, config)
    summary = build_matrix_summary(cases, config)
    summary = add_baseline_deltas(summary, baseline_case=config.baseline_case)
    checks = build_matrix_checks(cases, config, summary)
    passed = bool(checks["passed"].all()) if not checks.empty else True
    if config.fail_on_missing and not passed:
        missing = ", ".join(checks.loc[~checks["passed"], "case_id"].astype(str).tolist())
        raise ValueError(f"Experiment matrix is missing required artifacts for cases: {missing}")
    paths = write_matrix_artifacts(output_dir, config, manifest, summary, checks, passed)
    return {
        "manifest_path": paths["manifest"],
        "summary_path": paths["summary"],
        "checks_path": paths["checks"],
        "status_path": paths["status"],
        "config_path": paths["config"],
        "report_path": paths["report"],
        "manifest": manifest,
        "summary": summary,
        "checks": checks,
        "passed": passed,
    }


def validate_config(config: ExperimentMatrixConfig) -> None:
    case_ids = {case.case_id for case in default_matrix_cases()}
    if config.baseline_case not in case_ids:
        raise ValueError(f"Unknown baseline case {config.baseline_case!r}; expected one of {sorted(case_ids)}.")
    unknown_roles = sorted(set(config.artifact_paths) - set(KNOWN_ARTIFACT_ROLES))
    if unknown_roles:
        raise ValueError(f"Unknown artifact roles: {', '.join(unknown_roles)}")


def build_matrix_manifest(cases: tuple[MatrixCase, ...], config: ExperimentMatrixConfig) -> pd.DataFrame:
    rows = []
    for case in cases:
        available = [role for role in case.recommended_artifacts if role in config.artifact_paths]
        rows.append(
            {
                "case_id": case.case_id,
                "short_name": case.short_name,
                "method_name": case.method_name,
                "hypothesis": case.hypothesis,
                "recommended_artifacts": ";".join(case.recommended_artifacts),
                "provided_artifacts": ";".join(available),
                "command_hint": case.command_hint,
                "status": "configured" if available else "pending_artifacts",
            }
        )
    return pd.DataFrame(rows)


def build_matrix_summary(cases: tuple[MatrixCase, ...], config: ExperimentMatrixConfig) -> pd.DataFrame:
    rows = []
    for case in cases:
        row = base_summary_row(case)
        row.update(load_case_metrics(case.case_id, config))
        row["artifact_count"] = sum(1 for role in case.recommended_artifacts if role in config.artifact_paths)
        row["configured"] = row["artifact_count"] > 0
        rows.append(row)
    return pd.DataFrame(rows)


def base_summary_row(case: MatrixCase) -> dict[str, Any]:
    row = {
        "case_id": case.case_id,
        "short_name": case.short_name,
        "method_name": case.method_name,
        "configured": False,
        "artifact_count": 0,
        "source_artifact": None,
        "source_roles": "",
        "metric_source_roles": "",
        "metric_source_artifacts": "",
        "metric_conflicts": "",
        "variant": None,
        "variant_kind": None,
        "notes": "",
    }
    row.update({column: np.nan for column in METRIC_COLUMNS})
    return row


def load_case_metrics(case_id: str, config: ExperimentMatrixConfig) -> dict[str, Any]:
    if case_id == "A":
        return merge_metrics(
            load_render_metrics(config, "trained_3dgs_render_comparison", preferred_variant="baseline"),
            load_geometry_summary_metrics(config, "trained_3dgs_geometry_summary"),
        )
    if case_id == "B":
        return attach_metric_sources(load_gate_sweep_metrics(config, "raw_gate_sweep", note="raw GPIS gate sweep"))
    if case_id == "C":
        return merge_metrics(
            load_gate_sweep_metrics(config, "calibrated_gate_sweep", note="calibrated gate sweep"),
            load_filtering_metrics(config, "calibrated_filtering_comparison", preferred_kind="gate_scaled", note="calibrated confidence gate-scaled variant"),
            load_calibration_metrics(config, "calibrated_confidence_metrics"),
        )
    if case_id == "D":
        return attach_metric_sources(
            load_filtering_metrics(
                config,
                "calibrated_filtering_comparison",
                exclude_kinds={"baseline", "random_same_retention"},
                note="calibrated pruning/refinement",
            )
        )
    if case_id == "E":
        return merge_metrics(
            load_render_metrics(config, "regularized_3dgs_render_comparison", preferred_variant="baseline"),
            load_geometry_summary_metrics(config, "regularized_geometry_summary"),
        )
    if case_id == "F":
        return merge_metrics(
            load_render_metrics(config, "regularized_calibrated_render_comparison", preferred_variant="gate_scaled"),
            load_filtering_metrics(
                config,
                "regularized_calibrated_filtering_comparison",
                exclude_kinds={"baseline", "random_same_retention"},
                note="regularized plus calibrated filtering",
            ),
        )
    return {}


def merge_metrics(*metrics_list: dict[str, Any]) -> dict[str, Any]:
    """Merge artifact metrics without silent overwrites.

    Earlier arguments have higher precedence. Later artifacts fill missing
    fields only. If a later artifact contains a different non-missing value for
    an already-populated metric, the first value is kept and the ignored source
    is recorded in ``metric_conflicts``. The summary CSV also receives compact
    ``metric_source_roles`` and ``metric_source_artifacts`` maps so each metric's
    provenance remains auditable.
    """

    merged: dict[str, Any] = {}
    notes: list[str] = []
    roles: list[str] = []
    sources: list[str] = []
    metric_source_roles: dict[str, str] = {}
    metric_source_artifacts: dict[str, str] = {}
    conflicts: list[str] = []
    for raw_metrics in metrics_list:
        metrics = attach_metric_sources(raw_metrics)
        if not metrics:
            continue
        if metrics.get("notes"):
            notes.append(str(metrics["notes"]))
        if metrics.get("source_roles"):
            roles.extend(str(metrics["source_roles"]).split(";"))
        if metrics.get("source_artifact"):
            sources.append(str(metrics["source_artifact"]))
        if metrics.get("metric_conflicts"):
            conflicts.extend(split_encoded_list(str(metrics["metric_conflicts"])))
        role_map = parse_source_map(metrics.get("metric_source_roles"))
        artifact_map = parse_source_map(metrics.get("metric_source_artifacts"))
        default_role = first_nonempty(str(metrics.get("source_roles") or "").split(";"))
        default_artifact = first_nonempty(str(metrics.get("source_artifact") or "").split(";"))
        for key, value in metrics.items():
            if key in SOURCE_METADATA_KEYS:
                continue
            if is_missing(value):
                continue
            source_role = role_map.get(key) or default_role
            source_artifact = artifact_map.get(key) or default_artifact
            if key not in merged or is_missing(merged.get(key)):
                merged[key] = value
                if key in PROVENANCE_COLUMN_SET:
                    if source_role:
                        metric_source_roles[key] = source_role
                    if source_artifact:
                        metric_source_artifacts[key] = source_artifact
                continue
            if key in PROVENANCE_COLUMN_SET and not values_equivalent(merged[key], value):
                kept_source = metric_source_roles.get(key) or "unknown"
                ignored_source = source_role or "unknown"
                conflicts.append(f"{key}: kept {kept_source}, ignored {ignored_source}")
    if notes:
        merged["notes"] = "; ".join(dict.fromkeys(notes))
    if roles:
        merged["source_roles"] = ";".join(dict.fromkeys(role for role in roles if role))
    if sources:
        merged["source_artifact"] = ";".join(dict.fromkeys(sources))
    if metric_source_roles:
        merged["metric_source_roles"] = format_source_map(metric_source_roles)
    if metric_source_artifacts:
        merged["metric_source_artifacts"] = format_source_map(metric_source_artifacts)
    if conflicts:
        merged["metric_conflicts"] = "; ".join(dict.fromkeys(conflict for conflict in conflicts if conflict))
    return merged


def attach_metric_sources(metrics: dict[str, Any]) -> dict[str, Any]:
    if not metrics:
        return {}
    result = dict(metrics)
    role_map = parse_source_map(result.get("metric_source_roles"))
    artifact_map = parse_source_map(result.get("metric_source_artifacts"))
    default_role = first_nonempty(str(result.get("source_roles") or "").split(";"))
    default_artifact = first_nonempty(str(result.get("source_artifact") or "").split(";"))
    for key, value in result.items():
        if key in SOURCE_METADATA_KEYS or key not in PROVENANCE_COLUMN_SET or is_missing(value):
            continue
        if default_role:
            role_map.setdefault(key, default_role)
        if default_artifact:
            artifact_map.setdefault(key, default_artifact)
    if role_map:
        result["metric_source_roles"] = format_source_map(role_map)
    if artifact_map:
        result["metric_source_artifacts"] = format_source_map(artifact_map)
    return result


def parse_source_map(encoded: Any) -> dict[str, str]:
    if encoded is None:
        return {}
    text = str(encoded)
    if not text:
        return {}
    parsed: dict[str, str] = {}
    for item in text.split(";"):
        if not item or "=" not in item:
            continue
        key, source = item.split("=", 1)
        if key and source:
            parsed[key] = source
    return parsed


def format_source_map(source_map: dict[str, str]) -> str:
    return ";".join(f"{key}={value}" for key, value in source_map.items() if value)


def split_encoded_list(encoded: str) -> list[str]:
    return [item.strip() for item in encoded.split(";") if item.strip()]


def first_nonempty(values: Any) -> str:
    for value_ in values:
        text = str(value_).strip()
        if text:
            return text
    return ""


def values_equivalent(left: Any, right: Any) -> bool:
    if is_missing(left) and is_missing(right):
        return True
    if is_missing(left) or is_missing(right):
        return False
    try:
        return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12, equal_nan=True))
    except (TypeError, ValueError):
        return str(left) == str(right)


def load_render_metrics(config: ExperimentMatrixConfig, role: str, *, preferred_variant: str | None = None) -> dict[str, Any]:
    path = config.artifact_paths.get(role)
    if path is None:
        return {}
    table = read_csv(path)
    if table.empty:
        return source_only(path, role, "render comparison was empty")
    candidates = table.copy()
    if preferred_variant and "variant" in candidates and preferred_variant in set(candidates["variant"].astype(str)):
        candidates = candidates[candidates["variant"].astype(str) == preferred_variant]
    row = sort_and_pick(candidates, [("mean_psnr", False), ("mean_ssim", False)])
    return {
        **source_only(path, role, "render comparison"),
        "variant": value(row, "variant"),
        "variant_kind": value(row, "variant_kind"),
        "retention_fraction": number(row, "retention_fraction"),
        "retained_count": number(row, "retained_count"),
        "gate_threshold": number(row, "gate_threshold"),
        "mean_psnr": number(row, "mean_psnr"),
        "mean_ssim": number(row, "mean_ssim"),
        "mean_lpips_vgg": number(row, "mean_lpips_vgg"),
        "image_count": number(row, "image_count"),
    }


def load_gate_sweep_metrics(config: ExperimentMatrixConfig, role: str, *, note: str) -> dict[str, Any]:
    path = config.artifact_paths.get(role)
    if path is None:
        return {}
    table = select_geometry_threshold(read_csv(path), config.primary_geometry_threshold)
    if "selection" in table and "gate_ge" in set(table["selection"].astype(str)):
        table = table[table["selection"].astype(str) == "gate_ge"]
    row = select_best_geometry_row(table)
    return {
        **source_only(path, role, note),
        "variant": value(row, "group") or value(row, "selection"),
        "variant_kind": value(row, "selection"),
        "geometry_threshold": number(row, "geometry_threshold", fallback="threshold"),
        "gate_threshold": number(row, "gate_threshold"),
        "retention_fraction": number(row, "retention_fraction"),
        "retained_count": number(row, "selected_pred_point_count"),
        "precision": number(row, "precision"),
        "recall": number(row, "recall"),
        "f_score": number(row, "f_score"),
        "chamfer_l1": number(row, "chamfer_l1"),
    }


def load_filtering_metrics(
    config: ExperimentMatrixConfig,
    role: str,
    *,
    preferred_kind: str | None = None,
    exclude_kinds: set[str] | None = None,
    note: str,
) -> dict[str, Any]:
    path = config.artifact_paths.get(role)
    if path is None:
        return {}
    table = select_geometry_threshold(read_csv(path), config.primary_geometry_threshold)
    if preferred_kind and "variant_kind" in table and preferred_kind in set(table["variant_kind"].astype(str)):
        table = table[table["variant_kind"].astype(str) == preferred_kind]
    if exclude_kinds and "variant_kind" in table:
        filtered = table[~table["variant_kind"].astype(str).isin(exclude_kinds)]
        if not filtered.empty:
            table = filtered
    row = select_best_geometry_row(table)
    return {
        **source_only(path, role, note),
        "variant": value(row, "variant"),
        "variant_kind": value(row, "variant_kind"),
        "geometry_threshold": number(row, "geometry_threshold", fallback="threshold"),
        "gate_threshold": number(row, "gate_threshold"),
        "retention_fraction": number(row, "retention_fraction"),
        "retained_count": number(row, "retained_count"),
        "precision": number(row, "precision"),
        "recall": number(row, "recall"),
        "f_score": number(row, "f_score"),
        "chamfer_l1": number(row, "chamfer_l1"),
        "chamfer_l2": number(row, "chamfer_l2"),
        "mean_psnr": number(row, "mean_psnr"),
        "mean_ssim": number(row, "mean_ssim"),
    }


def load_geometry_summary_metrics(config: ExperimentMatrixConfig, role: str) -> dict[str, Any]:
    path = config.artifact_paths.get(role)
    if path is None:
        return {}
    table = read_csv(path)
    if "group" in table and "all" in set(table["group"].astype(str)):
        table = table[table["group"].astype(str) == "all"]
    row = table.iloc[0]
    return {
        **source_only(path, role, "geometry summary"),
        "variant": value(row, "group"),
        "retained_count": number(row, "pred_point_count"),
        "chamfer_l1": number(row, "chamfer_l1"),
        "chamfer_l2": number(row, "chamfer_l2"),
    }


def load_calibration_metrics(config: ExperimentMatrixConfig, role: str) -> dict[str, Any]:
    path = config.artifact_paths.get(role)
    if path is None:
        return {}
    table = read_csv(path)
    row = sort_and_pick(table, [("auc", False), ("auroc", False), ("average_precision", False), ("brier", True)])
    return {
        **source_only(path, role, "calibration metrics"),
        "calibration_auc": number(row, "auc", fallback="auroc"),
        "calibration_auprc": number(row, "auprc", fallback="average_precision"),
        "calibration_brier": number(row, "brier"),
        "calibration_ece": number(row, "ece"),
    }


def source_only(path: Path, role: str, note: str) -> dict[str, Any]:
    return {"source_artifact": str(path), "source_roles": role, "notes": note}


def read_csv(path: Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing matrix artifact: {csv_path}")
    return pd.read_csv(csv_path)


def select_geometry_threshold(table: pd.DataFrame, primary_geometry_threshold: float) -> pd.DataFrame:
    threshold_column = "geometry_threshold" if "geometry_threshold" in table else "threshold" if "threshold" in table else None
    if threshold_column is None or table.empty:
        return table
    distances = (table[threshold_column].astype(float) - float(primary_geometry_threshold)).abs()
    nearest = float(table.loc[distances.idxmin(), threshold_column])
    return table[table[threshold_column].astype(float) == nearest]


def select_best_geometry_row(table: pd.DataFrame) -> pd.Series:
    if table.empty:
        raise ValueError("Cannot select a matrix row from an empty table.")
    return sort_and_pick(table, [("f_score", False), ("precision", False), ("chamfer_l1", True)])


def sort_and_pick(table: pd.DataFrame, sort_order: list[tuple[str, bool]]) -> pd.Series:
    if table.empty:
        raise ValueError("Cannot select a matrix row from an empty table.")
    sort_columns = [column for column, _ascending in sort_order if column in table]
    if sort_columns:
        ascending = [ascending for column, ascending in sort_order if column in table]
        table = table.sort_values(sort_columns, ascending=ascending)
    return table.iloc[0]


def add_baseline_deltas(summary: pd.DataFrame, *, baseline_case: str) -> pd.DataFrame:
    result = summary.copy()
    baseline = result[result["case_id"] == baseline_case]
    for metric in DELTA_METRICS:
        delta_column = f"delta_{metric}_vs_baseline"
        result[delta_column] = np.nan
        if baseline.empty or metric not in result:
            continue
        baseline_value = baseline.iloc[0][metric]
        if is_missing(baseline_value):
            continue
        if metric in HIGHER_IS_BETTER:
            result[delta_column] = result[metric].astype(float) - float(baseline_value)
        else:
            result[delta_column] = float(baseline_value) - result[metric].astype(float)
    return result


def build_matrix_checks(cases: tuple[MatrixCase, ...], config: ExperimentMatrixConfig, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case in cases:
        artifact_count = sum(1 for role in case.recommended_artifacts if role in config.artifact_paths)
        row = summary[summary["case_id"] == case.case_id].iloc[0]
        has_metric = any(not is_missing(row.get(column)) for column in METRIC_COLUMNS)
        rows.append(
            {
                "case_id": case.case_id,
                "short_name": case.short_name,
                "passed": bool(artifact_count and has_metric),
                "artifact_count": artifact_count,
                "has_metric": bool(has_metric),
                "details": "configured with metrics" if artifact_count and has_metric else "pending artifact or no metric loaded",
            }
        )
    return pd.DataFrame(rows)


def write_matrix_artifacts(
    output_dir: Path,
    config: ExperimentMatrixConfig,
    manifest: pd.DataFrame,
    summary: pd.DataFrame,
    checks: pd.DataFrame,
    passed: bool,
) -> dict[str, Path]:
    name = config.matrix_name
    paths = {
        "manifest": output_dir / f"{name}_manifest.csv",
        "summary": output_dir / f"{name}_summary.csv",
        "checks": output_dir / f"{name}_checks.csv",
        "status": output_dir / f"{name}_status.json",
        "config": output_dir / f"{name}_config.json",
        "report": output_dir / f"{name}_report.md",
    }
    manifest.to_csv(paths["manifest"], index=False)
    summary.to_csv(paths["summary"], index=False)
    checks.to_csv(paths["checks"], index=False)
    available_cases = summary.loc[summary["configured"], "case_id"].astype(str).tolist()
    missing_cases = summary.loc[~summary["configured"], "case_id"].astype(str).tolist()
    status = {
        "schema_version": 1,
        "matrix_name": name,
        "passed": passed,
        "fail_on_missing": config.fail_on_missing,
        "baseline_case": config.baseline_case,
        "primary_geometry_threshold": config.primary_geometry_threshold,
        "scene_dir": str(config.scene_dir) if config.scene_dir is not None else None,
        "artifact_paths": {key: str(value) for key, value in sorted(config.artifact_paths.items())},
        "available_cases": available_cases,
        "missing_cases": missing_cases,
        "available_case_count": len(available_cases),
        "case_count": int(len(summary)),
        "manifest_path": str(paths["manifest"]),
        "summary_path": str(paths["summary"]),
        "checks_path": str(paths["checks"]),
        "report_path": str(paths["report"]),
    }
    config_payload = {
        "matrix_name": name,
        "output_dir": str(output_dir),
        "scene_dir": str(config.scene_dir) if config.scene_dir is not None else None,
        "primary_geometry_threshold": config.primary_geometry_threshold,
        "baseline_case": config.baseline_case,
        "fail_on_missing": config.fail_on_missing,
        "artifact_paths": {key: str(value) for key, value in sorted(config.artifact_paths.items())},
        "known_artifact_roles": KNOWN_ARTIFACT_ROLES,
        "cases": [case.__dict__ for case in default_matrix_cases()],
    }
    write_json(paths["status"], status)
    write_json(paths["config"], config_payload)
    paths["report"].write_text(format_matrix_report(status, manifest, summary, checks), encoding="utf-8")
    return paths


def format_matrix_report(status: dict[str, Any], manifest: pd.DataFrame, summary: pd.DataFrame, checks: pd.DataFrame) -> str:
    lines = [
        "# GPIS/3DGS Experiment Matrix",
        "",
        f"- Matrix: `{status['matrix_name']}`",
        f"- Scene: `{status.get('scene_dir') or 'n/a'}`",
        f"- Passed: `{status['passed']}`",
        f"- Available cases: `{', '.join(status['available_cases']) or 'none'}`",
        f"- Missing cases: `{', '.join(status['missing_cases']) or 'none'}`",
        "",
        "## A-F Matrix Summary",
        "",
        markdown_table(summary, summary_columns(summary)),
        "",
        "## Checks",
        "",
        markdown_table(checks, list(checks.columns)),
        "",
        "## Planned Cases",
        "",
        markdown_table(manifest, ["case_id", "short_name", "method_name", "status", "provided_artifacts", "command_hint"]),
    ]
    return "\n".join(lines) + "\n"


def summary_columns(summary: pd.DataFrame) -> list[str]:
    columns = [
        "case_id",
        "short_name",
        "configured",
        "variant",
        "variant_kind",
        "retention_fraction",
        "gate_threshold",
        "geometry_threshold",
        "precision",
        "recall",
        "f_score",
        "chamfer_l1",
        "mean_psnr",
        "mean_ssim",
        "mean_lpips_vgg",
        "delta_f_score_vs_baseline",
        "delta_chamfer_l1_vs_baseline",
        "delta_mean_psnr_vs_baseline",
        "source_roles",
        "metric_source_roles",
        "metric_conflicts",
    ]
    return [column for column in columns if column in summary]


def markdown_table(data: pd.DataFrame, columns: list[str]) -> str:
    if data.empty:
        return "No rows."
    present = [column for column in columns if column in data]
    lines = ["| " + " | ".join(present) + " |", "| " + " | ".join(["---"] * len(present)) + " |"]
    for _, row in data[present].iterrows():
        lines.append("| " + " | ".join(format_value(row[column]) for column in present) + " |")
    return "\n".join(lines)


def value(row: pd.Series, column: str) -> Any:
    if column not in row or is_missing(row[column]):
        return None
    return row[column]


def number(row: pd.Series, column: str, *, fallback: str | None = None) -> float:
    selected = column if column in row else fallback
    if selected is None or selected not in row or is_missing(row[selected]):
        return np.nan
    return float(row[selected])


def is_missing(value_: Any) -> bool:
    if value_ is None:
        return True
    if isinstance(value_, float) and (np.isnan(value_) or np.isinf(value_)):
        return True
    try:
        return bool(pd.isna(value_))
    except (TypeError, ValueError):
        return False


def format_value(value_: Any) -> str:
    if is_missing(value_):
        return ""
    if isinstance(value_, np.bool_):
        return str(bool(value_))
    if isinstance(value_, np.integer):
        return str(int(value_))
    if isinstance(value_, np.floating):
        value_ = float(value_)
    if isinstance(value_, float):
        return f"{value_:.6g}"
    return str(value_).replace("\n", " ").replace("|", "\\|")
