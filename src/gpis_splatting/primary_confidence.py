from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.confidence import write_calibrated_confidence_bundle, write_gate_compatible_confidences
from gpis_splatting.real_geometry import format_threshold_label
from gpis_splatting.real_score_calibration import (
    DEFAULT_BASELINE_SCORES,
    default_calibration_prefix,
    default_feature_sets,
    default_topk_fractions,
    deterministic_train_validation_split,
    fit_calibration_methods,
    prepare_feature_table,
    run_gpis_splat_score_calibration as run_legacy_gpis_splat_score_calibration,
)
from gpis_splatting.serialization import write_json

DEFAULT_ISOTONIC_SCORES = ("score_current_gate", "score_raw_surface_band", "score_variance_penalized_band")


def run_primary_calibrated_confidence(
    *,
    field_scores_path: str | Path,
    output_dir: str | Path | None = None,
    method_name: str | None = None,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.1),
    topk_fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
    feature_sets: tuple[str, ...] = tuple(default_feature_sets()),
    baseline_scores: tuple[str, ...] | None = DEFAULT_BASELINE_SCORES,
    isotonic_scores: tuple[str, ...] | None = DEFAULT_ISOTONIC_SCORES,
    validation_fraction: float = 0.35,
    seed: int = 13,
    logistic_iterations: int = 600,
    learning_rate: float = 0.05,
    regularization: float = 1e-3,
    num_bins: int = 10,
    gate_count: int | None = None,
    missing_gate_value: float = 0.0,
    primary_threshold: float | None = None,
) -> dict[str, Any]:
    """Run score calibration and promote calibrated confidence to the reusable primary output.

    The legacy calibration path already writes score summaries, selected confidence columns,
    and gate-compatible NPZs. This wrapper keeps those artifacts, then refits the selected
    per-threshold calibrators and serializes them as a reusable calibrated-confidence model
    bundle. The existing status/report files are updated so downstream workflows can treat
    calibrated confidence as the first-class splat signal instead of recovering it from a
    one-off CSV.
    """
    calibration = run_legacy_gpis_splat_score_calibration(
        field_scores_path=field_scores_path,
        output_dir=output_dir,
        method_name=method_name,
        thresholds=thresholds,
        topk_fractions=topk_fractions,
        feature_sets=feature_sets,
        baseline_scores=baseline_scores,
        isotonic_scores=isotonic_scores,
        validation_fraction=validation_fraction,
        seed=seed,
        logistic_iterations=logistic_iterations,
        learning_rate=learning_rate,
        regularization=regularization,
        num_bins=num_bins,
        gate_count=gate_count,
        missing_gate_value=missing_gate_value,
    )
    field_path = Path(field_scores_path)
    out_dir = Path(output_dir) if output_dir is not None else field_path.parent
    prefix = method_name or default_calibration_prefix(field_path)
    table = prepare_feature_table(pd.read_csv(field_path))
    baseline_scores = DEFAULT_BASELINE_SCORES if baseline_scores is None else baseline_scores
    isotonic_scores = DEFAULT_ISOTONIC_SCORES if isotonic_scores is None else isotonic_scores
    selected_methods = fit_selected_calibrated_methods(
        table=table,
        calibration_status=calibration["status"],
        thresholds=thresholds,
        feature_sets=feature_sets,
        baseline_scores=baseline_scores,
        isotonic_scores=isotonic_scores,
        validation_fraction=validation_fraction,
        seed=seed,
        logistic_iterations=logistic_iterations,
        learning_rate=learning_rate,
        regularization=regularization,
    )
    selected_primary_threshold = resolve_primary_threshold(primary_threshold, thresholds)
    primary_label = format_threshold_label(selected_primary_threshold)
    primary_method = selected_method_summary(calibration["status"], selected_primary_threshold)

    model_bundle_path = out_dir / f"{prefix}_calibrated_confidence_model.json"
    bundle = write_calibrated_confidence_bundle(
        model_bundle_path,
        selected_methods,
        metadata={
            "field_scores_path": str(field_path),
            "method": prefix,
            "thresholds": list(thresholds),
            "primary_threshold": selected_primary_threshold,
            "feature_sets": list(feature_sets),
            "baseline_scores": [column for column in baseline_scores if column in table.columns],
            "isotonic_scores": [column for column in isotonic_scores if column in table.columns],
            "validation_fraction": validation_fraction,
            "seed": seed,
            "logistic_iterations": logistic_iterations,
            "learning_rate": learning_rate,
            "regularization": regularization,
            "source_status_path": str(calibration["status_path"]),
            "source_predictions_path": str(calibration["predictions_path"]),
        },
    )

    primary_predictions = bundle.predict(table)
    primary_confidence_column = f"confidence_{primary_label}"
    primary_method_column = f"selected_method_{primary_label}"
    primary_predictions.insert(1, "primary_confidence", primary_predictions[primary_confidence_column].to_numpy(dtype=np.float64))
    primary_predictions.insert(2, "primary_geometry_threshold", np.full((len(primary_predictions),), selected_primary_threshold, dtype=np.float64))
    primary_predictions.insert(3, "primary_method", primary_predictions[primary_method_column].to_numpy(dtype=object))
    primary_confidence_path = out_dir / f"{prefix}_primary_calibrated_confidence.csv"
    primary_predictions.to_csv(primary_confidence_path, index=False)
    primary_gate_paths = write_gate_compatible_confidences(
        predictions=primary_predictions,
        out_dir=out_dir,
        prefix=f"{prefix}_primary_calibrated_confidence",
        thresholds=thresholds,
        predictions_path=primary_confidence_path,
        gate_count=gate_count,
        missing_gate_value=missing_gate_value,
    )
    primary_gate_path = primary_gate_paths.get(primary_label)

    augmented_status = dict(calibration["status"])
    augmented_status.update(
        {
            "calibrated_confidence_is_primary": True,
            "model_bundle_path": str(model_bundle_path),
            "primary_confidence_path": str(primary_confidence_path),
            "primary_threshold": selected_primary_threshold,
            "primary_threshold_label": primary_label,
            "primary_method": primary_method,
            "primary_gate_paths": {label: str(path) for label, path in primary_gate_paths.items()},
            "primary_gate_path": None if primary_gate_path is None else str(primary_gate_path),
        }
    )
    write_json(calibration["status_path"], augmented_status)
    append_primary_confidence_report(Path(calibration["report_path"]), augmented_status)

    calibration.update(
        {
            "model_bundle_path": model_bundle_path,
            "primary_confidence_path": primary_confidence_path,
            "primary_gate_paths": primary_gate_paths,
            "primary_gate_path": primary_gate_path,
            "primary_predictions": primary_predictions,
            "bundle": bundle,
            "status": augmented_status,
        }
    )
    return calibration


# Preserve the existing public function name for the CLI while changing the default
# artifact contract to calibrated confidence.
def run_gpis_splat_score_calibration(**kwargs: Any) -> dict[str, Any]:
    return run_primary_calibrated_confidence(**kwargs)


def fit_selected_calibrated_methods(
    *,
    table: pd.DataFrame,
    calibration_status: dict[str, Any],
    thresholds: tuple[float, ...],
    feature_sets: tuple[str, ...],
    baseline_scores: tuple[str, ...],
    isotonic_scores: tuple[str, ...],
    validation_fraction: float,
    seed: int,
    logistic_iterations: int,
    learning_rate: float,
    regularization: float,
) -> list[tuple[float, Any]]:
    split = deterministic_train_validation_split(len(table), validation_fraction=validation_fraction, seed=seed)
    selected: list[tuple[float, Any]] = []
    for threshold in thresholds:
        labels = (table["nearest_gt_distance"].to_numpy(dtype=np.float64) <= threshold).astype(np.float64)
        methods = fit_calibration_methods(
            table=table,
            labels=labels,
            train_mask=split.train_mask,
            feature_sets=feature_sets,
            baseline_scores=baseline_scores,
            isotonic_scores=isotonic_scores,
            logistic_iterations=logistic_iterations,
            learning_rate=learning_rate,
            regularization=regularization,
        )
        selected_name = selected_method_name(calibration_status, threshold)
        selected.append((float(threshold), next(method for method in methods if method.name == selected_name)))
    return selected


def selected_method_name(calibration_status: dict[str, Any], threshold: float) -> str:
    summary = selected_method_summary(calibration_status, threshold)
    if summary is None:
        raise ValueError(f"Calibration status does not contain a selected method for threshold {threshold:g}.")
    return str(summary["method_name"])


def selected_method_summary(calibration_status: dict[str, Any], threshold: float) -> dict[str, Any] | None:
    for row in calibration_status.get("best_by_threshold", []):
        if abs(float(row["geometry_threshold"]) - float(threshold)) <= 1e-12:
            return dict(row)
    return None


def resolve_primary_threshold(primary_threshold: float | None, thresholds: tuple[float, ...]) -> float:
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    if primary_threshold is not None:
        requested = float(primary_threshold)
        if any(abs(float(threshold) - requested) <= 1e-12 for threshold in thresholds):
            return requested
        available = ", ".join(f"{threshold:g}" for threshold in thresholds)
        raise ValueError(f"primary_threshold must be one of the calibrated thresholds. Available: {available}.")
    for threshold in thresholds:
        if abs(float(threshold) - 0.05) <= 1e-12:
            return float(threshold)
    return float(thresholds[0])


def append_primary_confidence_report(report_path: Path, status: dict[str, Any]) -> None:
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    primary = status.get("primary_method") or {}
    lines = [
        "",
        "## Primary Calibrated Confidence",
        "",
        "Calibrated GPIS posterior-field confidence is the primary splat-quality signal for downstream filtering and gate export.",
        "",
        f"- Model bundle: `{status.get('model_bundle_path', 'n/a')}`",
        f"- Primary confidence CSV: `{status.get('primary_confidence_path', 'n/a')}`",
        f"- Primary threshold: `{status.get('primary_threshold', 'n/a')}`",
        f"- Primary gate NPZ: `{status.get('primary_gate_path', 'n/a')}`",
        f"- Selected method: `{primary.get('method_name', 'n/a')}`",
        f"- Method family: `{primary.get('method_family', 'n/a')}`",
        f"- Feature set: `{primary.get('feature_set', 'n/a')}`",
        f"- Brier: `{format_report_metric(primary.get('brier'))}`",
        f"- ECE: `{format_report_metric(primary.get('ece'))}`",
        f"- AUC: `{format_report_metric(primary.get('auc'))}`",
        f"- Average precision: `{format_report_metric(primary.get('average_precision'))}`",
        "",
    ]
    report_path.write_text(existing.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")


def format_report_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except TypeError:
        pass
    return f"{float(value):.6g}"


def primary_default_topk_fractions() -> tuple[float, ...]:
    return tuple(default_topk_fractions())


def primary_default_feature_sets() -> tuple[str, ...]:
    return tuple(default_feature_sets())
