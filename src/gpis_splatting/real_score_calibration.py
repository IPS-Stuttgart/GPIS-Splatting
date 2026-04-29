from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.real_field_scores import format_optional, score_columns
from gpis_splatting.real_gate_diagnostics import correlation, finite_or_none, rankdata_average, validate_diagnostic_config
from gpis_splatting.real_geometry import format_threshold_label
from gpis_splatting.serialization import write_json


DEFAULT_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "gpis_field": ("abs_mu", "sigma", "grad_norm", "abs_signed_distance", "distance_std"),
    "gpis_with_gate": ("abs_mu", "sigma", "grad_norm", "abs_signed_distance", "distance_std", "score_current_gate", "score_raw_surface_band"),
    "gpis_scores": (
        "score_current_gate",
        "score_raw_surface_band",
        "score_exp_neg_abs_distance",
        "score_negative_abs_distance",
        "score_negative_distance_std",
        "score_variance_penalized_band",
        "score_variance_penalized_exp",
        "score_negative_abs_mu",
    ),
}
DEFAULT_BASELINE_SCORES = (
    "score_current_gate",
    "score_raw_surface_band",
    "score_variance_penalized_band",
    "score_variance_penalized_exp",
    "score_negative_abs_distance",
    "score_negative_distance_std",
)


@dataclass(frozen=True)
class TrainValidationSplit:
    train_mask: np.ndarray
    validation_mask: np.ndarray


@dataclass(frozen=True)
class ScoreTransform:
    column: str
    train_min: float
    train_max: float

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        values = table[self.column].to_numpy(dtype=np.float64)
        denom = max(self.train_max - self.train_min, 1e-12)
        return np.clip((sanitize_vector(values) - self.train_min) / denom, 0.0, 1.0)


@dataclass(frozen=True)
class LogisticCalibrationModel:
    feature_names: tuple[str, ...]
    fill_values: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    constant_probability: float | None = None

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        if self.constant_probability is not None:
            return np.full((len(table),), self.constant_probability, dtype=np.float64)
        features = build_feature_matrix(table, self.feature_names, fill_values=self.fill_values)
        standardized = (features - self.mean[None, :]) / self.scale[None, :]
        return sigmoid(standardized @ self.weights + self.bias)


@dataclass(frozen=True)
class IsotonicCalibrationModel:
    score_column: str
    thresholds: np.ndarray
    values: np.ndarray
    fallback_probability: float

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        scores = sanitize_vector(table[self.score_column].to_numpy(dtype=np.float64))
        if self.thresholds.size == 0:
            return np.full(scores.shape, self.fallback_probability, dtype=np.float64)
        indices = np.searchsorted(self.thresholds, scores, side="left")
        indices = np.clip(indices, 0, self.values.shape[0] - 1)
        return np.clip(self.values[indices], 1e-6, 1.0 - 1e-6)


@dataclass(frozen=True)
class CalibratedMethod:
    name: str
    family: str
    feature_set: str | None
    model: ScoreTransform | LogisticCalibrationModel | IsotonicCalibrationModel

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return self.model.predict(table)


def run_gpis_splat_score_calibration(
    *,
    field_scores_path: str | Path,
    output_dir: str | Path | None = None,
    method_name: str | None = None,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.1),
    topk_fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
    feature_sets: tuple[str, ...] = tuple(DEFAULT_FEATURE_SETS),
    baseline_scores: tuple[str, ...] | None = DEFAULT_BASELINE_SCORES,
    isotonic_scores: tuple[str, ...] | None = ("score_current_gate", "score_raw_surface_band", "score_variance_penalized_band"),
    validation_fraction: float = 0.35,
    seed: int = 13,
    logistic_iterations: int = 600,
    learning_rate: float = 0.05,
    regularization: float = 1e-3,
    num_bins: int = 10,
) -> dict[str, Any]:
    validate_calibration_config(
        thresholds=thresholds,
        topk_fractions=topk_fractions,
        feature_sets=feature_sets,
        validation_fraction=validation_fraction,
        logistic_iterations=logistic_iterations,
        learning_rate=learning_rate,
        regularization=regularization,
        num_bins=num_bins,
    )
    baseline_scores = DEFAULT_BASELINE_SCORES if baseline_scores is None else baseline_scores
    isotonic_scores = ("score_current_gate", "score_raw_surface_band", "score_variance_penalized_band") if isotonic_scores is None else isotonic_scores
    field_path = Path(field_scores_path)
    table = prepare_feature_table(pd.read_csv(field_path))
    split = deterministic_train_validation_split(len(table), validation_fraction=validation_fraction, seed=seed)
    out_dir = Path(output_dir) if output_dir is not None else field_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = method_name or default_calibration_prefix(field_path)

    summary_rows: list[dict[str, Any]] = []
    ranked_rows: list[dict[str, Any]] = []
    prediction_columns: dict[str, np.ndarray] = {
        "splat_index": table["splat_index"].to_numpy(dtype=np.int64) if "splat_index" in table.columns else np.arange(len(table), dtype=np.int64),
        "nearest_gt_distance": table["nearest_gt_distance"].to_numpy(dtype=np.float64),
        "split": np.where(split.validation_mask, "validation", "train"),
    }
    best_by_threshold: list[dict[str, Any]] = []

    for threshold in thresholds:
        label_name = f"label_within_{format_threshold_label(threshold)}"
        labels = (table["nearest_gt_distance"].to_numpy(dtype=np.float64) <= threshold).astype(np.float64)
        prediction_columns[label_name] = labels.astype(np.int64)
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
        threshold_summaries = []
        for method in methods:
            probabilities = np.clip(method.predict(table), 1e-6, 1.0 - 1e-6)
            validation_probabilities = probabilities[split.validation_mask]
            validation_labels = labels[split.validation_mask]
            metrics = classification_metrics(validation_labels, validation_probabilities, num_bins=num_bins)
            ranked = ranked_label_selection(
                probabilities=validation_probabilities,
                labels=validation_labels,
                topk_fractions=topk_fractions,
                geometry_threshold=threshold,
                method_name=method.name,
                family=method.family,
                feature_set=method.feature_set,
            )
            best = best_ranked_row(ranked)
            full = ranked[ranked["topk_fraction"] == 1.0].iloc[0]
            summary = {
                "geometry_threshold": float(threshold),
                "method_name": method.name,
                "method_family": method.family,
                "feature_set": method.feature_set,
                "train_count": int(split.train_mask.sum()),
                "validation_count": int(split.validation_mask.sum()),
                "train_positive_rate": float(labels[split.train_mask].mean()),
                "validation_positive_rate": float(validation_labels.mean()),
                **metrics,
                "best_topk_fraction": float(best["topk_fraction"]),
                "best_retention_fraction": float(best["retention_fraction"]),
                "best_precision": float(best["precision"]),
                "best_recall": float(best["recall"]),
                "best_f_score": float(best["f_score"]),
                "full_precision": float(full["precision"]),
                "full_recall": float(full["recall"]),
                "full_f_score": float(full["f_score"]),
                "delta_best_f_score_vs_full": float(best["f_score"] - full["f_score"]),
            }
            summary_rows.append(summary)
            threshold_summaries.append(summary)
            ranked_rows.extend(ranked.to_dict(orient="records"))
        best_method = select_best_method(threshold_summaries)
        best_by_threshold.append(best_method)
        selected_method = next(method for method in methods if method.name == best_method["method_name"])
        confidence_column = f"confidence_{format_threshold_label(threshold)}"
        prediction_columns[confidence_column] = np.clip(selected_method.predict(table), 1e-6, 1.0 - 1e-6)
        prediction_columns[f"selected_method_{format_threshold_label(threshold)}"] = np.full((len(table),), selected_method.name, dtype=object)

    summary = pd.DataFrame(summary_rows)
    ranked = pd.DataFrame(ranked_rows)
    predictions = pd.DataFrame(prediction_columns)
    summary_path = out_dir / f"{prefix}_calibration_summary.csv"
    ranked_path = out_dir / f"{prefix}_calibration_ranked.csv"
    predictions_path = out_dir / f"{prefix}_calibrated_splat_scores.csv"
    confidence_path = out_dir / f"{prefix}_calibrated_confidence.npz"
    status_path = out_dir / f"{prefix}_calibration_status.json"
    report_path = out_dir / f"{prefix}_calibration_report.md"
    summary.to_csv(summary_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    np.savez_compressed(
        confidence_path,
        splat_index=predictions["splat_index"].to_numpy(dtype=np.int64),
        **{
            column: predictions[column].to_numpy()
            for column in predictions.columns
            if column.startswith("confidence_") or column.startswith("label_within_")
        },
    )
    status = {
        "schema_version": 1,
        "field_scores_path": str(field_path),
        "method": prefix,
        "thresholds": list(thresholds),
        "topk_fractions": list(topk_fractions),
        "feature_sets": list(feature_sets),
        "baseline_scores": [column for column in baseline_scores if column in table.columns],
        "isotonic_scores": [column for column in isotonic_scores if column in table.columns],
        "validation_fraction": validation_fraction,
        "seed": seed,
        "row_count": int(len(table)),
        "train_count": int(split.train_mask.sum()),
        "validation_count": int(split.validation_mask.sum()),
        "summary_path": str(summary_path),
        "ranked_path": str(ranked_path),
        "predictions_path": str(predictions_path),
        "confidence_path": str(confidence_path),
        "report_path": str(report_path),
        "best_by_threshold": best_by_threshold,
    }
    write_json(status_path, status)
    report_path.write_text(format_calibration_report(status, summary), encoding="utf-8")
    return {
        "summary_path": summary_path,
        "ranked_path": ranked_path,
        "predictions_path": predictions_path,
        "confidence_path": confidence_path,
        "status_path": status_path,
        "report_path": report_path,
        "summary": summary,
        "ranked": ranked,
        "predictions": predictions,
        "status": status,
    }


def validate_calibration_config(
    *,
    thresholds: tuple[float, ...],
    topk_fractions: tuple[float, ...],
    feature_sets: tuple[str, ...],
    validation_fraction: float,
    logistic_iterations: int,
    learning_rate: float,
    regularization: float,
    num_bins: int,
) -> None:
    validate_diagnostic_config(thresholds=thresholds, topk_fractions=topk_fractions, num_bins=num_bins, distance_chunk_size=1)
    unknown = [name for name in feature_sets if name not in DEFAULT_FEATURE_SETS]
    if unknown:
        raise ValueError(f"Unknown feature set(s): {', '.join(unknown)}.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    if logistic_iterations < 1:
        raise ValueError("logistic_iterations must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative.")


def prepare_feature_table(table: pd.DataFrame) -> pd.DataFrame:
    required = {"nearest_gt_distance"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Field-score table is missing required column(s): {', '.join(missing)}.")
    prepared = table.copy()
    if "mu" in prepared.columns and "abs_mu" not in prepared.columns:
        prepared["abs_mu"] = np.abs(prepared["mu"].to_numpy(dtype=np.float64))
    if "signed_distance" in prepared.columns and "abs_signed_distance" not in prepared.columns:
        prepared["abs_signed_distance"] = np.abs(prepared["signed_distance"].to_numpy(dtype=np.float64))
    if "splat_index" not in prepared.columns:
        prepared["splat_index"] = np.arange(len(prepared), dtype=np.int64)
    available_scores = score_columns(prepared)
    if not available_scores:
        raise ValueError("Field-score table does not contain any score_* columns.")
    if len(prepared) < 3:
        raise ValueError("At least three rows are required for train/validation calibration.")
    return prepared


def deterministic_train_validation_split(n_rows: int, *, validation_fraction: float, seed: int) -> TrainValidationSplit:
    validation_count = int(np.ceil(validation_fraction * n_rows))
    validation_count = min(max(1, validation_count), n_rows - 1)
    rng = np.random.default_rng(seed)
    validation_indices = np.sort(rng.choice(n_rows, size=validation_count, replace=False))
    validation_mask = np.zeros((n_rows,), dtype=bool)
    validation_mask[validation_indices] = True
    train_mask = ~validation_mask
    return TrainValidationSplit(train_mask=train_mask, validation_mask=validation_mask)


def fit_calibration_methods(
    *,
    table: pd.DataFrame,
    labels: np.ndarray,
    train_mask: np.ndarray,
    feature_sets: tuple[str, ...],
    baseline_scores: tuple[str, ...],
    isotonic_scores: tuple[str, ...],
    logistic_iterations: int,
    learning_rate: float,
    regularization: float,
) -> list[CalibratedMethod]:
    methods: list[CalibratedMethod] = []
    for column in baseline_scores:
        if column not in table.columns:
            continue
        train_values = sanitize_vector(table.loc[train_mask, column].to_numpy(dtype=np.float64))
        methods.append(
            CalibratedMethod(
                name=f"minmax_{column}",
                family="score_minmax",
                feature_set=column,
                model=ScoreTransform(column=column, train_min=float(train_values.min()), train_max=float(train_values.max())),
            )
        )
    for column in isotonic_scores:
        if column not in table.columns:
            continue
        methods.append(
            CalibratedMethod(
                name=f"isotonic_{column}",
                family="isotonic",
                feature_set=column,
                model=fit_isotonic_model(
                    scores=table.loc[train_mask, column].to_numpy(dtype=np.float64),
                    labels=labels[train_mask],
                    score_column=column,
                ),
            )
        )
    for feature_set in feature_sets:
        feature_names = tuple(column for column in DEFAULT_FEATURE_SETS[feature_set] if column in table.columns)
        if not feature_names:
            continue
        methods.append(
            CalibratedMethod(
                name=f"logistic_{feature_set}",
                family="logistic",
                feature_set=feature_set,
                model=fit_logistic_model(
                    table=table.loc[train_mask],
                    labels=labels[train_mask],
                    feature_names=feature_names,
                    iterations=logistic_iterations,
                    learning_rate=learning_rate,
                    regularization=regularization,
                ),
            )
        )
    if not methods:
        raise ValueError("No calibration methods could be fit from the available columns.")
    return methods


def fit_logistic_model(
    *,
    table: pd.DataFrame,
    labels: np.ndarray,
    feature_names: tuple[str, ...],
    iterations: int,
    learning_rate: float,
    regularization: float,
) -> LogisticCalibrationModel:
    labels = np.asarray(labels, dtype=np.float64)
    positive_rate = float(np.clip(labels.mean(), 1e-6, 1.0 - 1e-6))
    fill_values = feature_fill_values(table, feature_names)
    features = build_feature_matrix(table, feature_names, fill_values=fill_values)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    if np.all(labels == labels[0]):
        return LogisticCalibrationModel(
            feature_names=feature_names,
            fill_values=fill_values,
            mean=mean,
            scale=scale,
            weights=np.zeros((len(feature_names),), dtype=np.float64),
            bias=logit(positive_rate),
            constant_probability=positive_rate,
        )
    standardized = (features - mean[None, :]) / scale[None, :]
    weights = np.zeros((standardized.shape[1],), dtype=np.float64)
    bias = logit(positive_rate)
    for _ in range(iterations):
        probabilities = sigmoid(standardized @ weights + bias)
        error = probabilities - labels
        grad_w = (standardized.T @ error) / labels.shape[0] + regularization * weights
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return LogisticCalibrationModel(feature_names=feature_names, fill_values=fill_values, mean=mean, scale=scale, weights=weights, bias=float(bias))


def fit_isotonic_model(*, scores: np.ndarray, labels: np.ndarray, score_column: str) -> IsotonicCalibrationModel:
    scores = sanitize_vector(scores)
    labels = np.asarray(labels, dtype=np.float64)
    fallback = float(np.clip(labels.mean(), 1e-6, 1.0 - 1e-6))
    if scores.size == 0:
        return IsotonicCalibrationModel(score_column=score_column, thresholds=np.asarray([], dtype=np.float64), values=np.asarray([], dtype=np.float64), fallback_probability=fallback)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    block_values: list[float] = []
    block_weights: list[int] = []
    block_max_scores: list[float] = []
    for score, label in zip(sorted_scores, sorted_labels, strict=False):
        block_values.append(float(label))
        block_weights.append(1)
        block_max_scores.append(float(score))
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            left_weight = block_weights[-2]
            right_weight = block_weights[-1]
            merged_weight = left_weight + right_weight
            merged_value = (block_values[-2] * left_weight + block_values[-1] * right_weight) / merged_weight
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            block_max_scores[-2:] = [block_max_scores[-1]]
    return IsotonicCalibrationModel(
        score_column=score_column,
        thresholds=np.asarray(block_max_scores, dtype=np.float64),
        values=np.clip(np.asarray(block_values, dtype=np.float64), 1e-6, 1.0 - 1e-6),
        fallback_probability=fallback,
    )


def build_feature_matrix(table: pd.DataFrame, feature_names: tuple[str, ...], *, fill_values: np.ndarray) -> np.ndarray:
    columns = []
    for index, name in enumerate(feature_names):
        values = table[name].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        columns.append(np.where(finite, values, fill_values[index]))
    return np.column_stack(columns).astype(np.float64)


def feature_fill_values(table: pd.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    values = []
    for name in feature_names:
        column = table[name].to_numpy(dtype=np.float64)
        finite = column[np.isfinite(column)]
        values.append(float(np.median(finite)) if finite.size else 0.0)
    return np.asarray(values, dtype=np.float64)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, *, num_bins: int) -> dict[str, float | None]:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return {
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "nll": float(-np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))),
        "ece": expected_calibration_error(labels, probabilities, num_bins=num_bins),
        "auc": roc_auc(labels, probabilities),
        "average_precision": average_precision(labels, probabilities),
        "spearman_probability_vs_label": finite_or_none(correlation(rankdata_average(probabilities), rankdata_average(labels))),
        "mean_probability": float(probabilities.mean()),
    }


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, *, num_bins: int) -> float:
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    error = 0.0
    for index in range(num_bins):
        lo = edges[index]
        hi = edges[index + 1]
        if index == num_bins - 1:
            mask = (probabilities >= lo) & (probabilities <= hi)
        else:
            mask = (probabilities >= lo) & (probabilities < hi)
        if not np.any(mask):
            continue
        error += float(mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean()))
    return error


def roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positive = labels == 1.0
    negative = labels == 0.0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata_average(probabilities)
    auc = (float(ranks[positive].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = float(labels.sum())
    if positives <= 0.0:
        return None
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / np.arange(1, sorted_labels.shape[0] + 1, dtype=np.float64)
    return float(np.sum(precision * sorted_labels) / positives)


def ranked_label_selection(
    *,
    probabilities: np.ndarray,
    labels: np.ndarray,
    topk_fractions: tuple[float, ...],
    geometry_threshold: float,
    method_name: str,
    family: str,
    feature_set: str | None,
) -> pd.DataFrame:
    order = np.argsort(-probabilities, kind="mergesort")
    n_points = int(labels.shape[0])
    positives = float(labels.sum())
    rows = []
    for fraction in sorted(set(topk_fractions)):
        count = max(1, min(n_points, int(np.ceil(fraction * n_points))))
        selected = order[:count]
        selected_labels = labels[selected]
        precision = float(selected_labels.mean())
        recall = 0.0 if positives <= 0.0 else float(selected_labels.sum() / positives)
        f_score = 0.0 if precision + recall <= 0.0 else float(2.0 * precision * recall / (precision + recall))
        rows.append(
            {
                "geometry_threshold": float(geometry_threshold),
                "method_name": method_name,
                "method_family": family,
                "feature_set": feature_set,
                "topk_fraction": float(fraction),
                "selected_splat_count": count,
                "retention_fraction": count / n_points,
                "probability_min": float(probabilities[selected].min()),
                "probability_mean": float(probabilities[selected].mean()),
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
            }
        )
    return pd.DataFrame(rows)


def best_ranked_row(ranked: pd.DataFrame) -> pd.Series:
    return ranked.sort_values(["f_score", "retention_fraction"], ascending=[False, True]).iloc[0]


def select_best_method(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = pd.DataFrame(summary_rows)
    sorted_summary = summary.sort_values(["brier", "best_f_score", "auc"], ascending=[True, False, False], na_position="last")
    row = sorted_summary.iloc[0]
    return {
        "geometry_threshold": float(row["geometry_threshold"]),
        "method_name": row["method_name"],
        "method_family": row["method_family"],
        "feature_set": row["feature_set"],
        "brier": float(row["brier"]),
        "nll": float(row["nll"]),
        "ece": float(row["ece"]),
        "auc": maybe_float(row["auc"]),
        "average_precision": maybe_float(row["average_precision"]),
        "best_topk_fraction": float(row["best_topk_fraction"]),
        "best_f_score": float(row["best_f_score"]),
    }


def format_calibration_report(status: dict[str, Any], summary: pd.DataFrame) -> str:
    lines = [
        "# GPIS Splat Score Calibration",
        "",
        f"- Method: `{status['method']}`",
        f"- Field-score CSV: `{status['field_scores_path']}`",
        f"- Rows: `{status['row_count']}`",
        f"- Train/validation: `{status['train_count']}` / `{status['validation_count']}`",
        f"- Summary CSV: `{status['summary_path']}`",
        f"- Ranked CSV: `{status['ranked_path']}`",
        f"- Calibrated scores CSV: `{status['predictions_path']}`",
        f"- Confidence NPZ: `{status['confidence_path']}`",
        "",
        "## Best Calibrators",
        "",
    ]
    for row in status["best_by_threshold"]:
        lines.append(
            f"- threshold `{row['geometry_threshold']:.6g}`: `{row['method_name']}` "
            f"Brier `{row['brier']:.6g}`, NLL `{row['nll']:.6g}`, ECE `{row['ece']:.6g}`, "
            f"AUC `{format_optional(row['auc'])}`, AP `{format_optional(row['average_precision'])}`, best F-score `{row['best_f_score']:.6g}`"
        )
    if not summary.empty:
        lines.extend(["", "## Summary Table", "", format_summary_table(summary)])
    return "\n".join(lines) + "\n"


def format_summary_table(summary: pd.DataFrame) -> str:
    columns = ["geometry_threshold", "method_name", "method_family", "brier", "nll", "ece", "auc", "average_precision", "best_topk_fraction", "best_f_score"]
    lines = [
        "| threshold | method | family | brier | nll | ece | auc | ap | top_k | best_f |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    sorted_summary = summary[columns].sort_values(["geometry_threshold", "brier", "best_f_score"], ascending=[True, True, False])
    for row in sorted_summary.itertuples(index=False):
        lines.append(
            f"| {row.geometry_threshold:.6g} | `{row.method_name}` | `{row.method_family}` | {row.brier:.6g} | {row.nll:.6g} | "
            f"{row.ece:.6g} | {format_optional(row.auc)} | {format_optional(row.average_precision)} | {row.best_topk_fraction:.6g} | {row.best_f_score:.6g} |"
        )
    return "\n".join(lines)


def default_calibration_prefix(path: Path) -> str:
    stem = path.stem
    suffix = "_gpis_field_scores"
    if stem.endswith(suffix):
        return f"{stem[: -len(suffix)]}_calibrated"
    return f"{stem}_calibrated"


def sanitize_vector(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    fill = float(np.median(finite)) if finite.size else 0.0
    return np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def logit(probability: float) -> float:
    probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(probability / (1.0 - probability)))


def maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def default_topk_fractions() -> tuple[float, ...]:
    return tuple(float(value) for value in (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0))


def default_feature_sets() -> tuple[str, ...]:
    return tuple(DEFAULT_FEATURE_SETS)
