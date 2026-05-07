from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd

from gpis_splatting.confidence import (
    CalibratedConfidenceBundle,
    StoredConfidenceMethod,
    brier_score,
    expected_calibration_error,
    format_threshold_label,
    read_calibrated_confidence_bundle,
    reliability_table,
    write_gate_compatible_confidences,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_LABEL_COLUMN = "nearest_gt_distance"
DEFAULT_THRESHOLDS = (0.02, 0.05, 0.1)
DEFAULT_SCORE_COLUMNS = (
    "score_current_gate",
    "score_raw_surface_band",
    "score_variance_penalized_band",
    "score_variance_penalized_exp",
    "score_negative_abs_distance",
    "score_negative_distance_std",
)
DEFAULT_GROUP_COLUMN_CANDIDATES = (
    "scene",
    "scene_id",
    "dataset_scene",
    "source_splat_index",
    "base_splat_index",
    "source_point3d_id",
    "point3d_id",
    "track_id",
)
DEFAULT_EXCLUDED_FEATURE_COLUMNS = frozenset(
    {
        "splat_index",
        "candidate_index",
        "source_splat_index",
        "base_splat_index",
        "point3d_id",
        "track_id",
        "scene",
        "scene_id",
        "dataset",
        "dataset_scene",
        "candidate_type",
        "split",
        "fold",
        "is_surface",
        "is_generated_hard_negative",
        DEFAULT_LABEL_COLUMN,
    }
)
DEFAULT_EXCLUDED_FEATURE_PREFIXES = ("label", "within_", "gt_", "ground_truth", "nearest_gt_")
COORDINATE_COLUMN_SETS = (("query_x", "query_y", "query_z"), ("eval_x", "eval_y", "eval_z"), ("x", "y", "z"), ("center_x", "center_y", "center_z"))
EPS = 1e-12


@dataclass(frozen=True)
class ConfidenceFeatureConfig:
    """Configuration for deterministic GPIS/splat feature extraction."""

    feature_columns: tuple[str, ...] | None = None
    extra_feature_columns: tuple[str, ...] = ()
    exclude_columns: tuple[str, ...] = ()
    include_score_columns: bool = True
    include_coordinate_columns: bool = False
    include_derived_features: bool = True
    allow_label_like_features: bool = False


@dataclass(frozen=True)
class ConfidenceSplitConfig:
    """Leakage-aware train/validation split configuration."""

    validation_fraction: float = 0.35
    seed: int = 13
    group_columns: tuple[str, ...] = ()
    auto_group_columns: bool = True
    spatial_cell_size: float | None = None
    coordinate_columns: tuple[str, str, str] | None = None


@dataclass(frozen=True)
class ConfidenceFitConfig:
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    label_column: str = DEFAULT_LABEL_COLUMN
    feature_config: ConfidenceFeatureConfig = field(default_factory=ConfidenceFeatureConfig)
    split_config: ConfidenceSplitConfig = field(default_factory=ConfidenceSplitConfig)
    baseline_score_columns: tuple[str, ...] = DEFAULT_SCORE_COLUMNS
    isotonic_score_columns: tuple[str, ...] = ("score_current_gate", "score_raw_surface_band", "score_variance_penalized_band")
    logistic_iterations: int = 600
    learning_rate: float = 0.05
    regularization: float = 1e-3
    num_bins: int = 10
    selection_metric: str = "brier"
    write_reliability_plots: bool = True
    gate_count: int | None = None
    missing_gate_value: float = 0.0


@dataclass(frozen=True)
class ConfidenceFeatureTable:
    table: pd.DataFrame
    feature_columns: tuple[str, ...]
    derived_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]


@dataclass(frozen=True)
class ConfidenceSplit:
    train_mask: np.ndarray
    validation_mask: np.ndarray
    group_key: np.ndarray
    group_count: int
    train_group_count: int
    validation_group_count: int
    strategy: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_index": np.arange(self.group_key.shape[0], dtype=np.int64),
                "split": np.where(self.validation_mask, "validation", "train"),
                "group_key": self.group_key.astype(str),
            }
        )


@dataclass(frozen=True)
class FittedConfidenceMethod:
    name: str
    family: str
    feature_set: str | None
    model: dict[str, Any]

    def stored(self, threshold: float) -> StoredConfidenceMethod:
        return StoredConfidenceMethod(float(threshold), self.name, self.family, self.feature_set, self.model)

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return self.stored(0.0).predict(table)


@dataclass(frozen=True)
class CalibratedConfidenceFitResult:
    bundle: CalibratedConfidenceBundle
    feature_table: pd.DataFrame
    split: ConfidenceSplit
    summary: pd.DataFrame
    predictions: pd.DataFrame
    reliability_tables: dict[str, pd.DataFrame]
    selected_methods: tuple[StoredConfidenceMethod, ...]
    feature_columns: tuple[str, ...]
    artifacts: dict[str, Any]
    status: dict[str, Any]


class ConfidenceFeatureExtractor:
    """Deterministic extractor for calibrated GPIS confidence features."""

    def __init__(self, config: ConfidenceFeatureConfig | None = None) -> None:
        self.config = config or ConfidenceFeatureConfig()
        self.feature_columns_: tuple[str, ...] | None = None
        self.derived_columns_: tuple[str, ...] = ()
        self.excluded_columns_: tuple[str, ...] = ()

    def fit(self, table: pd.DataFrame) -> "ConfidenceFeatureExtractor":
        extracted = self._build(table)
        self.feature_columns_ = extracted.feature_columns
        self.derived_columns_ = extracted.derived_columns
        self.excluded_columns_ = extracted.excluded_columns
        return self

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns_ is None:
            raise ValueError("ConfidenceFeatureExtractor must be fit before transform.")
        extracted = self._build(table)
        missing = [column for column in self.feature_columns_ if column not in extracted.table.columns]
        if missing:
            raise ValueError(f"Input table is missing calibrated-confidence feature column(s): {', '.join(missing)}.")
        return extracted.table

    def fit_transform(self, table: pd.DataFrame) -> ConfidenceFeatureTable:
        extracted = self._build(table)
        self.feature_columns_ = extracted.feature_columns
        self.derived_columns_ = extracted.derived_columns
        self.excluded_columns_ = extracted.excluded_columns
        return extracted

    def _build(self, table: pd.DataFrame) -> ConfidenceFeatureTable:
        prepared = table.copy()
        derived = add_derived_confidence_features(prepared) if self.config.include_derived_features else []
        features, excluded = select_confidence_feature_columns(prepared, self.config)
        return ConfidenceFeatureTable(prepared, tuple(features), tuple(derived), tuple(excluded))


def add_derived_confidence_features(table: pd.DataFrame) -> list[str]:
    added: list[str] = []

    def add(name: str, values: np.ndarray) -> None:
        if name not in table.columns:
            table[name] = np.asarray(values, dtype=np.float64)
            added.append(name)

    if "mu" in table.columns:
        mu = numeric_array(table["mu"])
        add("abs_mu", np.abs(mu))
    if "signed_distance" in table.columns:
        distance = numeric_array(table["signed_distance"])
        add("abs_signed_distance", np.abs(distance))
    if "sigma" in table.columns:
        sigma = positive_array(table["sigma"])
        add("log_sigma", np.log(sigma))
    if "variance" in table.columns:
        variance = positive_array(table["variance"])
        add("log_variance", np.log(variance))
    if "distance_std" in table.columns:
        distance_std = positive_array(table["distance_std"])
        add("log_distance_std", np.log(distance_std))
    if {"abs_signed_distance", "distance_std"}.issubset(table.columns):
        add("distance_snr", numeric_array(table["abs_signed_distance"]) / positive_array(table["distance_std"]))
    if {"distance_std", "grad_norm"}.issubset(table.columns):
        add("distance_uncertainty_ratio", numeric_array(table["distance_std"]) / positive_array(table["grad_norm"]))
    if {"abs_mu", "sigma"}.issubset(table.columns):
        add("abs_mu_over_sigma", numeric_array(table["abs_mu"]) / positive_array(table["sigma"]))
    if "grad_norm" in table.columns:
        add("log_grad_norm", np.log(positive_array(table["grad_norm"])))
    if {"signed_distance", "grad_norm"}.issubset(table.columns):
        add("signed_distance_over_grad_norm", numeric_array(table["signed_distance"]) / positive_array(table["grad_norm"]))
    if {"abs_signed_distance", "distance_std"}.issubset(table.columns):
        add("score_gpis_surface_likelihood", np.exp(-0.5 * (numeric_array(table["abs_signed_distance"]) / positive_array(table["distance_std"])) ** 2))
    return added


def select_confidence_feature_columns(table: pd.DataFrame, config: ConfidenceFeatureConfig) -> tuple[list[str], list[str]]:
    if config.feature_columns is not None:
        columns = list(config.feature_columns)
        missing = [column for column in columns if column not in table.columns]
        if missing:
            raise ValueError(f"Requested confidence feature column(s) are missing: {', '.join(missing)}.")
    else:
        columns = [column for column in table.columns if is_candidate_feature_column(column, table[column], config)]
    for column in config.extra_feature_columns:
        if column not in table.columns:
            raise ValueError(f"Requested extra confidence feature column is missing: {column}.")
        if column not in columns:
            columns.append(column)
    excluded = sorted(set(table.columns) - set(columns))
    leaky = [column for column in columns if is_label_like_column(column)]
    if leaky and not config.allow_label_like_features:
        raise ValueError("Refusing to use label-like calibrated-confidence feature column(s): " + ", ".join(leaky))
    if not columns:
        raise ValueError("No calibrated-confidence feature columns are available after leakage filtering.")
    return columns, excluded


def is_candidate_feature_column(column: str, series: pd.Series, config: ConfidenceFeatureConfig) -> bool:
    if column in DEFAULT_EXCLUDED_FEATURE_COLUMNS or column in config.exclude_columns:
        return False
    if is_label_like_column(column) and not config.allow_label_like_features:
        return False
    if not config.include_score_columns and column.startswith("score_"):
        return False
    if not config.include_coordinate_columns and any(column in coords for coords in COORDINATE_COLUMN_SETS):
        return False
    return pd.api.types.is_numeric_dtype(series)


def is_label_like_column(column: str) -> bool:
    return column == DEFAULT_LABEL_COLUMN or any(column.startswith(prefix) for prefix in DEFAULT_EXCLUDED_FEATURE_PREFIXES)


def make_leakage_free_split(table: pd.DataFrame, config: ConfidenceSplitConfig | None = None) -> ConfidenceSplit:
    cfg = config or ConfidenceSplitConfig()
    if len(table) < 3:
        raise ValueError("At least three rows are required for train/validation calibration.")
    if not 0.0 < cfg.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    group_key, strategy = build_group_keys(table, cfg)
    unique_groups = np.unique(group_key)
    if unique_groups.shape[0] < 2:
        return row_level_split(len(table), cfg)
    rng = np.random.default_rng(cfg.seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    validation_group_count = int(np.ceil(cfg.validation_fraction * shuffled.shape[0]))
    validation_group_count = min(max(1, validation_group_count), shuffled.shape[0] - 1)
    validation_groups = set(shuffled[:validation_group_count].tolist())
    validation_mask = np.asarray([key in validation_groups for key in group_key], dtype=bool)
    if validation_mask.all() or not validation_mask.any():
        return row_level_split(len(table), cfg)
    train_mask = ~validation_mask
    return ConfidenceSplit(train_mask, validation_mask, group_key, int(unique_groups.shape[0]), int(np.unique(group_key[train_mask]).shape[0]), int(np.unique(group_key[validation_mask]).shape[0]), strategy)


def build_group_keys(table: pd.DataFrame, config: ConfidenceSplitConfig) -> tuple[np.ndarray, str]:
    parts: list[np.ndarray] = []
    names = list(config.group_columns)
    if config.auto_group_columns:
        names.extend(column for column in DEFAULT_GROUP_COLUMN_CANDIDATES if column in table.columns and column not in names)
    for column in names:
        if column not in table.columns:
            raise ValueError(f"Requested split group column is missing: {column}.")
        values = table[column].astype(str).to_numpy()
        if column == "source_splat_index":
            row_ids = np.arange(len(table), dtype=np.int64).astype(str)
            values = np.where(values == "-1", np.char.add("ungrouped:", row_ids), np.char.add("source_splat_index:", values))
        else:
            values = np.char.add(f"{column}:", values)
        parts.append(values.astype(str))
    if config.spatial_cell_size is not None:
        coords = resolve_coordinate_columns(table, config.coordinate_columns)
        cells = np.floor(table.loc[:, list(coords)].to_numpy(dtype=np.float64) / float(config.spatial_cell_size)).astype(np.int64)
        parts.append(np.asarray([f"cell:{row[0]}:{row[1]}:{row[2]}" for row in cells], dtype=str))
    if not parts:
        return np.arange(len(table), dtype=np.int64).astype(str), "row"
    keys = parts[0]
    for part in parts[1:]:
        keys = np.char.add(np.char.add(keys, "|"), part)
    return keys.astype(str), f"columns={'+'.join(names)}" if names else "spatial_cell"


def row_level_split(n_rows: int, config: ConfidenceSplitConfig) -> ConfidenceSplit:
    rng = np.random.default_rng(config.seed)
    validation_count = int(np.ceil(config.validation_fraction * n_rows))
    validation_count = min(max(1, validation_count), n_rows - 1)
    validation_indices = rng.choice(n_rows, size=validation_count, replace=False)
    validation_mask = np.zeros((n_rows,), dtype=bool)
    validation_mask[validation_indices] = True
    group_key = np.arange(n_rows, dtype=np.int64).astype(str)
    return ConfidenceSplit(~validation_mask, validation_mask, group_key, n_rows, n_rows - validation_count, validation_count, "row")


def resolve_coordinate_columns(table: pd.DataFrame, requested: tuple[str, str, str] | None) -> tuple[str, str, str]:
    if requested is not None:
        if any(column not in table.columns for column in requested):
            raise ValueError("Requested spatial split coordinate columns are missing.")
        return requested
    for columns in COORDINATE_COLUMN_SETS:
        if all(column in table.columns for column in columns):
            return columns
    raise ValueError("Spatial splitting requires coordinate columns such as query_x/query_y/query_z.")


def fit_calibrated_confidence(table: pd.DataFrame, config: ConfidenceFitConfig | None = None) -> CalibratedConfidenceFitResult:
    cfg = config or ConfidenceFitConfig()
    validate_fit_config(cfg)
    if cfg.label_column not in table.columns:
        raise ValueError(f"Field-score table is missing required label column: {cfg.label_column}.")
    features = ConfidenceFeatureExtractor(cfg.feature_config).fit_transform(table)
    split = make_leakage_free_split(features.table, cfg.split_config)
    selected: list[StoredConfidenceMethod] = []
    summary_rows: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {
        "splat_index": features.table["splat_index"].to_numpy(dtype=np.int64) if "splat_index" in features.table.columns else np.arange(len(features.table), dtype=np.int64),
        "split": np.where(split.validation_mask, "validation", "train"),
    }
    reliability: dict[str, pd.DataFrame] = {}
    distances = features.table[cfg.label_column].to_numpy(dtype=np.float64)
    for threshold in cfg.thresholds:
        label = format_threshold_label(float(threshold))
        labels = (distances <= float(threshold)).astype(np.float64)
        methods = build_candidate_methods(features.table, labels, split.train_mask, features.feature_columns, cfg)
        scored_rows: list[dict[str, Any]] = []
        for method in methods:
            probabilities = np.clip(method.predict(features.table), 1e-6, 1.0 - 1e-6)
            metrics = classification_metrics(labels[split.validation_mask], probabilities[split.validation_mask], cfg.num_bins)
            row = {"geometry_threshold": float(threshold), "method_name": method.name, "method_family": method.family, "feature_set": method.feature_set, "train_count": int(split.train_mask.sum()), "validation_count": int(split.validation_mask.sum()), **metrics}
            scored_rows.append(row)
            summary_rows.append(row)
        chosen_row = select_best_summary(scored_rows, cfg.selection_metric)
        chosen = next(method for method in methods if method.name == chosen_row["method_name"])
        stored = chosen.stored(float(threshold))
        selected.append(stored)
        probabilities = np.clip(chosen.predict(features.table), 1e-6, 1.0 - 1e-6)
        predictions[f"label_within_{label}"] = labels.astype(np.int64)
        predictions[f"confidence_{label}"] = probabilities
        predictions[f"selected_method_{label}"] = np.full((len(features.table),), chosen.name, dtype=object)
        reliability[label] = reliability_table(labels[split.validation_mask], probabilities[split.validation_mask], num_bins=cfg.num_bins)
    bundle = CalibratedConfidenceBundle(
        methods=tuple(selected),
        metadata={
            "feature_columns": list(features.feature_columns),
            "derived_columns": list(features.derived_columns),
            "split": {"strategy": split.strategy, "group_count": split.group_count, "train_group_count": split.train_group_count, "validation_group_count": split.validation_group_count},
            "split_strategy": split.strategy,
            "validation_fraction": cfg.split_config.validation_fraction,
            "seed": cfg.split_config.seed,
            "thresholds": list(cfg.thresholds),
        },
    )
    status = {"schema_version": 1, "row_count": int(len(features.table)), "feature_columns": list(features.feature_columns), "thresholds": list(cfg.thresholds), "split_strategy": split.strategy, "train_count": int(split.train_mask.sum()), "validation_count": int(split.validation_mask.sum())}
    return CalibratedConfidenceFitResult(bundle, features.table, split, pd.DataFrame(summary_rows), pd.DataFrame(predictions), reliability, tuple(selected), features.feature_columns, {}, status)


def run_calibrated_confidence_fit(*, field_scores_path: str | Path, output_dir: str | Path | None = None, method_name: str | None = None, metadata_path: str | Path | None = None, config: ConfidenceFitConfig | None = None) -> CalibratedConfidenceFitResult:
    cfg = config or ConfidenceFitConfig()
    field_path = Path(field_scores_path)
    table = pd.read_csv(field_path)
    if metadata_path is not None:
        table = join_metadata(table, pd.read_csv(metadata_path))
    out_dir = Path(output_dir) if output_dir is not None else field_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = method_name or field_path.stem.replace("_gpis_field_scores", "")
    result = fit_calibrated_confidence(table, cfg)
    artifacts = write_confidence_artifacts(result, out_dir=out_dir, prefix=prefix, source_path=field_path, config=cfg)
    status = dict(result.status)
    status.update({"field_scores_path": str(field_path), "metadata_path": None if metadata_path is None else str(metadata_path), "method": prefix, "artifacts": {key: str(value) for key, value in artifacts.items()}})
    status_path = out_dir / f"{prefix}_confidence_api_status.json"
    status_path.write_text(json.dumps(json_ready(status), indent=2, sort_keys=True), encoding="utf-8")
    artifacts["status_path"] = status_path
    status["status_path"] = str(status_path)
    return CalibratedConfidenceFitResult(result.bundle, result.feature_table, result.split, result.summary, result.predictions, result.reliability_tables, result.selected_methods, result.feature_columns, artifacts, status)


def apply_calibrated_confidence_model(*, model_bundle_path: str | Path, field_scores_path: str | Path, output_path: str | Path | None = None, feature_config: ConfidenceFeatureConfig | None = None, threshold: float | None = None) -> pd.DataFrame:
    bundle = read_calibrated_confidence_bundle(model_bundle_path)
    table = pd.read_csv(field_scores_path)
    extractor = ConfidenceFeatureExtractor(feature_config or ConfidenceFeatureConfig())
    extracted = extractor.fit_transform(table)
    predictions = bundle.predict(extracted.table, threshold=threshold)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(path, index=False)
    return predictions


def write_confidence_artifacts(result: CalibratedConfidenceFitResult, *, out_dir: Path, prefix: str, source_path: Path, config: ConfidenceFitConfig) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    model_path = out_dir / f"{prefix}_calibrated_confidence_model.json"
    model_path.write_text(json.dumps(result.bundle.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    artifacts["model_bundle_path"] = model_path
    features_path = out_dir / f"{prefix}_confidence_features.csv"
    result.feature_table.to_csv(features_path, index=False)
    artifacts["features_path"] = features_path
    artifacts["feature_table_path"] = features_path
    split_path = out_dir / f"{prefix}_confidence_split.csv"
    result.split.to_frame().to_csv(split_path, index=False)
    artifacts["split_path"] = split_path
    summary_path = out_dir / f"{prefix}_confidence_summary.csv"
    result.summary.to_csv(summary_path, index=False)
    artifacts["summary_path"] = summary_path
    predictions_path = out_dir / f"{prefix}_calibrated_confidence.csv"
    result.predictions.to_csv(predictions_path, index=False)
    artifacts["predictions_path"] = predictions_path
    gate_paths = write_gate_compatible_confidences(predictions=result.predictions, out_dir=out_dir, prefix=prefix, thresholds=config.thresholds, predictions_path=predictions_path, gate_count=config.gate_count, missing_gate_value=config.missing_gate_value)
    artifacts.update({f"gate_{key}": path for key, path in gate_paths.items()})
    reliability_plot_paths: dict[str, Path] = {}
    for label, table in result.reliability_tables.items():
        reliability_path = out_dir / f"{prefix}_reliability_{label}.csv"
        table.to_csv(reliability_path, index=False)
        artifacts[f"reliability_{label}"] = reliability_path
        if config.write_reliability_plots:
            plot_path = out_dir / f"{prefix}_reliability_{label}.png"
            plot_reliability_table(table, plot_path, title=f"Reliability {prefix}@{label}")
            artifacts[f"reliability_plot_{label}"] = plot_path
            reliability_plot_paths[label] = plot_path
    artifacts["reliability_plot_paths"] = reliability_plot_paths
    report_path = out_dir / f"{prefix}_confidence_api_report.md"
    report_path.write_text(format_confidence_report(result, artifacts, source_path), encoding="utf-8")
    artifacts["report_path"] = report_path
    return artifacts


def join_metadata(field_table: pd.DataFrame, metadata_table: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ("splat_index", "candidate_index") if column in field_table.columns and column in metadata_table.columns]
    if keys:
        return field_table.merge(metadata_table, on=keys[0], how="left", suffixes=("", "_metadata"))
    if len(field_table) != len(metadata_table):
        raise ValueError("Metadata can only be row-concatenated when it has the same number of rows as the field-score table.")
    return pd.concat([field_table.reset_index(drop=True), metadata_table.reset_index(drop=True)], axis=1)


def build_candidate_methods(table: pd.DataFrame, labels: np.ndarray, train_mask: np.ndarray, feature_columns: tuple[str, ...], config: ConfidenceFitConfig) -> list[FittedConfidenceMethod]:
    methods: list[FittedConfidenceMethod] = []
    for column in config.baseline_score_columns:
        if column in table.columns:
            values = sanitize_vector(table.loc[train_mask, column].to_numpy(dtype=np.float64))
            methods.append(FittedConfidenceMethod(f"minmax_{column}", "score_minmax", column, {"kind": "score_minmax", "column": column, "train_min": float(values.min()), "train_max": float(values.max())}))
    for column in config.isotonic_score_columns:
        if column in table.columns:
            methods.append(FittedConfidenceMethod(f"isotonic_{column}", "isotonic", column, fit_isotonic_config(table.loc[train_mask, column].to_numpy(dtype=np.float64), labels[train_mask], column)))
    methods.append(FittedConfidenceMethod("logistic_confidence_features", "logistic", "confidence_features", fit_logistic_config(table.loc[train_mask], labels[train_mask], feature_columns, config)))
    return methods


def fit_logistic_config(table: pd.DataFrame, labels: np.ndarray, feature_names: tuple[str, ...], config: ConfidenceFitConfig) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float64)
    fill_values = feature_fill_values(table, feature_names)
    features = build_feature_matrix(table, feature_names, fill_values)
    mean = features.mean(axis=0)
    scale = np.where(features.std(axis=0) < EPS, 1.0, features.std(axis=0))
    positive_rate = float(np.clip(labels.mean(), 1e-6, 1.0 - 1e-6))
    if np.all(labels == labels[0]):
        weights = np.zeros((len(feature_names),), dtype=np.float64)
        return logistic_model_config(feature_names, fill_values, mean, scale, weights, logit(positive_rate), positive_rate)
    standardized = (features - mean[None, :]) / scale[None, :]
    weights = np.zeros((standardized.shape[1],), dtype=np.float64)
    bias = logit(positive_rate)
    for _ in range(config.logistic_iterations):
        probabilities = sigmoid(standardized @ weights + bias)
        error = probabilities - labels
        weights -= config.learning_rate * ((standardized.T @ error) / labels.shape[0] + config.regularization * weights)
        bias -= config.learning_rate * float(error.mean())
    return logistic_model_config(feature_names, fill_values, mean, scale, weights, float(bias), None)


def logistic_model_config(feature_names: tuple[str, ...], fill_values: np.ndarray, mean: np.ndarray, scale: np.ndarray, weights: np.ndarray, bias: float, constant: float | None) -> dict[str, Any]:
    return {"kind": "logistic", "feature_names": list(feature_names), "fill_values": fill_values.astype(float).tolist(), "mean": mean.astype(float).tolist(), "scale": scale.astype(float).tolist(), "weights": weights.astype(float).tolist(), "bias": float(bias), "constant_probability": constant}


def fit_isotonic_config(scores: np.ndarray, labels: np.ndarray, score_column: str) -> dict[str, Any]:
    scores = sanitize_vector(scores)
    labels = np.asarray(labels, dtype=np.float64)
    fallback = float(np.clip(labels.mean(), 1e-6, 1.0 - 1e-6))
    order = np.argsort(scores, kind="mergesort")
    block_values: list[float] = []
    block_weights: list[int] = []
    block_scores: list[float] = []
    for score, label in zip(scores[order], labels[order], strict=False):
        block_values.append(float(label))
        block_weights.append(1)
        block_scores.append(float(score))
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            left = block_weights[-2]
            right = block_weights[-1]
            weight = left + right
            block_values[-2:] = [(block_values[-2] * left + block_values[-1] * right) / weight]
            block_weights[-2:] = [weight]
            block_scores[-2:] = [block_scores[-1]]
    return {"kind": "isotonic", "score_column": score_column, "thresholds": block_scores, "values": np.clip(block_values, 1e-6, 1.0 - 1e-6).tolist(), "fallback_probability": fallback}


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, num_bins: int) -> dict[str, float | None]:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return {"brier": brier_score(labels, probabilities), "nll": float(-np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))), "ece": expected_calibration_error(labels, probabilities, num_bins=num_bins), "auc": roc_auc(labels, probabilities), "average_precision": average_precision(labels, probabilities), "mean_probability": float(probabilities.mean()), "positive_rate": float(labels.mean())}


def select_best_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    reverse = metric in {"auc", "average_precision"}
    return sorted(rows, key=lambda row: none_safe(row.get(metric), reverse=reverse), reverse=reverse)[0]


def none_safe(value: Any, *, reverse: bool) -> float:
    if value is None or pd.isna(value):
        return -np.inf if reverse else np.inf
    return float(value)


def plot_reliability_table(table: pd.DataFrame, output_path: str | Path, *, title: str = "Reliability") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(4.8, 4.2))
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0)
    non_empty = table[table["count"] > 0]
    if not non_empty.empty:
        axis.plot(non_empty["mean_probability"], non_empty["empirical_accuracy"], marker="o")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean predicted confidence")
    axis.set_ylabel("Empirical accuracy")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def format_confidence_report(result: CalibratedConfidenceFitResult, artifacts: dict[str, Any], source_path: Path) -> str:
    lines = ["# Calibrated Confidence API", "", f"- Source field scores: `{source_path}`", f"- Rows: `{len(result.feature_table)}`", f"- Train / validation rows: `{int(result.split.train_mask.sum())}` / `{int(result.split.validation_mask.sum())}`", f"- Split strategy: `{result.split.strategy}`", f"- Model bundle: `{artifacts['model_bundle_path']}`", f"- Predictions: `{artifacts['predictions_path']}`", "", "## Selected calibrators", "", "| threshold | method | family | feature set |", "| ---: | --- | --- | --- |"]
    for method in result.selected_methods:
        lines.append(f"| {method.geometry_threshold:.6g} | `{method.method_name}` | `{method.method_family}` | `{method.feature_set}` |")
    lines.extend(["", "## Feature columns", ""])
    lines.extend(f"- `{column}`" for column in result.feature_columns)
    return "\n".join(lines) + "\n"


def validate_fit_config(config: ConfidenceFitConfig) -> None:
    if not config.thresholds or any(threshold <= 0.0 for threshold in config.thresholds):
        raise ValueError("thresholds must contain positive values.")
    if config.logistic_iterations < 1:
        raise ValueError("logistic_iterations must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.regularization < 0.0:
        raise ValueError("regularization must be non-negative.")
    if config.num_bins < 1:
        raise ValueError("num_bins must be positive.")
    if config.selection_metric not in {"brier", "nll", "ece", "auc", "average_precision"}:
        raise ValueError("Unsupported selection_metric.")


def build_feature_matrix(table: pd.DataFrame, feature_names: Sequence[str], fill_values: np.ndarray) -> np.ndarray:
    columns = []
    for index, name in enumerate(feature_names):
        values = table[name].to_numpy(dtype=np.float64)
        columns.append(np.where(np.isfinite(values), values, fill_values[index]))
    return np.column_stack(columns).astype(np.float64)


def feature_fill_values(table: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    values = []
    for name in feature_names:
        column = table[name].to_numpy(dtype=np.float64)
        finite = column[np.isfinite(column)]
        values.append(float(np.median(finite)) if finite.size else 0.0)
    return np.asarray(values, dtype=np.float64)


def numeric_array(series: pd.Series) -> np.ndarray:
    return sanitize_vector(series.to_numpy(dtype=np.float64))


def positive_array(series: pd.Series) -> np.ndarray:
    return np.maximum(numeric_array(series), EPS)


def sanitize_vector(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    fill = float(np.median(finite)) if finite.size else 0.0
    return np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill).astype(np.float64)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0.0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return float(np.log(clipped / (1.0 - clipped)))


def roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.float64)
    positives = labels == 1.0
    negatives = labels == 0.0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(probabilities).rank(method="average").to_numpy(dtype=np.float64)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.float64)
    positives = float(labels.sum())
    if positives <= 0.0:
        return None
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, sorted_labels.shape[0] + 1, dtype=np.float64)
    return float(np.sum(precision * sorted_labels) / positives)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
