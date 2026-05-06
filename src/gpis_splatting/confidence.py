from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StoredConfidenceMethod:
    """Serializable calibrated-confidence method for one geometry threshold."""

    geometry_threshold: float
    method_name: str
    method_family: str
    feature_set: str | None
    model: dict[str, Any]

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        kind = str(self.model.get("kind", ""))
        if kind == "score_minmax":
            return _predict_score_minmax(table, self.model)
        if kind == "isotonic":
            return _predict_isotonic(table, self.model)
        if kind == "logistic":
            return _predict_logistic(table, self.model)
        raise ValueError(f"Unsupported stored confidence model kind: {kind!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry_threshold": float(self.geometry_threshold),
            "method_name": self.method_name,
            "method_family": self.method_family,
            "feature_set": self.feature_set,
            "model": _json_ready(self.model),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoredConfidenceMethod":
        return cls(
            geometry_threshold=float(data["geometry_threshold"]),
            method_name=str(data["method_name"]),
            method_family=str(data["method_family"]),
            feature_set=None if data.get("feature_set") is None else str(data["feature_set"]),
            model=dict(data["model"]),
        )


@dataclass(frozen=True)
class CalibratedConfidenceBundle:
    """Reusable GPIS splat-confidence calibrator bundle."""

    methods: tuple[StoredConfidenceMethod, ...]
    metadata: dict[str, Any]
    schema_version: int = SCHEMA_VERSION

    @property
    def thresholds(self) -> tuple[float, ...]:
        return tuple(method.geometry_threshold for method in self.methods)

    def method_for_threshold(self, threshold: float) -> StoredConfidenceMethod:
        for method in self.methods:
            if abs(method.geometry_threshold - float(threshold)) <= 1e-12:
                return method
        available = ", ".join(format_threshold_label(value) for value in self.thresholds)
        raise ValueError(f"No calibrated confidence model is available for threshold {threshold:g}. Available: {available}.")

    def predict(self, table: pd.DataFrame, *, threshold: float | None = None) -> pd.DataFrame:
        methods = (self.method_for_threshold(threshold),) if threshold is not None else self.methods
        predictions: dict[str, np.ndarray] = {
            "splat_index": table["splat_index"].to_numpy(dtype=np.int64)
            if "splat_index" in table.columns
            else np.arange(len(table), dtype=np.int64)
        }
        for method in methods:
            label = format_threshold_label(method.geometry_threshold)
            predictions[f"confidence_{label}"] = np.clip(method.predict(table), 1e-6, 1.0 - 1e-6)
            predictions[f"selected_method_{label}"] = np.full((len(table),), method.method_name, dtype=object)
        return pd.DataFrame(predictions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "metadata": _json_ready(self.metadata),
            "methods": [method.to_dict() for method in self.methods],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibratedConfidenceBundle":
        version = int(data.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported calibrated confidence bundle schema_version {version}; expected {SCHEMA_VERSION}.")
        methods = tuple(StoredConfidenceMethod.from_dict(item) for item in data.get("methods", []))
        if not methods:
            raise ValueError("Calibrated confidence bundle does not contain any methods.")
        return cls(methods=methods, metadata=dict(data.get("metadata", {})), schema_version=version)


@dataclass(frozen=True)
class ReliabilityBin:
    bin_lower: float
    bin_upper: float
    count: int
    mean_probability: float | None
    empirical_accuracy: float | None
    absolute_error: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "bin_lower": self.bin_lower,
            "bin_upper": self.bin_upper,
            "count": self.count,
            "mean_probability": self.mean_probability,
            "empirical_accuracy": self.empirical_accuracy,
            "absolute_error": self.absolute_error,
        }


def write_calibrated_confidence_bundle(
    path: str | Path,
    selected_methods: Iterable[tuple[float, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> CalibratedConfidenceBundle:
    bundle = CalibratedConfidenceBundle(
        methods=tuple(method_to_stored_confidence(threshold, method) for threshold, method in selected_methods),
        metadata={} if metadata is None else dict(metadata),
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return bundle


def read_calibrated_confidence_bundle(path: str | Path) -> CalibratedConfidenceBundle:
    return CalibratedConfidenceBundle.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def apply_calibrated_confidence_bundle(
    *,
    model_bundle_path: str | Path,
    field_scores_path: str | Path,
    output_path: str | Path | None = None,
    threshold: float | None = None,
    gate_output_dir: str | Path | None = None,
    gate_count: int | None = None,
    missing_gate_value: float = 0.0,
) -> dict[str, Any]:
    bundle = read_calibrated_confidence_bundle(model_bundle_path)
    field_path = Path(field_scores_path)
    predictions = bundle.predict(pd.read_csv(field_path), threshold=threshold)
    out_path = Path(output_path) if output_path is not None else field_path.with_name(f"{field_path.stem}_applied_calibrated_confidence.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_path, index=False)
    thresholds = (float(threshold),) if threshold is not None else bundle.thresholds
    gate_paths: dict[str, Path] = {}
    if gate_output_dir is not None or gate_count is not None:
        gate_paths = write_gate_compatible_confidences(
            predictions=predictions,
            out_dir=Path(gate_output_dir) if gate_output_dir is not None else out_path.parent,
            prefix=out_path.stem,
            thresholds=thresholds,
            predictions_path=out_path,
            gate_count=gate_count,
            missing_gate_value=missing_gate_value,
        )
    return {"predictions_path": out_path, "predictions": predictions, "gate_paths": gate_paths, "bundle": bundle}


def method_to_stored_confidence(threshold: float, method: Any) -> StoredConfidenceMethod:
    return StoredConfidenceMethod(
        geometry_threshold=float(threshold),
        method_name=str(method.name),
        method_family=str(method.family),
        feature_set=None if getattr(method, "feature_set", None) is None else str(method.feature_set),
        model=model_to_config(method.model),
    )


def model_to_config(model: Any) -> dict[str, Any]:
    if _has_attrs(model, "column", "train_min", "train_max"):
        return {"kind": "score_minmax", "column": str(model.column), "train_min": float(model.train_min), "train_max": float(model.train_max)}
    if _has_attrs(model, "score_column", "thresholds", "values", "fallback_probability"):
        return {
            "kind": "isotonic",
            "score_column": str(model.score_column),
            "thresholds": _array_to_float_list(model.thresholds),
            "values": _array_to_float_list(model.values),
            "fallback_probability": float(model.fallback_probability),
        }
    if _has_attrs(model, "feature_names", "fill_values", "mean", "scale", "weights", "bias", "constant_probability"):
        return {
            "kind": "logistic",
            "feature_names": [str(name) for name in model.feature_names],
            "fill_values": _array_to_float_list(model.fill_values),
            "mean": _array_to_float_list(model.mean),
            "scale": _array_to_float_list(model.scale),
            "weights": _array_to_float_list(model.weights),
            "bias": float(model.bias),
            "constant_probability": None if model.constant_probability is None else float(model.constant_probability),
        }
    raise TypeError(f"Unsupported calibrated confidence model type: {type(model).__name__}.")


def reliability_bins(labels: Sequence[float], probabilities: Sequence[float], *, num_bins: int = 10) -> list[ReliabilityBin]:
    if num_bins < 1:
        raise ValueError("num_bins must be positive.")
    label_array = np.asarray(labels, dtype=np.float64)
    probability_array = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    if label_array.shape != probability_array.shape:
        raise ValueError("labels and probabilities must have the same shape.")
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    rows: list[ReliabilityBin] = []
    for index in range(num_bins):
        lo = float(edges[index])
        hi = float(edges[index + 1])
        mask = (probability_array >= lo) & (probability_array <= hi if index == num_bins - 1 else probability_array < hi)
        if not np.any(mask):
            rows.append(ReliabilityBin(lo, hi, 0, None, None, None))
            continue
        mean_probability = float(probability_array[mask].mean())
        empirical_accuracy = float(label_array[mask].mean())
        rows.append(ReliabilityBin(lo, hi, int(mask.sum()), mean_probability, empirical_accuracy, abs(mean_probability - empirical_accuracy)))
    return rows


def reliability_table(labels: Sequence[float], probabilities: Sequence[float], *, num_bins: int = 10) -> pd.DataFrame:
    return pd.DataFrame([row.to_dict() for row in reliability_bins(labels, probabilities, num_bins=num_bins)])


def expected_calibration_error(labels: Sequence[float], probabilities: Sequence[float], *, num_bins: int = 10) -> float:
    total = len(np.asarray(labels))
    if total == 0:
        raise ValueError("At least one label is required to compute ECE.")
    return float(sum((row.count / total) * row.absolute_error for row in reliability_bins(labels, probabilities, num_bins=num_bins) if row.count and row.absolute_error is not None))


def brier_score(labels: Sequence[float], probabilities: Sequence[float]) -> float:
    label_array = np.asarray(labels, dtype=np.float64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if label_array.shape != probability_array.shape:
        raise ValueError("labels and probabilities must have the same shape.")
    return float(np.mean((probability_array - label_array) ** 2))


def write_gate_compatible_confidences(
    *,
    predictions: pd.DataFrame,
    out_dir: str | Path,
    prefix: str,
    thresholds: Sequence[float],
    predictions_path: str | Path,
    gate_count: int | None = None,
    missing_gate_value: float = 0.0,
) -> dict[str, Path]:
    if gate_count is not None and gate_count < 1:
        raise ValueError("gate_count must be positive when provided.")
    if not 0.0 <= missing_gate_value <= 1.0:
        raise ValueError("missing_gate_value must be in [0, 1].")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    scored_splat_index = predictions["splat_index"].to_numpy(dtype=np.int64) if "splat_index" in predictions.columns else np.arange(len(predictions), dtype=np.int64)
    output_splat_index, scored_mask = build_gate_index(scored_splat_index=scored_splat_index, gate_count=gate_count)
    gate_paths: dict[str, Path] = {}
    for threshold in thresholds:
        label = format_threshold_label(float(threshold))
        column = f"confidence_{label}"
        if column not in predictions.columns:
            continue
        scored_gate = np.clip(predictions[column].to_numpy(dtype=np.float64), 0.0, 1.0)
        gate = scored_gate if gate_count is None else np.full((gate_count,), missing_gate_value, dtype=np.float64)
        if gate_count is not None:
            gate[scored_splat_index] = scored_gate
        method_column = f"selected_method_{label}"
        selected_method = str(predictions[method_column].iloc[0]) if method_column in predictions.columns and len(predictions) else ""
        path = out_path / f"{prefix}_gate_{label}.npz"
        np.savez_compressed(
            path,
            gate=gate,
            raw_gate=gate,
            splat_index=output_splat_index,
            scored_splat_index=scored_splat_index,
            scored_mask=scored_mask,
            geometry_threshold=np.asarray(float(threshold), dtype=np.float64),
            selected_method=np.asarray(selected_method),
            predictions_path=np.asarray(str(predictions_path)),
            missing_gate_value=np.asarray(float(missing_gate_value), dtype=np.float64),
            scored_count=np.asarray(int(scored_splat_index.shape[0]), dtype=np.int64),
            missing_count=np.asarray(int(scored_mask.shape[0] - scored_mask.sum()), dtype=np.int64),
            scored_fraction=np.asarray(float(scored_mask.mean()), dtype=np.float64),
            missing_fraction=np.asarray(float(1.0 - scored_mask.mean()), dtype=np.float64),
        )
        gate_paths[label] = path
    return gate_paths


def build_gate_index(*, scored_splat_index: np.ndarray, gate_count: int | None) -> tuple[np.ndarray, np.ndarray]:
    if scored_splat_index.size and np.any(scored_splat_index < 0):
        raise ValueError("splat_index values must be non-negative to export gate-compatible confidence files.")
    if int(np.unique(scored_splat_index).shape[0]) != int(scored_splat_index.shape[0]):
        raise ValueError("splat_index values must be unique to export gate-compatible confidence files.")
    if gate_count is None:
        return scored_splat_index, np.ones(scored_splat_index.shape, dtype=bool)
    if scored_splat_index.size and int(scored_splat_index.max()) >= gate_count:
        raise ValueError("gate_count must be greater than every scored splat_index.")
    scored_mask = np.zeros((gate_count,), dtype=bool)
    scored_mask[scored_splat_index] = True
    return np.arange(gate_count, dtype=np.int64), scored_mask


def format_threshold_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _predict_score_minmax(table: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    column = str(config["column"])
    _require_columns(table, [column])
    values = sanitize_vector(table[column].to_numpy(dtype=np.float64))
    train_min = float(config["train_min"])
    train_max = float(config["train_max"])
    return np.clip((values - train_min) / max(train_max - train_min, 1e-12), 0.0, 1.0)


def _predict_isotonic(table: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    column = str(config["score_column"])
    _require_columns(table, [column])
    scores = sanitize_vector(table[column].to_numpy(dtype=np.float64))
    thresholds = np.asarray(config.get("thresholds", []), dtype=np.float64)
    values = np.asarray(config.get("values", []), dtype=np.float64)
    if thresholds.size == 0 or values.size == 0:
        return np.full(scores.shape, float(config.get("fallback_probability", 0.5)), dtype=np.float64)
    indices = np.clip(np.searchsorted(thresholds, scores, side="left"), 0, values.shape[0] - 1)
    return np.clip(values[indices], 1e-6, 1.0 - 1e-6)


def _predict_logistic(table: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    constant_probability = config.get("constant_probability")
    if constant_probability is not None:
        return np.full((len(table),), float(constant_probability), dtype=np.float64)
    feature_names = tuple(str(name) for name in config["feature_names"])
    _require_columns(table, feature_names)
    fill_values = np.asarray(config["fill_values"], dtype=np.float64)
    features = build_feature_matrix(table, feature_names, fill_values=fill_values)
    mean = np.asarray(config["mean"], dtype=np.float64)
    scale = np.where(np.abs(np.asarray(config["scale"], dtype=np.float64)) < 1e-12, 1.0, np.asarray(config["scale"], dtype=np.float64))
    weights = np.asarray(config["weights"], dtype=np.float64)
    standardized = (features - mean[None, :]) / scale[None, :]
    return sigmoid(standardized @ weights + float(config["bias"]))


def build_feature_matrix(table: pd.DataFrame, feature_names: Sequence[str], *, fill_values: np.ndarray) -> np.ndarray:
    if len(feature_names) != len(fill_values):
        raise ValueError("feature_names and fill_values must have the same length.")
    columns = []
    for index, name in enumerate(feature_names):
        values = table[name].to_numpy(dtype=np.float64)
        columns.append(np.where(np.isfinite(values), values, fill_values[index]))
    return np.column_stack(columns).astype(np.float64)


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


def _require_columns(table: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"Field-score table is missing required calibrated-confidence column(s): {', '.join(missing)}.")


def _has_attrs(value: Any, *names: str) -> bool:
    return all(hasattr(value, name) for name in names)


def _array_to_float_list(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=np.float64).ravel().tolist()]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
