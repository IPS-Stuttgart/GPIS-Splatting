from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gpis_splatting.confidence import (
    StoredConfidenceMethod,
    apply_calibrated_confidence_bundle,
    brier_score,
    expected_calibration_error,
    method_to_stored_confidence,
    read_calibrated_confidence_bundle,
    reliability_table,
    write_calibrated_confidence_bundle,
)


@dataclass(frozen=True)
class DummyMethod:
    name: str
    family: str
    feature_set: str | None
    model: object


@dataclass(frozen=True)
class DummyLogisticModel:
    feature_names: tuple[str, ...]
    fill_values: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    constant_probability: float | None = None


@dataclass(frozen=True)
class DummyScoreTransform:
    column: str
    train_min: float
    train_max: float


def test_bundle_round_trips_and_predicts_logistic_confidence(tmp_path: Path) -> None:
    method = DummyMethod(
        name="logistic_gpis_field",
        family="logistic",
        feature_set="gpis_field",
        model=DummyLogisticModel(
            feature_names=("abs_signed_distance", "distance_std"),
            fill_values=np.asarray([0.0, 0.0]),
            mean=np.asarray([0.0, 0.0]),
            scale=np.asarray([1.0, 1.0]),
            weights=np.asarray([-4.0, -2.0]),
            bias=2.0,
        ),
    )
    path = tmp_path / "confidence_model.json"
    write_calibrated_confidence_bundle(path, [(0.05, method)], metadata={"source": "unit-test"})

    bundle = read_calibrated_confidence_bundle(path)
    table = pd.DataFrame(
        {
            "splat_index": [3, 7],
            "abs_signed_distance": [0.01, 1.0],
            "distance_std": [0.01, 0.5],
        }
    )

    predictions = bundle.predict(table)

    assert predictions["splat_index"].tolist() == [3, 7]
    assert predictions["selected_method_0p05"].tolist() == ["logistic_gpis_field", "logistic_gpis_field"]
    assert predictions.loc[0, "confidence_0p05"] > predictions.loc[1, "confidence_0p05"]
    assert bundle.metadata["source"] == "unit-test"


def test_method_serializer_supports_score_transform_duck_type() -> None:
    method = DummyMethod(
        name="minmax_score_current_gate",
        family="score_minmax",
        feature_set="score_current_gate",
        model=DummyScoreTransform(column="score_current_gate", train_min=0.25, train_max=0.75),
    )

    stored = method_to_stored_confidence(0.1, method)
    predictions = stored.predict(pd.DataFrame({"score_current_gate": [0.25, 0.5, 0.75]}))

    assert isinstance(stored, StoredConfidenceMethod)
    np.testing.assert_allclose(predictions, [0.0, 0.5, 1.0])


def test_apply_bundle_writes_predictions_and_gate(tmp_path: Path) -> None:
    method = DummyMethod(
        name="minmax_score_current_gate",
        family="score_minmax",
        feature_set="score_current_gate",
        model=DummyScoreTransform(column="score_current_gate", train_min=0.0, train_max=1.0),
    )
    bundle_path = tmp_path / "bundle.json"
    field_scores_path = tmp_path / "scores.csv"
    write_calibrated_confidence_bundle(bundle_path, [(0.05, method)])
    pd.DataFrame({"splat_index": [0, 2], "score_current_gate": [0.2, 0.8]}).to_csv(field_scores_path, index=False)

    result = apply_calibrated_confidence_bundle(
        model_bundle_path=bundle_path,
        field_scores_path=field_scores_path,
        output_path=tmp_path / "predictions.csv",
        gate_output_dir=tmp_path,
        gate_count=3,
        missing_gate_value=1.0,
    )

    saved = pd.read_csv(result["predictions_path"])
    assert saved["confidence_0p05"].tolist() == [0.2, 0.8]
    gate = np.load(result["gate_paths"]["0p05"])
    np.testing.assert_allclose(gate["gate"], [0.2, 1.0, 0.8])
    assert int(gate["missing_count"]) == 1


def test_reliability_metrics_are_weighted_by_bin_counts() -> None:
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])

    table = reliability_table(labels, probabilities, num_bins=2)

    assert table["count"].tolist() == [2, 2]
    np.testing.assert_allclose(expected_calibration_error(labels, probabilities, num_bins=2), 0.15)
    np.testing.assert_allclose(brier_score(labels, probabilities), 0.025)


def test_missing_model_feature_fails_with_actionable_message() -> None:
    method = DummyMethod(
        name="logistic_gpis_field",
        family="logistic",
        feature_set="gpis_field",
        model=DummyLogisticModel(
            feature_names=("abs_signed_distance",),
            fill_values=np.asarray([0.0]),
            mean=np.asarray([0.0]),
            scale=np.asarray([1.0]),
            weights=np.asarray([-1.0]),
            bias=0.0,
        ),
    )
    stored = method_to_stored_confidence(0.05, method)

    try:
        stored.predict(pd.DataFrame({"score_current_gate": [1.0]}))
    except ValueError as exc:
        assert "abs_signed_distance" in str(exc)
    else:
        raise AssertionError("Expected missing feature column to fail.")
