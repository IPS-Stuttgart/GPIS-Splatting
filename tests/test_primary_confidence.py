from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gpis_splatting.confidence import read_calibrated_confidence_bundle
from gpis_splatting.primary_confidence import resolve_primary_threshold, run_primary_calibrated_confidence


def make_field_scores() -> pd.DataFrame:
    distance = np.asarray([0.01, 0.02, 0.03, 0.04, 0.08, 0.12, 0.2, 0.35], dtype=np.float64)
    confidence_like = np.asarray([0.95, 0.9, 0.82, 0.75, 0.5, 0.35, 0.2, 0.05], dtype=np.float64)
    return pd.DataFrame(
        {
            "splat_index": np.arange(distance.shape[0], dtype=np.int64),
            "nearest_gt_distance": distance,
            "mu": distance,
            "sigma": np.full_like(distance, 0.05),
            "grad_norm": np.ones_like(distance),
            "signed_distance": distance,
            "distance_std": np.full_like(distance, 0.05),
            "score_current_gate": confidence_like,
            "score_raw_surface_band": confidence_like,
            "score_variance_penalized_band": confidence_like * 0.9,
            "score_variance_penalized_exp": confidence_like * 0.85,
            "score_negative_abs_distance": -distance,
            "score_negative_distance_std": -np.full_like(distance, 0.05),
            "score_exp_neg_abs_distance": np.exp(-distance),
            "score_negative_abs_mu": -distance,
        }
    )


def test_run_primary_calibrated_confidence_writes_primary_artifacts(tmp_path: Path) -> None:
    field_scores_path = tmp_path / "field_scores.csv"
    make_field_scores().to_csv(field_scores_path, index=False)

    result = run_primary_calibrated_confidence(
        field_scores_path=field_scores_path,
        output_dir=tmp_path,
        method_name="unit",
        thresholds=(0.05,),
        topk_fractions=(0.5, 1.0),
        feature_sets=("gpis_field",),
        baseline_scores=("score_current_gate",),
        isotonic_scores=("score_current_gate",),
        validation_fraction=0.25,
        seed=3,
        logistic_iterations=20,
        learning_rate=0.05,
        regularization=1e-3,
        gate_count=10,
        missing_gate_value=1.0,
    )

    assert result["model_bundle_path"].exists()
    assert result["primary_confidence_path"].exists()
    assert result["primary_gate_path"].exists()
    predictions = pd.read_csv(result["primary_confidence_path"])
    assert "primary_confidence" in predictions.columns
    assert "confidence_0p05" in predictions.columns
    assert predictions["primary_confidence"].between(0.0, 1.0).all()

    bundle = read_calibrated_confidence_bundle(result["model_bundle_path"])
    assert bundle.thresholds == (0.05,)
    assert bundle.metadata["primary_threshold"] == 0.05

    status = json.loads(Path(result["status_path"]).read_text(encoding="utf-8"))
    assert status["calibrated_confidence_is_primary"] is True
    assert status["model_bundle_path"].endswith("unit_calibrated_confidence_model.json")
    assert status["primary_confidence_path"].endswith("unit_primary_calibrated_confidence.csv")
    assert status["primary_gate_path"].endswith("unit_primary_calibrated_confidence_gate_0p05.npz")


def test_resolve_primary_threshold_prefers_0p05_when_available() -> None:
    assert resolve_primary_threshold(None, (0.02, 0.05, 0.1)) == 0.05
    assert resolve_primary_threshold(None, (0.02, 0.1)) == 0.02
    assert resolve_primary_threshold(0.1, (0.02, 0.1)) == 0.1
