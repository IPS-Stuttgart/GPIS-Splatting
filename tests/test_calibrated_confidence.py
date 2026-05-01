from __future__ import annotations

import pandas as pd

from gpis_splatting.real_calibrated_confidence import (
    format_calibrated_confidence_report,
    select_best_calibrator,
    select_best_filtering_variant,
    select_calibrated_gate_path,
    validate_calibrated_confidence_config,
)


def test_select_calibrated_gate_path_uses_threshold_label() -> None:
    status = {"calibrated_gate_paths": {"0p05": "evaluations/gate_0p05.npz", "0p1": "evaluations/gate_0p1.npz"}}

    assert str(select_calibrated_gate_path(status, calibration_threshold=0.05)).endswith("gate_0p05.npz")


def test_select_best_calibrator_for_threshold() -> None:
    status = {
        "best_calibrators": [
            {"geometry_threshold": 0.02, "method_name": "weak", "auc": 0.6},
            {"geometry_threshold": 0.05, "method_name": "logistic_gpis_field", "auc": 0.9},
        ]
    }

    selected = select_best_calibrator(status, calibration_threshold=0.05)

    assert selected is not None
    assert selected["method_name"] == "logistic_gpis_field"
    assert selected["auc"] == 0.9


def test_select_best_filtering_variant_prefers_f_score_then_chamfer_then_retention() -> None:
    comparison = pd.DataFrame(
        [
            {"geometry_threshold": 0.05, "variant": "baseline", "variant_kind": "baseline", "f_score": 0.7, "chamfer_l1": 0.05, "retention_fraction": 1.0},
            {"geometry_threshold": 0.05, "variant": "gate_scaled", "variant_kind": "gate_scaled", "f_score": 0.8, "chamfer_l1": 0.08, "retention_fraction": 1.0},
            {"geometry_threshold": 0.05, "variant": "gate_ge_0p5", "variant_kind": "gate_threshold", "f_score": 0.8, "chamfer_l1": 0.04, "retention_fraction": 0.5},
            {"geometry_threshold": 0.1, "variant": "wrong_threshold", "variant_kind": "baseline", "f_score": 1.0, "chamfer_l1": 0.01, "retention_fraction": 1.0},
        ]
    )

    selected = select_best_filtering_variant(comparison, calibration_threshold=0.05)

    assert selected is not None
    assert selected["variant"] == "gate_ge_0p5"


def test_validate_requires_calibration_threshold_in_thresholds() -> None:
    validate_calibrated_confidence_config(calibration_threshold=0.05, thresholds=(0.02, 0.05, 0.1), render_max_frames=0)

    try:
        validate_calibrated_confidence_config(calibration_threshold=0.05, thresholds=(0.02, 0.1), render_max_frames=0)
    except ValueError as exc:
        assert "calibration_threshold" in str(exc)
    else:
        raise AssertionError("Expected missing calibration threshold to fail.")


def test_format_report_states_calibrated_confidence_interface() -> None:
    report = format_calibrated_confidence_report(
        {
            "method": "calibrated_confidence",
            "scene_dir": "real_scenes/ignatius_tnt64",
            "model_path": "preliminary_gpis_model.npz",
            "calibration_threshold": 0.05,
            "calibrated_gate_path": "evaluations/gate_0p05.npz",
            "generated_splats_path": "evaluations/hard_negative_splats.npz",
            "hard_negative_report_path": "evaluations/hard_negative_report.md",
            "filtering_report_path": "evaluations/filtering_report.md",
            "status_path": "evaluations/status.json",
            "best_calibrator": {
                "method_name": "logistic_gpis_field",
                "method_family": "logistic",
                "feature_set": "gpis_field",
                "brier": 0.12,
                "nll": 0.4,
                "ece": 0.03,
                "auc": 0.9,
                "average_precision": 0.88,
            },
            "best_filtering_variant": {
                "variant": "gate_scaled",
                "variant_kind": "gate_scaled",
                "retention_fraction": 1.0,
                "precision": 0.8,
                "recall": 0.7,
                "f_score": 0.75,
                "chamfer_l1": 0.04,
                "mean_psnr": None,
                "mean_ssim": None,
            },
        }
    )

    assert "calibrated GPIS posterior-field features as the primary splat-confidence interface" in report
    assert "logistic_gpis_field" in report
    assert "gate_scaled" in report
