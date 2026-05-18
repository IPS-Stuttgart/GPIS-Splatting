from __future__ import annotations

import numpy as np
import pytest

from gpis_splatting.trained_3dgs_evaluation import validate_trained_3dgs_gate_coverage, validate_trained_3dgs_scoring_request


def test_trained_3dgs_scoring_rejects_partial_cap_by_default() -> None:
    with pytest.raises(ValueError, match="Partial GPIS scoring"):
        validate_trained_3dgs_scoring_request(gaussian_count=10, max_pred_points=5, allow_partial_gpis_scores=False)


def test_trained_3dgs_scoring_accepts_full_cap() -> None:
    assert validate_trained_3dgs_scoring_request(gaussian_count=10, max_pred_points=10, allow_partial_gpis_scores=False) == 10
    assert validate_trained_3dgs_scoring_request(gaussian_count=10, max_pred_points=0, allow_partial_gpis_scores=False) is None


def test_trained_3dgs_gate_coverage_accepts_full_large_scale_gate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    gate_path = tmp_path / "full_gate.npz"
    np.savez_compressed(gate_path, gate=np.linspace(0.0, 1.0, 4))

    coverage = validate_trained_3dgs_gate_coverage(gate_path, expected_count=4)

    assert coverage["gate_count"] == 4
    assert coverage["scored_count"] == 4
    assert coverage["missing_count"] == 0
    assert coverage["scored_fraction"] == 1.0


def test_trained_3dgs_gate_coverage_rejects_fallback_filled_calibration_gate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    gate_path = tmp_path / "partial_gate.npz"
    np.savez_compressed(
        gate_path,
        gate=np.ones(5),
        scored_count=np.asarray(3, dtype=np.int64),
        missing_count=np.asarray(2, dtype=np.int64),
        scored_mask=np.asarray([True, False, True, False, True]),
    )

    with pytest.raises(ValueError, match="fallback gates"):
        validate_trained_3dgs_gate_coverage(gate_path, expected_count=5)


def test_trained_3dgs_gate_coverage_can_allow_partial_for_diagnostics(tmp_path) -> None:  # type: ignore[no-untyped-def]
    gate_path = tmp_path / "partial_gate.npz"
    np.savez_compressed(
        gate_path,
        gate=np.ones(5),
        scored_count=np.asarray(3, dtype=np.int64),
        missing_count=np.asarray(2, dtype=np.int64),
    )

    coverage = validate_trained_3dgs_gate_coverage(gate_path, expected_count=5, allow_partial_gpis_scores=True)

    assert coverage["scored_count"] == 3
    assert coverage["missing_count"] == 2
    assert coverage["missing_fraction"] == pytest.approx(0.4)
