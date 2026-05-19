from __future__ import annotations

import numpy as np
import pytest

from gpis_splatting.metrics import expected_calibration_error


def test_expected_calibration_error_uses_fixed_probability_bins_by_default() -> None:
    prob = np.array([0.05, 0.10, 0.20])
    labels = np.array([0.0, 0.0, 1.0])

    # All samples fall into the first fixed bin [0, 0.5), so the result is the
    # absolute gap between mean confidence 7/60 and mean accuracy 1/3.
    assert np.isclose(expected_calibration_error(prob, labels, bins=2), 13.0 / 60.0)


def test_expected_calibration_error_keeps_adaptive_binning_available() -> None:
    prob = np.array([0.05, 0.10, 0.20])
    labels = np.array([0.0, 0.0, 1.0])

    # Previous behavior: sort by confidence and split into equal-count chunks.
    assert np.isclose(expected_calibration_error(prob, labels, bins=2, binning="adaptive"), 19.0 / 60.0)


def test_expected_calibration_error_includes_right_edge_in_last_fixed_bin() -> None:
    prob = np.array([0.0, 0.5, 1.0])
    labels = np.array([0.0, 1.0, 1.0])

    assert np.isclose(expected_calibration_error(prob, labels, bins=2), 1.0 / 6.0)


def test_expected_calibration_error_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="bins"):
        expected_calibration_error(np.array([0.5]), np.array([1.0]), bins=0)
    with pytest.raises(ValueError, match="same number"):
        expected_calibration_error(np.array([0.5, 0.6]), np.array([1.0]))
    with pytest.raises(ValueError, match="binning"):
        expected_calibration_error(np.array([0.5]), np.array([1.0]), binning="quantile")
