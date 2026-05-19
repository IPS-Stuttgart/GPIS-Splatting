from __future__ import annotations

import numpy as np
import pytest

from gpis_splatting.metrics import equal_count_calibration_error, expected_calibration_error


def test_expected_calibration_error_uses_fixed_probability_bins() -> None:
    prob = np.array([0.01, 0.02, 0.03, 0.91])
    labels = np.array([0.0, 0.0, 1.0, 0.0])

    fixed_bin_expected = (3 / 4) * abs(prob[:3].mean() - labels[:3].mean()) + (1 / 4) * abs(0.91 - 0.0)

    assert expected_calibration_error(prob, labels, bins=2) == pytest.approx(fixed_bin_expected)
    assert equal_count_calibration_error(prob, labels, bins=2) != pytest.approx(fixed_bin_expected)


def test_expected_calibration_error_includes_probability_one_in_last_bin() -> None:
    prob = np.array([0.0, 1.0])
    labels = np.array([0.0, 1.0])

    assert expected_calibration_error(prob, labels, bins=10) == pytest.approx(0.0)


def test_expected_calibration_error_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        expected_calibration_error(np.array([0.1, 0.2]), np.array([1.0]))
    with pytest.raises(ValueError, match="positive"):
        expected_calibration_error(np.array([0.1]), np.array([1.0]), bins=0)
    with pytest.raises(ValueError, match="finite"):
        expected_calibration_error(np.array([np.nan]), np.array([1.0]))
