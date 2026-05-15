from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gpis_splatting.gpis_backends import InducingPointGPISBackend
from gpis_splatting.large_scale_gpis import (
    LargeScaleGPISScoreConfig,
    estimate_inducing_query_batch_size,
    score_large_scale_gpis,
    write_large_scale_scores_npz,
)


def make_training_data(n: int = 80) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)
    points = torch.rand((n, 3), generator=generator, dtype=torch.float64) * 2.0 - 1.0
    sdf = torch.linalg.norm(points, dim=-1) - 0.65
    return points, sdf


def test_large_scale_inducing_matches_standard_prediction() -> None:
    x_train, y_train = make_training_data()
    backend = InducingPointGPISBackend.fit(x_train, y_train, num_inducing=20, fit_batch_size=17)
    query = x_train[:11] * 0.9

    reference = backend.predict(query, batch_size=4)
    result = score_large_scale_gpis(
        backend,
        query,
        config=LargeScaleGPISScoreConfig(batch_size=4, prediction_device="cpu", prediction_dtype="float64", output_device="cpu"),
    )

    assert result.scores.mean is not None
    assert result.scores.variance is not None
    assert result.scores.gradient is not None
    torch.testing.assert_close(result.scores.mean, reference.mean)
    torch.testing.assert_close(result.scores.variance, reference.variance)
    torch.testing.assert_close(result.scores.gradient, reference.gradient)
    assert result.scores.gate.shape == (11,)
    assert result.stats["batch_size"] == 4


def test_large_scale_gate_only_writes_npz(tmp_path: Path) -> None:
    x_train, y_train = make_training_data(48)
    backend = InducingPointGPISBackend.fit(x_train, y_train, num_inducing=12, fit_batch_size=16)
    result = score_large_scale_gpis(
        backend,
        x_train[:9],
        config=LargeScaleGPISScoreConfig(batch_size=3, prediction_device="cpu", prediction_dtype="float32", output_device="cpu", include_prediction=False),
    )

    assert result.scores.mean is None
    assert result.scores.gradient is None
    out_path = tmp_path / "scores.npz"
    write_large_scale_scores_npz(out_path, result)
    saved = np.load(out_path)
    assert saved.files == ["gate"]
    assert saved["gate"].shape == (9,)


def test_estimate_inducing_query_batch_size_validates_inputs() -> None:
    assert estimate_inducing_query_batch_size(num_inducing=128, dtype="float32", memory_budget_mib=1) > 0
    with pytest.raises(ValueError, match="num_inducing"):
        estimate_inducing_query_batch_size(num_inducing=0)
    with pytest.raises(ValueError, match="prediction dtype"):
        estimate_inducing_query_batch_size(num_inducing=8, dtype="float16")
