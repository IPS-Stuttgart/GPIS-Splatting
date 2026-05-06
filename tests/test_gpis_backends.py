from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from gpis_splatting.gpis import fit_dense_gpis, predict_gpis, save_model, surface_band_probability
from gpis_splatting.gpis_backends import (
    DenseExactGPISBackend,
    GPISBackendName,
    LocalExactGPISBackend,
    fit_gpis_backend,
    load_gpis_backend,
    nearest_training_indices,
)


def make_training_data(n: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    points = torch.rand((n, 3), generator=generator, dtype=torch.float64) * 2.0 - 1.0
    sdf = torch.linalg.norm(points, dim=-1) - 0.65
    return points, sdf


def test_dense_backend_matches_legacy_dense_prediction() -> None:
    x_train, y_train = make_training_data()
    query = x_train[:8] * 0.9
    legacy_model = fit_dense_gpis(x_train, y_train, lengthscale=0.45, noise_std=0.03)
    backend = DenseExactGPISBackend.fit(x_train, y_train, lengthscale=0.45, noise_std=0.03)

    legacy_prediction = predict_gpis(legacy_model, query, batch_size=4)
    backend_prediction = backend.predict(query, batch_size=4)

    torch.testing.assert_close(backend_prediction.mean, legacy_prediction.mean)
    torch.testing.assert_close(backend_prediction.variance, legacy_prediction.variance)
    torch.testing.assert_close(backend_prediction.gradient, legacy_prediction.gradient)


def test_load_backend_supports_legacy_dense_model_file(tmp_path: Path) -> None:
    x_train, y_train = make_training_data(20)
    model = fit_dense_gpis(x_train, y_train, lengthscale=0.4, noise_std=0.04)
    path = tmp_path / "legacy_model.npz"
    save_model(str(path), model, metadata={"scene": "unit"})

    backend, metadata = load_gpis_backend(path)
    prediction = backend.predict(x_train[:5])

    assert isinstance(backend, DenseExactGPISBackend)
    assert metadata["scene"] == "unit"
    assert metadata["backend"] == "dense_exact"
    assert prediction.mean.shape == (5,)


def test_local_exact_backend_predicts_finite_quantities_and_saves(tmp_path: Path) -> None:
    x_train, y_train = make_training_data(45)
    backend = LocalExactGPISBackend.fit(x_train, y_train, lengthscale=0.55, noise_std=0.04, num_neighbors=12)
    query = x_train[:7] + 0.01

    prediction = backend.predict(query, batch_size=3)
    gate = surface_band_probability(prediction, epsilon=0.08)

    assert prediction.mean.shape == (7,)
    assert prediction.variance.shape == (7,)
    assert prediction.gradient.shape == (7, 3)
    assert torch.all(torch.isfinite(prediction.mean))
    assert torch.all(prediction.variance > 0.0)
    assert torch.all((gate >= 0.0) & (gate <= 1.0))

    path = tmp_path / "local_backend.npz"
    backend.save(path, metadata={"scene": "unit"})
    loaded, metadata = load_gpis_backend(path)
    loaded_prediction = loaded.predict(query, batch_size=2)

    assert isinstance(loaded, LocalExactGPISBackend)
    assert metadata["scene"] == "unit"
    assert metadata["backend"] == "local_exact"
    torch.testing.assert_close(loaded_prediction.mean, prediction.mean)
    torch.testing.assert_close(loaded_prediction.variance, prediction.variance)
    torch.testing.assert_close(loaded_prediction.gradient, prediction.gradient)


def test_fit_gpis_backend_dispatches_and_validates_unknown_backend() -> None:
    x_train, y_train = make_training_data(16)

    dense = fit_gpis_backend("dense_exact", x_train, y_train)
    local = fit_gpis_backend("local_exact", x_train, y_train, num_neighbors=5)

    assert isinstance(dense, DenseExactGPISBackend)
    assert isinstance(local, LocalExactGPISBackend)
    with pytest.raises(ValueError, match="Unknown GPIS backend"):
        fit_gpis_backend(cast(GPISBackendName, "not_a_backend"), x_train, y_train)


def test_nearest_training_indices_caps_requested_neighbor_count() -> None:
    x_train, _ = make_training_data(4)
    indices = nearest_training_indices(x_train[:2], x_train, num_neighbors=100)

    assert indices.shape == (2, 4)
    assert np.all(np.isin(indices.numpy(), np.arange(4)))
