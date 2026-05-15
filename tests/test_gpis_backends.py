from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from gpis_splatting.gpis import fit_dense_gpis, predict_gpis, save_model, surface_band_probability
from gpis_splatting.gpis_backends import (
    ARDInducingPointGPISBackend,
    DenseExactGPISBackend,
    GPISBackendName,
    InducingPointGPISBackend,
    LocalExactGPISBackend,
    MultiresInducingGPISBackend,
    SKIGridGPISBackend,
    fit_gpis_backend,
    load_gpis_backend,
    nearest_training_indices,
    select_inducing_indices,
)


def make_training_data(n: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    points = torch.rand((n, 3), generator=generator, dtype=torch.float64) * 2.0 - 1.0
    sdf = torch.linalg.norm(points, dim=-1) - 0.65
    return points, sdf


def assert_valid_prediction(prediction, n: int) -> None:
    assert prediction.mean.shape == (n,)
    assert prediction.variance.shape == (n,)
    assert prediction.gradient.shape == (n, 3)
    assert torch.all(torch.isfinite(prediction.mean))
    assert torch.all(torch.isfinite(prediction.gradient))
    assert torch.all(prediction.variance > 0.0)
    gate = surface_band_probability(prediction, epsilon=0.08)
    assert torch.all((gate >= 0.0) & (gate <= 1.0))


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
    assert_valid_prediction(prediction, 7)

    path = tmp_path / "local_backend.npz"
    backend.save(path, metadata={"scene": "unit"})
    loaded, metadata = load_gpis_backend(path)
    loaded_prediction = loaded.predict(query, batch_size=2)

    assert metadata["scene"] == "unit"
    assert metadata["backend"] == "local_exact"
    torch.testing.assert_close(loaded_prediction.mean, prediction.mean)
    torch.testing.assert_close(loaded_prediction.variance, prediction.variance)
    torch.testing.assert_close(loaded_prediction.gradient, prediction.gradient)


def test_kdtree_local_backend_matches_local_exact() -> None:
    x_train, y_train = make_training_data(50)
    query = x_train[:8] + 0.02
    local = fit_gpis_backend("local_exact", x_train, y_train, lengthscale=0.45, noise_std=0.04, num_neighbors=10)
    kdtree = fit_gpis_backend("local_kdtree", x_train, y_train, lengthscale=0.45, noise_std=0.04, num_neighbors=10, leaf_size=8)

    local_prediction = local.predict(query, batch_size=4)
    kdtree_prediction = kdtree.predict(query, batch_size=4)

    torch.testing.assert_close(kdtree_prediction.mean, local_prediction.mean)
    torch.testing.assert_close(kdtree_prediction.variance, local_prediction.variance)
    torch.testing.assert_close(kdtree_prediction.gradient, local_prediction.gradient)


def test_faiss_local_backend_matches_local_exact_when_available() -> None:
    pytest.importorskip("faiss")
    x_train, y_train = make_training_data(50)
    query = x_train[:8] + 0.02
    local = fit_gpis_backend("local_exact", x_train, y_train, lengthscale=0.45, noise_std=0.04, num_neighbors=10)
    faiss_backend = fit_gpis_backend("local_faiss", x_train, y_train, lengthscale=0.45, noise_std=0.04, num_neighbors=10)

    local_prediction = local.predict(query, batch_size=4)
    faiss_prediction = faiss_backend.predict(query, batch_size=4)

    torch.testing.assert_close(faiss_prediction.mean, local_prediction.mean)
    torch.testing.assert_close(faiss_prediction.variance, local_prediction.variance)
    torch.testing.assert_close(faiss_prediction.gradient, local_prediction.gradient)


def test_inducing_point_backend_predicts_finite_quantities_and_saves(tmp_path: Path) -> None:
    x_train, y_train = make_training_data(80)
    backend = InducingPointGPISBackend.fit(x_train, y_train, lengthscale=0.5, noise_std=0.05, num_inducing=18, inducing_selection="farthest", fit_batch_size=23)
    query = x_train[:9] * 0.95

    prediction = backend.predict(query, batch_size=4)
    assert backend.num_inducing == 18
    assert backend.training_count == 80
    assert_valid_prediction(prediction, 9)

    path = tmp_path / "inducing_backend.npz"
    backend.save(path, metadata={"scene": "unit"})
    loaded, metadata = load_gpis_backend(path)
    loaded_prediction = loaded.predict(query, batch_size=3)

    assert metadata["scene"] == "unit"
    assert metadata["backend"] == "inducing_points"
    torch.testing.assert_close(loaded_prediction.mean, prediction.mean)
    torch.testing.assert_close(loaded_prediction.variance, prediction.variance)
    torch.testing.assert_close(loaded_prediction.gradient, prediction.gradient)


def test_inducing_point_backend_matches_dense_when_all_points_are_inducing() -> None:
    x_train, y_train = make_training_data(24)
    query = x_train[:6] + 0.02
    dense = DenseExactGPISBackend.fit(x_train, y_train, lengthscale=0.7, noise_std=0.08, jitter=1e-8)
    inducing = InducingPointGPISBackend.fit(x_train, y_train, lengthscale=0.7, noise_std=0.08, jitter=1e-8, num_inducing=100, inducing_selection="first", fit_batch_size=7)

    dense_prediction = dense.predict(query, batch_size=3)
    inducing_prediction = inducing.predict(query, batch_size=3)

    torch.testing.assert_close(inducing_prediction.mean, dense_prediction.mean, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(inducing_prediction.variance, dense_prediction.variance, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(inducing_prediction.gradient, dense_prediction.gradient, rtol=5e-4, atol=5e-4)


def test_ard_inducing_backend_predicts_finite_quantities_and_saves(tmp_path: Path) -> None:
    x_train, y_train = make_training_data(70)
    backend = fit_gpis_backend("ard_inducing_points", x_train, y_train, num_inducing=16, fit_batch_size=19, ard_lengthscales=torch.tensor([0.4, 0.6, 0.5], dtype=torch.float64))
    query = x_train[:7] * 0.9

    prediction = backend.predict(query, batch_size=3)
    assert isinstance(backend, ARDInducingPointGPISBackend)
    assert backend.ard_lengthscales is not None
    assert_valid_prediction(prediction, 7)

    path = tmp_path / "ard_backend.npz"
    backend.save(path, metadata={"scene": "unit"})
    loaded, metadata = load_gpis_backend(path)
    loaded_prediction = loaded.predict(query, batch_size=3)

    assert metadata["backend"] == "ard_inducing_points"
    torch.testing.assert_close(loaded_prediction.mean, prediction.mean)
    torch.testing.assert_close(loaded_prediction.variance, prediction.variance)
    torch.testing.assert_close(loaded_prediction.gradient, prediction.gradient)


def test_ski_grid_backend_predicts_finite_quantities_and_saves(tmp_path: Path) -> None:
    x_train, y_train = make_training_data(55)
    backend = fit_gpis_backend("ski_grid", x_train, y_train, lengthscale=0.55, noise_std=0.05, fit_batch_size=17, ski_grid_size=4, ski_padding=0.02)
    query = x_train[:6]

    prediction = backend.predict(query, batch_size=3)
    assert isinstance(backend, SKIGridGPISBackend)
    assert backend.backend_name == "ski_grid"
    assert_valid_prediction(prediction, 6)

    path = tmp_path / "ski_backend.npz"
    backend.save(path, metadata={"scene": "unit"})
    loaded, metadata = load_gpis_backend(path)
    loaded_prediction = loaded.predict(query, batch_size=2)

    assert metadata["backend"] == "ski_grid"
    torch.testing.assert_close(loaded_prediction.mean, prediction.mean)
    torch.testing.assert_close(loaded_prediction.variance, prediction.variance)
    torch.testing.assert_close(loaded_prediction.gradient, prediction.gradient)


def test_multires_inducing_backend_predicts_finite_quantities_and_saves(tmp_path: Path) -> None:
    x_train, y_train = make_training_data(70)
    backend = fit_gpis_backend("multires_inducing", x_train, y_train, lengthscale=0.65, noise_std=0.06, num_inducing=8, fit_batch_size=21, multires_levels=2, multires_inducing_growth=1.5)
    query = x_train[:5]

    prediction = backend.predict(query, batch_size=3)
    assert isinstance(backend, MultiresInducingGPISBackend)
    assert backend.num_inducing >= 16
    assert_valid_prediction(prediction, 5)

    path = tmp_path / "multires_backend.npz"
    backend.save(path, metadata={"scene": "unit"})
    loaded, metadata = load_gpis_backend(path)
    loaded_prediction = loaded.predict(query, batch_size=2)

    assert metadata["backend"] == "multires_inducing"
    torch.testing.assert_close(loaded_prediction.mean, prediction.mean)
    torch.testing.assert_close(loaded_prediction.variance, prediction.variance)
    torch.testing.assert_close(loaded_prediction.gradient, prediction.gradient)


def test_fit_gpis_backend_dispatches_and_validates_unknown_backend() -> None:
    x_train, y_train = make_training_data(16)

    for backend_name in ("dense_exact", "local_exact", "local_kdtree", "inducing_points", "ard_inducing_points", "ski_grid", "multires_inducing"):
        backend = fit_gpis_backend(cast(GPISBackendName, backend_name), x_train, y_train, num_neighbors=5, num_inducing=6, fit_batch_size=5, ski_grid_size=3, multires_levels=1)
        assert backend.backend_name == backend_name

    with pytest.raises(ValueError, match="Unknown GPIS backend"):
        fit_gpis_backend(cast(GPISBackendName, "not_a_backend"), x_train, y_train)


def test_nearest_training_indices_caps_requested_neighbor_count() -> None:
    x_train, _ = make_training_data(4)
    indices = nearest_training_indices(x_train[:2], x_train, num_neighbors=100)

    assert indices.shape == (2, 4)
    assert np.all(np.isin(indices.numpy(), np.arange(4)))


def test_select_inducing_indices_supports_deterministic_strategies() -> None:
    x_train, _ = make_training_data(7)

    first = select_inducing_indices(x_train, num_inducing=3, method="first")
    uniform = select_inducing_indices(x_train, num_inducing=3, method="uniform")
    farthest = select_inducing_indices(x_train, num_inducing=3, method="farthest")

    torch.testing.assert_close(first, torch.tensor([0, 1, 2]))
    assert uniform.shape == (3,)
    assert farthest.shape == (3,)
    assert len(set(farthest.tolist())) == 3
    with pytest.raises(ValueError, match="inducing_selection"):
        select_inducing_indices(x_train, num_inducing=3, method="bad")


def test_ard_lengthscales_are_validated() -> None:
    x_train, y_train = make_training_data(20)
    with pytest.raises(ValueError, match="ard_lengthscales"):
        fit_gpis_backend("ard_inducing_points", x_train, y_train, ard_lengthscales=torch.tensor([0.4, 0.0, 0.5], dtype=torch.float64))
