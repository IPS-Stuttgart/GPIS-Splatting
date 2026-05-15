from __future__ import annotations

import torch

from gpis_splatting.scalable_gpis_backends import (
    ScalableInducingPointGPISBackend,
    ScalableLocalExactGPISBackend,
    make_grid_inducing_points,
    nearest_training_indices_chunked,
    nearest_training_indices_scalable,
)


def make_training_data(n: int = 48) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)
    points = torch.rand((n, 3), generator=generator, dtype=torch.float64) * 2.0 - 1.0
    sdf = torch.linalg.norm(points, dim=-1) - 0.65
    return points, sdf


def test_chunked_knn_matches_dense_cdist_distances() -> None:
    x_train, _ = make_training_data(37)
    query = x_train[:9] + 0.013
    dense = nearest_training_indices_scalable(query, x_train, num_neighbors=5, neighbor_backend="cdist")
    chunked = nearest_training_indices_chunked(query, x_train, num_neighbors=5, train_chunk_size=7)
    distances = torch.cdist(query, x_train)
    torch.testing.assert_close(torch.gather(distances, dim=1, index=chunked), torch.gather(distances, dim=1, index=dense))


def test_scalable_local_backend_predicts_finite_values() -> None:
    x_train, y_train = make_training_data(64)
    backend = ScalableLocalExactGPISBackend.fit(x_train, y_train, num_neighbors=12, neighbor_backend="chunked", neighbor_train_chunk_size=11, local_solve_batch_size=4)
    prediction = backend.predict(x_train[:10], batch_size=6)
    assert prediction.mean.shape == (10,)
    assert prediction.variance.shape == (10,)
    assert prediction.gradient.shape == (10, 3)
    assert torch.all(torch.isfinite(prediction.mean))
    assert torch.all(prediction.variance > 0.0)


def test_grid_inducing_backend_predicts_finite_values() -> None:
    x_train, y_train = make_training_data(80)
    grid = make_grid_inducing_points(x_train, num_inducing=17)
    assert grid.shape == (17, 3)
    backend = ScalableInducingPointGPISBackend.fit(x_train, y_train, num_inducing=17, inducing_selection="grid", fit_batch_size=23, compute_device="cpu")
    prediction = backend.predict(x_train[:8], batch_size=4)
    assert backend.num_inducing == 17
    assert prediction.mean.shape == (8,)
    assert prediction.variance.shape == (8,)
    assert prediction.gradient.shape == (8, 3)
    assert torch.all(torch.isfinite(prediction.mean))
    assert torch.all(prediction.variance > 0.0)
