from __future__ import annotations

import torch

from gpis_splatting.gpis import fit_dense_gpis
from gpis_splatting.gpis_regularization import (
    GPISRegularizationConfig,
    gpis_regularization_loss,
    predict_gpis_differentiable,
)


def make_plane_gpis() -> object:
    coords = torch.linspace(-0.6, 0.6, 4, dtype=torch.float64)
    points = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1).reshape(-1, 3)
    sdf = points[:, 2]
    return fit_dense_gpis(points, sdf, lengthscale=0.65, noise_std=0.02, mean_constant=0.0)


def test_differentiable_prediction_preserves_center_gradients() -> None:
    model = make_plane_gpis()
    centers = torch.tensor([[0.1, -0.1, 0.2], [0.0, 0.0, -0.25]], dtype=torch.float64, requires_grad=True)

    prediction = predict_gpis_differentiable(model, centers)
    loss = prediction.mean.square().mean()
    loss.backward()

    assert centers.grad is not None
    assert torch.linalg.norm(centers.grad) > 0.0


def test_surface_regularizer_pulls_off_surface_centers_more_than_surface_centers() -> None:
    model = make_plane_gpis()
    near = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.1, 0.01]], dtype=torch.float64)
    far = torch.tensor([[0.0, 0.0, 0.45], [0.2, -0.1, -0.45]], dtype=torch.float64)
    config = GPISRegularizationConfig(epsilon=0.12, surface_weight=1.0, opacity_weight=0.0, normal_weight=0.0)

    near_loss = gpis_regularization_loss(model, near, config=config).surface_loss
    far_loss = gpis_regularization_loss(model, far, config=config).surface_loss

    assert far_loss > near_loss


def test_regularization_combines_surface_opacity_and_normal_terms() -> None:
    model = make_plane_gpis()
    centers = torch.tensor([[0.0, 0.0, 0.02], [0.0, 0.0, 0.5]], dtype=torch.float64, requires_grad=True)
    opacities = torch.tensor([0.8, 0.8], dtype=torch.float64)
    normals = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    config = GPISRegularizationConfig(epsilon=0.12, surface_weight=1.0, opacity_weight=0.5, normal_weight=0.25)

    result = gpis_regularization_loss(model, centers, opacities=opacities, gaussian_normals=normals, config=config)
    result.loss.backward()

    assert result.surface_loss >= 0.0
    assert result.opacity_loss >= 0.0
    assert result.normal_loss >= 0.0
    assert centers.grad is not None
    assert torch.all(torch.isfinite(centers.grad))


def test_opacity_term_penalizes_low_confidence_splats_more() -> None:
    model = make_plane_gpis()
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.55]], dtype=torch.float64)
    opacities = torch.tensor([1.0, 1.0], dtype=torch.float64)
    config = GPISRegularizationConfig(epsilon=0.1, surface_weight=0.0, opacity_weight=1.0, normal_weight=0.0, reduction="none")

    result = gpis_regularization_loss(model, centers, opacities=opacities, config=config)

    assert result.opacity_loss[1] > result.opacity_loss[0]
