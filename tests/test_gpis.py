from __future__ import annotations

import torch

from gpis_splatting.gpis import fit_dense_gpis, predict_gpis, rbf_kernel, surface_band_probability
from gpis_splatting.scenes import sample_scene


def test_rbf_kernel_is_symmetric_and_psd() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.1, -0.1],
            [-0.3, 0.4, 0.2],
            [0.5, -0.2, 0.1],
        ],
        dtype=torch.float64,
    )
    kernel = rbf_kernel(points, points, lengthscale=0.4, variance=1.3)

    assert torch.allclose(kernel, kernel.T, atol=1e-12)
    eigvals = torch.linalg.eigvalsh(kernel)
    assert float(eigvals.min()) > -1e-10


def test_dense_gpis_predicts_stable_quantities() -> None:
    data = sample_scene("sphere", num_points=45, seed=3, noise_std=0.02)
    model = fit_dense_gpis(data["points"], data["observed_sdf"], lengthscale=0.42, noise_std=0.03)
    prediction = predict_gpis(model, data["points"][:12], batch_size=8)
    gate = surface_band_probability(prediction, epsilon=0.08)

    assert prediction.mean.shape == (12,)
    assert prediction.variance.shape == (12,)
    assert prediction.gradient.shape == (12, 3)
    assert torch.all(torch.isfinite(prediction.mean))
    assert torch.all(prediction.variance > 0.0)
    assert torch.all(prediction.grad_norm >= 1e-6)
    assert torch.all((prediction.inside_probability >= 0.0) & (prediction.inside_probability <= 1.0))
    assert torch.all((gate >= 0.0) & (gate <= 1.0))

