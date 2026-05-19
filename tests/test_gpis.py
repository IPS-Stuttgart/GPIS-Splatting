from __future__ import annotations

import torch

from gpis_splatting.feedback import refine_gpis_with_splat_feedback
from gpis_splatting.gpis import fit_dense_gpis, predict_gpis, rbf_kernel, surface_band_probability
from gpis_splatting.scenes import sample_scene
from gpis_splatting.splats import make_candidate_splats


def test_surface_jitter_is_scalar_per_point(monkeypatch) -> None:
    randn_shapes: list[tuple[int, ...]] = []
    original_randn = torch.randn

    def recording_randn(*args, **kwargs):
        shape_arg = args[0] if args else kwargs.get("size", ())
        if isinstance(shape_arg, torch.Size):
            shape = tuple(shape_arg)
        elif isinstance(shape_arg, (tuple, list)):
            shape = tuple(shape_arg)
        elif isinstance(shape_arg, int):
            shape = (shape_arg,)
        else:
            shape = tuple(shape_arg)
        randn_shapes.append(shape)
        return original_randn(*args, **kwargs)

    monkeypatch.setattr(torch, "randn", recording_randn)

    sample_scene("sphere", num_points=20, seed=3, noise_std=0.02)
    make_candidate_splats("sphere", num_splats=20, offsurface_fraction=0.28, seed=11)

    assert randn_shapes.count((14, 1)) == 2
    assert (14, 3) not in randn_shapes


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


def test_splat_feedback_adds_pseudo_observations() -> None:
    data = sample_scene("sphere", num_points=50, seed=8, noise_std=0.02)
    model = fit_dense_gpis(data["points"], data["observed_sdf"], lengthscale=0.5, noise_std=0.03)
    splats = make_candidate_splats("sphere", num_splats=70, offsurface_fraction=0.25, seed=11)

    feedback = refine_gpis_with_splat_feedback(
        model,
        splats,
        epsilon=0.12,
        iterations=1,
        pseudo_points_per_iteration=12,
        min_gate=0.0,
    )

    assert feedback.base_gate.shape == (splats.centers.shape[0],)
    assert feedback.feedback_gate.shape == feedback.base_gate.shape
    assert feedback.selected_mask.sum() == 12
    assert feedback.model.x_train.shape[0] == model.x_train.shape[0] + 12
    assert feedback.model.observation_noise_std is not None
    assert len(feedback.trace) == 1
    assert feedback.trace[0]["selector"] == "gate"
    assert torch.all((feedback.feedback_gate >= 0.0) & (feedback.feedback_gate <= 1.0))


def test_uncertainty_feedback_selector_records_uncertainty_scores() -> None:
    data = sample_scene("sphere", num_points=50, seed=10, noise_std=0.02)
    model = fit_dense_gpis(data["points"], data["observed_sdf"], lengthscale=0.5, noise_std=0.03)
    splats = make_candidate_splats("sphere", num_splats=70, offsurface_fraction=0.25, seed=13)

    feedback = refine_gpis_with_splat_feedback(
        model,
        splats,
        epsilon=0.12,
        iterations=1,
        pseudo_points_per_iteration=10,
        min_gate=0.0,
        selector="uncertainty",
    )

    assert feedback.selected_mask.sum() == 10
    assert feedback.trace[0]["selector"] == "uncertainty"
    assert feedback.trace[0]["selected_score_mean"] > 0.0
    assert feedback.trace[0]["selected_distance_std_mean"] > 0.0


def test_diverse_uncertainty_feedback_suppresses_nearby_splats() -> None:
    data = sample_scene("sphere", num_points=50, seed=12, noise_std=0.02)
    model = fit_dense_gpis(data["points"], data["observed_sdf"], lengthscale=0.5, noise_std=0.03)
    splats = make_candidate_splats("sphere", num_splats=70, offsurface_fraction=0.25, seed=15)

    feedback = refine_gpis_with_splat_feedback(
        model,
        splats,
        epsilon=0.12,
        iterations=1,
        pseudo_points_per_iteration=10,
        min_gate=0.0,
        selector="uncertainty_diverse",
        diversity_radius=10.0,
    )

    assert feedback.selected_mask.sum() == 1
    assert feedback.trace[0]["selector"] == "uncertainty_diverse"
