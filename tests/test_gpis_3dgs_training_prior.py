from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gpis_splatting.gpis_3dgs_training_prior import (
    GPIS3DGSTrainingPriorConfig,
    GPIS3DGSTrainingPriorRegularizer,
    align_training_prior_state,
    load_gpis_training_prior,
)


class DummyGaussians:
    def __init__(self, alpha: torch.Tensor) -> None:
        self._xyz = torch.zeros((alpha.numel(), 3), dtype=alpha.dtype)
        self._opacity = torch.logit(torch.clamp(alpha, 1e-5, 1.0 - 1e-5)).detach().clone().requires_grad_(True)
        self.xyz_gradient_accum = torch.ones((alpha.numel(), 1), dtype=alpha.dtype)
        self.pruned_mask: torch.Tensor | None = None

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    def prune_points(self, mask: torch.Tensor) -> None:
        self.pruned_mask = mask.detach().clone()


def write_prior(path: Path) -> None:
    np.savez_compressed(
        path,
        gate=np.array([0.95, 0.2, 0.8, 0.05], dtype=np.float32),
        densify_weight=np.array([0.1, 0.3, 0.9, 0.0], dtype=np.float32),
        densify_candidate_mask=np.array([False, False, True, False]),
        prune_weight=np.array([0.0, 0.6, 0.1, 0.95], dtype=np.float32),
        prune_candidate_mask=np.array([False, True, False, True]),
        opacity_target_alpha=np.array([0.8, 0.1, 0.7, 0.01], dtype=np.float32),
        opacity_regularization_weight=np.array([0.0, 1.0, 0.25, 2.0], dtype=np.float32),
    )


def test_training_prior_adds_differentiable_opacity_loss(tmp_path: Path) -> None:
    prior_path = tmp_path / "prior.npz"
    write_prior(prior_path)
    gaussians = DummyGaussians(torch.tensor([0.8, 0.8, 0.7, 0.8], dtype=torch.float64))
    regularizer = GPIS3DGSTrainingPriorRegularizer.from_prior_path(
        prior_path,
        config=GPIS3DGSTrainingPriorConfig(opacity_weight=2.0, max_regularized_gaussians=None, ramp_iterations=2),
        dtype=torch.float64,
    )

    step = regularizer.compute(gaussians, iteration=0)
    assert step is not None
    assert 0.0 < step.active_weight < 1.0
    assert step.opacity_loss.item() > 0.0
    step.loss.backward()

    assert gaussians._opacity.grad is not None
    assert torch.linalg.norm(gaussians._opacity.grad) > 0.0
    assert torch.isfinite(gaussians._opacity.grad).all()


def test_training_prior_prunes_high_weight_candidates_on_schedule(tmp_path: Path) -> None:
    prior_path = tmp_path / "prior.npz"
    write_prior(prior_path)
    gaussians = DummyGaussians(torch.tensor([0.8, 0.8, 0.7, 0.8], dtype=torch.float64))
    regularizer = GPIS3DGSTrainingPriorRegularizer.from_prior_path(
        prior_path,
        config=GPIS3DGSTrainingPriorConfig(prune_interval=1, prune_weight_threshold=0.5, max_prune_fraction=0.5, max_regularized_gaussians=None),
        dtype=torch.float64,
    )

    step = regularizer.compute(gaussians, iteration=0)
    assert step is not None
    mask = regularizer.maybe_prune(gaussians, step, iteration=0)

    assert mask is not None
    assert gaussians.pruned_mask is not None
    assert gaussians.pruned_mask.tolist() == [False, True, False, True]
    assert regularizer.state.gaussian_count == 2


def test_training_prior_boosts_densification_gradient_stats(tmp_path: Path) -> None:
    prior_path = tmp_path / "prior.npz"
    write_prior(prior_path)
    gaussians = DummyGaussians(torch.tensor([0.8, 0.8, 0.7, 0.8], dtype=torch.float64))
    regularizer = GPIS3DGSTrainingPriorRegularizer.from_prior_path(
        prior_path,
        config=GPIS3DGSTrainingPriorConfig(densification_boost_interval=1, densification_gradient_boost=2.0, max_regularized_gaussians=None),
        dtype=torch.float64,
    )

    step = regularizer.compute(gaussians, iteration=0)
    assert step is not None
    weights = regularizer.maybe_boost_densification_stats(gaussians, step, iteration=0)

    assert weights is not None
    assert gaussians.xyz_gradient_accum[0].item() == 1.0
    assert gaussians.xyz_gradient_accum[2].item() > 1.0
    assert gaussians.xyz_gradient_accum[3].item() == 1.0


def test_training_prior_pads_new_gaussians_neutrally(tmp_path: Path) -> None:
    prior_path = tmp_path / "prior.npz"
    write_prior(prior_path)
    state = load_gpis_training_prior(prior_path, dtype=torch.float64)

    aligned = align_training_prior_state(state, count=6, config=GPIS3DGSTrainingPriorConfig(count_mismatch="pad"))

    assert aligned.gaussian_count == 6
    assert torch.allclose(aligned.opacity_regularization_weight[-2:], torch.zeros(2, dtype=torch.float64))
    assert aligned.densify_candidate_mask[-2:].tolist() == [False, False]
    assert aligned.prune_candidate_mask[-2:].tolist() == [False, False]
