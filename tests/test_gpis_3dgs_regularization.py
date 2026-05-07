from __future__ import annotations

import torch

from gpis_splatting.gpis import fit_dense_gpis
from gpis_splatting.gpis_3dgs_regularization import (
    GPIS3DGSDensityControlConfig,
    GPIS3DGSRegularizationStep,
    GPIS3DGSRegularizerConfig,
    GPIS3DGSTrainingRegularizer,
    extract_3dgs_tensors,
    gaussian_normals_from_scale_rotation,
    sample_gaussian_indices,
)
from gpis_splatting.gpis_regularization import GPISRegularizationConfig


class DummyGaussians:
    def __init__(self, centers: torch.Tensor, opacities: torch.Tensor | None = None) -> None:
        self._xyz = centers
        default_opacity = torch.full((centers.shape[0],), 0.5, dtype=centers.dtype)
        opacity = opacities if opacities is not None else default_opacity
        self._opacity = torch.logit(torch.clamp(opacity, 1e-5, 1.0 - 1e-5))[:, None]
        self._scaling = torch.tensor([[0.05, 0.3, 0.4]] * centers.shape[0], dtype=centers.dtype)
        self._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * centers.shape[0], dtype=centers.dtype)
        self.xyz_gradient_accum = torch.ones((centers.shape[0], 1), dtype=centers.dtype)
        self.pruned_mask: torch.Tensor | None = None

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._rotation

    def prune_points(self, mask: torch.Tensor) -> None:
        self.pruned_mask = mask.detach().clone()


def make_plane_gpis() -> object:
    coords = torch.linspace(-0.6, 0.6, 4, dtype=torch.float64)
    points = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1).reshape(-1, 3)
    sdf = points[:, 2]
    return fit_dense_gpis(points, sdf, lengthscale=0.65, noise_std=0.02, mean_constant=0.0)


def test_training_regularizer_computes_differentiable_loss_from_3dgs_like_object() -> None:
    model = make_plane_gpis()
    centers = torch.tensor([[0.0, 0.0, 0.02], [0.0, 0.0, 0.45], [0.1, 0.2, -0.4]], dtype=torch.float64, requires_grad=True)
    gaussians = DummyGaussians(centers, torch.tensor([0.8, 0.5, 0.5], dtype=torch.float64))
    regularizer = GPIS3DGSTrainingRegularizer(
        model,
        loss_config=GPISRegularizationConfig(epsilon=0.12, surface_weight=1.0, opacity_weight=0.1, normal_weight=0.05),
        schedule_config=GPIS3DGSRegularizerConfig(start_iteration=2, ramp_iterations=4, max_regularized_gaussians=None),
    )

    step = regularizer.compute(gaussians, iteration=3)
    assert step is not None
    assert step.sampled_indices.numel() == centers.shape[0]
    assert 0.0 < step.active_weight < 1.0
    step.loss.backward()

    assert centers.grad is not None
    assert torch.all(torch.isfinite(centers.grad))
    assert torch.linalg.norm(centers.grad) > 0.0


def test_training_regularizer_returns_none_when_schedule_is_inactive() -> None:
    model = make_plane_gpis()
    centers = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    regularizer = GPIS3DGSTrainingRegularizer(model, schedule_config=GPIS3DGSRegularizerConfig(start_iteration=10))

    assert regularizer.compute(centers=centers, iteration=9) is None


def test_prune_mask_selects_low_confidence_low_opacity_subset() -> None:
    step = GPIS3DGSRegularizationStep(
        loss=torch.tensor(0.0),
        raw_loss=torch.tensor(0.0),
        surface_loss=torch.tensor(0.0),
        opacity_loss=torch.tensor(0.0),
        normal_loss=torch.tensor(0.0),
        confidence=torch.tensor([0.9, 0.01, 0.02, 0.5]),
        signed_distance=torch.zeros(4),
        distance_std=torch.ones(4),
        field_normal=torch.zeros((4, 3)),
        sampled_indices=torch.tensor([0, 1, 2, 3]),
        active_weight=1.0,
        gaussian_count=4,
    )
    regularizer = GPIS3DGSTrainingRegularizer(
        make_plane_gpis(),
        density_config=GPIS3DGSDensityControlConfig(prune_confidence_threshold=0.05, prune_opacity_threshold=0.05, max_prune_fraction=0.5),
    )
    mask = regularizer.build_prune_mask(step, opacities=torch.tensor([0.5, 0.01, 0.2, 0.01]))

    assert mask.tolist() == [False, True, False, False]


def test_maybe_prune_calls_3dgs_prune_points_on_schedule() -> None:
    centers = torch.zeros((3, 3), dtype=torch.float64)
    gaussians = DummyGaussians(centers, torch.tensor([0.01, 0.01, 0.5], dtype=torch.float64))
    step = GPIS3DGSRegularizationStep(
        loss=torch.tensor(0.0),
        raw_loss=torch.tensor(0.0),
        surface_loss=torch.tensor(0.0),
        opacity_loss=torch.tensor(0.0),
        normal_loss=torch.tensor(0.0),
        confidence=torch.tensor([0.01, 0.8, 0.01]),
        signed_distance=torch.zeros(3),
        distance_std=torch.ones(3),
        field_normal=torch.zeros((3, 3)),
        sampled_indices=torch.tensor([0, 1, 2]),
        active_weight=1.0,
        gaussian_count=3,
    )
    regularizer = GPIS3DGSTrainingRegularizer(
        make_plane_gpis(),
        density_config=GPIS3DGSDensityControlConfig(prune_interval=2, prune_confidence_threshold=0.05, prune_opacity_threshold=0.05, max_prune_fraction=1.0),
    )

    mask = regularizer.maybe_prune(gaussians, step, iteration=4)

    assert mask is not None
    assert gaussians.pruned_mask is not None
    assert gaussians.pruned_mask.tolist() == [True, False, False]


def test_densification_boost_multiplies_gradient_accumulator_for_uncertain_low_confidence_gaussians() -> None:
    centers = torch.zeros((3, 3), dtype=torch.float64)
    gaussians = DummyGaussians(centers)
    step = GPIS3DGSRegularizationStep(
        loss=torch.tensor(0.0),
        raw_loss=torch.tensor(0.0),
        surface_loss=torch.tensor(0.0),
        opacity_loss=torch.tensor(0.0),
        normal_loss=torch.tensor(0.0),
        confidence=torch.tensor([0.8, 0.1, 0.05], dtype=torch.float64),
        signed_distance=torch.zeros(3, dtype=torch.float64),
        distance_std=torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64),
        field_normal=torch.zeros((3, 3), dtype=torch.float64),
        sampled_indices=torch.tensor([0, 1, 2]),
        active_weight=1.0,
        gaussian_count=3,
    )
    regularizer = GPIS3DGSTrainingRegularizer(
        make_plane_gpis(),
        density_config=GPIS3DGSDensityControlConfig(
            densification_boost_interval=1,
            densification_confidence_threshold=0.2,
            densification_min_distance_std=0.2,
            densification_gradient_boost=2.0,
        ),
    )

    weights = regularizer.maybe_boost_densification_stats(gaussians, step, iteration=0)

    assert weights is not None
    assert gaussians.xyz_gradient_accum[0].item() == 1.0
    assert gaussians.xyz_gradient_accum[1].item() > 1.0
    assert gaussians.xyz_gradient_accum[2].item() > gaussians.xyz_gradient_accum[1].item()


def test_gaussian_normals_use_smallest_scale_axis_after_rotation() -> None:
    scaling = torch.tensor([[0.1, 0.5, 0.7], [0.5, 0.1, 0.7]], dtype=torch.float64)
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    normals = gaussian_normals_from_scale_rotation(scaling, rotation)

    assert torch.allclose(normals, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64))


def test_extract_3dgs_tensors_uses_activated_opacity_and_inferred_normals() -> None:
    centers = torch.zeros((2, 3), dtype=torch.float64)
    gaussians = DummyGaussians(centers, torch.tensor([0.2, 0.7], dtype=torch.float64))

    tensors = extract_3dgs_tensors(gaussians)

    assert torch.allclose(tensors.centers, centers)
    assert tensors.opacities is not None
    assert torch.allclose(tensors.opacities, torch.tensor([0.2, 0.7], dtype=torch.float64))
    assert tensors.gaussian_normals is not None
    assert tensors.gaussian_normals.shape == centers.shape


def test_strided_sampling_rotates_with_iteration() -> None:
    first = sample_gaussian_indices(10, max_gaussians=4, mode="strided", seed=0, iteration=0, device="cpu")
    second = sample_gaussian_indices(10, max_gaussians=4, mode="strided", seed=0, iteration=1, device="cpu")

    assert first.numel() == 4
    assert second.numel() == 4
    assert not torch.equal(first, second)
