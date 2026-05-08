from __future__ import annotations

import torch

from gpis_splatting.gpis_3dgs_optimization import GPIS3DGSOptimizationLoop, GPIS3DGSOptimizationLoopConfig
from gpis_splatting.gpis_3dgs_regularization import GPIS3DGSRegularizationStep


class DummyRegularizer:
    def __init__(self, *, active: bool = True) -> None:
        self.active = active
        self.compute_calls = 0
        self.boost_calls = 0
        self.prune_calls = 0

    def compute(self, gaussians, *, iteration: int, centers=None, opacities=None, gaussian_normals=None):
        self.compute_calls += 1
        if not self.active:
            return None
        target = gaussians.center_target if centers is None else centers.new_tensor([0.25])
        loss = (gaussians.parameter - target).square().mean()
        return GPIS3DGSRegularizationStep(
            loss=loss,
            raw_loss=loss,
            surface_loss=loss,
            opacity_loss=torch.zeros((), dtype=loss.dtype),
            normal_loss=torch.zeros((), dtype=loss.dtype),
            confidence=torch.tensor([0.1, 0.9], dtype=loss.dtype),
            signed_distance=torch.zeros(2, dtype=loss.dtype),
            distance_std=torch.tensor([0.6, 0.2], dtype=loss.dtype),
            field_normal=torch.zeros((2, 3), dtype=loss.dtype),
            sampled_indices=torch.tensor([0, 1]),
            active_weight=1.0,
            gaussian_count=2,
        )

    def maybe_boost_densification_stats(self, gaussians, step, *, iteration: int):
        self.boost_calls += 1
        gaussians.events.append(("boost", iteration))
        return torch.tensor([2.0, 1.0])

    def maybe_prune(self, gaussians, step, *, iteration: int):
        self.prune_calls += 1
        gaussians.events.append(("prune", iteration))
        return torch.tensor([True, False])


class DummyGaussians:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor([1.0]))
        self.center_target = torch.tensor([0.0])
        self.events: list[tuple[str, int]] = []


class LegacyOptimizer:
    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.param_groups = [{"params": [parameter]}]
        self.steps = 0
        self.zero_grad_calls = 0

    def step(self) -> None:
        self.steps += 1
        with torch.no_grad():
            for parameter in self.param_groups[0]["params"]:
                if parameter.grad is not None:
                    parameter -= 0.1 * parameter.grad

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1
        for parameter in self.param_groups[0]["params"]:
            parameter.grad = None


def test_step_combines_photometric_and_gpis_losses_in_one_backward_pass() -> None:
    gaussians = DummyGaussians()
    optimizer = torch.optim.SGD([gaussians.parameter], lr=0.1)
    loop = GPIS3DGSOptimizationLoop(
        DummyRegularizer(),
        GPIS3DGSOptimizationLoopConfig(zero_grad_after_step=True),
    )

    base_loss = gaussians.parameter.square().mean()
    step = loop.step(base_loss=base_loss, gaussians=gaussians, iteration=7, optimizer=optimizer)

    assert step.has_gpis
    assert step.optimizer_stepped
    assert step.gpis_step is not None
    assert torch.allclose(step.total_loss.detach(), base_loss.detach() + step.gpis_step.loss.detach())
    assert gaussians.parameter.item() < 1.0
    assert gaussians.parameter.grad is None
    assert gaussians.events == [("boost", 7), ("prune", 7)]


def test_inactive_regularizer_keeps_base_loss_and_skips_density_hooks() -> None:
    regularizer = DummyRegularizer(active=False)
    gaussians = DummyGaussians()
    loop = GPIS3DGSOptimizationLoop(regularizer)

    base_loss = gaussians.parameter.square().mean()
    step = loop.augment_loss(base_loss=base_loss, gaussians=gaussians, iteration=1)
    loop.after_backward(gaussians, step)
    loop.after_optimizer_step(gaussians, step)

    assert not step.has_gpis
    assert step.total_loss is base_loss
    assert regularizer.compute_calls == 1
    assert regularizer.boost_calls == 0
    assert regularizer.prune_calls == 0
    assert gaussians.events == []


def test_split_phase_api_allows_pruning_before_optimizer_step() -> None:
    regularizer = DummyRegularizer()
    gaussians = DummyGaussians()
    optimizer = torch.optim.SGD([gaussians.parameter], lr=0.1)
    loop = GPIS3DGSOptimizationLoop(
        regularizer,
        GPIS3DGSOptimizationLoopConfig(prune_after_optimizer_step=False),
    )

    train_step = loop.augment_loss(base_loss=gaussians.parameter.square().mean(), gaussians=gaussians, iteration=3)
    loop.backward(train_step, optimizer=optimizer)
    loop.after_backward(gaussians, train_step)
    optimizer.step()
    loop.after_optimizer_step(gaussians, train_step)

    assert regularizer.boost_calls == 1
    assert regularizer.prune_calls == 1
    assert gaussians.events == [("boost", 3), ("prune", 3)]
    assert train_step.prune_mask is not None
    assert train_step.prune_mask.tolist() == [True, False]


def test_step_supports_legacy_zero_grad_without_set_to_none_argument() -> None:
    gaussians = DummyGaussians()
    optimizer = LegacyOptimizer(gaussians.parameter)
    loop = GPIS3DGSOptimizationLoop(
        DummyRegularizer(),
        GPIS3DGSOptimizationLoopConfig(zero_grad_before_backward=True, zero_grad_after_step=True),
    )

    loop.step(base_loss=gaussians.parameter.square().mean(), gaussians=gaussians, iteration=2, optimizer=optimizer)

    assert optimizer.steps == 1
    assert optimizer.zero_grad_calls == 2


def test_clip_grad_norm_records_log_value() -> None:
    gaussians = DummyGaussians()
    optimizer = torch.optim.SGD([gaussians.parameter], lr=0.0)
    loop = GPIS3DGSOptimizationLoop(
        DummyRegularizer(),
        GPIS3DGSOptimizationLoopConfig(clip_grad_norm=0.1, step_optimizer=False),
    )

    step = loop.step(base_loss=(10.0 * gaussians.parameter).square().mean(), gaussians=gaussians, iteration=4, optimizer=optimizer)
    logs = step.log_dict()

    assert step.grad_norm is not None
    assert "train/grad_norm" in logs
    assert "gpis/pruned_gaussians" in logs
    assert "gpis/densification_boosted" in logs


def test_invalid_scalar_loss_is_rejected() -> None:
    loop = GPIS3DGSOptimizationLoop(DummyRegularizer(active=False))

    try:
        loop.augment_loss(base_loss=torch.ones(2), gaussians=None, iteration=0)
    except ValueError as exc:
        assert "must be scalar" in str(exc)
    else:
        raise AssertionError("Expected non-scalar base_loss to be rejected.")
