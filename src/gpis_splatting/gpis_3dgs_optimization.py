from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

Tensor = torch.Tensor
OptimizerLike = Any
StepCallback = Callable[["GPIS3DGSOptimizationStep"], None]


class GPIS3DGSRegularizationStepLike(Protocol):
    """Minimal scalar-loss result consumed by the 3DGS optimization adapter."""

    loss: Tensor

    def log_dict(self, prefix: str = "gpis") -> dict[str, Tensor]:
        ...


class GPIS3DGSRegularizerLike(Protocol):
    """Common interface shared by live GPIS and precomputed GPIS-prior regularizers."""

    def compute(
        self,
        gaussians: Any | None = None,
        *,
        iteration: int,
        centers: Tensor | None = None,
        opacities: Tensor | None = None,
        gaussian_normals: Tensor | None = None,
    ) -> GPIS3DGSRegularizationStepLike | None:
        ...

    def maybe_boost_densification_stats(
        self,
        gaussians: Any,
        step: GPIS3DGSRegularizationStepLike,
        *,
        iteration: int,
    ) -> Tensor | None:
        ...

    def maybe_prune(
        self,
        gaussians: Any,
        step: GPIS3DGSRegularizationStepLike,
        *,
        iteration: int,
    ) -> Tensor | None:
        ...


@dataclass(frozen=True)
class GPIS3DGSOptimizationLoopConfig:
    """Controls how GPIS regularization is applied inside a 3DGS optimizer step.

    The config intentionally does not own rendering or photometric loss construction. A
    3DGS trainer computes the normal image-space loss, then this loop helper adds the
    GPIS loss before backpropagation and applies optional density-control hooks around
    the optimizer step.
    """

    zero_grad_before_backward: bool = False
    zero_grad_after_step: bool = False
    set_to_none: bool = True
    retain_graph: bool = False
    step_optimizer: bool = True
    apply_densification_boost: bool = True
    apply_pruning: bool = True
    prune_after_optimizer_step: bool = True
    clip_grad_norm: float | None = None


@dataclass
class GPIS3DGSOptimizationStep:
    """State returned for one photometric-plus-GPIS optimization iteration."""

    iteration: int
    base_loss: Tensor
    total_loss: Tensor
    gpis_step: GPIS3DGSRegularizationStepLike | None
    density_boost_weights: Tensor | None = None
    prune_mask: Tensor | None = None
    grad_norm: Tensor | None = None
    optimizer_stepped: bool = False

    @property
    def has_gpis(self) -> bool:
        return self.gpis_step is not None

    def log_dict(self, *, train_prefix: str = "train", gpis_prefix: str = "gpis") -> dict[str, Tensor]:
        logs: dict[str, Tensor] = {
            f"{train_prefix}/base_loss": self.base_loss.detach(),
            f"{train_prefix}/total_loss": self.total_loss.detach(),
        }
        if self.grad_norm is not None:
            logs[f"{train_prefix}/grad_norm"] = _as_detached_tensor(self.grad_norm, device=self.total_loss.device)
        if self.density_boost_weights is not None:
            logs[f"{gpis_prefix}/densification_boosted"] = (self.density_boost_weights.detach() > 1.0).to(dtype=self.total_loss.dtype).sum()
        if self.prune_mask is not None:
            logs[f"{gpis_prefix}/pruned_gaussians"] = self.prune_mask.detach().to(dtype=self.total_loss.dtype).sum()
        if self.gpis_step is not None:
            logs.update(self.gpis_step.log_dict(prefix=gpis_prefix))
        return logs


class GPIS3DGSOptimizationLoop:
    """Combine a standard 3DGS loss with a GPIS-compatible loss and density hooks.

    ``regularizer`` may be the live :class:`GPIS3DGSTrainingRegularizer` or a
    precomputed-prior adapter such as ``GPIS3DGSTrainingPriorRegularizer``. Both expose
    the same small protocol: ``compute``, ``maybe_boost_densification_stats`` and
    ``maybe_prune``.

    Typical use inside a 3DGS trainer::

        loop = GPIS3DGSOptimizationLoop(gpis_regularizer)
        train_step = loop.augment_loss(base_loss=loss, gaussians=gaussians, iteration=iteration)
        train_step.total_loss.backward()
        loop.after_backward(gaussians, train_step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loop.after_optimizer_step(gaussians, train_step)

    For simple trainers, :meth:`step` performs the same sequence in one call.
    """

    def __init__(
        self,
        regularizer: GPIS3DGSRegularizerLike,
        config: GPIS3DGSOptimizationLoopConfig | None = None,
    ) -> None:
        self.regularizer = regularizer
        self.config = GPIS3DGSOptimizationLoopConfig() if config is None else config
        validate_optimization_loop_config(self.config)

    def augment_loss(
        self,
        *,
        base_loss: Tensor,
        gaussians: Any | None = None,
        iteration: int,
        centers: Tensor | None = None,
        opacities: Tensor | None = None,
        gaussian_normals: Tensor | None = None,
    ) -> GPIS3DGSOptimizationStep:
        """Add the scheduled GPIS-compatible loss to a scalar 3DGS photometric/base loss."""
        _validate_scalar_loss(base_loss, name="base_loss")
        gpis_step = self.regularizer.compute(
            gaussians,
            iteration=iteration,
            centers=centers,
            opacities=opacities,
            gaussian_normals=gaussian_normals,
        )
        if gpis_step is None:
            return GPIS3DGSOptimizationStep(iteration=iteration, base_loss=base_loss, total_loss=base_loss, gpis_step=None)
        _validate_scalar_loss(gpis_step.loss, name="gpis_step.loss")
        total_loss = base_loss + gpis_step.loss.to(dtype=base_loss.dtype, device=base_loss.device)
        return GPIS3DGSOptimizationStep(iteration=iteration, base_loss=base_loss, total_loss=total_loss, gpis_step=gpis_step)

    def backward(
        self,
        step: GPIS3DGSOptimizationStep,
        *,
        optimizer: OptimizerLike | None = None,
        parameters: Iterable[Tensor] | None = None,
        retain_graph: bool | None = None,
    ) -> GPIS3DGSOptimizationStep:
        """Backpropagate the combined loss and optionally clip gradients."""
        step.total_loss.backward(retain_graph=self.config.retain_graph if retain_graph is None else retain_graph)
        if self.config.clip_grad_norm is not None:
            resolved = _resolve_parameters(parameters=parameters, optimizer=optimizer)
            if not resolved:
                raise ValueError("parameters or an optimizer with param_groups must be provided when clip_grad_norm is configured.")
            step.grad_norm = torch.nn.utils.clip_grad_norm_(resolved, self.config.clip_grad_norm)
        return step

    def after_backward(self, gaussians: Any | None, step: GPIS3DGSOptimizationStep) -> GPIS3DGSOptimizationStep:
        """Apply hooks that belong after backward and before the trainer densifies."""
        if gaussians is None or step.gpis_step is None:
            return step
        if self.config.apply_densification_boost:
            step.density_boost_weights = self.regularizer.maybe_boost_densification_stats(gaussians, step.gpis_step, iteration=step.iteration)
        if self.config.apply_pruning and not self.config.prune_after_optimizer_step:
            step.prune_mask = self.regularizer.maybe_prune(gaussians, step.gpis_step, iteration=step.iteration)
        return step

    def after_optimizer_step(self, gaussians: Any | None, step: GPIS3DGSOptimizationStep) -> GPIS3DGSOptimizationStep:
        """Apply hooks that should run after the optimizer has consumed current gradients."""
        if gaussians is None or step.gpis_step is None:
            return step
        if self.config.apply_pruning and self.config.prune_after_optimizer_step:
            step.prune_mask = self.regularizer.maybe_prune(gaussians, step.gpis_step, iteration=step.iteration)
        return step

    def after_external_prune_mask(self, prune_mask: Tensor) -> None:
        """Notify stateful regularizers about pruning performed by an external trainer.

        Live GPIS regularization is stateless with respect to Gaussian identity, so this
        is a no-op there. Precomputed GPIS-prior regularizers use it to keep gate,
        densify, prune and opacity-target arrays aligned after the trainer removes
        Gaussians through its own pruning logic.
        """
        apply_prune_mask = getattr(self.regularizer, "apply_prune_mask", None)
        if callable(apply_prune_mask):
            apply_prune_mask(prune_mask)

    def step(
        self,
        *,
        base_loss: Tensor,
        gaussians: Any | None,
        iteration: int,
        optimizer: OptimizerLike | None = None,
        parameters: Iterable[Tensor] | None = None,
        centers: Tensor | None = None,
        opacities: Tensor | None = None,
        gaussian_normals: Tensor | None = None,
        before_optimizer_step: StepCallback | None = None,
        after_optimizer_step: StepCallback | None = None,
    ) -> GPIS3DGSOptimizationStep:
        """Run one combined optimization step for simple 3DGS trainers.

        Larger trainers can call :meth:`augment_loss`, :meth:`backward`,
        :meth:`after_backward`, and :meth:`after_optimizer_step` separately when they
        need to interleave custom visibility, densification, or logging code.
        """
        if optimizer is not None and self.config.zero_grad_before_backward:
            _zero_grad(optimizer, set_to_none=self.config.set_to_none)

        train_step = self.augment_loss(
            base_loss=base_loss,
            gaussians=gaussians,
            iteration=iteration,
            centers=centers,
            opacities=opacities,
            gaussian_normals=gaussian_normals,
        )
        self.backward(train_step, optimizer=optimizer, parameters=parameters)
        self.after_backward(gaussians, train_step)

        if before_optimizer_step is not None:
            before_optimizer_step(train_step)

        if optimizer is not None and self.config.step_optimizer:
            optimizer.step()
            train_step.optimizer_stepped = True

        if optimizer is not None and self.config.zero_grad_after_step:
            _zero_grad(optimizer, set_to_none=self.config.set_to_none)

        self.after_optimizer_step(gaussians, train_step)

        if after_optimizer_step is not None:
            after_optimizer_step(train_step)
        return train_step


def validate_optimization_loop_config(config: GPIS3DGSOptimizationLoopConfig) -> None:
    if config.clip_grad_norm is not None and config.clip_grad_norm <= 0.0:
        raise ValueError("clip_grad_norm must be positive when provided.")


def _validate_scalar_loss(loss: Tensor, *, name: str) -> None:
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if loss.numel() != 1:
        raise ValueError(f"{name} must be scalar; got shape {tuple(loss.shape)}.")


def _resolve_parameters(*, parameters: Iterable[Tensor] | None, optimizer: OptimizerLike | None) -> list[Tensor]:
    if parameters is not None:
        return [parameter for parameter in parameters if isinstance(parameter, torch.Tensor)]
    if optimizer is None or not hasattr(optimizer, "param_groups"):
        return []
    resolved: list[Tensor] = []
    for group in optimizer.param_groups:
        resolved.extend(parameter for parameter in group.get("params", []) if isinstance(parameter, torch.Tensor))
    return resolved


def _zero_grad(optimizer: OptimizerLike, *, set_to_none: bool) -> None:
    try:
        optimizer.zero_grad(set_to_none=set_to_none)
    except TypeError:
        optimizer.zero_grad()


def _as_detached_tensor(value: Tensor | float, *, device: torch.device) -> Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return torch.as_tensor(value, device=device)


__all__ = [
    "GPIS3DGSOptimizationLoop",
    "GPIS3DGSOptimizationLoopConfig",
    "GPIS3DGSOptimizationStep",
    "GPIS3DGSRegularizationStepLike",
    "GPIS3DGSRegularizerLike",
    "validate_optimization_loop_config",
]
