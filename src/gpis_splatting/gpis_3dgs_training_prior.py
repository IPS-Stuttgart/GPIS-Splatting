from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from gpis_splatting.gpis_3dgs_regularization import extract_3dgs_tensors, resolve_tensor_attribute, sample_gaussian_indices
from gpis_splatting.gpis_regularization import prepare_opacity, reduce_loss

Tensor = torch.Tensor
SampleMode = Literal["all", "strided"]
CountMismatchMode = Literal["error", "pad"]
AlphaLoss = Literal["l1", "l2", "charbonnier"]


@dataclass(frozen=True)
class GPIS3DGSTrainingPriorConfig:
    """Schedule and strength for a precomputed GPIS 3DGS training prior."""

    start_iteration: int = 0
    stop_iteration: int | None = None
    interval: int = 1
    ramp_iterations: int = 0
    max_regularized_gaussians: int | None = 65536
    sample_mode: SampleMode = "strided"
    sample_seed: int = 0
    count_mismatch: CountMismatchMode = "pad"
    opacity_weight: float = 1e-3
    opacity_loss: AlphaLoss = "l2"
    opacity_is_logit: bool = False
    charbonnier_eps: float = 1e-6
    neutral_gate: float = 1.0
    neutral_opacity_weight: float = 0.0
    neutral_densify_weight: float = 0.0
    prune_start_iteration: int = 0
    prune_interval: int = 0
    prune_weight_threshold: float = 0.5
    max_prune_fraction: float = 0.05
    densification_boost_start_iteration: int = 0
    densification_boost_interval: int = 0
    densification_gradient_boost: float = 0.0


@dataclass(frozen=True)
class GPIS3DGSTrainingPriorState:
    """Tensorized per-Gaussian policy exported from calibrated GPIS confidence."""

    gate: Tensor
    densify_weight: Tensor
    densify_candidate_mask: Tensor
    prune_weight: Tensor
    prune_candidate_mask: Tensor
    opacity_target_alpha: Tensor
    opacity_regularization_weight: Tensor

    @property
    def gaussian_count(self) -> int:
        return int(self.gate.numel())

    def to(self, *, device: torch.device | str, dtype: torch.dtype) -> GPIS3DGSTrainingPriorState:
        return GPIS3DGSTrainingPriorState(
            gate=self.gate.to(device=device, dtype=dtype),
            densify_weight=self.densify_weight.to(device=device, dtype=dtype),
            densify_candidate_mask=self.densify_candidate_mask.to(device=device),
            prune_weight=self.prune_weight.to(device=device, dtype=dtype),
            prune_candidate_mask=self.prune_candidate_mask.to(device=device),
            opacity_target_alpha=self.opacity_target_alpha.to(device=device, dtype=dtype),
            opacity_regularization_weight=self.opacity_regularization_weight.to(device=device, dtype=dtype),
        )


@dataclass
class GPIS3DGSTrainingPriorStep:
    """One scheduled runtime-prior contribution for a 3DGS training iteration."""

    loss: Tensor
    raw_loss: Tensor
    opacity_loss: Tensor
    gate: Tensor
    densify_weight: Tensor
    densify_candidate_mask: Tensor
    prune_weight: Tensor
    prune_candidate_mask: Tensor
    opacity_target_alpha: Tensor
    opacity_regularization_weight: Tensor
    sampled_indices: Tensor
    active_weight: float
    gaussian_count: int

    @property
    def terms(self) -> dict[str, Tensor]:
        return {"loss": self.loss, "raw_loss": self.raw_loss, "opacity": self.opacity_loss}

    def log_dict(self, prefix: str = "gpis_prior") -> dict[str, Tensor]:
        return {
            f"{prefix}/loss": self.loss.detach(),
            f"{prefix}/raw_loss": self.raw_loss.detach(),
            f"{prefix}/opacity_loss": self.opacity_loss.detach(),
            f"{prefix}/gate_mean": self.gate.detach().mean(),
            f"{prefix}/gate_min": self.gate.detach().min(),
            f"{prefix}/densify_weight_mean": self.densify_weight.detach().mean(),
            f"{prefix}/prune_weight_mean": self.prune_weight.detach().mean(),
            f"{prefix}/opacity_regularization_weight_mean": self.opacity_regularization_weight.detach().mean(),
            f"{prefix}/sampled_gaussians": torch.as_tensor(self.sampled_indices.numel(), device=self.loss.device),
        }


class GPIS3DGSTrainingPriorRegularizer:
    """Use an exported GPIS confidence policy inside a 3DGS training loop.

    This mirrors the live ``GPIS3DGSTrainingRegularizer`` shape: ``compute``,
    ``maybe_boost_densification_stats`` and ``maybe_prune``. The prior can therefore
    be plugged into ``GPIS3DGSOptimizationLoop`` without requiring a live GPIS query
    at every iteration.
    """

    def __init__(self, state: GPIS3DGSTrainingPriorState, config: GPIS3DGSTrainingPriorConfig | None = None) -> None:
        self.state = state
        self.config = GPIS3DGSTrainingPriorConfig() if config is None else config
        self._cached_state: GPIS3DGSTrainingPriorState | None = None
        self._cached_count: int | None = None
        self._cached_device: torch.device | None = None
        self._cached_dtype: torch.dtype | None = None
        validate_training_prior_config(self.config)
        validate_training_prior_state(self.state)

    @classmethod
    def from_prior_path(
        cls,
        prior_path: str | Path,
        *,
        config: GPIS3DGSTrainingPriorConfig | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> GPIS3DGSTrainingPriorRegularizer:
        return cls(load_gpis_training_prior(prior_path, device=device, dtype=dtype), config=config)

    def is_active(self, iteration: int) -> bool:
        cfg = self.config
        if iteration < cfg.start_iteration:
            return False
        if cfg.stop_iteration is not None and iteration > cfg.stop_iteration:
            return False
        return (iteration - cfg.start_iteration) % cfg.interval == 0

    def active_weight(self, iteration: int) -> float:
        cfg = self.config
        if not self.is_active(iteration):
            return 0.0
        if cfg.ramp_iterations <= 0:
            return 1.0
        elapsed = iteration - cfg.start_iteration + 1
        return min(1.0, max(0.0, elapsed / cfg.ramp_iterations))

    def compute(
        self,
        gaussians: Any | None = None,
        *,
        iteration: int,
        centers: Tensor | None = None,
        opacities: Tensor | None = None,
        gaussian_normals: Tensor | None = None,
    ) -> GPIS3DGSTrainingPriorStep | None:
        del gaussian_normals
        if not self.is_active(iteration):
            return None
        tensors = extract_3dgs_tensors(gaussians, centers=centers, opacities=opacities)
        count = int(tensors.centers.shape[0])
        sampled_indices = sample_gaussian_indices(
            count,
            max_gaussians=self.config.max_regularized_gaussians,
            mode=self.config.sample_mode,
            seed=self.config.sample_seed,
            iteration=iteration,
            device=tensors.centers.device,
        )
        if sampled_indices.numel() == 0:
            return None

        prior = self._state_for(count=count, device=tensors.centers.device, dtype=tensors.centers.dtype)
        selected = _select_state(prior, sampled_indices)
        zero = tensors.centers.sum() * 0.0
        if tensors.opacities is None:
            opacity_loss = zero
        else:
            alpha = prepare_opacity(tensors.opacities, opacity_is_logit=self.config.opacity_is_logit).to(dtype=tensors.centers.dtype, device=tensors.centers.device)
            alpha = alpha.index_select(0, sampled_indices)
            diff = alpha - selected.opacity_target_alpha
            opacity_values = alpha_loss_values(diff, loss=self.config.opacity_loss, eps=self.config.charbonnier_eps)
            opacity_loss = reduce_loss(selected.opacity_regularization_weight * opacity_values, "mean")

        raw_loss = self.config.opacity_weight * opacity_loss
        weight = self.active_weight(iteration)
        return GPIS3DGSTrainingPriorStep(
            loss=raw_loss * weight,
            raw_loss=raw_loss,
            opacity_loss=opacity_loss,
            gate=selected.gate,
            densify_weight=selected.densify_weight,
            densify_candidate_mask=selected.densify_candidate_mask,
            prune_weight=selected.prune_weight,
            prune_candidate_mask=selected.prune_candidate_mask,
            opacity_target_alpha=selected.opacity_target_alpha,
            opacity_regularization_weight=selected.opacity_regularization_weight,
            sampled_indices=sampled_indices,
            active_weight=weight,
            gaussian_count=count,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> GPIS3DGSTrainingPriorStep | None:
        return self.compute(*args, **kwargs)

    def build_prune_mask(self, step: GPIS3DGSTrainingPriorStep) -> Tensor:
        cfg = self.config
        full_mask = torch.zeros((step.gaussian_count,), dtype=torch.bool, device=step.sampled_indices.device)
        if cfg.max_prune_fraction <= 0.0:
            return full_mask
        candidate = step.prune_candidate_mask.detach() & (step.prune_weight.detach() >= cfg.prune_weight_threshold)
        candidate_indices = torch.nonzero(candidate, as_tuple=False).reshape(-1)
        if candidate_indices.numel() == 0:
            return full_mask
        max_count = int(step.gaussian_count * cfg.max_prune_fraction)
        if max_count < 1:
            return full_mask
        if candidate_indices.numel() > max_count:
            scores = step.prune_weight.detach().index_select(0, candidate_indices)
            candidate_indices = candidate_indices.index_select(0, torch.topk(scores, k=max_count, largest=True).indices)
        full_mask[step.sampled_indices.index_select(0, candidate_indices)] = True
        return full_mask

    def maybe_prune(self, gaussians: Any, step: GPIS3DGSTrainingPriorStep, *, iteration: int, opacities: Tensor | None = None) -> Tensor | None:
        del opacities
        if not self._should_prune(iteration):
            return None
        mask = self.build_prune_mask(step)
        if not bool(mask.any().item()):
            return mask
        prune = getattr(gaussians, "prune_points", None) or getattr(gaussians, "prune_gaussians", None)
        if prune is None or not callable(prune):
            raise AttributeError("gaussians must expose prune_points(mask) or prune_gaussians(mask) for GPIS-prior pruning.")
        with torch.no_grad():
            prune(mask)
        self.apply_prune_mask(mask)
        return mask

    def apply_prune_mask(self, prune_mask: Tensor) -> None:
        """Keep the per-Gaussian prior aligned after external trainer pruning."""
        mask = prune_mask.detach().reshape(-1).to(device=self.state.gate.device, dtype=torch.bool)
        aligned = align_training_prior_state(self.state, count=int(mask.numel()), config=self.config)
        keep = torch.nonzero(~mask, as_tuple=False).reshape(-1)
        self.state = _select_state(aligned, keep)
        self._clear_cache()

    def densification_boost_weights(self, step: GPIS3DGSTrainingPriorStep) -> Tensor:
        weights = torch.ones_like(step.densify_weight.detach())
        if self.config.densification_gradient_boost <= 0.0:
            return weights
        candidate = step.densify_candidate_mask.detach()
        if not bool(candidate.any().item()):
            return weights
        weights[candidate] = 1.0 + self.config.densification_gradient_boost * torch.clamp(step.densify_weight.detach()[candidate], min=0.0)
        return weights

    def maybe_boost_densification_stats(self, gaussians: Any, step: GPIS3DGSTrainingPriorStep, *, iteration: int) -> Tensor | None:
        if not self._should_boost_densification(iteration):
            return None
        gradient_accum = resolve_tensor_attribute(gaussians, ("xyz_gradient_accum",), required=False)
        if gradient_accum is None:
            return None
        weights = self.densification_boost_weights(step)
        boosted = weights > 1.0
        if not bool(boosted.any().item()):
            return weights
        indices = step.sampled_indices.index_select(0, torch.nonzero(boosted, as_tuple=False).reshape(-1))
        multipliers = weights[boosted]
        with torch.no_grad():
            while multipliers.ndim < gradient_accum.ndim:
                multipliers = multipliers.unsqueeze(-1)
            gradient_accum[indices] *= multipliers
        return weights

    def _clear_cache(self) -> None:
        self._cached_state = None
        self._cached_count = None
        self._cached_device = None
        self._cached_dtype = None

    def _state_for(self, *, count: int, device: torch.device, dtype: torch.dtype) -> GPIS3DGSTrainingPriorState:
        if self._cached_state is None or self._cached_count != count or self._cached_device != device or self._cached_dtype != dtype:
            self._cached_state = align_training_prior_state(self.state, count=count, config=self.config).to(device=device, dtype=dtype)
            self._cached_count = count
            self._cached_device = device
            self._cached_dtype = dtype
        return self._cached_state

    def _should_prune(self, iteration: int) -> bool:
        cfg = self.config
        return cfg.prune_interval > 0 and iteration >= cfg.prune_start_iteration and (iteration - cfg.prune_start_iteration) % cfg.prune_interval == 0

    def _should_boost_densification(self, iteration: int) -> bool:
        cfg = self.config
        return cfg.densification_gradient_boost > 0.0 and cfg.densification_boost_interval > 0 and iteration >= cfg.densification_boost_start_iteration and (iteration - cfg.densification_boost_start_iteration) % cfg.densification_boost_interval == 0


def load_gpis_training_prior(path: str | Path, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> GPIS3DGSTrainingPriorState:
    with np.load(path, allow_pickle=False) as data:
        gate = _load_signal(data, ("gate", "confidence", "calibrated_confidence", "raw_gate"), required=True, clip_max=1.0)
        count = int(gate.size)
        densify_weight = _load_signal(data, ("densify_weight",), default=gate, count=count)
        densify_candidate_mask = _load_mask(data, ("densify_candidate_mask",), default=densify_weight > 0.0, count=count)
        prune_weight = _load_signal(data, ("prune_weight",), default=1.0 - gate, count=count)
        prune_candidate_mask = _load_mask(data, ("prune_candidate_mask",), default=prune_weight > 0.0, count=count)
        opacity_target_alpha = _load_signal(data, ("opacity_target_alpha",), default=gate, count=count, clip_max=1.0)
        opacity_regularization_weight = _load_signal(data, ("opacity_regularization_weight",), default=1.0 - gate, count=count)
    return GPIS3DGSTrainingPriorState(
        gate=_to_tensor(gate, device=device, dtype=dtype),
        densify_weight=_to_tensor(densify_weight, device=device, dtype=dtype),
        densify_candidate_mask=torch.as_tensor(densify_candidate_mask, device=device, dtype=torch.bool),
        prune_weight=_to_tensor(prune_weight, device=device, dtype=dtype),
        prune_candidate_mask=torch.as_tensor(prune_candidate_mask, device=device, dtype=torch.bool),
        opacity_target_alpha=_to_tensor(opacity_target_alpha, device=device, dtype=dtype),
        opacity_regularization_weight=_to_tensor(opacity_regularization_weight, device=device, dtype=dtype),
    )


def align_training_prior_state(state: GPIS3DGSTrainingPriorState, *, count: int, config: GPIS3DGSTrainingPriorConfig) -> GPIS3DGSTrainingPriorState:
    if count < 0:
        raise ValueError("count must be non-negative.")
    current = state.gaussian_count
    if current == count:
        return state
    if config.count_mismatch == "error":
        raise ValueError(f"Training-prior Gaussian count {current} does not match current 3DGS count {count}.")
    if config.count_mismatch != "pad":
        raise ValueError("count_mismatch must be 'error' or 'pad'.")
    if count < current:
        return _slice_state(state, count)
    pad_count = count - current
    device = state.gate.device
    dtype = state.gate.dtype
    return GPIS3DGSTrainingPriorState(
        gate=torch.cat([state.gate, torch.full((pad_count,), config.neutral_gate, device=device, dtype=dtype)]),
        densify_weight=torch.cat([state.densify_weight, torch.full((pad_count,), config.neutral_densify_weight, device=device, dtype=dtype)]),
        densify_candidate_mask=torch.cat([state.densify_candidate_mask, torch.zeros((pad_count,), device=device, dtype=torch.bool)]),
        prune_weight=torch.cat([state.prune_weight, torch.zeros((pad_count,), device=device, dtype=dtype)]),
        prune_candidate_mask=torch.cat([state.prune_candidate_mask, torch.zeros((pad_count,), device=device, dtype=torch.bool)]),
        opacity_target_alpha=torch.cat([state.opacity_target_alpha, torch.ones((pad_count,), device=device, dtype=dtype)]),
        opacity_regularization_weight=torch.cat([state.opacity_regularization_weight, torch.full((pad_count,), config.neutral_opacity_weight, device=device, dtype=dtype)]),
    )


def alpha_loss_values(diff: Tensor, *, loss: AlphaLoss, eps: float) -> Tensor:
    if loss == "l1":
        return diff.abs()
    if loss == "l2":
        return diff.square()
    if loss == "charbonnier":
        if eps <= 0.0:
            raise ValueError("charbonnier_eps must be positive.")
        return torch.sqrt(diff.square() + eps**2) - eps
    raise ValueError("opacity_loss must be 'l1', 'l2', or 'charbonnier'.")


def validate_training_prior_config(config: GPIS3DGSTrainingPriorConfig) -> None:
    if config.start_iteration < 0:
        raise ValueError("start_iteration must be non-negative.")
    if config.stop_iteration is not None and config.stop_iteration < config.start_iteration:
        raise ValueError("stop_iteration must be greater than or equal to start_iteration.")
    if config.interval < 1:
        raise ValueError("interval must be positive.")
    if config.ramp_iterations < 0:
        raise ValueError("ramp_iterations must be non-negative.")
    if config.max_regularized_gaussians is not None and config.max_regularized_gaussians < 1:
        raise ValueError("max_regularized_gaussians must be positive when provided.")
    if config.sample_mode not in {"all", "strided"}:
        raise ValueError("sample_mode must be 'all' or 'strided'.")
    if config.count_mismatch not in {"error", "pad"}:
        raise ValueError("count_mismatch must be 'error' or 'pad'.")
    if config.opacity_weight < 0.0:
        raise ValueError("opacity_weight must be non-negative.")
    if config.opacity_loss not in {"l1", "l2", "charbonnier"}:
        raise ValueError("opacity_loss must be 'l1', 'l2', or 'charbonnier'.")
    if config.charbonnier_eps <= 0.0:
        raise ValueError("charbonnier_eps must be positive.")
    if not 0.0 <= config.neutral_gate <= 1.0:
        raise ValueError("neutral_gate must be in [0, 1].")
    if config.neutral_opacity_weight < 0.0 or config.neutral_densify_weight < 0.0:
        raise ValueError("neutral weights must be non-negative.")
    if config.prune_start_iteration < 0 or config.densification_boost_start_iteration < 0:
        raise ValueError("density-control start iterations must be non-negative.")
    if config.prune_interval < 0 or config.densification_boost_interval < 0:
        raise ValueError("density-control intervals must be non-negative.")
    if config.prune_weight_threshold < 0.0:
        raise ValueError("prune_weight_threshold must be non-negative.")
    if not 0.0 <= config.max_prune_fraction <= 1.0:
        raise ValueError("max_prune_fraction must be in [0, 1].")
    if config.densification_gradient_boost < 0.0:
        raise ValueError("densification_gradient_boost must be non-negative.")


def validate_training_prior_state(state: GPIS3DGSTrainingPriorState) -> None:
    count = state.gaussian_count
    for name in ("densify_weight", "prune_weight", "opacity_target_alpha", "opacity_regularization_weight"):
        value = getattr(state, name)
        if value.ndim != 1 or value.numel() != count:
            raise ValueError(f"{name} must have shape ({count},).")
    for name in ("densify_candidate_mask", "prune_candidate_mask"):
        value = getattr(state, name)
        if value.ndim != 1 or value.numel() != count:
            raise ValueError(f"{name} must have shape ({count},).")


def _select_state(state: GPIS3DGSTrainingPriorState, indices: Tensor) -> GPIS3DGSTrainingPriorState:
    return GPIS3DGSTrainingPriorState(
        gate=state.gate.index_select(0, indices),
        densify_weight=state.densify_weight.index_select(0, indices),
        densify_candidate_mask=state.densify_candidate_mask.index_select(0, indices),
        prune_weight=state.prune_weight.index_select(0, indices),
        prune_candidate_mask=state.prune_candidate_mask.index_select(0, indices),
        opacity_target_alpha=state.opacity_target_alpha.index_select(0, indices),
        opacity_regularization_weight=state.opacity_regularization_weight.index_select(0, indices),
    )


def _slice_state(state: GPIS3DGSTrainingPriorState, count: int) -> GPIS3DGSTrainingPriorState:
    indices = torch.arange(count, device=state.gate.device)
    return _select_state(state, indices)


def _load_signal(
    data: np.lib.npyio.NpzFile,
    names: tuple[str, ...],
    *,
    required: bool = False,
    default: np.ndarray | None = None,
    count: int | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    for name in names:
        if name in data.files:
            values = np.asarray(data[name], dtype=np.float64).reshape(-1)
            break
    else:
        if required:
            raise ValueError(f"Training-prior file is missing any of: {', '.join(names)}.")
        if default is None:
            raise ValueError("default is required for optional training-prior signals.")
        values = np.asarray(default, dtype=np.float64).reshape(-1)
    if count is not None and values.size != count:
        raise ValueError(f"Signal {names[0]} has {values.size} values, expected {count}.")
    values = np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, None)
    if clip_max is not None:
        values = np.clip(values, 0.0, clip_max)
    return values


def _load_mask(data: np.lib.npyio.NpzFile, names: tuple[str, ...], *, default: np.ndarray, count: int) -> np.ndarray:
    for name in names:
        if name in data.files:
            values = np.asarray(data[name], dtype=bool).reshape(-1)
            break
    else:
        values = np.asarray(default, dtype=bool).reshape(-1)
    if values.size != count:
        raise ValueError(f"Mask {names[0]} has {values.size} values, expected {count}.")
    return values


def _to_tensor(values: np.ndarray, *, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype)


__all__ = [
    "GPIS3DGSTrainingPriorConfig",
    "GPIS3DGSTrainingPriorRegularizer",
    "GPIS3DGSTrainingPriorState",
    "GPIS3DGSTrainingPriorStep",
    "align_training_prior_state",
    "alpha_loss_values",
    "load_gpis_training_prior",
    "validate_training_prior_config",
    "validate_training_prior_state",
]
