from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from gpis_splatting.gpis import GPISModel, load_model
from gpis_splatting.gpis_regularization import (
    GPISRegularizationConfig,
    gpis_regularization_loss,
    normalize_vectors,
    validate_regularization_config,
)

Tensor = torch.Tensor
SampleMode = Literal["all", "strided"]
PredictionDtype = Literal["centers", "model"]


@dataclass(frozen=True)
class GPIS3DGSRegularizerConfig:
    """Iteration schedule and batching for training-time 3DGS GPIS regularization."""

    start_iteration: int = 0
    stop_iteration: int | None = None
    interval: int = 1
    ramp_iterations: int = 0
    max_regularized_gaussians: int | None = 65536
    sample_mode: SampleMode = "strided"
    sample_seed: int = 0
    batch_size: int = 8192
    prediction_dtype: PredictionDtype = "centers"


@dataclass(frozen=True)
class GPIS3DGSDensityControlConfig:
    """Optional confidence-aware hooks for 3DGS density control."""

    prune_start_iteration: int = 0
    prune_interval: int = 0
    prune_confidence_threshold: float = 0.05
    prune_opacity_threshold: float | None = 0.01
    prune_only_low_opacity: bool = True
    max_prune_fraction: float = 0.05
    densification_boost_start_iteration: int = 0
    densification_boost_interval: int = 0
    densification_confidence_threshold: float = 0.35
    densification_min_distance_std: float | None = None
    densification_gradient_boost: float = 0.0


@dataclass(frozen=True)
class GPIS3DGSTensors:
    centers: Tensor
    opacities: Tensor | None = None
    gaussian_normals: Tensor | None = None


@dataclass(frozen=True)
class GPIS3DGSRegularizationStep:
    loss: Tensor
    raw_loss: Tensor
    surface_loss: Tensor
    opacity_loss: Tensor
    normal_loss: Tensor
    confidence: Tensor
    signed_distance: Tensor
    distance_std: Tensor
    field_normal: Tensor
    sampled_indices: Tensor
    active_weight: float
    gaussian_count: int

    @property
    def terms(self) -> dict[str, Tensor]:
        return {
            "loss": self.loss,
            "raw_loss": self.raw_loss,
            "surface": self.surface_loss,
            "opacity": self.opacity_loss,
            "normal": self.normal_loss,
        }

    def log_dict(self, prefix: str = "gpis") -> dict[str, Tensor]:
        return {
            f"{prefix}/loss": self.loss.detach(),
            f"{prefix}/raw_loss": self.raw_loss.detach(),
            f"{prefix}/surface_loss": self.surface_loss.detach(),
            f"{prefix}/opacity_loss": self.opacity_loss.detach(),
            f"{prefix}/normal_loss": self.normal_loss.detach(),
            f"{prefix}/confidence_mean": self.confidence.detach().mean(),
            f"{prefix}/confidence_min": self.confidence.detach().min(),
            f"{prefix}/distance_abs_mean": self.signed_distance.detach().abs().mean(),
            f"{prefix}/distance_std_mean": self.distance_std.detach().mean(),
            f"{prefix}/sampled_gaussians": torch.as_tensor(self.sampled_indices.numel(), device=self.loss.device),
        }


class GPIS3DGSTrainingRegularizer:
    """Adapter that makes a fixed GPIS field usable inside a 3DGS training loop.

    The adapter accepts explicit tensors or a 3DGS-like Gaussian model object. For the reference graphdeco-inria 3DGS implementation, it reads ``get_xyz``, ``get_opacity``, ``get_scaling`` and ``get_rotation``. Optional density-control hooks mutate ``xyz_gradient_accum`` and call ``prune_points(mask)`` when configured.
    """

    def __init__(
        self,
        model: GPISModel,
        *,
        loss_config: GPISRegularizationConfig | None = None,
        schedule_config: GPIS3DGSRegularizerConfig | None = None,
        density_config: GPIS3DGSDensityControlConfig | None = None,
    ) -> None:
        self.model = model
        self.loss_config = GPISRegularizationConfig() if loss_config is None else loss_config
        self.schedule_config = GPIS3DGSRegularizerConfig() if schedule_config is None else schedule_config
        self.density_config = GPIS3DGSDensityControlConfig() if density_config is None else density_config
        self._cached_model: GPISModel | None = None
        self._cached_device: torch.device | None = None
        self._cached_dtype: torch.dtype | None = None
        validate_regularization_config(self.loss_config)
        validate_3dgs_regularizer_config(self.schedule_config)
        validate_3dgs_density_config(self.density_config)

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        *,
        loss_config: GPISRegularizationConfig | None = None,
        schedule_config: GPIS3DGSRegularizerConfig | None = None,
        density_config: GPIS3DGSDensityControlConfig | None = None,
    ) -> "GPIS3DGSTrainingRegularizer":
        model, _metadata = load_model(str(model_path))
        return cls(model, loss_config=loss_config, schedule_config=schedule_config, density_config=density_config)

    def is_active(self, iteration: int) -> bool:
        cfg = self.schedule_config
        if iteration < cfg.start_iteration:
            return False
        if cfg.stop_iteration is not None and iteration > cfg.stop_iteration:
            return False
        return (iteration - cfg.start_iteration) % cfg.interval == 0

    def active_weight(self, iteration: int) -> float:
        cfg = self.schedule_config
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
    ) -> GPIS3DGSRegularizationStep | None:
        """Return a differentiable loss step, or ``None`` when the schedule is inactive."""
        if not self.is_active(iteration):
            return None
        tensors = extract_3dgs_tensors(gaussians, centers=centers, opacities=opacities, gaussian_normals=gaussian_normals)
        sampled_indices = sample_gaussian_indices(
            tensors.centers.shape[0],
            max_gaussians=self.schedule_config.max_regularized_gaussians,
            mode=self.schedule_config.sample_mode,
            seed=self.schedule_config.sample_seed,
            iteration=iteration,
            device=tensors.centers.device,
        )
        if sampled_indices.numel() == 0:
            return None

        sampled_centers = tensors.centers.index_select(0, sampled_indices)
        sampled_opacities = None if tensors.opacities is None else tensors.opacities.reshape(-1).index_select(0, sampled_indices)
        sampled_normals = None if tensors.gaussian_normals is None else tensors.gaussian_normals.index_select(0, sampled_indices)
        result = gpis_regularization_loss(
            self._model_for(sampled_centers),
            sampled_centers,
            opacities=sampled_opacities,
            gaussian_normals=sampled_normals,
            config=self.loss_config,
            batch_size=self.schedule_config.batch_size,
        )
        weight = self.active_weight(iteration)
        return GPIS3DGSRegularizationStep(
            loss=result.loss * weight,
            raw_loss=result.loss,
            surface_loss=result.surface_loss,
            opacity_loss=result.opacity_loss,
            normal_loss=result.normal_loss,
            confidence=result.confidence,
            signed_distance=result.signed_distance,
            distance_std=result.distance_std,
            field_normal=result.field_normal,
            sampled_indices=sampled_indices,
            active_weight=weight,
            gaussian_count=int(tensors.centers.shape[0]),
        )

    def __call__(self, *args: Any, **kwargs: Any) -> GPIS3DGSRegularizationStep | None:
        return self.compute(*args, **kwargs)

    def build_prune_mask(self, step: GPIS3DGSRegularizationStep, *, opacities: Tensor | None = None) -> Tensor:
        """Build a full-size boolean mask compatible with graphdeco 3DGS ``GaussianModel.prune_points``."""
        cfg = self.density_config
        full_mask = torch.zeros((step.gaussian_count,), dtype=torch.bool, device=step.sampled_indices.device)
        if cfg.max_prune_fraction <= 0.0:
            return full_mask
        candidate = step.confidence.detach() < cfg.prune_confidence_threshold
        if cfg.prune_opacity_threshold is not None and opacities is not None:
            sampled_opacities = opacities.reshape(-1).to(device=step.sampled_indices.device).index_select(0, step.sampled_indices).detach()
            opacity_candidate = sampled_opacities <= cfg.prune_opacity_threshold
            candidate = candidate & opacity_candidate if cfg.prune_only_low_opacity else candidate | opacity_candidate
        candidate_indices = torch.nonzero(candidate, as_tuple=False).reshape(-1)
        if candidate_indices.numel() == 0:
            return full_mask
        max_count = int(step.gaussian_count * cfg.max_prune_fraction)
        if max_count < 1:
            return full_mask
        if candidate_indices.numel() > max_count:
            scores = step.confidence.detach().index_select(0, candidate_indices)
            candidate_indices = candidate_indices.index_select(0, torch.topk(scores, k=max_count, largest=False).indices)
        full_mask[step.sampled_indices.index_select(0, candidate_indices)] = True
        return full_mask

    def maybe_prune(self, gaussians: Any, step: GPIS3DGSRegularizationStep, *, iteration: int, opacities: Tensor | None = None) -> Tensor | None:
        """Call the 3DGS model's prune method when the GPIS density-control schedule requests it."""
        if not self._should_prune(iteration):
            return None
        tensors = extract_3dgs_tensors(gaussians, opacities=opacities)
        mask = self.build_prune_mask(step, opacities=tensors.opacities)
        if not bool(mask.any().item()):
            return mask
        prune = getattr(gaussians, "prune_points", None) or getattr(gaussians, "prune_gaussians", None)
        if prune is None or not callable(prune):
            raise AttributeError("gaussians must expose prune_points(mask) or prune_gaussians(mask) for GPIS pruning.")
        with torch.no_grad():
            prune(mask)
        return mask

    def densification_boost_weights(self, step: GPIS3DGSRegularizationStep) -> Tensor:
        cfg = self.density_config
        weights = torch.ones_like(step.confidence.detach())
        if cfg.densification_gradient_boost <= 0.0:
            return weights
        candidate = step.confidence.detach() < cfg.densification_confidence_threshold
        if cfg.densification_min_distance_std is not None:
            candidate = candidate & (step.distance_std.detach() >= cfg.densification_min_distance_std)
        if not bool(candidate.any().item()):
            return weights
        uncertainty = step.distance_std.detach()[candidate]
        baseline = float(cfg.densification_min_distance_std or 0.0)
        normalized = torch.clamp(uncertainty - baseline, min=0.0)
        normalized = normalized / torch.clamp(normalized.max(), min=1e-12)
        weights[candidate] = 1.0 + cfg.densification_gradient_boost * normalized
        return weights

    def maybe_boost_densification_stats(self, gaussians: Any, step: GPIS3DGSRegularizationStep, *, iteration: int) -> Tensor | None:
        """Boost ``xyz_gradient_accum`` for uncertain sampled Gaussians before 3DGS densification."""
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

    def _prediction_dtype(self, centers: Tensor) -> torch.dtype:
        if self.schedule_config.prediction_dtype == "centers":
            return centers.dtype
        return self.model.dtype

    def _model_for(self, centers: Tensor) -> GPISModel:
        dtype = self._prediction_dtype(centers)
        device = centers.device
        if self._cached_model is None or self._cached_device != device or self._cached_dtype != dtype:
            self._cached_model = move_gpis_model(self.model, device=device, dtype=dtype)
            self._cached_device = device
            self._cached_dtype = dtype
        return self._cached_model

    def _should_prune(self, iteration: int) -> bool:
        cfg = self.density_config
        return cfg.prune_interval > 0 and iteration >= cfg.prune_start_iteration and (iteration - cfg.prune_start_iteration) % cfg.prune_interval == 0

    def _should_boost_densification(self, iteration: int) -> bool:
        cfg = self.density_config
        return cfg.densification_gradient_boost > 0.0 and cfg.densification_boost_interval > 0 and iteration >= cfg.densification_boost_start_iteration and (iteration - cfg.densification_boost_start_iteration) % cfg.densification_boost_interval == 0


def extract_3dgs_tensors(
    gaussians: Any | None = None,
    *,
    centers: Tensor | None = None,
    opacities: Tensor | None = None,
    gaussian_normals: Tensor | None = None,
) -> GPIS3DGSTensors:
    if centers is None:
        if gaussians is None:
            raise ValueError("Provide either gaussians or centers.")
        centers = resolve_tensor_attribute(gaussians, ("get_xyz", "xyz", "_xyz", "means3D", "means"))
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("3DGS centers must have shape (n_gaussians, 3).")

    if opacities is None and gaussians is not None:
        opacities = resolve_opacity_tensor(gaussians)
    if opacities is not None:
        opacities = opacities.reshape(-1)
        if opacities.shape[0] != centers.shape[0]:
            raise ValueError("opacities must have one value per Gaussian center.")

    if gaussian_normals is None and gaussians is not None:
        gaussian_normals = resolve_tensor_attribute(gaussians, ("get_normal", "get_normals", "normal", "normals", "_normal", "_normals"), required=False)
        if gaussian_normals is None:
            scaling = resolve_tensor_attribute(gaussians, ("get_scaling", "scaling", "scales", "_scaling"), required=False)
            rotation = resolve_tensor_attribute(gaussians, ("get_rotation", "rotation", "rotations", "_rotation"), required=False)
            if scaling is not None and rotation is not None:
                gaussian_normals = gaussian_normals_from_scale_rotation(scaling, rotation)
    if gaussian_normals is not None:
        if gaussian_normals.shape != centers.shape:
            raise ValueError("gaussian_normals must have shape (n_gaussians, 3).")
        gaussian_normals = normalize_vectors(gaussian_normals)
    return GPIS3DGSTensors(centers=centers, opacities=opacities, gaussian_normals=gaussian_normals)


def resolve_tensor_attribute(obj: Any, names: tuple[str, ...], *, required: bool = True) -> Tensor | None:
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        value = value() if callable(value) else value
        if isinstance(value, torch.Tensor):
            return value
    if required:
        raise AttributeError(f"Object does not expose any tensor attribute among: {', '.join(names)}.")
    return None


def resolve_opacity_tensor(gaussians: Any) -> Tensor | None:
    activated = resolve_tensor_attribute(gaussians, ("get_opacity", "opacity", "opacities", "alpha", "alphas"), required=False)
    if activated is not None:
        return activated
    raw = resolve_tensor_attribute(gaussians, ("_opacity", "opacity_logits"), required=False)
    return None if raw is None else torch.sigmoid(raw)


def gaussian_normals_from_scale_rotation(scaling: Tensor, rotation: Tensor) -> Tensor:
    if scaling.ndim != 2 or scaling.shape[1] != 3:
        raise ValueError("scaling must have shape (n_gaussians, 3).")
    if rotation.ndim != 2 or rotation.shape[1] != 4:
        raise ValueError("rotation must have shape (n_gaussians, 4) with quaternion order (w, x, y, z).")
    rotation_matrix = quaternion_to_rotation_matrix(rotation.to(dtype=scaling.dtype, device=scaling.device))
    local_axis = torch.argmin(scaling, dim=-1)
    gather_index = local_axis[:, None, None].expand(-1, 3, 1)
    normals = torch.gather(rotation_matrix, dim=2, index=gather_index).squeeze(-1)
    return normalize_vectors(normals)


def quaternion_to_rotation_matrix(quaternions: Tensor) -> Tensor:
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternions must have shape (n, 4) with order (w, x, y, z).")
    q = normalize_vectors(quaternions)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def sample_gaussian_indices(count: int, *, max_gaussians: int | None, mode: SampleMode, seed: int, iteration: int, device: torch.device | str) -> Tensor:
    if count < 0:
        raise ValueError("count must be non-negative.")
    if count == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if mode == "all" or max_gaussians is None or count <= max_gaussians:
        return torch.arange(count, dtype=torch.long, device=device)
    if mode != "strided":
        raise ValueError("sample_mode must be 'all' or 'strided'.")
    if max_gaussians < 1:
        raise ValueError("max_gaussians must be positive when provided.")
    step = count / float(max_gaussians)
    base = torch.floor(torch.arange(max_gaussians, dtype=torch.float64, device=device) * step).to(dtype=torch.long)
    offset = (int(seed) + int(iteration)) % count
    return (base + offset) % count


def move_gpis_model(model: GPISModel, *, device: torch.device | str, dtype: torch.dtype) -> GPISModel:
    return GPISModel(
        x_train=model.x_train.to(device=device, dtype=dtype),
        y_train=model.y_train.to(device=device, dtype=dtype),
        alpha=model.alpha.to(device=device, dtype=dtype),
        chol=model.chol.to(device=device, dtype=dtype),
        lengthscale=model.lengthscale,
        variance=model.variance,
        noise_std=model.noise_std,
        mean_constant=model.mean_constant,
        jitter=model.jitter,
        observation_noise_std=None if model.observation_noise_std is None else model.observation_noise_std.to(device=device, dtype=dtype),
    )


def validate_3dgs_regularizer_config(config: GPIS3DGSRegularizerConfig) -> None:
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
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if config.prediction_dtype not in {"centers", "model"}:
        raise ValueError("prediction_dtype must be 'centers' or 'model'.")


def validate_3dgs_density_config(config: GPIS3DGSDensityControlConfig) -> None:
    if config.prune_start_iteration < 0 or config.densification_boost_start_iteration < 0:
        raise ValueError("density-control start iterations must be non-negative.")
    if config.prune_interval < 0 or config.densification_boost_interval < 0:
        raise ValueError("density-control intervals must be non-negative.")
    if not 0.0 <= config.prune_confidence_threshold <= 1.0:
        raise ValueError("prune_confidence_threshold must be in [0, 1].")
    if config.prune_opacity_threshold is not None and not 0.0 <= config.prune_opacity_threshold <= 1.0:
        raise ValueError("prune_opacity_threshold must be in [0, 1] when provided.")
    if not 0.0 <= config.max_prune_fraction <= 1.0:
        raise ValueError("max_prune_fraction must be in [0, 1].")
    if not 0.0 <= config.densification_confidence_threshold <= 1.0:
        raise ValueError("densification_confidence_threshold must be in [0, 1].")
    if config.densification_min_distance_std is not None and config.densification_min_distance_std < 0.0:
        raise ValueError("densification_min_distance_std must be non-negative when provided.")
    if config.densification_gradient_boost < 0.0:
        raise ValueError("densification_gradient_boost must be non-negative.")
