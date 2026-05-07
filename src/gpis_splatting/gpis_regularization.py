from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from gpis_splatting.gpis import GPISModel, GPISPrediction, rbf_kernel, surface_band_probability
from gpis_splatting.splats import SplatCloud

Tensor = torch.Tensor
Reduction = Literal["mean", "sum", "none"]


@dataclass(frozen=True)
class GPISRegularizationConfig:
    epsilon: float = 0.08
    surface_weight: float = 1.0
    opacity_weight: float = 0.1
    normal_weight: float = 0.05
    surface_confidence_floor: float = 0.05
    distance_scale: float | None = None
    min_std: float = 1e-4
    charbonnier_eps: float = 1e-6
    opacity_is_logit: bool = False
    detach_confidence: bool = True
    detach_field_normal: bool = True
    reduction: Reduction = "mean"


@dataclass(frozen=True)
class GPISRegularizationResult:
    loss: Tensor
    surface_loss: Tensor
    opacity_loss: Tensor
    normal_loss: Tensor
    confidence: Tensor
    signed_distance: Tensor
    distance_std: Tensor
    field_normal: Tensor

    @property
    def terms(self) -> dict[str, Tensor]:
        return {"surface": self.surface_loss, "opacity": self.opacity_loss, "normal": self.normal_loss}


def predict_gpis_differentiable(model: GPISModel, x_query: Tensor, *, batch_size: int = 8192) -> GPISPrediction:
    """Predict a fixed dense GPIS while preserving gradients w.r.t. query points."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    x_query = x_query.to(dtype=model.dtype, device=model.device)
    if x_query.ndim != 2:
        raise ValueError("x_query must have shape (n_queries, n_dims).")
    if x_query.shape[1] != model.x_train.shape[1]:
        raise ValueError("x_query dimensionality must match model.x_train.")
    if x_query.shape[0] == 0:
        return GPISPrediction(
            mean=torch.empty((0,), dtype=model.dtype, device=model.device),
            variance=torch.empty((0,), dtype=model.dtype, device=model.device),
            gradient=torch.empty((0, model.x_train.shape[1]), dtype=model.dtype, device=model.device),
        )

    x_train = model.x_train.detach()
    alpha = model.alpha.detach()
    chol = model.chol.detach()
    means: list[Tensor] = []
    variances: list[Tensor] = []
    gradients: list[Tensor] = []
    for start in range(0, x_query.shape[0], batch_size):
        x_batch = x_query[start : start + batch_size]
        k_x = rbf_kernel(x_batch, x_train, model.lengthscale, model.variance)
        mean = model.mean_constant + k_x @ alpha
        solved = torch.cholesky_solve(k_x.T, chol)
        variance = torch.as_tensor(model.variance, dtype=model.dtype, device=model.device) - torch.sum(k_x.T * solved, dim=0)
        variance = torch.clamp(variance, min=1e-12)
        diff = x_train[None, :, :] - x_batch[:, None, :]
        gradient = torch.sum((k_x * alpha[None, :])[:, :, None] * diff, dim=1) / (model.lengthscale**2)
        means.append(mean)
        variances.append(variance)
        gradients.append(gradient)
    return GPISPrediction(mean=torch.cat(means), variance=torch.cat(variances), gradient=torch.cat(gradients))


def gpis_regularization_loss(
    model: GPISModel,
    centers: Tensor,
    *,
    opacities: Tensor | None = None,
    gaussian_normals: Tensor | None = None,
    config: GPISRegularizationConfig | None = None,
    batch_size: int = 8192,
) -> GPISRegularizationResult:
    """Differentiable GPIS terms for use inside a Gaussian-splat training loss."""
    cfg = GPISRegularizationConfig() if config is None else config
    validate_regularization_config(cfg)
    prediction = predict_gpis_differentiable(model, centers, batch_size=batch_size)
    confidence = surface_band_probability(prediction, cfg.epsilon, min_std=cfg.min_std)
    loss_weight = confidence.detach() if cfg.detach_confidence else confidence
    distance_scale = float(cfg.distance_scale if cfg.distance_scale is not None else cfg.epsilon)
    surface_weight = torch.clamp(cfg.surface_confidence_floor + (1.0 - cfg.surface_confidence_floor) * loss_weight, 0.0, 1.0)
    surface_loss = reduce_loss(surface_weight * charbonnier(prediction.distance, scale=distance_scale, eps=cfg.charbonnier_eps), cfg.reduction)

    zero = torch.zeros((), dtype=prediction.mean.dtype, device=prediction.mean.device)
    if opacities is None:
        opacity_loss = zero
    else:
        opacity_values = prepare_opacity(opacities, opacity_is_logit=cfg.opacity_is_logit).to(dtype=prediction.mean.dtype, device=prediction.mean.device)
        if opacity_values.shape != prediction.mean.shape:
            raise ValueError("opacities must have one value per Gaussian center.")
        opacity_loss = reduce_loss((1.0 - loss_weight) * opacity_values, cfg.reduction)

    field_normal = normalize_vectors(prediction.gradient)
    normal_ref = field_normal.detach() if cfg.detach_field_normal else field_normal
    if gaussian_normals is None:
        normal_loss = zero
    else:
        gaussian_normals = gaussian_normals.to(dtype=prediction.mean.dtype, device=prediction.mean.device)
        if gaussian_normals.shape != prediction.gradient.shape:
            raise ValueError("gaussian_normals must have the same shape as centers.")
        alignment = torch.abs(torch.sum(normalize_vectors(gaussian_normals) * normal_ref, dim=-1)).clamp(0.0, 1.0)
        normal_loss = reduce_loss(loss_weight * (1.0 - alignment), cfg.reduction)

    total = cfg.surface_weight * surface_loss + cfg.opacity_weight * opacity_loss + cfg.normal_weight * normal_loss
    return GPISRegularizationResult(total, surface_loss, opacity_loss, normal_loss, confidence, prediction.distance, prediction.distance_std, field_normal)


def gpis_regularization_for_splats(
    model: GPISModel,
    splats: SplatCloud,
    *,
    gaussian_normals: Tensor | None = None,
    config: GPISRegularizationConfig | None = None,
    batch_size: int = 8192,
) -> GPISRegularizationResult:
    opacity = torch.clamp(1.0 - torch.exp(-torch.clamp(splats.tau, min=0.0)), 0.0, 1.0)
    return gpis_regularization_loss(model, splats.centers, opacities=opacity, gaussian_normals=gaussian_normals, config=config, batch_size=batch_size)


def charbonnier(values: Tensor, *, scale: float, eps: float) -> Tensor:
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    normalized = values / scale
    return scale * (torch.sqrt(normalized.square() + eps**2) - eps)


def prepare_opacity(opacities: Tensor, *, opacity_is_logit: bool) -> Tensor:
    values = opacities.reshape(-1)
    return torch.sigmoid(values) if opacity_is_logit else torch.clamp(values, 0.0, 1.0)


def normalize_vectors(vectors: Tensor, *, eps: float = 1e-12) -> Tensor:
    return vectors / torch.clamp(torch.linalg.norm(vectors, dim=-1, keepdim=True), min=eps)


def reduce_loss(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError("reduction must be 'mean', 'sum', or 'none'.")


def validate_regularization_config(config: GPISRegularizationConfig) -> None:
    if config.epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if config.distance_scale is not None and config.distance_scale <= 0.0:
        raise ValueError("distance_scale must be positive when provided.")
    if config.min_std <= 0.0:
        raise ValueError("min_std must be positive.")
    if not 0.0 <= config.surface_confidence_floor <= 1.0:
        raise ValueError("surface_confidence_floor must be in [0, 1].")
    if config.charbonnier_eps <= 0.0:
        raise ValueError("charbonnier_eps must be positive.")
    if min(config.surface_weight, config.opacity_weight, config.normal_weight) < 0.0:
        raise ValueError("regularization weights must be non-negative.")
    if config.reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
