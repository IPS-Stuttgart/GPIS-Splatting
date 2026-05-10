from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


TRAINING_PRIOR_REQUIRED_KEYS = (
    "densify_weight",
    "prune_weight",
    "opacity_target_alpha",
    "opacity_regularization_weight",
    "initialization_points",
)


def load_training_prior(path: str | Path) -> dict[str, np.ndarray]:
    """Load an exported calibrated-GPIS training prior NPZ.

    The prior is produced by ``export_gpis_training_prior`` and contains three groups of
    trainer-facing signals:

    - initialization candidates: ``initialization_points`` and confidence/weight metadata;
    - densification/pruning policy: ``densify_weight``, ``densify_candidate_mask``,
      ``prune_weight``, and ``prune_candidate_mask``;
    - opacity regularization targets: ``opacity_target_alpha`` and
      ``opacity_regularization_weight``.
    """
    with np.load(path, allow_pickle=False) as data:
        prior = {key: data[key] for key in data.files}
    missing = sorted(set(TRAINING_PRIOR_REQUIRED_KEYS) - set(prior))
    if missing:
        raise ValueError(f"Training prior {path} is missing required arrays: {', '.join(missing)}.")
    return prior


def initialization_points_from_prior(prior: dict[str, np.ndarray], *, max_count: int | None = None, min_confidence: float = 0.0) -> np.ndarray:
    """Return GPIS-supported candidate positions for initialization or later densification.

    The returned rows are sorted by ``initialization_weight`` when available and filtered
    by ``initialization_confidence`` when present. This lets a trainer seed new Gaussians
    from high-confidence GPIS surface candidates instead of only pruning at the end.
    """
    points = np.asarray(prior["initialization_points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("initialization_points must have shape Nx3.")
    confidence = np.asarray(prior.get("initialization_confidence", np.ones(points.shape[0])), dtype=np.float64).reshape(-1)
    weight = np.asarray(prior.get("initialization_weight", confidence), dtype=np.float64).reshape(-1)
    if confidence.shape[0] != points.shape[0] or weight.shape[0] != points.shape[0]:
        raise ValueError("Initialization confidence/weight arrays must match initialization_points.")
    keep = np.flatnonzero(confidence >= min_confidence)
    ranked = keep[np.argsort(-weight[keep], kind="mergesort")]
    if max_count is not None:
        ranked = ranked[: max(0, int(max_count))]
    return points[ranked]


def initialization_metadata_from_prior(prior: dict[str, np.ndarray], *, max_count: int | None = None, min_confidence: float = 0.0) -> dict[str, np.ndarray]:
    """Return sorted initialization points plus optional source/candidate metadata."""
    points = np.asarray(prior["initialization_points"], dtype=np.float64)
    confidence = np.asarray(prior.get("initialization_confidence", np.ones(points.shape[0])), dtype=np.float64).reshape(-1)
    weight = np.asarray(prior.get("initialization_weight", confidence), dtype=np.float64).reshape(-1)
    keep = np.flatnonzero(confidence >= min_confidence)
    ranked = keep[np.argsort(-weight[keep], kind="mergesort")]
    if max_count is not None:
        ranked = ranked[: max(0, int(max_count))]
    result = {
        "points": points[ranked],
        "confidence": confidence[ranked],
        "weight": weight[ranked],
    }
    for key in ("initialization_source_splat_index", "initialization_candidate_index"):
        if key in prior:
            result[key] = np.asarray(prior[key])[ranked]
    return result


def densification_indices_from_prior(prior: dict[str, np.ndarray], *, max_count: int | None = None, min_weight: float = 0.0) -> np.ndarray:
    """Return Gaussian indices to promote during densification.

    The priority is high calibrated confidence combined with high GPIS uncertainty/evidence,
    as encoded by ``densify_weight``. The optional mask lets the exporter enforce a
    confidence and uncertainty threshold before ranking.
    """
    weights = np.asarray(prior["densify_weight"], dtype=np.float64).reshape(-1)
    mask = np.asarray(prior.get("densify_candidate_mask", np.ones(weights.shape[0], dtype=bool)), dtype=bool).reshape(-1)
    if mask.shape[0] != weights.shape[0]:
        raise ValueError("densify_candidate_mask must match densify_weight.")
    candidates = np.flatnonzero(mask & (weights >= min_weight))
    ranked = candidates[np.argsort(-weights[candidates], kind="mergesort")]
    if max_count is not None:
        ranked = ranked[: max(0, int(max_count))]
    return ranked.astype(np.int64)


def pruning_mask_from_prior(prior: dict[str, np.ndarray], *, min_weight: float = 0.0) -> np.ndarray:
    """Return a boolean mask for low-confidence floaters to suppress or prune."""
    weights = np.asarray(prior["prune_weight"], dtype=np.float64).reshape(-1)
    mask = np.asarray(prior.get("prune_candidate_mask", np.ones(weights.shape[0], dtype=bool)), dtype=bool).reshape(-1)
    if mask.shape[0] != weights.shape[0]:
        raise ValueError("prune_candidate_mask must match prune_weight.")
    return mask & (weights >= min_weight)


def opacity_targets_from_prior(prior: dict[str, np.ndarray], *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Return opacity target alpha and weights as tensors for a training loop."""
    target = torch.as_tensor(np.asarray(prior["opacity_target_alpha"], dtype=np.float32), dtype=dtype, device=device)
    weight = torch.as_tensor(np.asarray(prior["opacity_regularization_weight"], dtype=np.float32), dtype=dtype, device=device)
    if target.shape != weight.shape:
        raise ValueError("opacity_target_alpha and opacity_regularization_weight must have the same shape.")
    return target, weight


def gpis_opacity_regularization_loss(
    alpha: torch.Tensor,
    opacity_target_alpha: torch.Tensor | np.ndarray,
    opacity_regularization_weight: torch.Tensor | np.ndarray,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Softly penalize opacity for geometrically implausible Gaussians.

    This implements the trainer-side loss

    ``weight * (alpha - target_alpha)^2``

    and deliberately keeps low-confidence Gaussians differentiable instead of requiring
    immediate hard deletion.
    """
    target = torch.as_tensor(opacity_target_alpha, dtype=alpha.dtype, device=alpha.device).reshape_as(alpha)
    weight = torch.as_tensor(opacity_regularization_weight, dtype=alpha.dtype, device=alpha.device).reshape_as(alpha)
    penalty = weight * (alpha - target).square()
    if reduction == "none":
        return penalty
    if reduction == "sum":
        return penalty.sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    denom = torch.clamp(weight.sum(), min=torch.finfo(alpha.dtype).eps)
    return penalty.sum() / denom


def training_policy_summary(prior: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compact counts and ranges for logging a GPIS prior inside a trainer."""
    densify = densification_indices_from_prior(prior)
    prune = pruning_mask_from_prior(prior)
    init_points = np.asarray(prior["initialization_points"])
    gate = np.asarray(prior.get("gate", prior.get("per_gaussian_gate", np.empty((0,)))), dtype=np.float64)
    return {
        "gaussian_count": int(np.asarray(prior["densify_weight"]).shape[0]),
        "initialization_candidate_count": int(init_points.shape[0]),
        "densify_candidate_count": int(densify.shape[0]),
        "prune_candidate_count": int(prune.sum()),
        "gate_min": None if gate.size == 0 else float(gate.min()),
        "gate_mean": None if gate.size == 0 else float(gate.mean()),
        "gate_max": None if gate.size == 0 else float(gate.max()),
    }
