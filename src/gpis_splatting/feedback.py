from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .gpis import GPISModel, fit_dense_gpis, predict_gpis, surface_band_probability
from .splats import SplatCloud, gpis_gate_for_splats

Tensor = torch.Tensor
FeedbackSelector = Literal["gate", "uncertainty", "uncertainty_diverse"]
FEEDBACK_SELECTORS: tuple[FeedbackSelector, ...] = ("gate", "uncertainty", "uncertainty_diverse")

FEEDBACK_TRACE_FIELDS = (
    "iteration",
    "selector",
    "eligible_splats",
    "selected_splats",
    "pseudo_splats_total",
    "gate_mean",
    "gate_max",
    "score_mean",
    "score_max",
    "selected_gate_mean",
    "selected_score_mean",
    "selected_distance_std_mean",
    "train_points",
)


@dataclass
class FeedbackResult:
    model: GPISModel
    base_gate: Tensor
    feedback_gate: Tensor
    selected_mask: Tensor
    trace: list[dict[str, float | int | str]]


def refine_gpis_with_splat_feedback(
    base_model: GPISModel,
    splats: SplatCloud,
    epsilon: float,
    *,
    iterations: int = 2,
    pseudo_points_per_iteration: int = 80,
    min_gate: float = 0.55,
    pseudo_noise_std: float | None = None,
    selector: FeedbackSelector = "gate",
    diversity_radius: float = 0.16,
    batch_size: int = 4096,
) -> FeedbackResult:
    """Close the GPIS/splat loop by feeding high-confidence splats back as surface constraints.

    The one-way gate treats GPIS as fixed and only scales splat optical thickness. This loop keeps that
    gate, then promotes selected splats to heteroscedastic zero-level pseudo observations and refits GPIS.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if pseudo_points_per_iteration < 1:
        raise ValueError("pseudo_points_per_iteration must be positive.")
    if not 0.0 <= min_gate <= 1.0:
        raise ValueError("min_gate must be in [0, 1].")
    if pseudo_noise_std is not None and pseudo_noise_std <= 0.0:
        raise ValueError("pseudo_noise_std must be positive when provided.")
    if selector not in FEEDBACK_SELECTORS:
        raise ValueError(f"Unknown feedback selector '{selector}'. Expected one of {', '.join(FEEDBACK_SELECTORS)}.")
    if diversity_radius <= 0.0:
        raise ValueError("diversity_radius must be positive.")

    base_gate = gpis_gate_for_splats(splats, base_model, epsilon, batch_size=batch_size)
    current_model = base_model
    selected_mask = torch.zeros(splats.centers.shape[0], dtype=torch.bool)
    pseudo_noise_by_splat = torch.full((splats.centers.shape[0],), float("nan"), dtype=base_model.dtype)
    trace: list[dict[str, float | int | str]] = []
    noise_floor = float(pseudo_noise_std if pseudo_noise_std is not None else base_model.noise_std)

    for iteration in range(1, iterations + 1):
        gate, distance_std = _feedback_quantities(current_model, splats, epsilon, batch_size)
        eligible = (gate >= min_gate) & ~selected_mask
        score = _selector_score(selector, gate, distance_std, splats)
        if selector == "uncertainty_diverse":
            selected_this = _diverse_topk_mask(
                score,
                splats.centers,
                eligible,
                pseudo_points_per_iteration,
                diversity_radius,
            )
        else:
            selected_this = _topk_mask(score, eligible, pseudo_points_per_iteration)
        selected_count = int(selected_this.sum().item())

        if selected_count > 0:
            selected_gate = torch.clamp(gate[selected_this], min=0.05)
            pseudo_noise_by_splat[selected_this] = noise_floor / torch.sqrt(selected_gate)
            selected_mask |= selected_this
            current_model = _fit_with_selected_splats(base_model, splats, selected_mask, pseudo_noise_by_splat)
            selected_gate_mean = float(gate[selected_this].mean().item())
            selected_score_mean = float(score[selected_this].mean().item())
            selected_distance_std_mean = float(distance_std[selected_this].mean().item())
        else:
            selected_gate_mean = 0.0
            selected_score_mean = 0.0
            selected_distance_std_mean = 0.0

        trace.append(
            {
                "iteration": iteration,
                "selector": selector,
                "eligible_splats": int(eligible.sum().item()),
                "selected_splats": selected_count,
                "pseudo_splats_total": int(selected_mask.sum().item()),
                "gate_mean": float(gate.mean().item()),
                "gate_max": float(gate.max().item()),
                "score_mean": float(score.mean().item()),
                "score_max": float(score.max().item()),
                "selected_gate_mean": selected_gate_mean,
                "selected_score_mean": selected_score_mean,
                "selected_distance_std_mean": selected_distance_std_mean,
                "train_points": int(current_model.x_train.shape[0]),
            }
        )
        if selected_count == 0:
            break

    feedback_gate = gpis_gate_for_splats(splats, current_model, epsilon, batch_size=batch_size)
    return FeedbackResult(current_model, base_gate, feedback_gate, selected_mask, trace)


def save_feedback_trace(path: str | Path, trace: list[dict[str, float | int | str]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEEDBACK_TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(trace)


def _topk_mask(score: Tensor, eligible: Tensor, k: int) -> Tensor:
    selected = torch.zeros_like(eligible)
    eligible_count = int(eligible.sum().item())
    if eligible_count == 0:
        return selected

    masked_score = score.clone()
    masked_score[~eligible] = -torch.inf
    selected_indices = torch.topk(masked_score, k=min(k, eligible_count), largest=True).indices
    selected[selected_indices] = True
    return selected


def _diverse_topk_mask(score: Tensor, centers: Tensor, eligible: Tensor, k: int, radius: float) -> Tensor:
    selected = torch.zeros_like(eligible)
    eligible_indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    if eligible_indices.numel() == 0:
        return selected

    ranked_indices = eligible_indices[torch.argsort(score[eligible_indices], descending=True)]
    selected_indices: list[int] = []
    for index in ranked_indices.tolist():
        if len(selected_indices) >= k:
            break
        if not selected_indices:
            selected_indices.append(index)
            continue

        selected_centers = centers[torch.tensor(selected_indices, dtype=torch.long, device=centers.device)]
        distance = torch.linalg.norm(centers[index].to(dtype=selected_centers.dtype) - selected_centers, dim=-1)
        if bool(torch.all(distance >= radius)):
            selected_indices.append(index)

    if selected_indices:
        selected[torch.tensor(selected_indices, dtype=torch.long, device=selected.device)] = True
    return selected


def _feedback_quantities(model: GPISModel, splats: SplatCloud, epsilon: float, batch_size: int) -> tuple[Tensor, Tensor]:
    prediction = predict_gpis(model, splats.centers, batch_size=batch_size)
    gate = surface_band_probability(prediction, epsilon)
    distance_std = torch.nan_to_num(prediction.distance_std, nan=0.0, posinf=1e6, neginf=0.0)
    return gate, torch.clamp(distance_std, min=0.0)


def _selector_score(selector: FeedbackSelector, gate: Tensor, distance_std: Tensor, splats: SplatCloud) -> Tensor:
    optical_mass = torch.clamp(splats.tau.to(dtype=gate.dtype), min=0.0)
    if selector == "gate":
        return gate * optical_mass
    return gate * distance_std.to(dtype=gate.dtype) * optical_mass


def _fit_with_selected_splats(
    base_model: GPISModel,
    splats: SplatCloud,
    selected_mask: Tensor,
    pseudo_noise_by_splat: Tensor,
) -> GPISModel:
    pseudo_x = splats.centers[selected_mask].to(dtype=base_model.dtype)
    pseudo_y = torch.zeros(pseudo_x.shape[0], dtype=base_model.dtype)
    base_noise = _base_observation_noise(base_model)
    pseudo_noise = pseudo_noise_by_splat[selected_mask].to(dtype=base_model.dtype)

    x_train = torch.cat((base_model.x_train, pseudo_x), dim=0)
    y_train = torch.cat((base_model.y_train, pseudo_y), dim=0)
    observation_noise = torch.cat((base_noise, pseudo_noise), dim=0)
    return fit_dense_gpis(
        x_train,
        y_train,
        lengthscale=base_model.lengthscale,
        variance=base_model.variance,
        noise_std=base_model.noise_std,
        observation_noise_std=observation_noise,
        mean_constant=base_model.mean_constant,
        jitter=base_model.jitter,
    )


def _base_observation_noise(model: GPISModel) -> Tensor:
    if model.observation_noise_std is not None:
        return model.observation_noise_std.to(dtype=model.dtype)
    return torch.full_like(model.y_train, model.noise_std)
