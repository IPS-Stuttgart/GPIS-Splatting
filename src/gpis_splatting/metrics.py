from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .gpis import GPISPrediction
from .renderer import load_image


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def iou(pred_inside: np.ndarray, target_inside: np.ndarray) -> float:
    intersection = np.logical_and(pred_inside, target_inside).sum()
    union = np.logical_or(pred_inside, target_inside).sum()
    return float(intersection / max(union, 1))


def brier_score(prob: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((prob - labels.astype(np.float64)) ** 2))


def _flatten_calibration_inputs(prob: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if prob.shape != labels.shape:
        raise ValueError(f"prob and labels must have the same shape, got {prob.shape} and {labels.shape}")
    if not np.isfinite(prob).all():
        raise ValueError("prob must contain only finite values")
    if not np.isfinite(labels).all():
        raise ValueError("labels must contain only finite values")
    return np.clip(prob, 0.0, 1.0), labels


def expected_calibration_error(prob: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Fixed-width probability-bin expected calibration error.

    This is the conventional ECE definition: partition [0, 1] into fixed
    confidence intervals, compute the confidence/accuracy gap per non-empty
    interval, and weight each gap by that interval's sample fraction. The last
    bin includes probability 1.0.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    prob, labels = _flatten_calibration_inputs(prob, labels)
    if prob.size == 0:
        return 0.0

    bin_ids = np.minimum((prob * bins).astype(np.int64), bins - 1)
    ece = 0.0
    total = prob.shape[0]
    for bin_index in range(bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        confidence = float(prob[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += (int(mask.sum()) / total) * abs(confidence - accuracy)
    return float(ece)


def equal_count_calibration_error(prob: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Adaptive/equal-count calibration error retained for diagnostics."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    prob, labels = _flatten_calibration_inputs(prob, labels)
    if prob.size == 0:
        return 0.0

    order = np.argsort(prob)
    prob = prob[order]
    labels = labels[order]
    chunks = np.array_split(np.arange(prob.shape[0]), bins)
    ece = 0.0
    total = prob.shape[0]
    for chunk in chunks:
        if chunk.size == 0:
            continue
        confidence = float(prob[chunk].mean())
        accuracy = float(labels[chunk].mean())
        ece += (chunk.size / total) * abs(confidence - accuracy)
    return float(ece)


def gaussian_nll(residual: np.ndarray, variance: np.ndarray) -> float:
    variance = np.clip(variance, 1e-8, None)
    return float(0.5 * np.mean((residual**2) / variance + np.log(2.0 * math.pi * variance)))


def psnr(path_pred: str | Path, path_ref: str | Path) -> float:
    pred = load_image(path_pred)
    ref = load_image(path_ref)
    mse = float(np.mean((pred - ref) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def gpis_metric_row(
    prediction: GPISPrediction,
    true_sdf: torch.Tensor,
    *,
    render_dir: str | Path | None = None,
) -> dict[str, float | str]:
    pred_np = prediction.mean.detach().cpu().numpy()
    true_np = true_sdf.detach().cpu().numpy()
    inside_prob = prediction.inside_probability.detach().cpu().numpy()
    pred_inside = pred_np <= 0.0
    true_inside = true_np <= 0.0

    distance = prediction.distance.detach().cpu().numpy()
    distance_var = prediction.distance_std.detach().cpu().numpy() ** 2
    row: dict[str, float | str] = {
        "rmse_sdf": rmse(pred_np, true_np),
        "iou_inside": iou(pred_inside, true_inside),
        "nll_distance": gaussian_nll(true_np - distance, distance_var),
        "brier_inside": brier_score(inside_prob, true_inside),
        "ece_inside": expected_calibration_error(inside_prob, true_inside, bins=10),
        "ece_inside_equal_count": equal_count_calibration_error(inside_prob, true_inside, bins=10),
    }

    if render_dir is not None:
        render_path = Path(render_dir)
        for view in ("front", "side", "top"):
            ref = render_path / f"render_reference_{view}.png"
            plain = render_path / f"render_plain_{view}.png"
            gated = render_path / f"render_gpis_{view}.png"
            feedback = render_path / f"render_feedback_{view}.png"
            if ref.exists() and plain.exists():
                row[f"psnr_plain_{view}"] = psnr(plain, ref)
            if ref.exists() and gated.exists():
                row[f"psnr_gpis_{view}"] = psnr(gated, ref)
            if ref.exists() and feedback.exists():
                row[f"psnr_feedback_{view}"] = psnr(feedback, ref)
    return row


def save_metrics_csv(path: str | Path, row: dict[str, float | str]) -> None:
    pd.DataFrame([row]).to_csv(path, index=False)

