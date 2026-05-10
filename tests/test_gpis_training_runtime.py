from __future__ import annotations

import numpy as np
import torch

from gpis_splatting.gpis_training_runtime import (
    densification_indices_from_prior,
    gpis_opacity_regularization_loss,
    initialization_metadata_from_prior,
    initialization_points_from_prior,
    opacity_targets_from_prior,
    pruning_mask_from_prior,
    training_policy_summary,
)


def test_initialization_points_are_confidence_filtered_and_weight_sorted() -> None:
    prior = {
        "initialization_points": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        "initialization_confidence": np.asarray([0.8, 0.95, 0.4], dtype=np.float32),
        "initialization_weight": np.asarray([0.2, 0.9, 1.0], dtype=np.float32),
        "initialization_source_splat_index": np.asarray([10, 11, 12], dtype=np.int64),
        "initialization_candidate_index": np.asarray([0, 1, 2], dtype=np.int64),
    }

    points = initialization_points_from_prior(prior, min_confidence=0.75)
    metadata = initialization_metadata_from_prior(prior, min_confidence=0.75)

    assert points.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert metadata["source_splat_index"].tolist() == [11, 10]
    assert metadata["candidate_index"].tolist() == [1, 0]


def test_densification_indices_and_pruning_mask_use_weights_and_masks() -> None:
    prior = {
        "densify_weight": np.asarray([0.1, 0.8, 0.7, 0.9], dtype=np.float32),
        "densify_candidate_mask": np.asarray([True, True, False, True]),
        "prune_weight": np.asarray([0.0, 0.3, 0.6, 0.2], dtype=np.float32),
        "prune_candidate_mask": np.asarray([False, True, True, True]),
    }

    assert densification_indices_from_prior(prior, min_weight=0.5).tolist() == [3, 1]
    assert densification_indices_from_prior(prior, max_count=1, min_weight=0.5).tolist() == [3]
    assert pruning_mask_from_prior(prior, min_weight=0.25).tolist() == [False, True, True, False]


def test_opacity_regularization_loss_is_weighted_and_differentiable() -> None:
    alpha = torch.tensor([0.8, 0.4, 0.2], dtype=torch.float32, requires_grad=True)
    target = np.asarray([0.5, 0.4, 0.0], dtype=np.float32)
    weight = np.asarray([2.0, 0.0, 1.0], dtype=np.float32)

    loss = gpis_opacity_regularization_loss(alpha, target, weight)
    expected = (2.0 * (0.8 - 0.5) ** 2 + 1.0 * (0.2 - 0.0) ** 2) / 3.0

    assert torch.isclose(loss, torch.tensor(expected, dtype=torch.float32), atol=1e-6)
    loss.backward()
    assert alpha.grad is not None
    assert alpha.grad[1].item() == 0.0


def test_opacity_targets_and_policy_summary() -> None:
    prior = {
        "densify_weight": np.asarray([0.1, 0.9], dtype=np.float32),
        "densify_candidate_mask": np.asarray([False, True]),
        "prune_weight": np.asarray([0.8, 0.0], dtype=np.float32),
        "prune_candidate_mask": np.asarray([True, False]),
        "opacity_target_alpha": np.asarray([0.2, 0.7], dtype=np.float32),
        "opacity_regularization_weight": np.asarray([1.0, 0.5], dtype=np.float32),
        "initialization_points": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        "gate": np.asarray([0.1, 0.9], dtype=np.float32),
    }

    target, weight = opacity_targets_from_prior(prior)
    summary = training_policy_summary(prior)

    assert torch.allclose(target, torch.tensor([0.2, 0.7]))
    assert torch.allclose(weight, torch.tensor([1.0, 0.5]))
    assert summary["initialization_candidate_count"] == 1
    assert summary["densify_candidate_count"] == 1
    assert summary["prune_candidate_count"] == 1
    assert summary["gate_mean"] == 0.5
