from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gpis_splatting.external_3dgs import load_3dgs_ply, opacity_to_alpha
from gpis_splatting.gpis import fit_dense_gpis
from gpis_splatting.gpis_initialization import (
    GPISAwareInitializationConfig,
    initialize_gpis_aware_gaussians,
    write_3dgs_initialization_ply,
)


def test_gpis_aware_initialization_projects_and_orients_plane_candidates() -> None:
    model = fit_dense_gpis(
        torch.from_numpy(_plane_training_points()),
        torch.from_numpy(_plane_training_sdf()),
        lengthscale=0.35,
        noise_std=0.01,
    )
    seeds = np.asarray(
        [
            [-0.2, -0.2, 0.04],
            [0.0, -0.2, -0.03],
            [0.2, -0.2, 0.02],
            [-0.2, 0.1, -0.02],
            [0.1, 0.2, 0.03],
        ],
        dtype=np.float64,
    )
    colors = np.linspace(0.2, 0.8, seeds.shape[0] * 3, dtype=np.float64).reshape(seeds.shape[0], 3)
    config = GPISAwareInitializationConfig(
        target_count=4,
        proposals_per_seed=1,
        projection_iterations=5,
        epsilon=0.05,
        max_abs_distance=0.05,
        normal_scale=0.01,
        tangent_scale=0.05,
        min_separation=0.03,
        seed=3,
    )

    initialization = initialize_gpis_aware_gaussians(model, seeds, seed_colors=colors, config=config)

    assert 1 <= initialization.count <= 4
    assert np.max(np.abs(initialization.centers[:, 2])) < 0.06
    assert initialization.scales.shape == (initialization.count, 3)
    assert np.all(initialization.scales[:, 0] < initialization.scales[:, 1])
    assert np.allclose(np.linalg.norm(initialization.rotations, axis=1), 1.0)
    assert np.all((initialization.opacity >= config.min_opacity) & (initialization.opacity <= config.max_opacity))
    assert initialization.field_scores["selected"].sum() == initialization.count


def test_gpis_aware_initialization_writes_3dgs_ply(tmp_path: Path) -> None:
    model = fit_dense_gpis(
        torch.from_numpy(_plane_training_points()),
        torch.from_numpy(_plane_training_sdf()),
        lengthscale=0.35,
        noise_std=0.01,
    )
    seeds = np.asarray([[-0.1, 0.0, 0.02], [0.1, 0.0, -0.02]], dtype=np.float64)
    colors = np.asarray([[0.1, 0.2, 0.3], [0.7, 0.6, 0.5]], dtype=np.float64)
    initialization = initialize_gpis_aware_gaussians(
        model,
        seeds,
        seed_colors=colors,
        config=GPISAwareInitializationConfig(
            target_count=2,
            proposals_per_seed=0,
            projection_iterations=4,
            epsilon=0.05,
            normal_scale=0.01,
            tangent_scale=0.05,
        ),
    )

    ply_path = tmp_path / "gpis_init_3dgs.ply"
    write_3dgs_initialization_ply(ply_path, initialization, sh_degree=0)
    ply = load_3dgs_ply(ply_path)

    assert ply.vertex_count == initialization.count
    names = set(ply.vertices.dtype.names or ())
    required = {"x", "y", "z", "nx", "ny", "nz", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    assert required.issubset(names)
    assert np.allclose(np.exp(ply.vertices["scale_0"].astype(np.float64)), initialization.scales[:, 0])
    assert np.allclose(opacity_to_alpha(ply.vertices["opacity"].astype(np.float64), opacity_mode="logit"), initialization.opacity, atol=1e-6)


def _plane_training_points() -> np.ndarray:
    xy = np.asarray([[-0.35, -0.35], [0.0, -0.35], [0.35, -0.35], [-0.35, 0.0], [0.0, 0.0], [0.35, 0.0], [-0.35, 0.35], [0.0, 0.35], [0.35, 0.35]], dtype=np.float64)
    rows = []
    for z in (0.0, 0.12, -0.12):
        rows.append(np.column_stack([xy, np.full((xy.shape[0],), z, dtype=np.float64)]))
    return np.vstack(rows)


def _plane_training_sdf() -> np.ndarray:
    return np.concatenate([np.zeros((9,), dtype=np.float64), np.full((9,), 0.12, dtype=np.float64), np.full((9,), -0.12, dtype=np.float64)])
