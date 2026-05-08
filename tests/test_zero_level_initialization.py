from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gpis_splatting.cli.initialize_gpis_zero_level_splats import main as initialize_gpis_zero_level_splats_main
from gpis_splatting.gpis_backends import DenseExactGPISBackend
from gpis_splatting.splats import SplatCloud, load_splats, save_splats
from gpis_splatting.zero_level_initialization import ZeroLevelInitializationConfig, initialize_zero_level_splats, write_zero_level_3dgs_ply


def test_zero_level_initialization_projects_to_plane_and_writes_anisotropic_fields() -> None:
    backend, seed_points, seed_colors = _fit_plane_backend()
    result = initialize_zero_level_splats(
        backend,
        seed_points=seed_points,
        seed_colors=seed_colors,
        config=ZeroLevelInitializationConfig(
            num_candidates=80,
            target_count=12,
            seed=4,
            projection_iterations=5,
            max_projection_step=0.12,
            surface_band=0.04,
            min_confidence=0.001,
            min_gradient_norm=1e-5,
            nms_radius=0.04,
            tangent_scale=0.05,
            normal_scale=0.008,
            batch_size=256,
        ),
    )

    splats = result.splats
    assert 1 <= splats.centers.shape[0] <= 12
    assert torch.max(torch.abs(splats.centers[:, 2])) < 0.08
    assert splats.normals is not None
    assert splats.scales is not None
    assert splats.rotations is not None
    assert splats.covariances is not None
    assert splats.confidence is not None
    assert torch.all(torch.abs(splats.normals[:, 2]) > 0.65)
    assert torch.allclose(splats.scales[:, 0], torch.full_like(splats.scales[:, 0], 0.05))
    assert torch.allclose(splats.scales[:, 1], torch.full_like(splats.scales[:, 1], 0.05))
    assert torch.allclose(splats.scales[:, 2], torch.full_like(splats.scales[:, 2], 0.008))
    eigvals = torch.linalg.eigvalsh(splats.covariances)
    assert torch.all(eigvals > 0.0)
    assert torch.all((splats.confidence >= 0.0) & (splats.confidence <= 1.0))
    assert result.report["initializer"] == "gpis_zero_level"
    assert result.report["selected_splat_count"] == int(splats.centers.shape[0])


def test_zero_level_splats_roundtrip_and_3dgs_ply_export(tmp_path: Path) -> None:
    backend, seed_points, seed_colors = _fit_plane_backend()
    result = initialize_zero_level_splats(
        backend,
        seed_points=seed_points,
        seed_colors=seed_colors,
        config=ZeroLevelInitializationConfig(num_candidates=60, target_count=8, seed=2, surface_band=0.04, min_confidence=0.001, nms_radius=0.05, batch_size=128),
    )
    splats_path = tmp_path / "zero_splats.npz"
    save_splats(str(splats_path), result.splats)
    loaded = load_splats(str(splats_path))
    assert loaded.scales is not None
    assert loaded.rotations is not None
    assert loaded.covariances is not None
    assert loaded.confidence is not None
    assert torch.allclose(loaded.scales, result.splats.scales)
    assert torch.allclose(loaded.rotations, result.splats.rotations)
    assert torch.allclose(loaded.covariances, result.splats.covariances)
    assert torch.allclose(loaded.confidence, result.splats.confidence)

    ply_path = tmp_path / "point_cloud.ply"
    write_zero_level_3dgs_ply(ply_path, loaded)
    data = ply_path.read_bytes()
    header = data[: data.index(b"end_header")].decode("ascii")
    assert "property float scale_0" in header
    assert "property float scale_1" in header
    assert "property float scale_2" in header
    assert "property float rot_0" in header
    assert f"element vertex {loaded.centers.shape[0]}" in header


def test_initialize_gpis_zero_level_splats_cli(tmp_path: Path) -> None:
    backend, seed_points, seed_colors = _fit_plane_backend()
    model_path = tmp_path / "model.npz"
    seed_splats_path = tmp_path / "seed_splats.npz"
    output_splats = tmp_path / "initialized_splats.npz"
    output_ply = tmp_path / "initialized.ply"
    backend.save(model_path)
    save_splats(
        str(seed_splats_path),
        SplatCloud(
            centers=seed_points,
            colors=seed_colors,
            tau=torch.ones((seed_points.shape[0],), dtype=torch.float64),
            sigma=torch.full((seed_points.shape[0],), 0.02, dtype=torch.float64),
            is_surface=torch.ones((seed_points.shape[0],), dtype=torch.bool),
        ),
    )

    initialize_gpis_zero_level_splats_main(
        [
            "--model-path",
            str(model_path),
            "--seed-splats-path",
            str(seed_splats_path),
            "--output-splats",
            str(output_splats),
            "--output-ply",
            str(output_ply),
            "--num-candidates",
            "60",
            "--target-count",
            "6",
            "--surface-band",
            "0.04",
            "--min-confidence",
            "0.001",
            "--nms-radius",
            "0.05",
            "--batch-size",
            "128",
        ]
    )

    loaded = load_splats(str(output_splats))
    assert output_splats.exists()
    assert output_splats.with_name("initialized_splats_confidence_gate.npz").exists()
    assert output_splats.with_name("initialized_splats_report.json").exists()
    assert output_ply.exists()
    assert loaded.scales is not None
    assert loaded.rotations is not None
    assert loaded.centers.shape[0] <= 6


def _fit_plane_backend() -> tuple[DenseExactGPISBackend, torch.Tensor, torch.Tensor]:
    grid = torch.linspace(-0.5, 0.5, 3, dtype=torch.float64)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    surface = torch.cat((xy, torch.zeros((xy.shape[0], 1), dtype=torch.float64)), dim=1)
    front = surface + torch.tensor([0.0, 0.0, 0.18], dtype=torch.float64)
    back = surface - torch.tensor([0.0, 0.0, 0.18], dtype=torch.float64)
    x_train = torch.cat((surface, front, back), dim=0)
    y_train = torch.cat(
        (
            torch.zeros(surface.shape[0], dtype=torch.float64),
            torch.full((front.shape[0],), 0.18, dtype=torch.float64),
            torch.full((back.shape[0],), -0.18, dtype=torch.float64),
        )
    )
    colors = torch.stack((surface[:, 0] + 0.5, surface[:, 1] + 0.5, torch.full((surface.shape[0],), 0.5, dtype=torch.float64)), dim=1).clamp(0.0, 1.0)
    backend = DenseExactGPISBackend.fit(x_train, y_train, lengthscale=0.35, variance=1.0, noise_std=0.01, jitter=1e-6)
    return backend, surface, colors
