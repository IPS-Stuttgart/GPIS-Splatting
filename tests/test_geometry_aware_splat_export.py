from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gpis_splatting.geometry_aware_splat_export import finalize_gpis_aware_splat_export, normal_axis_variance
from gpis_splatting.splats import load_splats


def test_finalize_gpis_aware_splat_export_preserves_geometry(tmp_path: Path) -> None:
    gaussians_path = tmp_path / "gpis_init_gaussians.npz"
    output_splats = tmp_path / "gpis_init_splats.npz"

    centers = np.asarray([[0.0, 0.0, 0.0], [0.2, -0.1, 0.0]], dtype=np.float64)
    colors = np.asarray([[0.2, 0.4, 0.6], [0.8, 0.7, 0.1]], dtype=np.float64)
    opacity = np.asarray([0.25, 0.5], dtype=np.float64)
    scales = np.asarray([[0.01, 0.05, 0.06], [0.02, 0.07, 0.08]], dtype=np.float64)
    rotations = np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    normals = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    confidence = np.asarray([0.9, 0.4], dtype=np.float64)
    np.savez_compressed(
        gaussians_path,
        centers=centers,
        colors=colors,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        normals=normals,
        confidence=confidence,
    )

    result = finalize_gpis_aware_splat_export(gaussians_path=gaussians_path, output_splats_path=output_splats)
    splats = load_splats(str(output_splats))

    assert result["status"]["splat_count"] == 2
    assert splats.normals is not None
    assert splats.scales is not None
    assert splats.rotations is not None
    assert splats.covariances is not None
    assert splats.confidence is not None
    assert torch.allclose(splats.confidence, torch.from_numpy(confidence).to(torch.float64))
    assert torch.allclose(splats.sigma, torch.from_numpy(np.mean(scales[:, 1:3], axis=1)).to(torch.float64))
    assert torch.allclose(splats.tau, torch.from_numpy(-np.log1p(-opacity)).to(torch.float64))
    normal_variance = normal_axis_variance(normals, splats.covariances.numpy())
    assert np.allclose(normal_variance, scales[:, 0] ** 2, atol=1e-10)
    eigvals = torch.linalg.eigvalsh(splats.covariances)
    assert torch.all(eigvals > 0.0)
