from __future__ import annotations

import numpy as np
import pytest
import torch

from gpis_splatting.colmap_camera_models import project_normalized_points_with_intrinsics
from gpis_splatting.gsplat_adapter import frame_to_gsplat_camera
from gpis_splatting.prepared_colmap_export import build_colmap_cameras
from gpis_splatting.real_pipeline import project_splats_to_frame
from gpis_splatting.real_scene import _pinhole_intrinsics_from_colmap
from gpis_splatting.splats import SplatCloud


def test_colmap_opencv_intrinsics_are_parsed_without_dropping_params() -> None:
    params = [100.0, 120.0, 10.0, 20.0, 0.2, 0.05, 0.01, -0.02]

    fx, fy, cx, cy = _pinhole_intrinsics_from_colmap("OPENCV", params)

    assert (fx, fy, cx, cy) == (100.0, 120.0, 10.0, 20.0)


def test_project_normalized_points_with_opencv_distortion() -> None:
    intrinsics = {
        "model": "OPENCV",
        "fx": 100.0,
        "fy": 120.0,
        "cx": 10.0,
        "cy": 20.0,
        "params": [100.0, 120.0, 10.0, 20.0, 0.2, 0.05, 0.01, -0.02],
    }
    x = np.asarray([0.5], dtype=np.float64)
    y = np.asarray([0.25], dtype=np.float64)

    u, v = project_normalized_points_with_intrinsics(x, y, intrinsics)

    r2 = x * x + y * y
    radial = 0.2 * r2 + 0.05 * r2 * r2
    x_expected = x + x * radial + 2.0 * 0.01 * x * y + (-0.02) * (r2 + 2.0 * x * x)
    y_expected = y + y * radial + 2.0 * (-0.02) * x * y + 0.01 * (r2 + 2.0 * y * y)
    assert np.allclose(u, 100.0 * x_expected + 10.0)
    assert np.allclose(v, 120.0 * y_expected + 20.0)
    assert not np.allclose(u, 100.0 * x + 10.0)
    assert not np.allclose(v, 120.0 * y + 20.0)


def test_cpu_splat_projection_uses_colmap_distortion() -> None:
    splats = SplatCloud(
        centers=torch.tensor([[1.0, 0.5, 2.0]], dtype=torch.float64),
        colors=torch.ones((1, 3), dtype=torch.float64),
        tau=torch.ones(1, dtype=torch.float64),
        sigma=torch.ones(1, dtype=torch.float64),
        is_surface=torch.ones(1, dtype=torch.bool),
    )
    intrinsics = {
        "model": "OPENCV",
        "width": 200,
        "height": 160,
        "fx": 100.0,
        "fy": 120.0,
        "cx": 10.0,
        "cy": 20.0,
        "params": [100.0, 120.0, 10.0, 20.0, 0.2, 0.05, 0.01, -0.02],
    }
    frame = {"world_to_camera": np.eye(4).tolist(), "intrinsics": intrinsics}

    projected = project_splats_to_frame(splats, frame, projection_convention="opencv", near_plane=1e-4)

    expected_u, expected_v = project_normalized_points_with_intrinsics(
        np.asarray([0.5], dtype=np.float64),
        np.asarray([0.25], dtype=np.float64),
        intrinsics,
    )
    assert projected["valid"][0]
    assert np.allclose(projected["centers_px"][0], [expected_u[0], expected_v[0]])
    assert not np.allclose(projected["centers_px"][0], [60.0, 50.0])


def test_prepared_colmap_export_preserves_distorted_camera_model() -> None:
    params = (100.0, 120.0, 10.0, 20.0, 0.2, 0.05, 0.01, -0.02)
    frames = [
        {
            "width": 200,
            "height": 160,
            "intrinsics": {
                "model": "OPENCV",
                "width": 200,
                "height": 160,
                "fx": 100.0,
                "fy": 120.0,
                "cx": 10.0,
                "cy": 20.0,
                "params": list(params),
            },
        }
    ]

    cameras, camera_ids = build_colmap_cameras(frames)

    assert camera_ids == [1]
    assert cameras[0]["model"] == "OPENCV"
    assert cameras[0]["params"] == params


def test_gsplat_camera_rejects_distorted_colmap_camera_until_supported() -> None:
    frame = {
        "width": 200,
        "height": 160,
        "world_to_camera": np.eye(4).tolist(),
        "intrinsics": {
            "model": "OPENCV",
            "width": 200,
            "height": 160,
            "fx": 100.0,
            "fy": 120.0,
            "cx": 10.0,
            "cy": 20.0,
            "params": [100.0, 120.0, 10.0, 20.0, 0.2, 0.05, 0.01, -0.02],
        },
    }

    with pytest.raises(ValueError, match="distortion"):
        frame_to_gsplat_camera(frame, projection_convention="opencv", device="cpu", dtype="float64")
