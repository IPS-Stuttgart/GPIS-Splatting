from __future__ import annotations

from pathlib import Path

import numpy as np

from gpis_splatting.real_bootstrap import SAMPLE_TYPE_IDS
from gpis_splatting.robust_real_bootstrap import (
    RobustSparsePointCloud,
    build_multiview_ray_bootstrap_samples,
    filter_point_cloud_by_error,
    load_colmap_points3d_with_tracks,
)


def test_colmap_points3d_loader_preserves_track_image_ids(tmp_path: Path) -> None:
    points_path = tmp_path / "points3D.txt"
    points_path.write_text(
        "\n".join(
            [
                "# POINT3D_ID X Y Z R G B ERROR TRACK[]",
                "17 1.0 2.0 3.0 255 128 0 0.25 10 2 20 3 30 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cloud = load_colmap_points3d_with_tracks(points_path)

    assert cloud.points.shape == (1, 3)
    assert cloud.point_ids is not None
    assert cloud.point_ids.tolist() == [17]
    assert cloud.errors is not None
    assert cloud.errors.tolist() == [0.25]
    assert cloud.track_image_ids == ((10, 20, 30),)


def test_error_filter_drops_high_reprojection_error_and_keeps_metadata() -> None:
    cloud = RobustSparsePointCloud(
        points=np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=np.float64),
        colors=np.ones((3, 3), dtype=np.float64),
        errors=np.asarray([0.1, 0.2, 5.0], dtype=np.float64),
        point_ids=np.asarray([1, 2, 3], dtype=np.int64),
        track_image_ids=((10,), (20,), (30,)),
    )

    filtered, report = filter_point_cloud_by_error(cloud, max_error=1.0, error_percentile=None)

    assert filtered.points.shape[0] == 2
    assert filtered.point_ids is not None
    assert filtered.point_ids.tolist() == [1, 2]
    assert filtered.track_image_ids == ((10,), (20,))
    assert report["enabled"] is True
    assert report["dropped_count"] == 1


def test_multiview_bootstrap_uses_colmap_track_cameras_once_per_surface() -> None:
    cloud = RobustSparsePointCloud(
        points=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        colors=np.ones((1, 3), dtype=np.float64),
        errors=np.asarray([0.2], dtype=np.float64),
        point_ids=np.asarray([1], dtype=np.int64),
        track_image_ids=((20, 30),),
    )
    frames = [_frame(10, 0, [0.0, 0.0, 0.0]), _frame(20, 1, [1.0, 0.0, 0.0]), _frame(30, 2, [0.0, 1.0, 0.0])]

    samples = build_multiview_ray_bootstrap_samples(
        cloud,
        train_frames=frames,
        free_space_samples_per_point=1,
        free_space_min_fraction=0.5,
        free_space_max_fraction=0.5,
        add_behind_surface_samples=True,
        behind_surface_fraction=1.1,
        max_sample_distance=0.35,
        surface_noise_std=0.03,
        free_space_noise_std=0.08,
        behind_surface_noise_std=0.12,
        max_views_per_point=2,
    )

    assert int((samples["sample_type"] == SAMPLE_TYPE_IDS["surface"]).sum()) == 1
    assert int((samples["sample_type"] == SAMPLE_TYPE_IDS["free_space"]).sum()) == 2
    assert int((samples["sample_type"] == SAMPLE_TYPE_IDS["behind_surface"]).sum()) == 2
    assert samples["points"].shape[0] == 5
    free_space_camera_indices = samples["camera_index"][samples["sample_type"] == SAMPLE_TYPE_IDS["free_space"]]
    assert set(free_space_camera_indices.tolist()) == {1, 2}
    assert samples["ray_view_count"].tolist() == [2, 2, 2, 2, 2]
    assert samples["track_length"].tolist() == [2, 2, 2, 2, 2]


def test_multiview_bootstrap_falls_back_to_nearest_cameras_without_tracks() -> None:
    cloud = RobustSparsePointCloud(
        points=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        colors=np.ones((1, 3), dtype=np.float64),
    )
    frames = [_frame(10, 0, [0.0, 0.0, 0.0]), _frame(20, 1, [0.1, 0.0, 0.0]), _frame(30, 2, [5.0, 0.0, 0.0])]

    samples = build_multiview_ray_bootstrap_samples(
        cloud,
        train_frames=frames,
        free_space_samples_per_point=1,
        free_space_min_fraction=0.5,
        free_space_max_fraction=0.5,
        add_behind_surface_samples=False,
        behind_surface_fraction=1.1,
        max_sample_distance=0.35,
        surface_noise_std=0.03,
        free_space_noise_std=0.08,
        behind_surface_noise_std=0.12,
        max_views_per_point=2,
        visibility_distance_factor=2.0,
    )

    assert int((samples["sample_type"] == SAMPLE_TYPE_IDS["surface"]).sum()) == 1
    assert int((samples["sample_type"] == SAMPLE_TYPE_IDS["free_space"]).sum()) == 2
    free_space_camera_indices = samples["camera_index"][samples["sample_type"] == SAMPLE_TYPE_IDS["free_space"]]
    assert set(free_space_camera_indices.tolist()) == {0, 1}


def _frame(image_id: int, index: int, center: list[float]) -> dict[str, object]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = center
    return {
        "image_id": str(image_id),
        "index": index,
        "camera_to_world": matrix.tolist(),
    }
