from __future__ import annotations

from pathlib import Path

import numpy as np

from gpis_splatting.cli.augment_depth_normal_supervision import main as augment_depth_normal_supervision_main
from gpis_splatting.depth_normal_supervision import SAMPLE_TYPE_IDS, augment_samples_with_depth_normal_confidence, confidence_to_noise
from gpis_splatting.serialization import read_json, write_json


def test_depth_normal_confidence_samples_noise_and_offsets(tmp_path: Path) -> None:
    scene_dir = write_prepared_depth_scene(tmp_path)
    depth_dir, depth_conf_dir, normal_dir, normal_conf_dir = write_supervision_maps(scene_dir)

    result = augment_samples_with_depth_normal_confidence(
        scene_dir=scene_dir,
        depth_dir=depth_dir,
        depth_confidence_dir=depth_conf_dir,
        normal_dir=normal_dir,
        normal_confidence_dir=normal_conf_dir,
        include_base_samples=False,
        output_samples_path="depth_normal_samples.npz",
        max_pixels_per_frame=4,
        seed=3,
        add_free_space_samples=True,
        free_space_samples_per_depth=1,
        add_normal_offset_samples=True,
        normal_offset_distance=0.1,
        surface_noise_min=0.01,
        surface_noise_max=0.11,
        normal_noise_min=0.02,
        normal_noise_max=0.12,
    )

    with np.load(result["samples_path"], allow_pickle=False) as samples:
        points = samples["points"]
        sdf = samples["sdf"]
        sample_type = samples["sample_type"]
        noise = samples["observation_noise_std"]
        depth_confidence = samples["depth_confidence"]
        sample_confidence = samples["sample_confidence"]

    assert points.shape == (16, 3)
    assert int((sample_type == SAMPLE_TYPE_IDS["depth_surface"]).sum()) == 4
    assert int((sample_type == SAMPLE_TYPE_IDS["depth_free_space"]).sum()) == 4
    assert int((sample_type == SAMPLE_TYPE_IDS["depth_normal_positive"]).sum()) == 4
    assert int((sample_type == SAMPLE_TYPE_IDS["depth_normal_negative"]).sum()) == 4
    assert np.allclose(sdf[sample_type == SAMPLE_TYPE_IDS["depth_surface"]], 0.0)
    assert np.all(sdf[sample_type == SAMPLE_TYPE_IDS["depth_free_space"]] > 0.0)
    assert np.allclose(sdf[sample_type == SAMPLE_TYPE_IDS["depth_normal_positive"]], 0.1)
    assert np.allclose(sdf[sample_type == SAMPLE_TYPE_IDS["depth_normal_negative"]], -0.1)

    surface = sample_type == SAMPLE_TYPE_IDS["depth_surface"]
    assert np.all((depth_confidence >= 0.25) & (depth_confidence <= 1.0))
    assert np.allclose(noise[surface], confidence_to_noise(depth_confidence[surface], min_noise=0.01, max_noise=0.11))
    assert np.all(sample_confidence[sample_type == SAMPLE_TYPE_IDS["depth_normal_positive"]] <= depth_confidence[sample_type == SAMPLE_TYPE_IDS["depth_normal_positive"]])

    positive = points[sample_type == SAMPLE_TYPE_IDS["depth_normal_positive"]]
    negative = points[sample_type == SAMPLE_TYPE_IDS["depth_normal_negative"]]
    assert np.all(positive[:, 2] < 2.0)
    assert np.all(negative[:, 2] > 2.0)
    assert result["report"]["sample_type_counts"]["depth_surface"] == 4


def test_depth_normal_augmentation_merges_base_samples_and_cli(tmp_path: Path) -> None:
    scene_dir = write_prepared_depth_scene(tmp_path)
    depth_dir, depth_conf_dir, normal_dir, normal_conf_dir = write_supervision_maps(scene_dir)
    np.savez_compressed(
        scene_dir / "real_samples.npz",
        points=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        sdf=np.asarray([0.0], dtype=np.float64),
        observation_noise_std=np.asarray([0.03], dtype=np.float64),
        sample_type=np.asarray([SAMPLE_TYPE_IDS["surface"]], dtype=np.int64),
        source_point_index=np.asarray([0], dtype=np.int64),
        camera_index=np.asarray([0], dtype=np.int64),
        ray_distance=np.asarray([1.0], dtype=np.float64),
    )

    augment_depth_normal_supervision_main(
        [
            "--scene-dir",
            str(scene_dir),
            "--depth-dir",
            str(depth_dir),
            "--depth-confidence-dir",
            str(depth_conf_dir),
            "--normal-dir",
            str(normal_dir),
            "--normal-confidence-dir",
            str(normal_conf_dir),
            "--output-samples-path",
            "merged_depth_normal_samples.npz",
            "--max-pixels-per-frame",
            "2",
            "--free-space-samples-per-depth",
            "0",
            "--normal-offset-distance",
            "0.05",
        ]
    )

    output = scene_dir / "merged_depth_normal_samples.npz"
    report = read_json(scene_dir / "merged_depth_normal_samples.json")
    with np.load(output, allow_pickle=False) as samples:
        assert samples["points"].shape[0] == 1 + 2 + 2 + 2
        assert set(samples["sample_type"].tolist()) >= {SAMPLE_TYPE_IDS["surface"], SAMPLE_TYPE_IDS["depth_surface"]}
        assert "depth_confidence" in samples.files
        assert "sample_confidence" in samples.files
    assert report["base_sample_count"] == 1
    assert report["depth_normal_sample_count"] == 6
    assert report["sample_count"] == 7


def write_prepared_depth_scene(tmp_path: Path) -> Path:
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    write_json(
        scene_dir / "real_scene.json",
        {
            "schema_version": 1,
            "scene": "depth_fixture",
            "dataset": "test",
            "source_format": "colmap_text",
            "image_count": 1,
        },
    )
    frame = {
        "index": 0,
        "image_id": "0",
        "file_name": "image_0.png",
        "image_path": "images/image_0.png",
        "width": 4,
        "height": 3,
        "intrinsics": {"model": "PINHOLE", "width": 4, "height": 3, "fx": 2.0, "fy": 2.0, "cx": 1.5, "cy": 1.0, "params": []},
        "camera_to_world": np.eye(4, dtype=np.float64).tolist(),
        "world_to_camera": np.eye(4, dtype=np.float64).tolist(),
    }
    write_json(scene_dir / "cameras.json", {"schema_version": 1, "frames": [frame]})
    write_json(scene_dir / "splits.json", {"schema_version": 1, "train": [0], "test": []})
    return scene_dir


def write_supervision_maps(scene_dir: Path) -> tuple[Path, Path, Path, Path]:
    depth_dir = scene_dir / "depth"
    depth_conf_dir = scene_dir / "depth_confidence"
    normal_dir = scene_dir / "normals"
    normal_conf_dir = scene_dir / "normal_confidence"
    for path in (depth_dir, depth_conf_dir, normal_dir, normal_conf_dir):
        path.mkdir()
    depth = np.full((3, 4), 2.0, dtype=np.float64)
    depth_confidence = np.asarray(
        [
            [1.0, 0.8, 0.6, 0.4],
            [0.3, 0.25, 0.9, 0.7],
            [0.5, 0.45, 0.35, 0.95],
        ],
        dtype=np.float64,
    )
    normals = np.zeros((3, 4, 3), dtype=np.float64)
    normals[..., 2] = -1.0
    normal_confidence = np.full((3, 4), 0.5, dtype=np.float64)
    np.save(depth_dir / "image_0.npy", depth)
    np.save(depth_conf_dir / "image_0.npy", depth_confidence)
    np.save(normal_dir / "image_0.npy", normals)
    np.save(normal_conf_dir / "image_0.npy", normal_confidence)
    return depth_dir, depth_conf_dir, normal_dir, normal_conf_dir
