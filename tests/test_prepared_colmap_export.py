from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from gpis_splatting.cli.export_prepared_scene_to_colmap_3dgs import main as export_prepared_scene_to_colmap_3dgs_main
from gpis_splatting.cli.map_3dgs_renders_to_prepared_scene import main as map_3dgs_renders_to_prepared_scene_main
from gpis_splatting.cli.prepare_real_scene import main as prepare_real_scene_main
from gpis_splatting.prepared_colmap_export import export_prepared_scene_to_colmap_3dgs, rotation_matrix_to_quaternion
from gpis_splatting.serialization import read_json, write_json


def test_export_prepared_scene_to_colmap_3dgs_roundtrips_cameras(tmp_path: Path) -> None:
    scene_dir = write_prepared_scene_fixture(tmp_path / "real_scenes" / "toy_scene")
    output_dir = tmp_path / "colmap_for_3dgs"

    result = export_prepared_scene_to_colmap_3dgs(
        scene_dir=scene_dir,
        output_dir=output_dir,
        split="all",
        points_path="real_splats.npz",
        max_points=None,
    )

    assert result["status"]["frame_count"] == 2
    assert result["status"]["point_count"] == 3
    assert result["status"]["point_source"] == "splat_npz"
    assert (output_dir / "images" / "frame.png").exists()
    assert (output_dir / "images" / "000002_frame.png").exists()

    cameras_text = (output_dir / "sparse" / "0" / "cameras.txt").read_text(encoding="utf-8")
    images_text = (output_dir / "sparse" / "0" / "images.txt").read_text(encoding="utf-8")
    points_text = (output_dir / "sparse" / "0" / "points3D.txt").read_text(encoding="utf-8")
    assert "1 PINHOLE 8 6 10 11 4 3" in cameras_text
    assert "1 1 0 0 0 0 0 0 1 frame.png\n\n" in images_text
    assert "2 1 0 0 0 -1 0 0 1 000002_frame.png\n\n" in images_text
    assert "1 0 0 1 255 0 0 1" in points_text
    assert "3 0.20000000000000001 0 1.2 0 0 255 1" in points_text
    render_map = (output_dir / "render_name_map.csv").read_text(encoding="utf-8")
    assert "render_index,render_name,colmap_image_id,colmap_image_name,prepared_frame_index,prepared_file_name,prepared_image_path,split" in render_map
    assert "0,00000.png,1,frame.png,0,frame.png,images/frame.png,all" in render_map
    assert "1,00001.png,2,000002_frame.png,1,frame.png,images/duplicate.png,all" in render_map

    roundtrip_root = tmp_path / "roundtrip"
    prepare_real_scene_main(
        [
            "--input-dir",
            str(output_dir),
            "--scene",
            "roundtrip_scene",
            "--output-root",
            str(roundtrip_root),
            "--input-format",
            "colmap_text",
            "--train-view-count",
            "1",
        ]
    )
    roundtrip_frames = {frame["file_name"]: frame for frame in read_json(roundtrip_root / "roundtrip_scene" / "cameras.json")["frames"]}
    assert np.allclose(roundtrip_frames["frame.png"]["camera_to_world"], np.eye(4))
    assert np.isclose(roundtrip_frames["000002_frame.png"]["camera_to_world"][0][3], 1.0)


def test_export_prepared_scene_to_colmap_3dgs_cli_subsamples_points(tmp_path: Path) -> None:
    scene_dir = write_prepared_scene_fixture(tmp_path / "real_scenes" / "toy_scene")
    output_dir = tmp_path / "colmap_train_only"

    export_prepared_scene_to_colmap_3dgs_main(
        [
            "--scene-dir",
            str(scene_dir),
            "--output-dir",
            str(output_dir),
            "--split",
            "train",
            "--points-path",
            "real_splats.npz",
            "--max-points",
            "2",
            "--seed",
            "3",
        ]
    )

    status = read_json(output_dir / "export_status.json")
    images_text = (output_dir / "sparse" / "0" / "images.txt").read_text(encoding="utf-8")
    point_rows = [line for line in (output_dir / "sparse" / "0" / "points3D.txt").read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    assert status["frame_count"] == 1
    assert status["point_count"] == 2
    assert Path(status["render_name_map_path"]).name == "render_name_map.csv"
    assert "frame.png" in images_text
    assert "000002_frame.png" not in images_text
    assert len(point_rows) == 2
    assert (output_dir / "export_report.md").exists()


def test_map_3dgs_renders_to_prepared_scene_cli_writes_prepared_layout(tmp_path: Path) -> None:
    scene_dir = write_prepared_scene_fixture(tmp_path / "real_scenes" / "toy_scene")
    colmap_dir = tmp_path / "colmap_for_rendering"
    renders_dir = tmp_path / "renders"
    predictions_dir = tmp_path / "mapped_predictions"
    renders_dir.mkdir()

    export_prepared_scene_to_colmap_3dgs(scene_dir=scene_dir, output_dir=colmap_dir, split="all", points_path="real_splats.npz", max_points=None)
    write_image(renders_dir / "00000.png", value=120)
    write_image(renders_dir / "00001.png", value=180)

    map_3dgs_renders_to_prepared_scene_main(
        [
            "--map-path",
            str(colmap_dir / "render_name_map.csv"),
            "--renders-dir",
            str(renders_dir),
            "--output-dir",
            str(predictions_dir),
        ]
    )

    assert read_first_red_pixel(predictions_dir / "images" / "frame.png") == 120
    assert read_first_red_pixel(predictions_dir / "images" / "duplicate.png") == 180
    status = read_json(predictions_dir / "mapped_render_images_status.json")
    assert status["mapped_count"] == 2
    assert status["missing_count"] == 0


def test_rotation_matrix_to_quaternion_outputs_colmap_order() -> None:
    identity = np.eye(3, dtype=np.float64)
    assert np.allclose(rotation_matrix_to_quaternion(identity), (1.0, 0.0, 0.0, 0.0))

    rotation_z_180 = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(np.abs(rotation_matrix_to_quaternion(rotation_z_180)), (0.0, 0.0, 0.0, 1.0))


def write_prepared_scene_fixture(scene_dir: Path) -> Path:
    (scene_dir / "images").mkdir(parents=True)
    write_image(scene_dir / "images" / "frame.png", value=60)
    write_image(scene_dir / "images" / "duplicate.png", value=90)
    frame0 = {
        "index": 0,
        "image_id": "10",
        "file_name": "frame.png",
        "image_path": "images/frame.png",
        "width": 8,
        "height": 6,
        "camera_id": "1",
        "intrinsics": {"model": "PINHOLE", "width": 8, "height": 6, "fx": 10.0, "fy": 11.0, "cx": 4.0, "cy": 3.0, "params": []},
        "camera_to_world": np.eye(4, dtype=np.float64).tolist(),
        "world_to_camera": np.eye(4, dtype=np.float64).tolist(),
    }
    world_to_camera1 = np.eye(4, dtype=np.float64)
    world_to_camera1[0, 3] = -1.0
    camera_to_world1 = np.linalg.inv(world_to_camera1)
    frame1 = {
        **frame0,
        "index": 1,
        "image_id": "11",
        "file_name": "frame.png",
        "image_path": "images/duplicate.png",
        "camera_to_world": camera_to_world1.tolist(),
        "world_to_camera": world_to_camera1.tolist(),
    }
    write_json(
        scene_dir / "real_scene.json",
        {
            "schema_version": 1,
            "scene": "toy_scene",
            "dataset": "fixture",
            "source_format": "fixture",
            "image_count": 2,
            "train_view_count": 1,
            "test_view_count": 1,
        },
    )
    write_json(scene_dir / "cameras.json", {"schema_version": 1, "frames": [frame0, frame1]})
    write_json(scene_dir / "splits.json", {"schema_version": 1, "train": [0], "test": [1]})
    np.savez_compressed(
        scene_dir / "real_splats.npz",
        centers=np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.1], [0.2, 0.0, 1.2]], dtype=np.float64),
        colors=np.asarray([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    )
    return scene_dir


def write_image(path: Path, *, value: int) -> None:
    data = np.full((6, 8, 3), value, dtype=np.uint8)
    Image.fromarray(data, mode="RGB").save(path)


def read_first_red_pixel(path: Path) -> int:
    with Image.open(path) as image:
        return int(np.asarray(image)[0, 0, 0])
