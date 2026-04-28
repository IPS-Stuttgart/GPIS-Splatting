from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gpis_splatting.cli.bootstrap_real_gpis import main as bootstrap_real_gpis_main
from gpis_splatting.cli.evaluate_real_renders import main as evaluate_real_renders_main
from gpis_splatting.cli.prepare_real_scene import main as prepare_real_scene_main
from gpis_splatting.cli.validate_real_scene import main as validate_real_scene_main
from gpis_splatting.real_scene import build_sparse_split
from gpis_splatting.serialization import read_json, write_json


def test_prepare_validate_and_evaluate_transforms_scene(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    for index in range(4):
        _write_image(images / f"image_{index}.png", value=40 + index * 30)
    write_json(
        dataset / "transforms.json",
        {
            "camera_angle_x": 0.7,
            "frames": [
                {
                    "file_path": f"images/image_{index}.png",
                    "transform_matrix": _translated_identity(float(index), 0.0, 0.0),
                }
                for index in range(4)
            ],
        },
    )

    root = tmp_path / "real_scenes"
    prepare_real_scene_main(
        [
            "--input-dir",
            str(dataset),
            "--scene",
            "tiny_sparse",
            "--output-root",
            str(root),
            "--train-view-count",
            "2",
        ]
    )
    validate_real_scene_main(["--scene", "tiny_sparse", "--prepared-root", str(root)])

    scene_dir = root / "tiny_sparse"
    scene_meta = read_json(scene_dir / "real_scene.json")
    cameras = read_json(scene_dir / "cameras.json")
    splits = read_json(scene_dir / "splits.json")
    assert scene_meta["image_count"] == 4
    assert scene_meta["source_format"] == "transforms"
    assert splits["train"] == [0, 3]
    assert splits["test"] == [1, 2]
    assert len(cameras["frames"]) == 4
    assert (scene_dir / "images" / "image_0.png").exists()

    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for index in splits["test"]:
        target = np.asarray(Image.open(scene_dir / cameras["frames"][index]["image_path"]).convert("RGB"), dtype=np.uint8)
        prediction = np.clip(target.astype(np.int16) + 2, 0, 255).astype(np.uint8)
        Image.fromarray(prediction, mode="RGB").save(predictions / cameras["frames"][index]["file_name"])

    target_manifest = tmp_path / "target.json"
    write_json(
        target_manifest,
        {
            "name": "tiny_target",
            "dataset": "tiny",
            "primary_baseline": "vanilla",
            "reference_baselines": {"vanilla": {"psnr": 20.0, "ssim": 0.5, "lpips_vgg": 0.4}},
        },
    )
    evaluate_real_renders_main(
        [
            "--scene",
            "tiny_sparse",
            "--prepared-root",
            str(root),
            "--predictions-dir",
            str(predictions),
            "--method-name",
            "toy_method",
            "--benchmark-target",
            str(target_manifest),
        ]
    )

    eval_dir = scene_dir / "evaluations"
    metrics = pd.read_csv(eval_dir / "toy_method_test_image_metrics.csv")
    summary = pd.read_csv(eval_dir / "toy_method_test_summary.csv").iloc[0]
    assert len(metrics) == 2
    assert metrics["psnr"].min() > 35.0
    assert metrics["ssim"].between(-1.0, 1.0).all()
    assert summary["image_count"] == 2
    assert summary["missing_count"] == 0
    assert summary["psnr_delta_vs_target_baseline"] > 15.0
    assert (eval_dir / "toy_method_test_report.md").exists()


def test_prepare_colmap_text_scene(tmp_path: Path) -> None:
    dataset = tmp_path / "colmap_dataset"
    images = dataset / "images"
    sparse = dataset / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    _write_image(images / "a.png", value=70)
    _write_image(images / "b.png", value=90)
    (sparse / "cameras.txt").write_text(
        "\n".join(
            [
                "# Camera list",
                "1 PINHOLE 8 6 10 11 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "\n".join(
            [
                "# Image list",
                "1 1 0 0 0 0 0 0 1 a.png",
                "0 0 -1",
                "2 1 0 0 0 1 0 0 1 b.png",
                "0 0 -1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    root = tmp_path / "real_scenes"
    prepare_real_scene_main(
        [
            "--input-dir",
            str(dataset),
            "--scene",
            "colmap_sparse",
            "--output-root",
            str(root),
            "--input-format",
            "colmap_text",
            "--train-view-count",
            "1",
        ]
    )

    scene_meta = read_json(root / "colmap_sparse" / "real_scene.json")
    cameras = read_json(root / "colmap_sparse" / "cameras.json")
    splits = read_json(root / "colmap_sparse" / "splits.json")
    assert scene_meta["source_format"] == "colmap_text"
    assert splits["train"] == [0]
    assert splits["test"] == [1]
    assert cameras["frames"][0]["intrinsics"]["fx"] == 10.0
    assert cameras["frames"][0]["intrinsics"]["fy"] == 11.0
    assert cameras["frames"][1]["camera_to_world"][0][3] == -1.0


def test_sparse_split_is_deterministic() -> None:
    assert build_sparse_split(5, 3)["train"] == [0, 2, 4]
    assert build_sparse_split(2, 12)["train"] == [0, 1]


def test_bootstrap_real_gpis_from_colmap_points(tmp_path: Path) -> None:
    root = _prepare_colmap_scene_with_points(tmp_path)

    bootstrap_real_gpis_main(
        [
            "--scene",
            "colmap_bootstrap",
            "--prepared-root",
            str(root),
            "--point-source",
            "colmap",
            "--free-space-samples-per-point",
            "2",
            "--add-behind-surface-samples",
            "true",
            "--max-points",
            "10",
        ]
    )

    scene_dir = root / "colmap_bootstrap"
    samples = np.load(scene_dir / "real_samples.npz")
    splats = np.load(scene_dir / "real_splats.npz")
    report = read_json(scene_dir / "real_bootstrap_report.json")
    assert samples["points"].shape == (12, 3)
    assert samples["sdf"].shape == (12,)
    assert set(samples["sample_type"].tolist()) == {0, 1, 2}
    assert int((samples["sample_type"] == 0).sum()) == 3
    assert int((samples["sample_type"] == 1).sum()) == 6
    assert int((samples["sample_type"] == 2).sum()) == 3
    assert np.all(samples["sdf"][samples["sample_type"] == 1] > 0.0)
    assert np.all(samples["sdf"][samples["sample_type"] == 2] < 0.0)
    assert splats["centers"].shape == (3, 3)
    assert np.allclose(splats["colors"][0], [1.0, 0.0, 0.0])
    assert report["surface_point_count"] == 3
    assert report["sample_count"] == 12


def test_bootstrap_real_gpis_from_ply_points(tmp_path: Path) -> None:
    root = _prepare_colmap_scene_with_points(tmp_path)
    scene_dir = root / "colmap_bootstrap"
    ply_path = tmp_path / "points.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0.0 0.0 1.0 10 20 30",
                "0.2 0.0 1.2 40 50 60",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bootstrap_real_gpis_main(
        [
            "--scene-dir",
            str(scene_dir),
            "--point-source",
            "ply",
            "--point-path",
            str(ply_path),
            "--free-space-samples-per-point",
            "1",
            "--add-behind-surface-samples",
            "false",
            "--output-prefix",
            "ply",
        ]
    )

    samples = np.load(scene_dir / "ply_samples.npz")
    splats = np.load(scene_dir / "ply_splats.npz")
    assert samples["points"].shape == (4, 3)
    assert set(samples["sample_type"].tolist()) == {0, 1}
    assert splats["centers"].shape == (2, 3)
    assert np.allclose(splats["colors"][0], [10 / 255.0, 20 / 255.0, 30 / 255.0])


def _write_image(path: Path, *, value: int) -> None:
    data = np.full((6, 8, 3), value, dtype=np.uint8)
    Image.fromarray(data, mode="RGB").save(path)


def _translated_identity(x: float, y: float, z: float) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [x, y, z]
    return matrix.tolist()


def _prepare_colmap_scene_with_points(tmp_path: Path) -> Path:
    dataset = tmp_path / "colmap_points_dataset"
    images = dataset / "images"
    sparse = dataset / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    _write_image(images / "a.png", value=70)
    _write_image(images / "b.png", value=90)
    (sparse / "cameras.txt").write_text(
        "\n".join(
            [
                "# Camera list",
                "1 PINHOLE 8 6 10 11 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "\n".join(
            [
                "# Image list",
                "1 1 0 0 0 0 0 0 1 a.png",
                "0 0 -1",
                "2 1 0 0 0 1 0 0 1 b.png",
                "0 0 -1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text(
        "\n".join(
            [
                "# 3D point list",
                "1 0.0 0.0 1.0 255 0 0 0.1 1 0",
                "2 0.5 0.0 1.0 0 255 0 0.1 1 0",
                "3 0.0 0.5 1.5 0 0 255 0.1 2 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    root = tmp_path / "real_scenes"
    prepare_real_scene_main(
        [
            "--input-dir",
            str(dataset),
            "--scene",
            "colmap_bootstrap",
            "--output-root",
            str(root),
            "--input-format",
            "colmap_text",
            "--train-view-count",
            "1",
        ]
    )
    return root
