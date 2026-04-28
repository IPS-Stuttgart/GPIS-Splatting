from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

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


def _write_image(path: Path, *, value: int) -> None:
    data = np.full((6, 8, 3), value, dtype=np.uint8)
    Image.fromarray(data, mode="RGB").save(path)


def _translated_identity(x: float, y: float, z: float) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [x, y, z]
    return matrix.tolist()
