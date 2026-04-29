from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gpis_splatting.real_download import write_scaled_nerfstudio_transforms
from gpis_splatting.real_workflow import run_real_evaluation
from gpis_splatting.serialization import read_json, write_json


def test_scaled_nerfstudio_transform_filters_downloaded_images(tmp_path: Path) -> None:
    source = tmp_path / "transforms_fullres.json"
    output = tmp_path / "transforms.json"
    write_json(
        source,
        {
            "fl_x": 80.0,
            "fl_y": 96.0,
            "cx": 40.0,
            "cy": 48.0,
            "w": 80,
            "h": 96,
            "frames": [
                {"file_path": "./images/frame_000.png", "transform_matrix": _translated_identity(0.0, 0.0, 0.0)},
                {"file_path": "./images/frame_001.png", "transform_matrix": _translated_identity(0.0, 0.0, 0.0)},
            ],
        },
    )

    report = write_scaled_nerfstudio_transforms(
        source_path=source,
        output_path=output,
        image_dir="images_4",
        image_scale=4,
        available_images={"frame_001.png"},
    )

    data = read_json(output)
    assert data["fl_x"] == 20.0
    assert data["fl_y"] == 24.0
    assert data["cx"] == 10.0
    assert data["cy"] == 12.0
    assert data["w"] == 20
    assert data["h"] == 24
    assert len(data["frames"]) == 1
    assert data["frames"][0]["file_path"] == "./images_4/frame_001.png"
    assert report["original_frame_count"] == 2
    assert report["kept_frame_count"] == 1
    assert report["missing_frame_count"] == 1


def test_run_real_evaluation_workflow_on_local_nerfstudio_fixture(tmp_path: Path) -> None:
    source = _write_tiny_nerfstudio_source(tmp_path / "source", image_scale=2)
    prepared_root = tmp_path / "real_scenes"

    result = run_real_evaluation(
        scene="tiny_real_workflow",
        prepared_root=prepared_root,
        source_dir=source,
        download_dataset=False,
        image_scale=2,
        train_view_count=2,
        max_points=4,
        max_train_points=6,
        seed=3,
        lengthscale=0.5,
        noise_std=0.05,
        splat_sigmas=(0.04,),
        epsilons=(0.2,),
        gate_floors=(0.0, 0.25),
        max_frames=1,
    )

    comparison = pd.read_csv(result["comparison_path"])
    status = read_json(result["status_path"])
    assert len(comparison) == 3
    assert set(comparison["use_gpis_gate"].tolist()) == {False, True}
    assert comparison["image_count"].min() == 1
    assert np.isfinite(comparison["mean_psnr"]).all()
    assert np.isfinite(comparison["mean_ssim"]).all()
    assert (result["scene_dir"] / "real_gpis_model.npz").exists()
    assert (result["scene_dir"] / "evaluations" / "real_evaluation_report.md").exists()
    floored = comparison[comparison["gate_floor"] == 0.25].iloc[0]
    assert floored["gate_min"] >= 0.25
    assert status["row_count"] == 3
    assert status["best_psnr"] is not None


def _write_tiny_nerfstudio_source(root: Path, *, image_scale: int) -> Path:
    image_dir = root / f"images_{image_scale}"
    image_dir.mkdir(parents=True)
    width, height = 8, 6
    for index in range(4):
        data = np.zeros((height, width, 3), dtype=np.uint8)
        data[..., 0] = 30 + index * 20
        data[..., 1] = np.arange(height, dtype=np.uint8)[:, None] * 20
        data[..., 2] = np.arange(width, dtype=np.uint8)[None, :] * 20
        Image.fromarray(data, mode="RGB").save(image_dir / f"frame_{index:03d}.png")

    write_json(
        root / "transforms.json",
        {
            "fl_x": 10.0,
            "fl_y": 10.0,
            "cx": width / 2.0,
            "cy": height / 2.0,
            "w": width,
            "h": height,
            "frames": [
                {
                    "file_path": f"./images_{image_scale}/frame_{index:03d}.png",
                    "transform_matrix": _translated_identity(0.03 * index, 0.0, 0.0),
                }
                for index in range(4)
            ],
        },
    )
    (root / "sparse_pc.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "-0.1 -0.1 -1.0 255 0 0",
                "0.1 -0.1 -1.1 0 255 0",
                "-0.1 0.1 -1.2 0 0 255",
                "0.1 0.1 -1.3 255 255 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _translated_identity(x: float, y: float, z: float) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [x, y, z]
    return matrix.tolist()
