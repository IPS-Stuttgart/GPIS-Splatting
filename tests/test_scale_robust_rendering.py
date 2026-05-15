from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from gpis_splatting.scale_robust_rendering import run_scale_robust_3dgs_experiment, scale_prepared_frame
from gpis_splatting.serialization import read_json


def test_scale_prepared_frame_scales_dimensions_and_intrinsics() -> None:
    frame = {
        "width": 10,
        "height": 8,
        "intrinsics": {"fx": 5.0, "fy": 6.0, "cx": 4.0, "cy": 3.0, "width": 10, "height": 8},
    }

    scaled = scale_prepared_frame(frame, 0.5)

    assert scaled["width"] == 5
    assert scaled["height"] == 4
    assert scaled["intrinsics"]["fx"] == 2.5
    assert scaled["intrinsics"]["fy"] == 3.0
    assert scaled["intrinsics"]["cx"] == 2.0
    assert scaled["intrinsics"]["cy"] == 1.5
    assert frame["width"] == 10


def test_run_scale_robust_experiment_writes_metrics_and_summary(tmp_path: Path) -> None:
    scene_dir = tmp_path / "real_scenes" / "toy"
    write_tiny_prepared_scene(scene_dir)
    ply_path = tmp_path / "model" / "point_cloud" / "iteration_7" / "point_cloud.ply"
    ply_path.parent.mkdir(parents=True)
    write_tiny_3dgs_ply(ply_path)
    output_dir = tmp_path / "scale_robust"

    result = run_scale_robust_3dgs_experiment(
        input_ply_path=ply_path,
        scene_dir=scene_dir,
        output_dir=output_dir,
        method_name="toy_scale",
        scales=(0.5, 1.0),
        rasterize_modes=("classic", "antialiased"),
        device="cpu",
        max_frames=1,
        rasterization_fn=fake_rasterizer,
    )

    assert result["status"]["render_count"] == 4
    assert result["status"]["image_metric_count"] == 4
    assert result["render_manifest_path"].exists()
    assert result["metrics_path"].exists()
    assert result["summary_path"].exists()
    assert result["report_path"].exists()

    manifest = pd.read_csv(result["render_manifest_path"])
    metrics = pd.read_csv(result["metrics_path"])
    summary = pd.read_csv(result["summary_path"])
    assert set(manifest["rasterize_mode"]) == {"classic", "antialiased"}
    assert set(manifest["scale_label"]) == {"scale_0p5", "scale_1"}
    assert sorted(metrics["width"].unique().tolist()) == [5, 10]
    assert sorted(metrics["height"].unique().tolist()) == [4, 8]
    assert len(summary) == 4
    assert "delta_psnr_vs_classic" in summary.columns
    assert (output_dir / "renders" / "classic" / "scale_0p5" / "baseline" / "test" / "ours_7" / "renders" / "frame_000001.png").exists()

    status = read_json(result["status_path"])
    assert status["scales"] == [0.5, 1.0]
    assert status["rasterize_modes"] == ["classic", "antialiased"]


def fake_rasterizer(**kwargs: Any) -> torch.Tensor:
    height = int(kwargs["height"])
    width = int(kwargs["width"])
    value = 0.5 if kwargs.get("rasterize_mode") == "classic" else 0.55
    return torch.full((1, height, width, 3), value, dtype=torch.float32)


def write_tiny_prepared_scene(scene_dir: Path) -> None:
    (scene_dir / "images").mkdir(parents=True)
    write_image(scene_dir / "images" / "frame_000001.png", width=10, height=8, value=128)
    (scene_dir / "real_scene.json").write_text(
        '{"schema_version":1,"scene":"toy","dataset":"fixture","source_format":"fixture"}\n',
        encoding="utf-8",
    )
    (scene_dir / "cameras.json").write_text(
        "\n".join(
            [
                "{",
                '  "schema_version": 1,',
                '  "frames": [',
                "    {",
                '      "index": 1,',
                '      "file_name": "frame_000001.png",',
                '      "image_path": "images/frame_000001.png",',
                '      "width": 10,',
                '      "height": 8,',
                '      "intrinsics": {"fx": 6.0, "fy": 6.0, "cx": 5.0, "cy": 4.0, "width": 10, "height": 8},',
                '      "world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]',
                "    }",
                "  ]",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (scene_dir / "splits.json").write_text('{"schema_version":1,"train":[],"test":[0]}\n', encoding="utf-8")


def write_tiny_3dgs_ply(path: Path) -> None:
    rows = [
        [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, -2.0, -4.0, -4.0, -4.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.2, 0.3, 0.4, -1.0, -3.0, -3.0, -3.0, 1.0, 0.0, 0.0, 0.0],
    ]
    header = [
        "ply",
        "format ascii 1.0",
        "element vertex 2",
        "property float x",
        "property float y",
        "property float z",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]
    body = [" ".join(str(value) for value in row) for row in rows]
    path.write_text("\n".join([*header, *body]) + "\n", encoding="ascii")


def write_image(path: Path, *, width: int, height: int, value: int) -> None:
    data = np.full((height, width, 3), value, dtype=np.uint8)
    Image.fromarray(data, mode="RGB").save(path)
