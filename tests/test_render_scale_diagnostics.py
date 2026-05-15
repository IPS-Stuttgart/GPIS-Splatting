from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from gpis_splatting.render_scale_diagnostics import run_render_scale_diagnostics, validate_render_scale_factors, write_scaled_prepared_scene
from gpis_splatting.serialization import read_json


def test_write_scaled_prepared_scene_scales_resolution_and_intrinsics(tmp_path: Path) -> None:
    scene_dir = write_tiny_prepared_scene(tmp_path / "scene")
    scaled = write_scaled_prepared_scene(scene_dir, tmp_path / "scaled", 2.0)

    cameras = read_json(scaled / "cameras.json")
    frame = cameras["frames"][0]
    assert frame["width"] == 14
    assert frame["height"] == 10
    assert frame["intrinsics"]["fx"] == 8.0
    assert frame["intrinsics"]["fy"] == 8.0
    assert frame["intrinsics"]["cx"] == 6.0
    assert frame["intrinsics"]["cy"] == 4.0


def test_run_render_scale_diagnostics_with_injected_rasterizer(tmp_path: Path) -> None:
    scene_dir = write_tiny_prepared_scene(tmp_path / "scene")
    point_cloud = tmp_path / "point_cloud.ply"
    write_tiny_3dgs_ply(point_cloud)

    result = run_render_scale_diagnostics(
        scene_dir=scene_dir,
        input_ply_path=point_cloud,
        output_dir=tmp_path / "diag",
        method_name="unit_scale_aa",
        split="test",
        render_scale_factors=(1.0, 2.0),
        include_gsplat_antialiased=True,
        output_resolution="target",
        max_frames=2,
        device="cpu",
        rasterization_fn=fake_rasterization,
    )

    manifest = pd.read_csv(result["manifest_path"])
    assert set(manifest["variant"]) == {"scale1_classic", "scale1_antialiased", "scale2_classic", "scale2_antialiased"}
    assert Path(result["status_path"]).exists()
    assert Path(result["report_path"]).exists()
    summary = result["status"]["summary"]
    assert summary["scale_variant_count"] == 3
    assert summary["scale_image_count"] == 6
    assert summary["aa_image_count"] == 2


def test_validate_render_scale_factors_rejects_nonpositive_values() -> None:
    try:
        validate_render_scale_factors((1.0, 0.0))
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError for nonpositive scale factor.")


def fake_rasterization(**kwargs: object) -> torch.Tensor:
    height = int(kwargs["height"])
    width = int(kwargs["width"])
    mode = str(kwargs.get("rasterize_mode", "classic"))
    value = 0.25 if mode == "classic" else 0.35
    return torch.ones((1, height, width, 3), dtype=torch.float32) * value


def write_tiny_prepared_scene(scene_dir: Path) -> Path:
    (scene_dir / "images").mkdir(parents=True)
    for index in range(2):
        Image.fromarray(np.full((5, 7, 3), 32 + 8 * index, dtype=np.uint8), mode="RGB").save(scene_dir / "images" / f"frame_{index}.png")
    (scene_dir / "real_scene.json").write_text('{"schema_version":1,"scene":"toy","source_format":"transforms"}\n', encoding="utf-8")
    (scene_dir / "cameras.json").write_text(
        "\n".join(
            [
                "{",
                '  "schema_version": 1,',
                '  "frames": [',
                '    {"index": 0, "file_name": "frame_0.png", "image_path": "images/frame_0.png", "width": 7, "height": 5, '
                '"world_to_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]], '
                '"intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 3.0, "cy": 2.0, "width": 7, "height": 5}},',
                '    {"index": 1, "file_name": "frame_1.png", "image_path": "images/frame_1.png", "width": 7, "height": 5, '
                '"world_to_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,2.1],[0,0,0,1]], '
                '"intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 3.0, "cy": 2.0, "width": 7, "height": 5}}',
                "  ]",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (scene_dir / "splits.json").write_text('{"schema_version":1,"train":[],"test":[0,1]}\n', encoding="utf-8")
    return scene_dir


def write_tiny_3dgs_ply(path: Path) -> None:
    header = [
        "ply",
        "format ascii 1.0",
        "element vertex 1",
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
    row = "0 0 0 0.1 0.2 0.3 -1 -4 -4 -4 1 0 0 0"
    path.write_text("\n".join([*header, row]) + "\n", encoding="ascii")
