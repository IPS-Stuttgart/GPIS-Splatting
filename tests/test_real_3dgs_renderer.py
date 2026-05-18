from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gpis_splatting.external_3dgs import opacity_to_alpha
from gpis_splatting.real_3dgs_renderer import render_real_3dgs_splats
from gpis_splatting.render_metadata import render_metadata_from_report


def test_render_real_3dgs_splats_scales_opacity_and_preserves_fields(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    write_tiny_prepared_scene(scene_dir)
    point_cloud_path = tmp_path / "model" / "point_cloud" / "iteration_7" / "point_cloud.ply"
    point_cloud_path.parent.mkdir(parents=True)
    write_tiny_3dgs_ply(point_cloud_path)
    gate_path = tmp_path / "gate.npz"
    np.savez_compressed(gate_path, gate=np.asarray([1.0, 0.5], dtype=np.float64))
    calls: list[dict[str, object]] = []

    def recorder(**kwargs: object) -> torch.Tensor:
        calls.append(kwargs)
        height = int(kwargs["height"])
        width = int(kwargs["width"])
        return torch.ones((1, height, width, 3), dtype=torch.float32) * 0.25

    result = render_real_3dgs_splats(
        scene_dir=scene_dir,
        input_ply_path=point_cloud_path,
        gate_path=gate_path,
        use_gpis_gate=False,
        method_name="faithful_gated",
        split="test",
        gsplat_device="cpu",
        gsplat_dtype="float32",
        gsplat_rasterization_fn=recorder,
    )

    assert (Path(result["output_dir"]) / "frame_000001.png").exists()
    assert Path(result["report"]["gated_ply_path"]).exists()
    assert result["report"]["renderer_backend"] == "gsplat"
    assert result["report"]["backend"] == "gsplat"
    assert result["report"]["render_fidelity"] == "faithful_3dgs"
    assert result["report"]["photometric_metrics_allowed"] is True
    assert result["report"]["photometric_metrics_policy"] == "allowed"
    assert result["report"]["gate_summary"]["source"] == "external"

    metadata = render_metadata_from_report(result["report"], report_path=Path(result["report_path"]))
    assert metadata["backend"] == "gsplat"
    assert metadata["render_fidelity"] == "faithful_3dgs"
    assert metadata["photometric_metrics_allowed"] is True
    assert metadata["photometric_metrics_policy"] == "allowed"

    assert calls
    call = calls[0]
    assert call["scales"].shape == (2, 3)
    assert call["quats"].shape == (2, 4)
    assert call["colors"].shape == (2, 3)
    expected_opacity = torch.as_tensor(
        opacity_to_alpha(np.asarray([-1.0, 1.0]), opacity_mode="logit") * np.asarray([1.0, 0.5]),
        dtype=torch.float32,
    )
    assert torch.allclose(call["opacities"].detach().cpu(), expected_opacity, atol=1e-6)


def write_tiny_prepared_scene(scene_dir: Path) -> None:
    scene_dir.mkdir(parents=True)
    (scene_dir / "real_scene.json").write_text('{"schema_version":1,"scene":"toy","source_format":"transforms"}\n', encoding="utf-8")
    (scene_dir / "cameras.json").write_text(
        "\n".join(
            [
                "{",
                '  "schema_version": 1,',
                '  "frames": [',
                '    {"index": 1, "file_name": "frame_000001.png", "image_path": "images/frame_000001.png", '
                '"width": 7, "height": 5, "world_to_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]], '
                '"intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 3.0, "cy": 2.0, "width": 7, "height": 5}}',
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
        [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, -1.0, -4.0, -4.0, -4.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.2, 0.3, 0.4, 1.0, -3.0, -3.0, -3.0, 1.0, 0.0, 0.0, 0.0],
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
