from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gpis_splatting.external_3dgs import load_3dgs_ply, opacity_to_alpha
from gpis_splatting.gsplat_adapter import frame_to_gsplat_camera
from gpis_splatting.gsplat_fidelity_adapter import gaussian_ply_to_gsplat_tensors, render_3dgs_manifest_with_gsplat


def test_gaussian_ply_to_gsplat_tensors_decodes_common_3dgs_fields(tmp_path: Path) -> None:
    ply_path = tmp_path / "point_cloud.ply"
    write_tiny_3dgs_ply(ply_path)

    tensors, color_info = gaussian_ply_to_gsplat_tensors(load_3dgs_ply(ply_path), device="cpu", dtype="float32", opacity_mode="logit")

    expected_opacity = torch.as_tensor(opacity_to_alpha(np.asarray([-1.0, 1.0]), opacity_mode="logit"), dtype=torch.float32)
    assert tensors.means.shape == (2, 3)
    assert tensors.scales.shape == (2, 3)
    assert tensors.quats.shape == (2, 4)
    assert tensors.colors.shape == (2, 3)
    assert color_info["effective_color_mode"] == "rgb"
    assert torch.allclose(tensors.opacities.cpu(), expected_opacity)
    assert torch.allclose(torch.linalg.norm(tensors.quats, dim=1), torch.ones(2))


def test_gaussian_ply_to_gsplat_tensors_decodes_full_sh_coefficients(tmp_path: Path) -> None:
    ply_path = tmp_path / "point_cloud_sh.ply"
    write_tiny_3dgs_sh_ply(ply_path)

    tensors, color_info = gaussian_ply_to_gsplat_tensors(load_3dgs_ply(ply_path), device="cpu", dtype="float32", opacity_mode="logit", color_mode="sh", sh_degree=1)

    assert tensors.colors.shape == (1, 4, 3)
    assert color_info["effective_color_mode"] == "sh"
    assert color_info["effective_sh_degree"] == 1
    assert torch.allclose(tensors.colors[0, 0].cpu(), torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(tensors.colors[0, 1].cpu(), torch.tensor([1.0, 4.0, 7.0]))
    assert torch.allclose(tensors.colors[0, 2].cpu(), torch.tensor([2.0, 5.0, 8.0]))
    assert torch.allclose(tensors.colors[0, 3].cpu(), torch.tensor([3.0, 6.0, 9.0]))


def test_frame_to_gsplat_camera_converts_opengl_to_opencv() -> None:
    frame = {
        "width": 7,
        "height": 5,
        "world_to_camera": np.eye(4).tolist(),
        "intrinsics": {"fx": 4.0, "fy": 5.0, "cx": 3.0, "cy": 2.0, "width": 7, "height": 5},
    }

    camera = frame_to_gsplat_camera(frame, projection_convention="opengl", device="cpu", dtype="float64")

    assert camera.width == 7
    assert camera.height == 5
    assert np.allclose(camera.viewmat.cpu().numpy(), np.diag([1.0, -1.0, -1.0, 1.0]))
    assert np.allclose(camera.K.cpu().numpy(), [[4.0, 0.0, 3.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]])


def test_render_3dgs_manifest_with_injected_rasterizer(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    write_tiny_prepared_scene(scene_dir)
    point_cloud_path = tmp_path / "model" / "point_cloud" / "iteration_7" / "point_cloud.ply"
    point_cloud_path.parent.mkdir(parents=True)
    write_tiny_3dgs_ply(point_cloud_path)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{"variant": "baseline", "variant_kind": "baseline", "point_cloud_path": str(point_cloud_path)}]).to_csv(manifest_path, index=False)
    calls: list[dict[str, object]] = []

    def recorder(**kwargs: object) -> torch.Tensor:
        calls.append(kwargs)
        return fake_rasterization(**kwargs)

    result = render_3dgs_manifest_with_gsplat(
        manifest_path=manifest_path,
        scene_dir=scene_dir,
        output_root=tmp_path / "renders",
        method_name="unit_gsplat",
        split="test",
        device="cpu",
        strict_3dgs_fidelity=True,
        rasterization_fn=recorder,
    )

    expected = tmp_path / "renders" / "baseline" / "test" / "ours_7" / "renders" / "frame_000001.png"
    assert expected.exists()
    assert result["status"]["variant_count"] == 1
    assert (tmp_path / "renders" / "unit_gsplat_gsplat_render_manifest.csv").exists()
    assert calls[0]["backgrounds"].shape == (3,)


def test_render_3dgs_manifest_passes_sh_degree_to_rasterizer(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    write_tiny_prepared_scene(scene_dir)
    point_cloud_path = tmp_path / "model" / "point_cloud" / "iteration_7" / "point_cloud.ply"
    point_cloud_path.parent.mkdir(parents=True)
    write_tiny_3dgs_sh_ply(point_cloud_path)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{"variant": "baseline", "variant_kind": "baseline", "point_cloud_path": str(point_cloud_path)}]).to_csv(manifest_path, index=False)
    calls: list[dict[str, object]] = []

    def recorder(**kwargs: object) -> torch.Tensor:
        calls.append(kwargs)
        return fake_rasterization(**kwargs)

    result = render_3dgs_manifest_with_gsplat(
        manifest_path=manifest_path,
        scene_dir=scene_dir,
        output_root=tmp_path / "renders",
        method_name="unit_gsplat",
        split="test",
        device="cpu",
        color_mode="auto",
        sh_degree="auto",
        strict_3dgs_fidelity=True,
        rasterization_fn=recorder,
    )

    assert result["manifest"].iloc[0]["color_mode"] == "sh"
    assert result["manifest"].iloc[0]["sh_degree"] == 1
    assert calls
    assert calls[0]["sh_degree"] == 1
    assert calls[0]["colors"].shape == (1, 4, 3)


def fake_rasterization(**kwargs: object) -> torch.Tensor:
    height = int(kwargs["height"])
    width = int(kwargs["width"])
    return torch.ones((1, height, width, 3), dtype=torch.float32) * 0.25


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
    header = base_3dgs_ply_header(2)
    body = [" ".join(str(value) for value in row) for row in rows]
    path.write_text("\n".join([*header, *body]) + "\n", encoding="ascii")


def write_tiny_3dgs_sh_ply(path: Path) -> None:
    row = [
        0.0,
        0.0,
        0.0,
        0.1,
        0.2,
        0.3,
        *list(range(1, 10)),
        -1.0,
        -4.0,
        -4.0,
        -4.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
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
        *[f"property float f_rest_{index}" for index in range(9)],
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
    path.write_text("\n".join([*header, " ".join(str(value) for value in row)]) + "\n", encoding="ascii")


def base_3dgs_ply_header(vertex_count: int) -> list[str]:
    return [
        "ply",
        "format ascii 1.0",
        f"element vertex {vertex_count}",
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
