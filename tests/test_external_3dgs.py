from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gpis_splatting.cli.convert_3dgs_ply_to_splats import main as convert_3dgs_ply_to_splats_main
from gpis_splatting.cli.evaluate_3dgs_variant_renders import main as evaluate_3dgs_variant_renders_main
from gpis_splatting.cli.export_3dgs_gpis_variants import main as export_3dgs_gpis_variants_main
from gpis_splatting.external_3dgs import load_3dgs_ply, opacity_to_alpha


def test_convert_3dgs_ply_to_internal_splats(tmp_path: Path) -> None:
    ply_path = tmp_path / "point_cloud.ply"
    write_tiny_3dgs_ply(ply_path)
    output_splats = tmp_path / "trained_3dgs_splats.npz"

    convert_3dgs_ply_to_splats_main(["--input-ply", str(ply_path), "--output-splats", str(output_splats)])

    with np.load(output_splats) as splats:
        assert splats["centers"].shape == (4, 3)
        assert np.allclose(splats["centers"][2], [2.0, 0.0, 0.0])
        assert np.allclose(splats["tau"], opacity_to_alpha(np.asarray([-2.0, -1.0, 0.0, 1.0]), opacity_mode="logit"))
        assert np.all(splats["sigma"] > 0.0)
    assert (tmp_path / "trained_3dgs_splats.json").exists()


def test_export_3dgs_gpis_variants_preserves_and_gates_ply(tmp_path: Path) -> None:
    ply_path = tmp_path / "point_cloud.ply"
    write_tiny_3dgs_ply(ply_path)
    gate_path = tmp_path / "gate.npz"
    np.savez_compressed(gate_path, gate=np.asarray([0.1, 0.3, 0.8, 0.9], dtype=np.float64))
    output_dir = tmp_path / "variants"

    export_3dgs_gpis_variants_main(
        [
            "--input-ply",
            str(ply_path),
            "--gate-path",
            str(gate_path),
            "--output-dir",
            str(output_dir),
            "--method-name",
            "paper_gate",
            "--iteration",
            "7",
            "--gate-thresholds",
            "0.5",
        ]
    )

    baseline = load_3dgs_ply(output_dir / "paper_gate_baseline" / "point_cloud" / "iteration_7" / "point_cloud.ply")
    filtered = load_3dgs_ply(output_dir / "paper_gate_gate_ge_0p5" / "point_cloud" / "iteration_7" / "point_cloud.ply")
    scaled = load_3dgs_ply(output_dir / "paper_gate_gate_scaled" / "point_cloud" / "iteration_7" / "point_cloud.ply")
    manifest = (output_dir / "paper_gate_3dgs_variant_manifest.csv").read_text(encoding="utf-8")

    assert baseline.vertex_count == 4
    assert filtered.vertex_count == 2
    assert np.allclose(filtered.vertices["x"], [2.0, 3.0])
    assert np.allclose(filtered.vertices["f_dc_0"], [0.3, 0.4])
    assert "gate_scaled" in manifest

    source_alpha = opacity_to_alpha(baseline.vertices["opacity"].astype(np.float64), opacity_mode="logit")
    scaled_alpha = opacity_to_alpha(scaled.vertices["opacity"].astype(np.float64), opacity_mode="logit")
    assert np.allclose(scaled_alpha, source_alpha * np.asarray([0.1, 0.3, 0.8, 0.9]), atol=1e-6)


def test_export_3dgs_gpis_variants_rejects_gate_count_mismatch(tmp_path: Path) -> None:
    ply_path = tmp_path / "point_cloud.ply"
    write_tiny_3dgs_ply(ply_path)
    gate_path = tmp_path / "bad_gate.npz"
    np.savez_compressed(gate_path, gate=np.asarray([0.1, 0.2], dtype=np.float64))

    try:
        export_3dgs_gpis_variants_main(["--input-ply", str(ply_path), "--gate-path", str(gate_path), "--output-dir", str(tmp_path / "variants")])
    except ValueError as exc:
        assert "Gate count" in str(exc)
    else:
        raise AssertionError("Expected gate-count mismatch to fail.")


def test_export_3dgs_gpis_variants_preserves_binary_ply(tmp_path: Path) -> None:
    ply_path = tmp_path / "binary_point_cloud.ply"
    write_tiny_binary_3dgs_ply(ply_path)
    gate_path = tmp_path / "gate.npz"
    np.savez_compressed(gate_path, raw_gate=np.asarray([0.25, 0.75], dtype=np.float64))
    output_dir = tmp_path / "variants"

    export_3dgs_gpis_variants_main(
        [
            "--input-ply",
            str(ply_path),
            "--gate-path",
            str(gate_path),
            "--output-dir",
            str(output_dir),
            "--method-name",
            "binary_gate",
            "--iteration",
            "11",
            "--gate-thresholds",
            "0.5",
            "--write-scaled",
            "false",
        ]
    )

    filtered = load_3dgs_ply(output_dir / "binary_gate_gate_ge_0p5" / "point_cloud" / "iteration_11" / "point_cloud.ply")
    assert filtered.ply_format == "binary_little_endian"
    assert filtered.vertex_count == 1
    assert np.allclose(filtered.vertices["x"], [1.0])
    assert np.allclose(filtered.vertices["f_rest_0"], [0.33])


def test_evaluate_3dgs_variant_renders_writes_comparison(tmp_path: Path) -> None:
    scene_dir = tmp_path / "real_scenes" / "toy"
    write_tiny_prepared_scene(scene_dir)
    manifest_path = tmp_path / "variants" / "paper_gate_3dgs_variant_manifest.csv"
    manifest_path.parent.mkdir()
    pd.DataFrame(
        [
            {
                "variant": "baseline",
                "variant_kind": "baseline",
                "model_dir": str(tmp_path / "variants" / "paper_gate_baseline"),
                "point_cloud_path": str(tmp_path / "variants" / "paper_gate_baseline" / "point_cloud" / "iteration_7" / "point_cloud.ply"),
                "retained_count": 4,
                "retention_fraction": 1.0,
                "gate_threshold": np.nan,
                "opacity_scaled": False,
                "gate_min": 0.1,
                "gate_max": 0.9,
                "gate_mean": 0.5,
            },
            {
                "variant": "gate_scaled",
                "variant_kind": "gate_scaled",
                "model_dir": str(tmp_path / "variants" / "paper_gate_gate_scaled"),
                "point_cloud_path": str(tmp_path / "variants" / "paper_gate_gate_scaled" / "point_cloud" / "iteration_7" / "point_cloud.ply"),
                "retained_count": 4,
                "retention_fraction": 1.0,
                "gate_threshold": np.nan,
                "opacity_scaled": True,
                "gate_min": 0.1,
                "gate_max": 0.9,
                "gate_mean": 0.5,
            },
        ]
    ).to_csv(manifest_path, index=False)
    predictions_root = tmp_path / "rendered"
    for variant in ("baseline", "gate_scaled"):
        variant_dir = predictions_root / variant / "test" / "ours_7" / "renders"
        variant_dir.mkdir(parents=True)
        write_image(variant_dir / "frame_000001.png", value=80)

    evaluate_3dgs_variant_renders_main(
        [
            "--manifest-path",
            str(manifest_path),
            "--scene-dir",
            str(scene_dir),
            "--predictions-root",
            str(predictions_root),
            "--prediction-subdir",
            "test/ours_7/renders",
            "--output-dir",
            str(tmp_path / "evaluations"),
            "--method-name",
            "paper_gate",
        ]
    )

    comparison = pd.read_csv(tmp_path / "evaluations" / "paper_gate_3dgs_render_comparison.csv")
    assert comparison["variant"].tolist() == ["baseline", "gate_scaled"]
    assert comparison["mean_ssim"].tolist() == [1.0, 1.0]
    assert comparison["image_count"].tolist() == [1, 1]
    assert (tmp_path / "evaluations" / "paper_gate_3dgs_render_evaluation_report.md").exists()


def write_tiny_3dgs_ply(path: Path) -> None:
    rows = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, -2.0, -4.0, -4.0, -4.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.4, -1.0, -3.0, -3.0, -3.0, 1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.5, 0.0, -2.0, -2.0, -2.0, 1.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.6, 1.0, -1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0],
    ]
    header = [
        "ply",
        "format ascii 1.0",
        "element vertex 4",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
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


def write_tiny_binary_3dgs_ply(path: Path) -> None:
    properties = [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("f_dc_0", "float"),
        ("f_dc_1", "float"),
        ("f_dc_2", "float"),
        ("f_rest_0", "float"),
        ("opacity", "float"),
    ]
    dtype = np.dtype([(name, "<f4") for name, _property_type in properties])
    rows = np.zeros((2,), dtype=dtype)
    rows["x"] = [0.0, 1.0]
    rows["f_dc_0"] = [0.1, 0.2]
    rows["f_dc_1"] = [0.2, 0.3]
    rows["f_dc_2"] = [0.3, 0.4]
    rows["f_rest_0"] = [0.11, 0.33]
    rows["opacity"] = [-1.0, 1.0]
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 2",
            *[f"property {property_type} {name}" for name, property_type in properties],
            "end_header",
            "",
        ]
    ).encode("ascii")
    path.write_bytes(header + rows.tobytes())


def write_tiny_prepared_scene(scene_dir: Path) -> None:
    (scene_dir / "images").mkdir(parents=True)
    write_image(scene_dir / "images" / "frame_000001.png", value=80)
    (scene_dir / "real_scene.json").write_text(
        '{"scene":"toy","dataset":"fixture","source_format":"fixture"}\n',
        encoding="utf-8",
    )
    (scene_dir / "cameras.json").write_text(
        "\n".join(
            [
                "{",
                '  "frames": [',
                '    {"index": 1, "file_name": "frame_000001.png", "image_path": "images/frame_000001.png"}',
                "  ]",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (scene_dir / "splits.json").write_text('{"train": [], "test": [0]}\n', encoding="utf-8")


def write_image(path: Path, *, value: int) -> None:
    data = np.full((5, 7, 3), value, dtype=np.uint8)
    Image.fromarray(data, mode="RGB").save(path)
