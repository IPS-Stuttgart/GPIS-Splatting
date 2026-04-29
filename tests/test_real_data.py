from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gpis_splatting.cli.bootstrap_real_gpis import main as bootstrap_real_gpis_main
from gpis_splatting.cli.diagnose_real_render import main as diagnose_real_render_main
from gpis_splatting.cli.evaluate_real_renders import main as evaluate_real_renders_main
from gpis_splatting.cli.fit_real_gpis import main as fit_real_gpis_main
from gpis_splatting.cli.prepare_real_scene import main as prepare_real_scene_main
from gpis_splatting.cli.prepare_tanks_temples_scene import main as prepare_tanks_temples_scene_main
from gpis_splatting.cli.render_real_splats import main as render_real_splats_main
from gpis_splatting.cli.validate_real_scene import main as validate_real_scene_main
from gpis_splatting.real_bootstrap import load_ply_point_cloud
from gpis_splatting.real_scene import build_sparse_split
from gpis_splatting.serialization import read_json, write_json
from gpis_splatting.tanks_temples import google_drive_confirm_url_from_html, read_tanks_temples_log


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


def test_load_binary_little_endian_ply_point_cloud(tmp_path: Path) -> None:
    ply_path = tmp_path / "binary_points.ply"
    vertices = np.asarray(
        [
            (0.0, 0.1, 1.0, 10, 20, 30),
            (0.2, 0.3, 1.2, 40, 50, 60),
        ],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    ply_path.write_bytes(
        "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
            ]
        ).encode("ascii")
        + b"\n"
        + vertices.tobytes()
    )

    cloud = load_ply_point_cloud(ply_path)
    assert cloud.points.shape == (2, 3)
    assert np.allclose(cloud.points[1], [0.2, 0.3, 1.2])
    assert np.allclose(cloud.colors[0], [10 / 255.0, 20 / 255.0, 30 / 255.0])


def test_real_gpis_fit_render_and_evaluate_loop(tmp_path: Path) -> None:
    root = _prepare_colmap_scene_with_points(tmp_path)
    scene_dir = root / "colmap_bootstrap"

    bootstrap_real_gpis_main(
        [
            "--scene-dir",
            str(scene_dir),
            "--point-source",
            "colmap",
            "--free-space-samples-per-point",
            "1",
            "--add-behind-surface-samples",
            "false",
            "--max-points",
            "10",
        ]
    )
    fit_real_gpis_main(
        [
            "--scene-dir",
            str(scene_dir),
            "--max-train-points",
            "6",
            "--lengthscale",
            "0.5",
            "--noise-std",
            "0.05",
        ]
    )
    render_real_splats_main(
        [
            "--scene-dir",
            str(scene_dir),
            "--split",
            "train",
            "--epsilon",
            "0.2",
        ]
    )

    render_dir = scene_dir / "renders" / "real_gpis_gate"
    report = read_json(render_dir / "real_render_report.json")
    gates = np.load(render_dir / "real_splat_gates.npz")["gate"]
    assert (scene_dir / "real_gpis_model.npz").exists()
    assert (scene_dir / "real_gpis_model_fit_report.json").exists()
    assert (render_dir / "a.png").exists()
    assert report["image_count"] == 1
    assert report["outputs"][0]["drawn_splat_count"] > 0
    assert np.all((gates >= 0.0) & (gates <= 1.0))

    evaluate_real_renders_main(
        [
            "--scene-dir",
            str(scene_dir),
            "--predictions-dir",
            str(render_dir),
            "--method-name",
            "real_gpis_gate",
            "--split",
            "train",
        ]
    )
    metrics = pd.read_csv(scene_dir / "evaluations" / "real_gpis_gate_train_image_metrics.csv")
    assert len(metrics) == 1
    assert np.isfinite(metrics["ssim"]).all()


def test_real_render_diagnostics_outputs_visuals_and_metrics(tmp_path: Path) -> None:
    root = _prepare_colmap_scene_with_points(tmp_path)
    scene_dir = root / "colmap_bootstrap"

    bootstrap_real_gpis_main(
        [
            "--scene-dir",
            str(scene_dir),
            "--point-source",
            "colmap",
            "--free-space-samples-per-point",
            "1",
            "--add-behind-surface-samples",
            "false",
            "--max-points",
            "10",
        ]
    )
    fit_real_gpis_main(
        [
            "--scene-dir",
            str(scene_dir),
            "--max-train-points",
            "6",
            "--lengthscale",
            "0.5",
            "--noise-std",
            "0.05",
        ]
    )
    diagnose_real_render_main(
        [
            "--scene-dir",
            str(scene_dir),
            "--split",
            "train",
            "--max-frames",
            "1",
            "--epsilon",
            "0.2",
            "--gate-floor",
            "0.25",
        ]
    )

    diagnostics_dir = scene_dir / "diagnostics" / "real_render"
    frame_metrics = pd.read_csv(diagnostics_dir / "real_render_diagnostics.csv")
    status = read_json(diagnostics_dir / "real_render_diagnostics.json")
    assert len(frame_metrics) == 1
    row = frame_metrics.iloc[0]
    assert row["projected_splat_count"] > 0
    assert row["in_frame_splat_count"] > 0
    assert row["visible_splat_count"] == row["in_frame_splat_count"]
    assert row["plain_drawn_splat_count"] > 0
    assert np.isfinite(row["plain_psnr"])
    assert np.isfinite(row["gated_ssim"])
    assert row["gate_min"] >= 0.25
    assert (diagnostics_dir / "gate_histogram.png").exists()
    assert Path(row["target_plain_gated_panel"]).exists()
    assert Path(row["projected_splat_overlay"]).exists()
    assert Path(row["depth_visualization"]).exists()
    assert Path(row["gate_overlay"]).exists()
    assert Path(row["gate_heatmap"]).exists()
    assert Path(row["gate_histogram"]).exists()
    assert status["metric_summary"]["frame_count"] == 1
    assert (diagnostics_dir / "real_render_diagnostics.md").exists()


def test_prepare_tanks_temples_scene_from_log_fixture(tmp_path: Path) -> None:
    source = _write_tanks_temples_fixture(tmp_path / "tanks_temples" / "Ignatius")
    root = tmp_path / "real_scenes"

    prepare_tanks_temples_scene_main(
        [
            "--input-dir",
            str(source),
            "--prepared-scene",
            "ignatius_fixture",
            "--output-root",
            str(root),
            "--train-view-count",
            "2",
        ]
    )

    scene_dir = root / "ignatius_fixture"
    scene_meta = read_json(scene_dir / "real_scene.json")
    cameras = read_json(scene_dir / "cameras.json")
    splits = read_json(scene_dir / "splits.json")
    validation = read_json(scene_dir / "validation.json")
    assert scene_meta["dataset"] == "tanks_temples"
    assert scene_meta["source_format"] == "tanks_temples_log"
    assert scene_meta["tanks_temples"]["intrinsics_source"] == "tanks_temples_download_page_recommended_pinhole"
    assert Path(scene_meta["tanks_temples"]["reconstruction_path"]).parts[-2:] == ("reconstruction", "Ignatius.ply")
    assert Path(scene_meta["tanks_temples"]["ground_truth_path"]).parts[-2:] == ("ground_truth", "Ignatius.ply")
    assert validation["passed"] is True
    assert splits["train"] == [0, 2]
    assert splits["test"] == [1]
    assert len(cameras["frames"]) == 3
    first = cameras["frames"][0]
    assert first["intrinsics"]["fx"] == 5.6
    assert first["intrinsics"]["fy"] == 5.6
    assert first["intrinsics"]["cx"] == 4.0
    assert first["intrinsics"]["cy"] == 3.0
    assert first["camera_to_world"][2][3] == 1.0
    assert first["world_to_camera"][2][3] == -1.0
    assert (scene_dir / "images" / "000000.jpg").exists()


def test_read_tanks_temples_log_rejects_incomplete_pose(tmp_path: Path) -> None:
    log_path = tmp_path / "bad.log"
    log_path.write_text("0 0 0\n1 0 0 0\n", encoding="utf-8")
    try:
        read_tanks_temples_log(log_path)
    except ValueError as exc:
        assert "five lines per pose" in str(exc)
    else:
        raise AssertionError("Expected malformed Tanks and Temples log to fail.")


def test_google_drive_confirm_url_from_html_form() -> None:
    html = """
    <html>
      <body>
        <form id="download-form" action="https://drive.usercontent.google.com/download" method="get">
          <input type="hidden" name="id" value="abc123">
          <input type="hidden" name="export" value="download">
          <input type="hidden" name="confirm" value="t">
          <input type="hidden" name="uuid" value="uuid-123">
        </form>
      </body>
    </html>
    """

    confirm_url = google_drive_confirm_url_from_html(html, base_url="https://drive.google.com/uc?export=download&id=abc123")
    assert confirm_url is not None
    assert confirm_url.startswith("https://drive.usercontent.google.com/download?")
    assert "id=abc123" in confirm_url
    assert "confirm=t" in confirm_url
    assert "uuid=uuid-123" in confirm_url


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


def _write_tanks_temples_fixture(root: Path) -> Path:
    image_dir = root / "image_sets" / "Ignatius"
    image_dir.mkdir(parents=True)
    width, height = 8, 6
    for index in range(3):
        data = np.zeros((height, width, 3), dtype=np.uint8)
        data[..., 0] = 40 + index * 20
        data[..., 1] = np.arange(height, dtype=np.uint8)[:, None] * 20
        data[..., 2] = np.arange(width, dtype=np.uint8)[None, :] * 20
        Image.fromarray(data, mode="RGB").save(image_dir / f"{index:06d}.jpg")

    (root / "camera_poses").mkdir()
    (root / "camera_poses" / "Ignatius.log").write_text(
        "\n".join(
            [
                "0 0 0",
                "1 0 0 0",
                "0 1 0 0",
                "0 0 1 1",
                "0 0 0 1",
                "1 1 0",
                "1 0 0 0.1",
                "0 1 0 0",
                "0 0 1 1",
                "0 0 0 1",
                "2 2 0",
                "1 0 0 0.2",
                "0 1 0 0",
                "0 0 1 1",
                "0 0 0 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for directory, suffix, content in (
        ("reconstruction", ".ply", _tiny_ascii_ply()),
        ("ground_truth", ".ply", _tiny_ascii_ply()),
        ("alignment", ".txt", "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"),
        ("crop", ".json", '{"min": [-1, -1, -1], "max": [1, 1, 1]}'),
    ):
        target_dir = root / directory
        target_dir.mkdir()
        (target_dir / f"Ignatius{suffix}").write_text(content, encoding="utf-8")
    return root


def _tiny_ascii_ply() -> str:
    return (
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
                "0 0 1 255 0 0",
                "0.1 0 1 0 255 0",
            ]
        )
        + "\n"
    )
