from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gpis_splatting.cli.evaluate_render_consistency import main as evaluate_render_consistency_main
from gpis_splatting.render_consistency import antialias_roundtrip, camera_pose_delta
from gpis_splatting.serialization import read_json, write_json


def test_evaluate_render_consistency_reports_temporal_scale_view_and_antialiasing_metrics(tmp_path: Path) -> None:
    scene_dir = tmp_path / "real_scenes" / "tiny"
    image_dir = scene_dir / "images"
    predictions_dir = tmp_path / "predictions"
    lowres_dir = tmp_path / "predictions_lowres"
    image_dir.mkdir(parents=True)
    predictions_dir.mkdir()
    lowres_dir.mkdir()

    frames = []
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    for index in range(4):
        frame_name = f"frame_{index}.png"
        target = np.full((8, 10, 3), 40 + 20 * index, dtype=np.uint8)
        _write_image(image_dir / frame_name, target)
        pose = np.eye(4)
        pose[0, 3] = 0.05 * index
        frames.append(
            {
                "index": index,
                "image_id": str(index),
                "file_name": frame_name,
                "image_path": f"images/{frame_name}",
                "width": 10,
                "height": 8,
                "camera_id": "0",
                "intrinsics": {"model": "PINHOLE", "width": 10, "height": 8, "fx": 5.0, "fy": 5.0, "cx": 5.0, "cy": 4.0, "params": []},
                "camera_to_world": pose.tolist(),
                "world_to_camera": np.linalg.inv(pose).tolist(),
            }
        )
        prediction = np.clip(target.astype(np.int16) + 2, 0, 255).astype(np.uint8)
        if index == 2:
            prediction[::2, ::2] = 255
        _write_image(predictions_dir / frame_name, prediction)
        lowres = np.asarray(Image.fromarray(prediction, mode="RGB").resize((5, 4), resample=resampling), dtype=np.uint8)
        _write_image(lowres_dir / frame_name, lowres)

    write_json(scene_dir / "real_scene.json", {"schema_version": 1, "scene": "tiny", "dataset": "unit", "image_count": 4})
    write_json(scene_dir / "cameras.json", {"schema_version": 1, "frames": frames})
    write_json(scene_dir / "splits.json", {"schema_version": 1, "train": [0], "test": [0, 1, 2, 3]})

    evaluate_render_consistency_main(
        [
            "--scene-dir",
            str(scene_dir),
            "--predictions-dir",
            str(predictions_dir),
            "--method-name",
            "toy",
            "--split",
            "test",
            "--scale-predictions-dir",
            f"lowres={lowres_dir}",
            "--aa-downsample-factor",
            "2",
            "--max-view-translation",
            "0.06",
        ]
    )

    eval_dir = scene_dir / "evaluations"
    temporal = pd.read_csv(eval_dir / "toy_test_temporal_consistency.csv")
    scale = pd.read_csv(eval_dir / "toy_test_scale_consistency.csv")
    aa = pd.read_csv(eval_dir / "toy_test_antialiasing_consistency.csv")
    status = read_json(eval_dir / "toy_test_render_consistency_status.json")
    summary = status["summary"]

    assert temporal.shape[0] == 3
    assert scale.shape[0] == 4
    assert aa.shape[0] == 4
    assert status["schema_version"] == 2
    assert status["aa_downsample_factors"] == [2]
    assert summary["image_count"] == 4
    assert summary["temporal_pair_count"] == 3
    assert summary["temporal_skipped_view_filter_count"] == 0
    assert summary["scale_variant_count"] == 1
    assert summary["scale_image_count"] == 4
    assert summary["aa_factor_count"] == 1
    assert summary["aa_image_count"] == 4
    assert summary["prediction_resized_to_target_count"] == 0
    assert temporal["temporal_instability_score"].max() > 0.0
    assert np.allclose(temporal["camera_translation_delta"].to_numpy(), 0.05)
    assert np.isfinite(temporal["view_instability_score"]).all()
    assert scale["variant_resized_to_base"].all()
    assert np.isfinite(scale["scale_mad"]).all()
    assert aa["aa_instability_score"].max() > 0.0
    assert np.isfinite(aa["aa_mad"]).all()
    assert (eval_dir / "toy_test_render_consistency_report.md").exists()


def test_view_motion_filter_skips_large_camera_steps(tmp_path: Path) -> None:
    scene_dir = tmp_path / "real_scenes" / "filter"
    predictions_dir = tmp_path / "predictions"
    (scene_dir / "images").mkdir(parents=True)
    predictions_dir.mkdir()
    frames = []
    translations = [0.0, 0.05, 0.30]
    for index, tx in enumerate(translations):
        frame_name = f"frame_{index}.png"
        image = np.full((6, 6, 3), 32 + index, dtype=np.uint8)
        _write_image(scene_dir / "images" / frame_name, image)
        _write_image(predictions_dir / frame_name, image)
        pose = np.eye(4)
        pose[0, 3] = tx
        frames.append(
            {
                "index": index,
                "image_id": str(index),
                "file_name": frame_name,
                "image_path": f"images/{frame_name}",
                "width": 6,
                "height": 6,
                "camera_id": "0",
                "intrinsics": {"model": "PINHOLE", "width": 6, "height": 6, "fx": 3.0, "fy": 3.0, "cx": 3.0, "cy": 3.0, "params": []},
                "camera_to_world": pose.tolist(),
                "world_to_camera": np.linalg.inv(pose).tolist(),
            }
        )
    write_json(scene_dir / "real_scene.json", {"schema_version": 1, "scene": "filter", "dataset": "unit", "image_count": 3})
    write_json(scene_dir / "cameras.json", {"schema_version": 1, "frames": frames})
    write_json(scene_dir / "splits.json", {"schema_version": 1, "train": [0], "test": [0, 1, 2]})

    evaluate_render_consistency_main(
        [
            "--scene-dir",
            str(scene_dir),
            "--predictions-dir",
            str(predictions_dir),
            "--method-name",
            "toy",
            "--split",
            "test",
            "--max-view-translation",
            "0.06",
            "--disable-aa-roundtrip",
            "true",
        ]
    )

    eval_dir = scene_dir / "evaluations"
    temporal = pd.read_csv(eval_dir / "toy_test_temporal_consistency.csv")
    status = read_json(eval_dir / "toy_test_render_consistency_status.json")
    assert temporal.shape[0] == 1
    assert temporal.iloc[0]["left_frame_index"] == 0
    assert temporal.iloc[0]["right_frame_index"] == 1
    assert status["summary"]["temporal_skipped_view_filter_count"] == 1
    assert status["summary"]["aa_image_count"] == 0


def test_antialiasing_roundtrip_exposes_checkerboard_high_frequency() -> None:
    checker = np.zeros((16, 16, 3), dtype=np.float64)
    checker[::2, ::2] = 1.0
    checker[1::2, 1::2] = 1.0
    smooth = np.full((16, 16, 3), 0.5, dtype=np.float64)

    checker_lowpass = antialias_roundtrip(checker, 2)
    smooth_lowpass = antialias_roundtrip(smooth, 2)

    checker_mad = float(np.abs(checker - checker_lowpass).mean())
    smooth_mad = float(np.abs(smooth - smooth_lowpass).mean())
    assert checker_mad > 0.1
    assert smooth_mad < checker_mad


def test_camera_pose_delta_uses_world_to_camera_fallback() -> None:
    left_c2w = np.eye(4)
    right_c2w = np.eye(4)
    right_c2w[2, 3] = 0.25
    delta = camera_pose_delta({"world_to_camera": np.linalg.inv(left_c2w).tolist()}, {"world_to_camera": np.linalg.inv(right_c2w).tolist()})
    assert np.isclose(delta["translation_delta"], 0.25)
    assert np.isclose(delta["rotation_delta_deg"], 0.0)


def _write_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array, mode="RGB").save(path)
