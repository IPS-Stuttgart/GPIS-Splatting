from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gpis_splatting.cli.evaluate_render_consistency import main as evaluate_render_consistency_main
from gpis_splatting.serialization import read_json, write_json


def test_evaluate_render_consistency_reports_temporal_and_scale_metrics(tmp_path: Path) -> None:
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
                "camera_to_world": np.eye(4).tolist(),
                "world_to_camera": np.eye(4).tolist(),
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
        ]
    )

    eval_dir = scene_dir / "evaluations"
    temporal = pd.read_csv(eval_dir / "toy_test_temporal_consistency.csv")
    scale = pd.read_csv(eval_dir / "toy_test_scale_consistency.csv")
    status = read_json(eval_dir / "toy_test_render_consistency_status.json")
    summary = status["summary"]

    assert temporal.shape[0] == 3
    assert scale.shape[0] == 4
    assert summary["image_count"] == 4
    assert summary["temporal_pair_count"] == 3
    assert summary["scale_variant_count"] == 1
    assert summary["scale_image_count"] == 4
    assert summary["prediction_resized_to_target_count"] == 0
    assert temporal["temporal_instability_score"].max() > 0.0
    assert scale["variant_resized_to_base"].all()
    assert np.isfinite(scale["scale_mad"]).all()
    assert (eval_dir / "toy_test_render_consistency_report.md").exists()


def _write_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array, mode="RGB").save(path)
