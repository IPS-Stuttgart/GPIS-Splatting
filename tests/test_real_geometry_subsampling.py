from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gpis_splatting.real_geometry import evaluate_tanks_temples_geometry
from gpis_splatting.serialization import read_json, write_json
from gpis_splatting.splats import SplatCloud, save_splats


def test_evaluate_tanks_temples_geometry_subsamples_predictions_after_crop(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    write_json(
        scene_dir / "real_scene.json",
        {
            "schema_version": 1,
            "scene": "crop_order_scene",
            "dataset": "unit_test",
            "tanks_temples": {
                "ground_truth_path": "gt.ply",
                "alignment_path": "identity.txt",
                "crop_path": "crop.json",
            },
        },
    )
    write_json(scene_dir / "cameras.json", {"schema_version": 1, "frames": []})
    write_json(scene_dir / "splits.json", {"schema_version": 1, "train": [], "test": []})
    (scene_dir / "identity.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")
    write_json(scene_dir / "crop.json", {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.1]})
    (scene_dir / "gt.ply").write_text(
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
        + "\n",
        encoding="utf-8",
    )

    # With seed 13, deterministic_subsample(..., max_points=1) selects index 5
    # from a six-point input.  If predicted points are sampled before the crop,
    # the only sampled point lies outside the crop and the evaluator has no
    # predictions left.  The intended protocol crops first, then samples from the
    # in-crop predictions.
    splats = SplatCloud(
        centers=torch.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=torch.float64,
        ),
        colors=torch.ones((6, 3), dtype=torch.float64),
        tau=torch.ones((6,), dtype=torch.float64),
        sigma=torch.full((6,), 0.04, dtype=torch.float64),
        is_surface=torch.ones((6,), dtype=torch.bool),
    )
    save_splats(str(scene_dir / "crop_order_splats.npz"), splats)
    np.savez_compressed(scene_dir / "crop_order_gates.npz", gate=np.asarray([0.9, 0.8, 0.1, 0.2, 0.3, 0.4], dtype=np.float64))

    result = evaluate_tanks_temples_geometry(
        scene_dir=scene_dir,
        splats_path="crop_order_splats.npz",
        gate_path="crop_order_gates.npz",
        method_name="crop_order",
        thresholds=(0.05,),
        max_gt_points=0,
        max_pred_points=1,
        distance_chunk_size=2,
    )

    summary = pd.read_csv(result["summary_path"])
    thresholds = pd.read_csv(result["threshold_metrics_path"])
    status = read_json(result["status_path"])
    all_005 = thresholds[(thresholds["group"] == "all") & (thresholds["threshold"] == 0.05)].iloc[0]
    assert status["pred_count_input"] == 6
    assert status["crop"]["pred_total"] == 6
    assert status["crop"]["pred_kept"] == 2
    assert status["pred_count_after_crop"] == 2
    assert status["pred_count_sampled"] == 1
    assert status["pred_sample_indices_count"] == 1
    assert status["pred_count_evaluated"] == 1
    assert summary[summary["group"] == "all"].iloc[0]["pred_point_count"] == 1
    assert all_005["precision"] == 1.0
    assert all_005["recall"] == 0.5
