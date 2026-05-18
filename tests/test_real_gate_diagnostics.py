from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gpis_splatting.real_gate_diagnostics import prepare_gate_diagnostic_inputs
from gpis_splatting.real_geometry import deterministic_subsample
from gpis_splatting.serialization import write_json
from gpis_splatting.splats import SplatCloud, save_splats


def test_gate_diagnostics_subsample_from_cropped_population(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    write_json(
        scene_dir / "real_scene.json",
        {
            "schema_version": 1,
            "scene": "synthetic_crop",
            "dataset": "tanks_temples",
            "tanks_temples": {
                "ground_truth_path": "gt.ply",
                "crop_path": "crop.json",
            },
        },
    )
    write_json(scene_dir / "cameras.json", {"schema_version": 1, "frames": []})
    write_json(scene_dir / "splits.json", {"schema_version": 1, "train": [], "test": []})
    write_json(scene_dir / "crop.json", {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]})

    centers = np.asarray(
        [
            [-4.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [-1.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    splats = SplatCloud(
        centers=torch.from_numpy(centers),
        colors=torch.ones((centers.shape[0], 3), dtype=torch.float64),
        tau=torch.ones((centers.shape[0],), dtype=torch.float64),
        sigma=torch.full((centers.shape[0],), 0.04, dtype=torch.float64),
        is_surface=torch.ones((centers.shape[0],), dtype=torch.bool),
    )
    save_splats(str(scene_dir / "toy_splats.npz"), splats)
    gates = np.arange(centers.shape[0], dtype=np.float64) / 10.0
    np.savez_compressed(scene_dir / "toy_gates.npz", gate=gates)
    _write_ascii_ply(scene_dir / "gt.ply", centers[4:])

    inputs = prepare_gate_diagnostic_inputs(
        scene_dir=scene_dir,
        splats_path="toy_splats.npz",
        ground_truth_path=None,
        alignment_path=None,
        crop_path=None,
        method_name="toy",
        max_pred_points=2,
        max_gt_points=0,
        seed=1,
        apply_alignment=False,
        invert_alignment=False,
        use_crop=True,
        gate_path="toy_gates.npz",
        model_path=None,
        epsilon=0.24,
        gate_floor=0.0,
        gate_batch_size=16,
    )

    cropped_source_indices = np.arange(4, 8, dtype=np.int64)
    _, expected_sample_indices = deterministic_subsample(centers[cropped_source_indices], max_points=2, seed=1)
    expected_splat_indices = cropped_source_indices[expected_sample_indices]

    assert inputs.crop_summary["pred_total"] == 8
    assert inputs.crop_summary["pred_kept"] == 4
    assert inputs.pred_count_input == 8
    assert inputs.pred_count_sampled == 2
    assert inputs.pred_count_evaluated == 2
    assert inputs.splat_indices.tolist() == expected_splat_indices.tolist()
    assert np.allclose(inputs.gates, gates[expected_splat_indices])
    assert np.all((inputs.pred_points >= -1.0) & (inputs.pred_points <= 1.0))


def _write_ascii_ply(path: Path, points: np.ndarray) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines.extend(f"{x} {y} {z}" for x, y, z in points)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
