from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gpis_splatting.gpis_training_prior import GpisTrainingPriorConfig, export_gpis_training_prior


def test_export_gpis_training_prior_writes_soft_training_artifacts(tmp_path: Path) -> None:
    ply_path = tmp_path / "point_cloud.ply"
    write_tiny_3dgs_ply(ply_path)
    gate_path = tmp_path / "gate.npz"
    np.savez_compressed(gate_path, gate=np.asarray([0.9, 0.2, 0.8], dtype=np.float32))
    scores_path = tmp_path / "scores.csv"
    pd.DataFrame({"distance_std": [0.1, 0.8, 0.9], "score_gpis_surface_likelihood": [0.9, 0.1, 0.7]}).to_csv(scores_path, index=False)

    result = export_gpis_training_prior(
        input_ply_path=ply_path,
        gate_path=gate_path,
        field_scores_path=scores_path,
        output_dir=tmp_path / "prior",
        method_name="unit_prior",
        config=GpisTrainingPriorConfig(clone_top_count=1),
    )

    assert result["prior_path"].exists()
    assert result["initialization_seed_ply_path"].exists()
    assert result["trainer_hooks_path"].exists()
    with np.load(result["prior_path"], allow_pickle=False) as data:
        assert data["densify_weight"].shape == (3,)
        assert data["prune_candidate_mask"].tolist() == [False, True, False]
        assert "opacity_regularization_weight" in data.files
        assert "opacity_target_alpha" in data.files
    assert result["status"]["initialization_candidate_count"] == 2
    assert result["status"]["seed_count"] == 3
    assert "lambda_gpis_opacity" in result["trainer_hooks_path"].read_text(encoding="utf-8")


def write_tiny_3dgs_ply(path: Path) -> None:
    rows = [
        [0.0, 0.0, 0.0, -1.0, -4.0, -4.0, -4.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, -3.0, -3.0, -3.0, 1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 1.0, -2.0, -2.0, -2.0, 1.0, 0.0, 0.0, 0.0],
    ]
    header = [
        "ply",
        "format ascii 1.0",
        "element vertex 3",
        "property float x",
        "property float y",
        "property float z",
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
    path.write_text("\n".join([*header, *(" ".join(str(value) for value in row) for row in rows)]) + "\n", encoding="ascii")
