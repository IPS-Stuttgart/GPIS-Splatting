from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gpis_splatting.gaussian_native_features import (
    append_3dgs_native_features_to_field_scores,
    build_3dgs_native_feature_table,
    default_trained_3dgs_feature_sets,
)
from gpis_splatting.primary_confidence import run_primary_calibrated_confidence


def test_build_3dgs_native_feature_table_extracts_gaussian_attributes(tmp_path: Path) -> None:
    ply_path = tmp_path / "point_cloud.ply"
    write_feature_rich_3dgs_ply(ply_path)

    features = build_3dgs_native_feature_table(ply_path=ply_path)

    assert features["splat_index"].tolist() == [0, 1, 2, 3]
    assert features["gaussian_alpha"].between(0.0, 1.0).all()
    assert np.allclose(features["gaussian_opacity_logit"], [-2.0, -0.5, 0.5, 2.0])
    assert np.all(features["gaussian_scale_mean"] > 0.0)
    assert np.all(features["gaussian_scale_anisotropy"] >= 1.0)
    assert features["gaussian_sh_rest_energy"].iloc[-1] > features["gaussian_sh_rest_energy"].iloc[0]
    assert "score_negative_gaussian_scale_max" in features.columns


def test_append_3dgs_native_features_aligns_by_splat_index_and_calibrates(tmp_path: Path) -> None:
    ply_path = tmp_path / "point_cloud.ply"
    write_feature_rich_3dgs_ply(ply_path)
    field_scores_path = tmp_path / "unit_gpis_field_scores.csv"
    make_field_scores(splat_index=np.asarray([3, 1, 2, 0], dtype=np.int64)).to_csv(field_scores_path, index=False)

    augmented = append_3dgs_native_features_to_field_scores(
        field_scores_path=field_scores_path,
        ply_path=ply_path,
        output_path=tmp_path / "unit_gpis_field_scores_with_3dgs_native.csv",
    )
    table = pd.read_csv(augmented["field_scores_path"])

    assert table["splat_index"].tolist() == [3, 1, 2, 0]
    assert np.allclose(table["gaussian_opacity_logit"].to_numpy(), [2.0, -0.5, 0.5, -2.0])
    assert set(augmented["status"]["registered_feature_sets"]) >= {"gpis_3dgs_native", "gpis_3dgs_fused", "gpis_3dgs_scores"}

    result = run_primary_calibrated_confidence(
        field_scores_path=augmented["field_scores_path"],
        output_dir=tmp_path,
        method_name="unit_native",
        thresholds=(0.05,),
        topk_fractions=(0.5, 1.0),
        feature_sets=default_trained_3dgs_feature_sets(),
        baseline_scores=("score_current_gate",),
        isotonic_scores=("score_current_gate",),
        validation_fraction=0.25,
        seed=1,
        logistic_iterations=5,
        learning_rate=0.05,
        regularization=1e-3,
        gate_count=4,
        missing_gate_value=1.0,
    )

    feature_sets = set(result["status"]["feature_sets"])
    assert "gpis_3dgs_native" in feature_sets
    assert "gpis_3dgs_fused" in feature_sets
    assert result["primary_gate_path"].exists()


def make_field_scores(*, splat_index: np.ndarray) -> pd.DataFrame:
    nearest_gt_distance_by_index = np.asarray([0.08, 0.04, 0.03, 0.01], dtype=np.float64)
    distance = nearest_gt_distance_by_index[splat_index]
    confidence_like = 1.0 - np.clip(distance / 0.1, 0.0, 1.0)
    return pd.DataFrame(
        {
            "splat_index": splat_index,
            "nearest_gt_distance": distance,
            "mu": distance,
            "sigma": np.full_like(distance, 0.05),
            "grad_norm": np.ones_like(distance),
            "signed_distance": distance,
            "distance_std": np.full_like(distance, 0.05),
            "score_current_gate": confidence_like,
            "score_raw_surface_band": confidence_like,
            "score_variance_penalized_band": confidence_like * 0.9,
            "score_variance_penalized_exp": confidence_like * 0.85,
            "score_negative_abs_distance": -distance,
            "score_negative_distance_std": -np.full_like(distance, 0.05),
            "score_exp_neg_abs_distance": np.exp(-distance),
            "score_negative_abs_mu": -distance,
        }
    )


def write_feature_rich_3dgs_ply(path: Path) -> None:
    rows = [
        [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.00, 0.00, 0.00, -2.0, -4.0, -4.0, -4.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.2, 0.3, 0.4, 0.01, 0.00, 0.00, -0.5, -3.0, -3.5, -3.0, 1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.3, 0.4, 0.5, 0.02, 0.01, 0.00, 0.5, -2.0, -2.5, -2.0, 1.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.03, 0.02, 0.01, 2.0, -1.0, -1.5, -1.0, 1.0, 0.0, 0.0, 0.0],
    ]
    header = [
        "ply",
        "format ascii 1.0",
        "element vertex 4",
        "property float x",
        "property float y",
        "property float z",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float f_rest_0",
        "property float f_rest_1",
        "property float f_rest_2",
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
