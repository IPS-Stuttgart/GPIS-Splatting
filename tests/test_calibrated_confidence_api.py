from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gpis_splatting.calibrated_confidence_api import (
    ConfidenceFeatureConfig,
    ConfidenceFeatureExtractor,
    ConfidenceFitConfig,
    ConfidenceSplitConfig,
    fit_calibrated_confidence,
    make_leakage_free_split,
    run_calibrated_confidence_fit,
)
from gpis_splatting.confidence import read_calibrated_confidence_bundle


def confidence_table() -> pd.DataFrame:
    rows = []
    for source in range(12):
        for copy in range(2):
            good = source < 6
            distance = 0.01 + 0.002 * copy if good else 0.12 + 0.01 * copy
            rows.append(
                {
                    "splat_index": len(rows),
                    "scene": "ignatius",
                    "source_splat_index": source,
                    "query_x": float(source),
                    "query_y": float(copy),
                    "query_z": 0.0,
                    "mu": distance * (1.0 if good else 2.0),
                    "variance": 0.01 + 0.001 * copy,
                    "grad_norm": 1.0,
                    "signed_distance": distance,
                    "distance_std": 0.02 + 0.002 * copy,
                    "score_current_gate": 1.0 - distance,
                    "score_raw_surface_band": np.exp(-distance),
                    "score_variance_penalized_band": np.exp(-distance) / 1.1,
                    "nearest_gt_distance": distance,
                    "within_0p05": distance <= 0.05,
                }
            )
    return pd.DataFrame(rows)


def test_feature_extractor_derives_features_and_blocks_label_leakage() -> None:
    extracted = ConfidenceFeatureExtractor().fit_transform(confidence_table())

    assert "abs_mu" in extracted.table.columns
    assert "abs_signed_distance" in extracted.table.columns
    assert "distance_snr" in extracted.table.columns
    assert "nearest_gt_distance" not in extracted.feature_columns
    assert "within_0p05" not in extracted.feature_columns
    assert "query_x" not in extracted.feature_columns
    assert "score_current_gate" in extracted.feature_columns


def test_explicit_leaky_feature_is_rejected() -> None:
    extractor = ConfidenceFeatureExtractor(ConfidenceFeatureConfig(feature_columns=("nearest_gt_distance",)))

    try:
        extractor.fit_transform(confidence_table())
    except ValueError as exc:
        assert "Refusing to use" in str(exc)
    else:
        raise AssertionError("Expected explicit label-like feature to fail.")


def test_leakage_free_split_keeps_source_groups_intact() -> None:
    table = confidence_table()
    split = make_leakage_free_split(
        table,
        ConfidenceSplitConfig(validation_fraction=0.35, seed=3, group_columns=("source_splat_index",), auto_group_columns=False),
    )

    train_sources = set(table.loc[split.train_mask, "source_splat_index"].tolist())
    validation_sources = set(table.loc[split.validation_mask, "source_splat_index"].tolist())

    assert train_sources
    assert validation_sources
    assert train_sources.isdisjoint(validation_sources)


def test_fit_serializes_bundle_and_predictions_round_trip() -> None:
    result = fit_calibrated_confidence(
        confidence_table(),
        config=ConfidenceFitConfig(
            thresholds=(0.05,),
            split_config=ConfidenceSplitConfig(validation_fraction=0.35, seed=5, group_columns=("source_splat_index",), auto_group_columns=False),
            num_bins=5,
            logistic_iterations=50,
        ),
    )

    assert result.feature_columns
    assert "confidence_0p05" in result.predictions.columns
    assert not result.reliability_tables["0p05"].empty
    assert result.bundle.metadata["split"]["strategy"] == "columns=source_splat_index"
    assert result.selected_methods[0].geometry_threshold == 0.05


def test_run_writes_reliability_plot_and_reusable_model(tmp_path: Path) -> None:
    scores_path = tmp_path / "method_gpis_field_scores.csv"
    confidence_table().to_csv(scores_path, index=False)

    result = run_calibrated_confidence_fit(
        field_scores_path=scores_path,
        output_dir=tmp_path,
        method_name="method",
        config=ConfidenceFitConfig(
            thresholds=(0.05,),
            split_config=ConfidenceSplitConfig(validation_fraction=0.35, seed=7, group_columns=("source_splat_index",), auto_group_columns=False),
            num_bins=5,
            logistic_iterations=50,
        ),
    )

    model_path = result.artifacts["model_bundle_path"]
    plot_path = result.artifacts["reliability_plot_paths"]["0p05"]
    predictions_path = result.artifacts["predictions_path"]
    bundle = read_calibrated_confidence_bundle(model_path)
    predictions = bundle.predict(pd.read_csv(result.artifacts["feature_table_path"]))

    assert model_path.exists()
    assert plot_path.exists()
    assert predictions_path.exists()
    assert "confidence_0p05" in predictions.columns
