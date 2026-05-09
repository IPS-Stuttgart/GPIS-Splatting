from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gpis_splatting.cli.run_experiment_matrix import collect_artifact_paths
from gpis_splatting.experiment_matrix import ExperimentMatrixConfig, run_experiment_matrix


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_run_experiment_matrix_aggregates_available_cases(tmp_path: Path) -> None:
    raw_gate = write_csv(
        tmp_path / "raw_gate_sweep.csv",
        [
            {"selection": "all", "geometry_threshold": 0.05, "retention_fraction": 1.0, "precision": 0.4, "recall": 0.8, "f_score": 0.53, "chamfer_l1": 0.12},
            {
                "selection": "gate_ge",
                "gate_threshold": 0.25,
                "geometry_threshold": 0.05,
                "retention_fraction": 0.7,
                "precision": 0.6,
                "recall": 0.7,
                "f_score": 0.64,
                "chamfer_l1": 0.09,
            },
            {
                "selection": "gate_ge",
                "gate_threshold": 0.5,
                "geometry_threshold": 0.05,
                "retention_fraction": 0.4,
                "precision": 0.8,
                "recall": 0.45,
                "f_score": 0.58,
                "chamfer_l1": 0.08,
            },
        ],
    )
    filtering = write_csv(
        tmp_path / "calibrated_filtering.csv",
        [
            {
                "variant": "baseline",
                "variant_kind": "baseline",
                "geometry_threshold": 0.05,
                "retention_fraction": 1.0,
                "precision": 0.45,
                "recall": 0.82,
                "f_score": 0.58,
                "chamfer_l1": 0.11,
                "mean_psnr": 24.0,
            },
            {
                "variant": "gate_scaled",
                "variant_kind": "gate_scaled",
                "geometry_threshold": 0.05,
                "retention_fraction": 1.0,
                "precision": 0.50,
                "recall": 0.84,
                "f_score": 0.63,
                "chamfer_l1": 0.10,
                "mean_psnr": 24.5,
            },
            {
                "variant": "gate_ge_0p5",
                "variant_kind": "gate_threshold",
                "geometry_threshold": 0.05,
                "retention_fraction": 0.5,
                "precision": 0.80,
                "recall": 0.60,
                "f_score": 0.69,
                "chamfer_l1": 0.07,
                "mean_psnr": 23.6,
            },
            {
                "variant": "random_same_retention_0p5_seed0",
                "variant_kind": "random_same_retention",
                "geometry_threshold": 0.05,
                "retention_fraction": 0.5,
                "precision": 0.99,
                "recall": 0.99,
                "f_score": 0.99,
                "chamfer_l1": 0.01,
                "mean_psnr": 30.0,
            },
        ],
    )
    render = write_csv(
        tmp_path / "trained_3dgs_render.csv",
        [
            {"variant": "baseline", "variant_kind": "baseline", "retention_fraction": 1.0, "retained_count": 100, "mean_psnr": 25.0, "mean_ssim": 0.82},
            {"variant": "gate_scaled", "variant_kind": "gate_scaled", "retention_fraction": 1.0, "retained_count": 100, "mean_psnr": 26.0, "mean_ssim": 0.84},
        ],
    )

    result = run_experiment_matrix(
        ExperimentMatrixConfig(
            output_dir=tmp_path / "matrix",
            artifact_paths={
                "raw_gate_sweep": raw_gate,
                "calibrated_filtering_comparison": filtering,
                "trained_3dgs_render_comparison": render,
            },
        )
    )

    summary = result["summary"].set_index("case_id")
    checks = result["checks"].set_index("case_id")
    assert result["summary_path"].exists()
    assert result["report_path"].exists()
    assert checks.loc["A", "passed"]
    assert checks.loc["B", "passed"]
    assert checks.loc["C", "passed"]
    assert checks.loc["D", "passed"]
    assert not checks.loc["E", "passed"]
    assert summary.loc["A", "mean_psnr"] == 25.0
    assert summary.loc["B", "gate_threshold"] == 0.25
    assert summary.loc["C", "variant_kind"] == "gate_scaled"
    assert summary.loc["D", "variant"] == "gate_ge_0p5"
    assert summary.loc["D", "f_score"] == 0.69
    assert summary.loc["C", "delta_mean_psnr_vs_baseline"] == -0.5
    assert "GPIS/3DGS Experiment Matrix" in result["report_path"].read_text(encoding="utf-8")


def test_run_experiment_matrix_writes_placeholders_without_artifacts(tmp_path: Path) -> None:
    result = run_experiment_matrix(ExperimentMatrixConfig(output_dir=tmp_path / "matrix"))

    assert result["summary"].shape[0] == 6
    assert result["checks"]["passed"].sum() == 0
    assert result["manifest_path"].exists()
    assert not result["passed"]


def test_run_experiment_matrix_can_fail_on_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required artifacts"):
        run_experiment_matrix(ExperimentMatrixConfig(output_dir=tmp_path / "matrix", fail_on_missing=True))


def test_collect_artifact_paths_rejects_unknown_role() -> None:
    class Args:
        trained_3dgs_render_comparison = None
        trained_3dgs_geometry_summary = None
        raw_gate_sweep = None
        calibrated_gate_sweep = None
        calibrated_confidence_metrics = None
        calibrated_filtering_comparison = None
        regularized_3dgs_render_comparison = None
        regularized_geometry_summary = None
        regularized_calibrated_render_comparison = None
        regularized_calibrated_filtering_comparison = None
        artifact = ["unknown_role=foo.csv"]

    with pytest.raises(ValueError, match="Unknown artifact role"):
        collect_artifact_paths(Args())
