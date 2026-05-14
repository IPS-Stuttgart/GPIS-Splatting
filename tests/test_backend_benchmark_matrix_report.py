from __future__ import annotations

from pathlib import Path

import pandas as pd

from gpis_splatting.cli.append_backend_benchmark_to_matrix_report import append_backend_benchmark_to_report


def test_append_backend_benchmark_to_matrix_report(tmp_path: Path) -> None:
    report = tmp_path / "matrix_report.md"
    benchmark = tmp_path / "backend_benchmark.csv"
    output = tmp_path / "combined_report.md"
    report.write_text("# GPIS/3DGS Experiment Matrix\n\nExisting report.\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "backend": "dense_exact",
                "status": "success",
                "fit_time_sec": 0.1,
                "predict_time_sec": 0.02,
                "queries_per_sec": 5000.0,
                "model_storage_mib": 1.25,
                "mean_rmse_vs_dense": 0.0,
                "gate_rmse_vs_dense": 0.0,
            },
            {
                "backend": "inducing_points",
                "status": "success",
                "fit_time_sec": 0.04,
                "predict_time_sec": 0.01,
                "queries_per_sec": 10000.0,
                "model_storage_mib": 0.5,
                "mean_rmse_vs_dense": 0.02,
                "gate_rmse_vs_dense": 0.01,
            },
        ]
    ).to_csv(benchmark, index=False)

    append_backend_benchmark_to_report(matrix_report=report, backend_benchmark=benchmark, output_report=output)

    text = output.read_text(encoding="utf-8")
    assert "Existing report." in text
    assert "## GPIS Backend Benchmark" in text
    assert "dense_exact" in text
    assert "inducing_points" in text
    assert "gate_rmse_vs_dense" in text
