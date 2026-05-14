from __future__ import annotations

import csv
from pathlib import Path

import torch

from gpis_splatting.backend_benchmark import BackendBenchmarkConfig, make_benchmark_samples, run_backend_benchmark


def test_make_benchmark_samples_is_deterministic() -> None:
    points_a, sdf_a = make_benchmark_samples(12, shape="torus", seed=5)
    points_b, sdf_b = make_benchmark_samples(12, shape="torus", seed=5)

    torch.testing.assert_close(points_a, points_b)
    torch.testing.assert_close(sdf_a, sdf_b)
    assert points_a.shape == (12, 3)
    assert sdf_a.shape == (12,)


def test_backend_benchmark_writes_artifacts(tmp_path: Path) -> None:
    result = run_backend_benchmark(
        BackendBenchmarkConfig(
            output_dir=tmp_path,
            benchmark_name="unit_backend_benchmark",
            backends=("dense_exact", "local_exact", "inducing_points"),
            n_train=32,
            n_query=10,
            batch_size=5,
            num_neighbors=8,
            num_inducing=12,
            fit_batch_size=16,
        )
    )

    assert result["csv_path"].exists()
    assert result["config_path"].exists()
    assert result["status_path"].exists()
    assert result["report_path"].exists()
    rows = list(csv.DictReader(result["csv_path"].open(encoding="utf-8")))

    assert [row["backend"] for row in rows] == ["dense_exact", "local_exact", "inducing_points"]
    assert all(row["status"] == "success" for row in rows)
    assert rows[0]["mean_rmse_vs_dense"] != ""
    assert float(rows[0]["mean_rmse_vs_dense"]) < 1e-10
    assert result["status"]["ok"] is True
