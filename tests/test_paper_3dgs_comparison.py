from __future__ import annotations

from pathlib import Path

import pandas as pd

from gpis_splatting.paper_3dgs_comparison import Paper3DGSComparisonConfig, run_paper_3dgs_comparison


def test_paper_3dgs_comparison_aggregates_photometry_geometry_and_performance(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    manifest_path = tmp_path / "manifest.csv"
    photometry_path = tmp_path / "photometry.csv"
    geometry_path = tmp_path / "geometry.csv"
    performance_path = tmp_path / "performance.csv"

    pd.DataFrame(
        [
            {
                "variant": "baseline",
                "variant_kind": "baseline",
                "point_cloud_path": str(tmp_path / "baseline.ply"),
                "model_dir": str(tmp_path / "baseline"),
                "retained_count": 100,
                "retention_fraction": 1.0,
                "gate_threshold": None,
            },
            {
                "variant": "gate_scaled",
                "variant_kind": "gate_scaled",
                "point_cloud_path": str(tmp_path / "gate_scaled.ply"),
                "model_dir": str(tmp_path / "gate_scaled"),
                "retained_count": 80,
                "retention_fraction": 0.8,
                "gate_threshold": None,
            },
        ]
    ).to_csv(manifest_path, index=False)
    pd.DataFrame(
        [
            {"variant": "baseline", "mean_psnr": 25.0, "mean_ssim": 0.81, "mean_lpips_vgg": 0.18, "image_count": 3},
            {"variant": "gate_scaled", "mean_psnr": 25.5, "mean_ssim": 0.83, "mean_lpips_vgg": 0.16, "image_count": 3},
        ]
    ).to_csv(photometry_path, index=False)
    pd.DataFrame(
        [
            {"variant": "baseline", "geometry_threshold": 0.05, "precision": 0.6, "recall": 0.5, "f_score": 0.545, "chamfer_l1": 0.08, "chamfer_l2": 0.003},
            {"variant": "gate_scaled", "geometry_threshold": 0.05, "precision": 0.7, "recall": 0.55, "f_score": 0.616, "chamfer_l1": 0.06, "chamfer_l2": 0.002},
        ]
    ).to_csv(geometry_path, index=False)
    pd.DataFrame(
        [
            {"variant": "baseline", "fps": 72.0, "peak_vram_mb": 2300.0, "rendered_gaussian_count": 100, "device": "cuda:0"},
            {"variant": "gate_scaled", "fps": 85.0, "peak_vram_mb": 1800.0, "rendered_gaussian_count": 80, "device": "cuda:0"},
        ]
    ).to_csv(performance_path, index=False)

    result = run_paper_3dgs_comparison(
        Paper3DGSComparisonConfig(
            output_dir=tmp_path / "out",
            scenes=(
                {
                    "scene": "toy",
                    "scene_dir": str(scene_dir),
                    "manifest_path": str(manifest_path),
                    "render_comparison_path": str(photometry_path),
                    "geometry_comparison_path": str(geometry_path),
                    "performance_path": str(performance_path),
                },
            ),
            require_lpips=True,
            require_performance=True,
            fail_on_missing=True,
        )
    )

    comparison = pd.read_csv(result["comparison_path"])
    table = pd.read_csv(result["paper_table_path"])
    checks = pd.read_csv(result["checks_path"])

    assert result["passed"] is True
    assert comparison.shape[0] == 2
    assert set(comparison["variant"]) == {"baseline", "gate_scaled"}
    assert float(comparison.loc[comparison["variant"] == "gate_scaled", "f_score"].iloc[0]) == 0.616
    assert "PSNR↑" in table.columns
    assert "VRAM MB↓" in table.columns
    assert checks["passed"].all()
    assert result["report_path"].exists()
