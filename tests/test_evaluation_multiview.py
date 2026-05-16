from __future__ import annotations

from pathlib import Path

import pandas as pd

from gpis_splatting.evaluation import evaluate_ablation_artifacts


def test_evaluation_checks_all_preset_views_for_all_view_run(tmp_path: Path) -> None:
    root = tmp_path / "all_view_ablation"
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "shape": "sphere",
                "feedback_iterations": 0,
                "feedback_selector": "none",
                "scene": "sphere_fb0",
                "rmse_sdf": 0.2,
                "iou_inside": 0.7,
                "nll_distance": 0.1,
                "brier_inside": 0.1,
                "ece_inside": 0.1,
                "psnr_gpis_front": 12.0,
                "psnr_gpis_side": 3.0,
                "psnr_gpis_top": 11.0,
            }
        ]
    ).to_csv(root / "ablation_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "shape": "sphere",
                "feedback_iterations": 0,
                "feedback_selector": "none",
                "scene": "sphere_fb0",
                "psnr_delta": 0.0,
            }
        ]
    ).to_csv(summary_dir / "ablation_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "shape": "sphere",
                "feedback_iterations": 0,
                "feedback_selector": "none",
                "psnr_delta": 0.0,
            }
        ]
    ).to_csv(summary_dir / "ablation_winners.csv", index=False)

    status = evaluate_ablation_artifacts(
        ablation_root=root,
        preset_name="all_view_test",
        preset={
            "description": "All-view smoke test",
            "ablation": {
                "shapes": ["sphere"],
                "feedback_iterations": [0],
                "feedback_selectors": ["gate"],
                "view": "all",
            },
            "targets": {
                "max_rmse_sdf": 1.0,
                "min_iou_inside": 0.0,
                "min_psnr_gpis": 5.0,
            },
        },
        primary_metric="psnr_delta",
    )

    checks = {check["check"]: check for check in status["checks"]}
    assert status["views"] == ["front", "side", "top"]
    assert not status["passed"]
    assert checks["min_psnr_gpis_front"]["passed"]
    assert not checks["min_psnr_gpis_side"]["passed"]
    assert checks["min_psnr_gpis_top"]["passed"]
