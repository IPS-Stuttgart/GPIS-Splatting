from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gpis_splatting.cli.evaluate import main as evaluate_main
from gpis_splatting.cli.fit_gpis import main as fit_main
from gpis_splatting.cli.generate_scene import main as generate_main
from gpis_splatting.cli.render_splats import main as render_main
from gpis_splatting.cli.run_ablation import main as ablation_main
from gpis_splatting.cli.run_evaluation import main as evaluation_main
from gpis_splatting.cli.summarize_ablation import main as summarize_main
from gpis_splatting.serialization import read_json
from gpis_splatting.splats import load_splats


def test_small_end_to_end_pipeline(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    scene = "sphere_regression"

    generate_main(
        [
            "--shape",
            "sphere",
            "--scene",
            scene,
            "--num-points",
            "80",
            "--noise-std",
            "0.03",
            "--seed",
            "5",
            "--output-root",
            str(root),
        ]
    )
    fit_main(
        [
            "--scene",
            scene,
            "--grid-size",
            "12",
            "--output-root",
            str(root),
        ]
    )
    render_main(
        [
            "--scene",
            scene,
            "--view",
            "front",
            "--image-size",
            "48",
            "--num-splats",
            "120",
            "--epsilon",
            "0.11",
            "--feedback-iterations",
            "1",
            "--feedback-pseudo-points",
            "16",
            "--feedback-min-gate",
            "0.0",
            "--feedback-selector",
            "uncertainty",
            "--output-root",
            str(root),
        ]
    )
    evaluate_main(["--scene", scene, "--output-root", str(root)])

    out_dir = root / scene
    assert (out_dir / "config.json").exists()
    assert (out_dir / "samples.npz").exists()
    assert (out_dir / "gpis_model.npz").exists()
    assert (out_dir / "posterior_grid.npz").exists()
    assert (out_dir / "render_plain_front.png").exists()
    assert (out_dir / "render_gpis_front.png").exists()
    assert (out_dir / "render_feedback_front.png").exists()
    assert (out_dir / "feedback_gpis_model.npz").exists()
    assert (out_dir / "feedback_trace.csv").exists()
    assert (out_dir / "feedback_splat_gates.npz").exists()
    assert (out_dir / "metrics.csv").exists()

    metrics = pd.read_csv(out_dir / "metrics.csv").iloc[0]
    assert metrics["rmse_sdf"] < 0.35
    assert metrics["iou_inside"] > 0.35
    assert metrics["psnr_gpis_front"] >= metrics["psnr_plain_front"]
    assert "psnr_feedback_front" in metrics
    assert "feedback_rmse_sdf" in metrics


def test_render_splats_regenerates_cache_when_generation_arguments_change(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    scene = "sphere_splat_cache_regression"

    generate_main(
        [
            "--shape",
            "sphere",
            "--scene",
            scene,
            "--num-points",
            "32",
            "--noise-std",
            "0.03",
            "--seed",
            "5",
            "--output-root",
            str(root),
        ]
    )

    common_args = [
        "--scene",
        scene,
        "--view",
        "front",
        "--image-size",
        "16",
        "--use-gpis-gate",
        "false",
        "--output-root",
        str(root),
    ]

    render_main(
        [
            *common_args,
            "--num-splats",
            "12",
            "--seed",
            "1",
        ]
    )
    out_dir = root / scene
    splat_path = out_dir / "splats.npz"
    metadata_path = out_dir / "splats_metadata.json"

    first = load_splats(str(splat_path))
    first_centers = first.centers.detach().cpu().numpy().copy()
    assert first.centers.shape[0] == 12
    assert read_json(metadata_path) == {
        "schema_version": 1,
        "generator": "make_candidate_splats",
        "shape": "sphere",
        "num_splats": 12,
        "seed": 1,
    }

    render_main(
        [
            *common_args,
            "--num-splats",
            "12",
            "--seed",
            "2",
        ]
    )
    second = load_splats(str(splat_path))
    second_centers = second.centers.detach().cpu().numpy()
    assert second.centers.shape[0] == 12
    assert not np.allclose(first_centers, second_centers)
    assert read_json(metadata_path)["seed"] == 2

    render_main(
        [
            *common_args,
            "--num-splats",
            "9",
            "--seed",
            "2",
        ]
    )
    third = load_splats(str(splat_path))
    metadata = read_json(metadata_path)
    assert third.centers.shape[0] == 9
    assert metadata["num_splats"] == 9
    assert metadata["seed"] == 2


def test_feedback_ablation_runner_writes_comparison_table(tmp_path: Path) -> None:
    root = tmp_path / "experiments"

    ablation_main(
        [
            "--shapes",
            "sphere",
            "--feedback-iterations",
            "0",
            "1",
            "--feedback-selectors",
            "gate",
            "uncertainty",
            "--num-points",
            "45",
            "--grid-size",
            "8",
            "--image-size",
            "32",
            "--num-splats",
            "55",
            "--epsilon",
            "0.11",
            "--feedback-pseudo-points",
            "6",
            "--feedback-min-gate",
            "0.0",
            "--output-root",
            str(root),
            "--experiment-name",
            "tiny_ablation",
        ]
    )

    out_dir = root / "tiny_ablation"
    metrics_path = out_dir / "ablation_metrics.csv"
    assert (out_dir / "ablation_config.json").exists()
    assert metrics_path.exists()
    assert (out_dir / "sphere_fb0" / "metrics.csv").exists()
    assert (out_dir / "sphere_fb1_gate" / "feedback_trace.csv").exists()
    assert (out_dir / "sphere_fb1_uncertainty" / "feedback_trace.csv").exists()

    metrics = pd.read_csv(metrics_path)
    assert list(metrics["feedback_iterations"]) == [0, 1, 1]
    assert list(metrics["feedback_selector"]) == ["none", "gate", "uncertainty"]
    assert set(metrics["shape"]) == {"sphere"}
    assert metrics.loc[metrics["feedback_iterations"] == 0, "feedback_selected_splats"].iloc[0] == 0
    assert (metrics.loc[metrics["feedback_iterations"] == 1, "feedback_selected_splats"] == 6).all()
    assert "psnr_gpis_front" in metrics
    assert "psnr_feedback_front" in metrics


def test_ablation_summarizer_writes_plots_and_winners(tmp_path: Path) -> None:
    root = tmp_path / "experiments" / "tiny_ablation"
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "shape": "sphere",
                "feedback_iterations": 0,
                "feedback_selector": "none",
                "scene": "sphere_fb0",
                "psnr_gpis_front": 18.0,
                "rmse_sdf": 0.24,
                "iou_inside": 0.55,
                "feedback_selected_splats": 0,
            },
            {
                "shape": "sphere",
                "feedback_iterations": 1,
                "feedback_selector": "gate",
                "scene": "sphere_fb1_gate",
                "psnr_gpis_front": 18.0,
                "psnr_feedback_front": 18.5,
                "rmse_sdf": 0.24,
                "feedback_rmse_sdf": 0.22,
                "iou_inside": 0.55,
                "feedback_iou_inside": 0.57,
                "feedback_selected_splats": 6,
            },
            {
                "shape": "sphere",
                "feedback_iterations": 1,
                "feedback_selector": "uncertainty",
                "scene": "sphere_fb1_uncertainty",
                "psnr_gpis_front": 18.0,
                "psnr_feedback_front": 19.2,
                "rmse_sdf": 0.24,
                "feedback_rmse_sdf": 0.20,
                "iou_inside": 0.55,
                "feedback_iou_inside": 0.59,
                "feedback_selected_splats": 6,
            },
        ]
    ).to_csv(root / "ablation_metrics.csv", index=False)

    summarize_main(["--ablation-root", str(root)])

    summary_dir = root / "summary"
    assert (summary_dir / "ablation_summary.csv").exists()
    assert (summary_dir / "ablation_winners.csv").exists()
    assert (summary_dir / "ablation_summary.md").exists()
    assert (summary_dir / "psnr_delta_by_selector.png").exists()
    assert (summary_dir / "rmse_delta_by_selector.png").exists()
    assert (summary_dir / "selected_splats_by_selector.png").exists()
    assert (summary_dir / "feedback_iteration_trend.png").exists()

    summary = pd.read_csv(summary_dir / "ablation_summary.csv")
    winners = pd.read_csv(summary_dir / "ablation_winners.csv")
    assert {round(value, 6) for value in summary["psnr_delta"]} == {0.0, 0.5, 1.2}
    assert winners.iloc[0]["shape"] == "sphere"
    assert winners.iloc[0]["feedback_selector"] == "uncertainty"
    assert round(winners.iloc[0]["psnr_delta"], 6) == 1.2


def test_evaluation_runner_writes_report_and_checks(tmp_path: Path) -> None:
    root = tmp_path / "experiments"

    evaluation_main(
        [
            "--preset",
            "synthetic_ci",
            "--output-root",
            str(root),
            "--experiment-name",
            "tiny_evaluation",
        ]
    )

    out_dir = root / "tiny_evaluation"
    assert (out_dir / "ablation_metrics.csv").exists()
    assert (out_dir / "summary" / "ablation_summary.csv").exists()
    assert (out_dir / "summary" / "ablation_winners.csv").exists()
    assert (out_dir / "evaluation_config.json").exists()
    assert (out_dir / "evaluation_checks.csv").exists()
    assert (out_dir / "evaluation_status.json").exists()
    assert (out_dir / "evaluation_report.md").exists()

    checks = pd.read_csv(out_dir / "evaluation_checks.csv")
    assert checks["passed"].all()
    assert set(checks["check"]) >= {
        "expected_row_count",
        "base_metric_columns_present",
        "feedback_metric_columns_present",
        "winner_rows_available",
    }
