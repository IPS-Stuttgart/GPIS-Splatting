from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from gpis_splatting.cli.evaluate import main as evaluate_main
from gpis_splatting.cli.fit_gpis import main as fit_main
from gpis_splatting.cli.generate_scene import main as generate_main
from gpis_splatting.cli.render_splats import main as render_main
from gpis_splatting.paths import scene_dir
from gpis_splatting.scenes import available_shapes
from gpis_splatting.serialization import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a synthetic feedback-iteration ablation.")
    parser.add_argument("--shapes", nargs="+", choices=available_shapes(), default=list(available_shapes()))
    parser.add_argument("--feedback-iterations", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--experiment-name", default="feedback_ablation")
    parser.add_argument("--num-points", type=int, default=140)
    parser.add_argument("--noise-std", type=float, default=0.035)
    parser.add_argument("--grid-size", type=int, default=22)
    parser.add_argument("--lengthscale", type=float, default=0.8)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--num-splats", type=int, default=360)
    parser.add_argument("--epsilon", type=float, default=0.09)
    parser.add_argument("--view", default="front", help="all, front, side, or top")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--feedback-pseudo-points", type=int, default=80)
    parser.add_argument("--feedback-min-gate", type=float, default=0.55)
    parser.add_argument("--feedback-pseudo-noise-std", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    experiment_root = Path(args.output_root) / args.experiment_name
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_json(experiment_root / "ablation_config.json", _config_from_args(args))

    rows = []
    for shape in args.shapes:
        for feedback_iterations in args.feedback_iterations:
            rows.append(_run_case(args, experiment_root, shape, feedback_iterations))

    metrics_path = experiment_root / "ablation_metrics.csv"
    pd.DataFrame(rows).sort_values(["shape", "feedback_iterations"]).to_csv(metrics_path, index=False)
    print(f"Wrote {metrics_path}")


def _validate_args(args: argparse.Namespace) -> None:
    if any(iteration < 0 for iteration in args.feedback_iterations):
        raise ValueError("feedback iterations must be non-negative.")
    if args.feedback_pseudo_points < 1:
        raise ValueError("feedback pseudo-points must be positive.")
    if not 0.0 <= args.feedback_min_gate <= 1.0:
        raise ValueError("feedback min-gate must be in [0, 1].")
    if args.feedback_pseudo_noise_std is not None and args.feedback_pseudo_noise_std <= 0.0:
        raise ValueError("feedback pseudo-noise std must be positive when provided.")


def _run_case(
    args: argparse.Namespace,
    experiment_root: Path,
    shape: str,
    feedback_iterations: int,
) -> dict[str, Any]:
    scene_name = f"{shape}_fb{feedback_iterations}"
    case_dir = scene_dir(scene_name, experiment_root)

    generate_main(
        [
            "--shape",
            shape,
            "--scene",
            scene_name,
            "--num-points",
            str(args.num_points),
            "--noise-std",
            str(args.noise_std),
            "--seed",
            str(args.seed),
            "--output-root",
            str(experiment_root),
        ]
    )
    fit_main(
        [
            "--scene",
            scene_name,
            "--grid-size",
            str(args.grid_size),
            "--lengthscale",
            str(args.lengthscale),
            "--variance",
            str(args.variance),
            "--output-root",
            str(experiment_root),
        ]
    )

    (case_dir / "splats.npz").unlink(missing_ok=True)
    render_args = [
        "--scene",
        scene_name,
        "--view",
        args.view,
        "--image-size",
        str(args.image_size),
        "--num-splats",
        str(args.num_splats),
        "--epsilon",
        str(args.epsilon),
        "--seed",
        str(args.seed),
        "--feedback-iterations",
        str(feedback_iterations),
        "--feedback-pseudo-points",
        str(args.feedback_pseudo_points),
        "--feedback-min-gate",
        str(args.feedback_min_gate),
        "--output-root",
        str(experiment_root),
    ]
    if args.feedback_pseudo_noise_std is not None:
        render_args.extend(["--feedback-pseudo-noise-std", str(args.feedback_pseudo_noise_std)])
    render_main(render_args)
    evaluate_main(["--scene", scene_name, "--output-root", str(experiment_root)])

    metrics = pd.read_csv(case_dir / "metrics.csv").iloc[0].to_dict()
    row: dict[str, Any] = {
        "shape": shape,
        "feedback_iterations": feedback_iterations,
        "scene": scene_name,
        "scene_dir": str(case_dir),
        **metrics,
        **_feedback_trace_summary(case_dir),
    }
    return row


def _feedback_trace_summary(case_dir: Path) -> dict[str, Any]:
    trace_path = case_dir / "feedback_trace.csv"
    if not trace_path.exists():
        return {
            "feedback_selected_splats": 0,
            "feedback_final_train_points": None,
            "feedback_final_gate_mean": None,
        }

    trace = pd.read_csv(trace_path)
    if trace.empty:
        return {
            "feedback_selected_splats": 0,
            "feedback_final_train_points": None,
            "feedback_final_gate_mean": None,
        }
    final = trace.iloc[-1]
    return {
        "feedback_selected_splats": int(trace["selected_splats"].sum()),
        "feedback_final_train_points": int(final["train_points"]),
        "feedback_final_gate_mean": float(final["gate_mean"]),
    }


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "shapes": list(args.shapes),
        "feedback_iterations": list(args.feedback_iterations),
        "num_points": args.num_points,
        "noise_std": args.noise_std,
        "grid_size": args.grid_size,
        "lengthscale": args.lengthscale,
        "variance": args.variance,
        "image_size": args.image_size,
        "num_splats": args.num_splats,
        "epsilon": args.epsilon,
        "view": args.view,
        "seed": args.seed,
        "feedback_pseudo_points": args.feedback_pseudo_points,
        "feedback_min_gate": args.feedback_min_gate,
        "feedback_pseudo_noise_std": args.feedback_pseudo_noise_std,
    }


if __name__ == "__main__":
    main()
