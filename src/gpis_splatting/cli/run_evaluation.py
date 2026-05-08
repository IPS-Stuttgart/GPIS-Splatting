from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.cli.run_ablation import main as run_ablation_main
from gpis_splatting.cli.summarize_ablation import main as summarize_ablation_main
from gpis_splatting.evaluation import (
    build_ablation_args,
    evaluate_ablation_artifacts,
    get_evaluation_preset,
    preset_names,
    write_evaluation_artifacts,
)
from gpis_splatting.evaluation_config import load_evaluation_preset_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible GPIS-splatting evaluation preset.")
    parser.add_argument("--preset", choices=preset_names(), default="synthetic_quick")
    parser.add_argument("--preset-config", default=None, help="Optional JSON preset file. When set, this overrides --preset.")
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--experiment-name", default=None, help="Defaults to evaluation_<preset>.")
    parser.add_argument("--seed", type=int, default=None, help="Override the preset seed.")
    parser.add_argument("--primary-metric", choices=("psnr_delta", "rmse_delta", "iou_delta"), default="psnr_delta")
    parser.add_argument(
        "--benchmark-target",
        default=None,
        help="Optional JSON target manifest to include in the evaluation report.",
    )
    parser.add_argument(
        "--no-fail-on-checks",
        action="store_true",
        help="Write artifacts but return success even if evaluation checks fail.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.preset_config:
        preset = load_evaluation_preset_file(args.preset_config)
        preset_name = str(preset.get("name") or Path(args.preset_config).stem)
    else:
        preset = get_evaluation_preset(args.preset)
        preset_name = args.preset
    experiment_name = args.experiment_name or f"evaluation_{preset_name}"
    ablation_root = Path(args.output_root) / experiment_name

    ablation_args = build_ablation_args(
        preset,
        output_root=args.output_root,
        experiment_name=experiment_name,
        seed=args.seed,
    )
    run_ablation_main(ablation_args)
    summarize_ablation_main(
        [
            "--ablation-root",
            str(ablation_root),
            "--primary-metric",
            args.primary_metric,
        ]
    )

    status = evaluate_ablation_artifacts(
        ablation_root=ablation_root,
        preset_name=preset_name,
        preset=preset,
        primary_metric=args.primary_metric,
        benchmark_target=args.benchmark_target,
    )
    paths = write_evaluation_artifacts(
        output_dir=ablation_root,
        status=status,
        preset=preset,
        preset_name=preset_name,
        ablation_args=ablation_args,
    )
    print(f"Wrote {paths['config']}")
    print(f"Wrote {paths['checks']}")
    print(f"Wrote {paths['status']}")
    print(f"Wrote {paths['report']}")

    if not status["passed"] and not args.no_fail_on_checks:
        failed = [check["check"] for check in status["checks"] if not check["passed"]]
        raise SystemExit(f"Evaluation checks failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
