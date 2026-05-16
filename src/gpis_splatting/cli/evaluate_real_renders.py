from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.real_benchmark import evaluate_real_renders


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate predicted render images against a prepared real scene split.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/evaluations.")
    parser.add_argument("--method-name", default="method")
    parser.add_argument("--split", default="test")
    parser.add_argument("--benchmark-target", default=None)
    parser.add_argument("--compute-lpips", type=str_to_bool, default=False)
    parser.add_argument("--require-all", type=str_to_bool, default=True)
    parser.add_argument(
        "--allow-diagnostic-proxy",
        type=str_to_bool,
        default=False,
        help="Allow PSNR/SSIM/LPIPS evaluation of render_real_splats CPU proxy outputs. Keep false for photometric claims.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    output_dir = Path(args.output_dir) if args.output_dir is not None else scene_dir / "evaluations"
    status = evaluate_real_renders(
        scene_dir=scene_dir,
        predictions_dir=args.predictions_dir,
        output_dir=output_dir,
        method_name=args.method_name,
        split=args.split,
        benchmark_target=args.benchmark_target,
        compute_lpips=args.compute_lpips,
        require_all=args.require_all,
        allow_diagnostic_proxy=args.allow_diagnostic_proxy,
    )
    print(f"Wrote {status['metrics_path']}")
    print(f"Wrote {status['summary_path']}")
    print(f"Wrote {status['report_path']}")
    print(f"mean_psnr: {status['summary']['mean_psnr']:.6g}")
    print(f"mean_ssim: {status['summary']['mean_ssim']:.6g}")


if __name__ == "__main__":
    main()
