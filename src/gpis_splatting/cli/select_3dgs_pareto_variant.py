from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.variant_selection import DEFAULT_PARETO_PSNR_DROP_TOLERANCE, select_psnr_constrained_pareto_variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select a trained-3DGS GPIS variant on the PSNR-constrained Pareto frontier.")
    parser.add_argument("--comparison-path", required=True, help="CSV written by evaluate_3dgs_variant_renders.")
    parser.add_argument("--output-dir", default=None, help="Defaults to the comparison CSV directory.")
    parser.add_argument("--method-name", default=None, help="Prefix for output filenames. Defaults to the comparison CSV prefix.")
    parser.add_argument("--baseline-variant", default="baseline", help="Variant name used as the PSNR reference.")
    parser.add_argument("--psnr-drop-tolerance", type=float, default=DEFAULT_PARETO_PSNR_DROP_TOLERANCE, help="Maximum allowed PSNR drop from the baseline in dB.")
    parser.add_argument(
        "--objective",
        default="min_retained",
        help="Tie-breaking objective on the feasible Pareto frontier: min_retained, max_psnr, max_ssim, or min_lpips. Hyphenated aliases are accepted.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = select_psnr_constrained_pareto_variant(
        comparison_path=args.comparison_path,
        output_dir=Path(args.output_dir) if args.output_dir is not None else None,
        method_name=args.method_name,
        baseline_variant=args.baseline_variant,
        psnr_drop_tolerance=args.psnr_drop_tolerance,
        objective=args.objective,
    )
    print(f"Wrote {result['selection_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"selected variant: {result['status']['selected_variant']}")


if __name__ == "__main__":
    main()
