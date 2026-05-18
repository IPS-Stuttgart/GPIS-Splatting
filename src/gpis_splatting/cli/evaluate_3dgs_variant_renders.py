from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.cli.evaluate_real_renders import str_to_bool
from gpis_splatting.external_3dgs import evaluate_3dgs_variant_renders
from gpis_splatting.variant_selection import DEFAULT_PARETO_PSNR_DROP_TOLERANCE, select_psnr_constrained_pareto_variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate rendered trained-3DGS GPIS variant folders against a prepared real scene split.")
    parser.add_argument("--manifest-path", required=True, help="CSV written by export_3dgs_gpis_variants.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--predictions-root", required=True, help="Directory containing one rendered image directory per manifest variant.")
    parser.add_argument("--prediction-subdir", default="", help="Optional relative image subdirectory below each variant directory.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/evaluations.")
    parser.add_argument("--method-name", default="trained_3dgs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--benchmark-target", default=None)
    parser.add_argument("--compute-lpips", type=str_to_bool, default=False)
    parser.add_argument("--require-all-images", type=str_to_bool, default=True)
    parser.add_argument("--require-all-variants", type=str_to_bool, default=True)
    parser.add_argument("--write-pareto-selection", type=str_to_bool, default=True, help="Also write a PSNR-constrained Pareto selection CSV/status/report.")
    parser.add_argument("--pareto-baseline-variant", default="baseline", help="Variant name used as the PSNR reference for Pareto selection.")
    parser.add_argument("--pareto-psnr-drop-tolerance", type=float, default=DEFAULT_PARETO_PSNR_DROP_TOLERANCE, help="Maximum allowed PSNR drop from the baseline in dB.")
    parser.add_argument(
        "--pareto-objective",
        default="min_retained",
        help="Tie-breaking objective on the feasible Pareto frontier: min_retained, max_psnr, max_ssim, or min_lpips. Hyphenated aliases are accepted.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    output_dir = Path(args.output_dir) if args.output_dir is not None else scene_dir / "evaluations"
    result = evaluate_3dgs_variant_renders(
        manifest_path=args.manifest_path,
        scene_dir=scene_dir,
        predictions_root=args.predictions_root,
        output_dir=output_dir,
        method_name=args.method_name,
        split=args.split,
        prediction_subdir=args.prediction_subdir,
        compute_lpips=args.compute_lpips,
        require_all_images=args.require_all_images,
        require_all_variants=args.require_all_variants,
        benchmark_target=args.benchmark_target,
    )
    print(f"Wrote {result['comparison_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"variants: {result['status']['variant_count']}")
    if args.write_pareto_selection:
        selection = select_psnr_constrained_pareto_variant(
            comparison_path=result["comparison_path"],
            output_dir=output_dir,
            method_name=args.method_name,
            baseline_variant=args.pareto_baseline_variant,
            psnr_drop_tolerance=args.pareto_psnr_drop_tolerance,
            objective=args.pareto_objective,
        )
        print(f"Wrote {selection['selection_path']}")
        print(f"Wrote {selection['status_path']}")
        print(f"Wrote {selection['report_path']}")
        print(f"selected variant: {selection['status']['selected_variant']}")


if __name__ == "__main__":
    main()
