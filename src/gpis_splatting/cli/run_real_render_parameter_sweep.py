from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.cli.evaluate_real_renders import str_to_bool
from gpis_splatting.real_pipeline import PROJECTION_CONVENTIONS, parse_rgb_triplet
from gpis_splatting.real_render_sweep import run_real_render_parameter_sweep


def rgb_triplet(value: str) -> tuple[float, float, float]:
    try:
        return parse_rgb_triplet(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep real-scene splat-renderer appearance parameters and rank by held-out render metrics.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--splats-path", default=None, help="Defaults to <scene-dir>/real_splats.npz.")
    parser.add_argument("--method-name", default="render_parameter_sweep")
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/evaluations/<method-name>.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-frames", type=int, default=0, help="Use 0 for the full split.")
    parser.add_argument("--sigma-scales", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--tau-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--min-sigma-pxs", type=float, nargs="+", default=[0.6, 1.0])
    parser.add_argument("--kernel-radii", type=float, nargs="+", default=[2.0, 3.0])
    parser.add_argument("--background-colors", type=rgb_triplet, nargs="+", default=[(0.0, 0.0, 0.0)], help="One or more RGB triplets, e.g. 0,0,0 1,1,1.")
    parser.add_argument("--projection-conventions", choices=PROJECTION_CONVENTIONS, nargs="+", default=["auto"])
    parser.add_argument("--near-plane", type=float, default=1e-4)
    parser.add_argument("--selection-metric", choices=["mean_psnr", "mean_ssim"], default="mean_psnr")
    parser.add_argument("--run-alignment", type=str_to_bool, default=True)
    parser.add_argument("--alignment-coverage-downsample", type=int, default=8)
    parser.add_argument("--alignment-max-overlay-splats", type=int, default=1000)
    parser.add_argument("--audit-max-panels", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    max_frames = None if args.max_frames <= 0 else args.max_frames
    result = run_real_render_parameter_sweep(
        scene_dir=scene_dir,
        splats_path=args.splats_path,
        method_name=args.method_name,
        output_dir=args.output_dir,
        split=args.split,
        max_frames=max_frames,
        sigma_scales=tuple(args.sigma_scales),
        tau_scales=tuple(args.tau_scales),
        min_sigma_pxs=tuple(args.min_sigma_pxs),
        kernel_radii=tuple(args.kernel_radii),
        background_colors=tuple(args.background_colors),
        projection_conventions=tuple(args.projection_conventions),
        near_plane=args.near_plane,
        selection_metric=args.selection_metric,
        run_alignment=args.run_alignment,
        alignment_coverage_downsample=args.alignment_coverage_downsample,
        alignment_max_overlay_splats=args.alignment_max_overlay_splats,
        audit_max_panels=args.audit_max_panels,
    )
    best = result["status"]["best"]["best"]
    print(f"Wrote {result['sweep_path']}")
    print(f"Wrote {result['ranked_path']}")
    print(f"Wrote {result['best_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"best_variant: {best['variant']}")
    print(f"best_mean_psnr: {best['mean_psnr']:.6g}")
    print(f"best_mean_ssim: {best['mean_ssim']:.6g}")


if __name__ == "__main__":
    main()
