from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.cli.evaluate_real_renders import str_to_bool
from gpis_splatting.real_alignment import diagnose_real_alignment
from gpis_splatting.real_pipeline import PROJECTION_CONVENTIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose real-scene render alignment with projection coverage and ranked failure modes.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--render-dir", required=True, help="Directory containing rendered prediction images and optional real_render_report.json.")
    parser.add_argument("--splats-path", default=None, help="Defaults to the render report splats_path, then <scene-dir>/real_splats.npz.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/diagnostics/real_alignment/<render-dir-name>.")
    parser.add_argument("--split", default="test", help="Scene split to diagnose, or all.")
    parser.add_argument("--max-frames", type=int, default=0, help="Frame cap. Use 0 for the whole split.")
    parser.add_argument("--projection-convention", choices=PROJECTION_CONVENTIONS, default="auto")
    parser.add_argument("--near-plane", type=float, default=1e-4)
    parser.add_argument("--kernel-radius", type=float, default=3.0)
    parser.add_argument("--min-sigma-px", type=float, default=0.8)
    parser.add_argument("--coverage-downsample", type=int, default=4)
    parser.add_argument("--max-overlay-splats", type=int, default=2000)
    parser.add_argument("--require-predictions", type=str_to_bool, default=False)
    parser.add_argument("--low-psnr-threshold", type=float, default=12.0)
    parser.add_argument("--min-valid-depth-fraction", type=float, default=0.05)
    parser.add_argument("--min-in-frame-fraction", type=float, default=0.01)
    parser.add_argument("--min-coverage-fraction", type=float, default=0.01)
    parser.add_argument("--min-prediction-nonblack-fraction", type=float, default=0.001)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    max_frames = None if args.max_frames <= 0 else args.max_frames
    result = diagnose_real_alignment(
        scene_dir=scene_dir,
        render_dir=args.render_dir,
        splats_path=args.splats_path,
        output_dir=args.output_dir,
        split=args.split,
        max_frames=max_frames,
        projection_convention=args.projection_convention,
        near_plane=args.near_plane,
        kernel_radius=args.kernel_radius,
        min_sigma_px=args.min_sigma_px,
        coverage_downsample=args.coverage_downsample,
        max_overlay_splats=args.max_overlay_splats,
        require_predictions=args.require_predictions,
        low_psnr_threshold=args.low_psnr_threshold,
        min_valid_depth_fraction=args.min_valid_depth_fraction,
        min_in_frame_fraction=args.min_in_frame_fraction,
        min_coverage_fraction=args.min_coverage_fraction,
        min_prediction_nonblack_fraction=args.min_prediction_nonblack_fraction,
    )
    summary = result["status"]["summary"]
    print(f"Wrote {result['frames_path']}")
    print(f"Wrote {result['ranked_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"evaluated_predictions: {summary['evaluated_count']}")
    print(f"missing_predictions: {summary['missing_prediction_count']}")
    print(f"failure_counts: {summary['failure_counts']}")


if __name__ == "__main__":
    main()
