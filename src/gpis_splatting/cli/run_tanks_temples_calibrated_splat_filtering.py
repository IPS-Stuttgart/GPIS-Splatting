from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.cli.evaluate_tanks_temples_geometry import optional_bool, optional_positive_int, str_to_bool
from gpis_splatting.real_splat_filtering import run_tanks_temples_calibrated_splat_filtering


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter or tau-scale splats with calibrated confidence gates and evaluate each variant.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--splats-path", default=None, help="Defaults to <scene-dir>/real_splats.npz.")
    parser.add_argument("--gate-path", required=True, help="Calibrated gate .npz containing gate or raw_gate.")
    parser.add_argument("--method-name", default="calibrated_splat_filtering")
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/evaluations.")
    parser.add_argument("--gate-thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--include-baseline", type=str_to_bool, default=True)
    parser.add_argument("--write-scaled", type=str_to_bool, default=True)
    parser.add_argument("--write-filtered", type=str_to_bool, default=True)
    parser.add_argument("--tau-scale-floor", type=float, default=0.0)
    parser.add_argument("--ground-truth-path", default=None, help="Defaults to the Tanks and Temples path stored in real_scene.json.")
    parser.add_argument("--alignment-path", default=None, help="Defaults to the Tanks and Temples alignment path stored in real_scene.json.")
    parser.add_argument("--crop-path", default=None, help="Defaults to the Tanks and Temples crop path stored in real_scene.json.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--max-pred-points", type=optional_positive_int, default=100000, help="Prediction subsample cap. Use 0 for all.")
    parser.add_argument("--max-gt-points", type=optional_positive_int, default=100000, help="Ground-truth subsample cap. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--apply-alignment", type=optional_bool, default=None, help="true, false, or auto. Auto applies a resolved Tanks and Temples alignment.")
    parser.add_argument("--invert-alignment", type=str_to_bool, default=False)
    parser.add_argument("--use-crop", type=str_to_bool, default=True)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    parser.add_argument("--render-split", default="test")
    parser.add_argument("--render-max-frames", type=int, default=0, help="Use 0 to skip rendering.")
    parser.add_argument("--evaluate-render-metrics", type=str_to_bool, default=True)
    parser.add_argument("--benchmark-target", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    result = run_tanks_temples_calibrated_splat_filtering(
        scene_dir=scene_dir,
        splats_path=args.splats_path,
        gate_path=args.gate_path,
        method_name=args.method_name,
        output_dir=args.output_dir,
        gate_thresholds=tuple(args.gate_thresholds),
        include_baseline=args.include_baseline,
        write_scaled=args.write_scaled,
        write_filtered=args.write_filtered,
        tau_scale_floor=args.tau_scale_floor,
        ground_truth_path=args.ground_truth_path,
        alignment_path=args.alignment_path,
        crop_path=args.crop_path,
        thresholds=tuple(args.thresholds),
        max_pred_points=args.max_pred_points,
        max_gt_points=args.max_gt_points,
        seed=args.seed,
        apply_alignment=args.apply_alignment,
        invert_alignment=args.invert_alignment,
        use_crop=args.use_crop,
        distance_chunk_size=args.distance_chunk_size,
        render_split=args.render_split,
        render_max_frames=args.render_max_frames,
        evaluate_render_metrics=args.evaluate_render_metrics,
        benchmark_target=args.benchmark_target,
    )
    print(f"Wrote {result['comparison_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"variants: {result['status']['variant_count']}")


if __name__ == "__main__":
    main()
