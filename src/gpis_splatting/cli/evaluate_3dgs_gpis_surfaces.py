from __future__ import annotations

import argparse

from gpis_splatting.cli.evaluate_tanks_temples_geometry import optional_positive_int, str_to_bool
from gpis_splatting.gaussian_surface import evaluate_gaussian_surface_geometry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate extracted GPIS-gated Gaussian surfaces against ground-truth geometry.")
    parser.add_argument("--manifest-path", required=True, help="Surface manifest written by extract_3dgs_gpis_surfaces.")
    parser.add_argument("--ground-truth-path", required=True, help="Ground-truth PLY point cloud or mesh vertices used as geometry reference.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="gpis_gated_3dgs_surface")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--max-pred-points", type=optional_positive_int, default=100000, help="Prediction point/sample cap. Use 0 for all point-cloud vertices.")
    parser.add_argument("--max-gt-points", type=optional_positive_int, default=100000, help="Ground-truth subsample cap. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--alignment-path", default=None, help="Optional 4x4 transform applied to extracted surfaces before evaluation.")
    parser.add_argument("--invert-alignment", type=str_to_bool, default=False)
    parser.add_argument("--crop-path", default=None, help="Optional crop JSON using the existing real_geometry crop formats.")
    parser.add_argument("--use-crop", type=str_to_bool, default=True)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = evaluate_gaussian_surface_geometry(
        manifest_path=args.manifest_path,
        ground_truth_path=args.ground_truth_path,
        output_dir=args.output_dir,
        method_name=args.method_name,
        thresholds=tuple(args.thresholds),
        max_pred_points=args.max_pred_points,
        max_gt_points=args.max_gt_points,
        seed=args.seed,
        alignment_path=args.alignment_path,
        invert_alignment=args.invert_alignment,
        crop_path=args.crop_path,
        use_crop=args.use_crop,
        distance_chunk_size=args.distance_chunk_size,
    )
    first = result["summary"][0]
    print(f"Wrote {result['summary_path']}")
    print(f"Wrote {result['threshold_metrics_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"chamfer_l1: {first['chamfer_l1']:.6g}")
    print(f"chamfer_l2: {first['chamfer_l2']:.6g}")


if __name__ == "__main__":
    main()
