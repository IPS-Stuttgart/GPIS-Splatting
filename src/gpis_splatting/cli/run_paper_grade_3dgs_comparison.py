from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from gpis_splatting.cli.evaluate_real_renders import str_to_bool
from gpis_splatting.paper_3dgs_comparison import Paper3DGSComparisonConfig, config_from_json, run_paper_3dgs_comparison


def optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    return None if parsed <= 0 else parsed


def optional_bool(value: str) -> bool | None:
    lowered = value.lower()
    if lowered in {"auto", "none"}:
        return None
    return str_to_bool(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate paper-grade trained-3DGS comparison metrics across scenes.")
    parser.add_argument("--scene-config", required=True, help="JSON configuration with scene entries and metric artifact paths.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--comparison-name", default=None)
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--prediction-subdir", default=None)
    parser.add_argument("--primary-geometry-threshold", type=float, default=None)
    parser.add_argument("--geometry-thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--compute-photometry", type=str_to_bool, default=None)
    parser.add_argument("--compute-geometry", type=str_to_bool, default=None)
    parser.add_argument("--compute-lpips", type=str_to_bool, default=None)
    parser.add_argument("--require-all-images", type=str_to_bool, default=None)
    parser.add_argument("--require-all-variants", type=str_to_bool, default=None)
    parser.add_argument("--max-pred-points", type=optional_positive_int, default=None, help="Geometry prediction subsample cap. Use 0 for all.")
    parser.add_argument("--max-gt-points", type=optional_positive_int, default=None, help="Geometry ground-truth subsample cap. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--apply-alignment", type=optional_bool, default=None, help="true, false, or auto/none to use scene/default behavior.")
    parser.add_argument("--invert-alignment", type=str_to_bool, default=None)
    parser.add_argument("--use-crop", type=str_to_bool, default=None)
    parser.add_argument("--distance-chunk-size", type=int, default=None)
    parser.add_argument("--fail-on-missing", type=str_to_bool, default=None)
    parser.add_argument("--require-lpips", type=str_to_bool, default=None)
    parser.add_argument("--require-performance", type=str_to_bool, default=None)
    return parser


def override_config(config: Paper3DGSComparisonConfig, args: argparse.Namespace) -> Paper3DGSComparisonConfig:
    updates: dict[str, Any] = {}
    candidates = {
        "output_dir": Path(args.output_dir) if args.output_dir is not None else None,
        "comparison_name": args.comparison_name,
        "prepared_root": Path(args.prepared_root) if args.prepared_root is not None else None,
        "split": args.split,
        "method_name": args.method_name,
        "prediction_subdir": args.prediction_subdir,
        "primary_geometry_threshold": args.primary_geometry_threshold,
        "geometry_thresholds": tuple(args.geometry_thresholds) if args.geometry_thresholds is not None else None,
        "compute_photometry": args.compute_photometry,
        "compute_geometry": args.compute_geometry,
        "compute_lpips": args.compute_lpips,
        "require_all_images": args.require_all_images,
        "require_all_variants": args.require_all_variants,
        "max_pred_points": args.max_pred_points,
        "max_gt_points": args.max_gt_points,
        "seed": args.seed,
        "apply_alignment": args.apply_alignment,
        "invert_alignment": args.invert_alignment,
        "use_crop": args.use_crop,
        "distance_chunk_size": args.distance_chunk_size,
        "fail_on_missing": args.fail_on_missing,
        "require_lpips": args.require_lpips,
        "require_performance": args.require_performance,
    }
    for name, value in candidates.items():
        if value is not None:
            updates[name] = value
    return replace(config, **updates)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = override_config(config_from_json(args.scene_config), args)
    result = run_paper_3dgs_comparison(config)
    print(f"Wrote {result['comparison_path']}")
    print(f"Wrote {result['aggregate_path']}")
    print(f"Wrote {result['paper_table_path']}")
    print(f"Wrote {result['checks_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"passed: {result['passed']}")


if __name__ == "__main__":
    main()
