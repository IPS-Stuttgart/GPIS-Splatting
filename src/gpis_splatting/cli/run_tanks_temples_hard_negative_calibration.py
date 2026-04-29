from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.real_gate_diagnostics import default_topk_fractions
from gpis_splatting.real_hard_negatives import run_tanks_temples_hard_negative_calibration


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def optional_bool(value: str) -> bool | None:
    lowered = value.lower()
    if lowered in {"auto", "none"}:
        return None
    return str_to_bool(value)


def optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    return None if parsed <= 0 else parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate hard-negative splat candidates, score them with GPIS, and calibrate splat confidence.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--splats-path", default=None, help="Defaults to <scene-dir>/real_splats.npz.")
    parser.add_argument("--model-path", required=True, help="GPIS model used to score generated candidates.")
    parser.add_argument("--ground-truth-path", default=None, help="Defaults to the Tanks and Temples path stored in real_scene.json.")
    parser.add_argument("--alignment-path", default=None, help="Defaults to the Tanks and Temples alignment path stored in real_scene.json.")
    parser.add_argument("--crop-path", default=None, help="Defaults to the Tanks and Temples crop path stored in real_scene.json.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/evaluations.")
    parser.add_argument("--method-name", default="hard_negative")
    parser.add_argument("--max-source-splats", type=optional_positive_int, default=5000, help="Source splat subsample cap. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--include-source", type=str_to_bool, default=True)
    parser.add_argument("--jitter-copies", type=int, default=1)
    parser.add_argument("--ray-copies", type=int, default=1)
    parser.add_argument("--behind-copies", type=int, default=1)
    parser.add_argument("--random-count", type=nonnegative_int, default=None, help="Random crop-volume candidate count. Defaults to source count. Use 0 to disable.")
    parser.add_argument("--jitter-std", type=float, default=0.03)
    parser.add_argument("--ray-shift-min", type=float, default=0.03)
    parser.add_argument("--ray-shift-max", type=float, default=0.15)
    parser.add_argument("--behind-shift-min", type=float, default=0.08)
    parser.add_argument("--behind-shift-max", type=float, default=0.25)
    parser.add_argument("--random-bounds-scale", type=float, default=1.25)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--topk-fractions", type=float, nargs="+", default=list(default_topk_fractions()))
    parser.add_argument("--calibration-validation-fraction", type=float, default=0.35)
    parser.add_argument("--max-pred-points", type=optional_positive_int, default=100000, help="Field-score prediction cap. Use 0 for all.")
    parser.add_argument("--max-gt-points", type=optional_positive_int, default=100000, help="Ground-truth cap. Use 0 for all.")
    parser.add_argument("--apply-alignment", type=optional_bool, default=None, help="true, false, or auto. Auto applies a resolved Tanks and Temples alignment.")
    parser.add_argument("--invert-alignment", type=str_to_bool, default=False)
    parser.add_argument("--use-crop", type=str_to_bool, default=True)
    parser.add_argument("--epsilon", type=float, default=0.24)
    parser.add_argument("--gate-floor", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    result = run_tanks_temples_hard_negative_calibration(
        scene_dir=scene_dir,
        model_path=args.model_path,
        splats_path=args.splats_path,
        method_name=args.method_name,
        output_dir=args.output_dir,
        ground_truth_path=args.ground_truth_path,
        alignment_path=args.alignment_path,
        crop_path=args.crop_path,
        max_source_splats=args.max_source_splats,
        seed=args.seed,
        include_source=args.include_source,
        jitter_copies=args.jitter_copies,
        ray_copies=args.ray_copies,
        behind_copies=args.behind_copies,
        random_count=args.random_count,
        jitter_std=args.jitter_std,
        ray_shift_min=args.ray_shift_min,
        ray_shift_max=args.ray_shift_max,
        behind_shift_min=args.behind_shift_min,
        behind_shift_max=args.behind_shift_max,
        random_bounds_scale=args.random_bounds_scale,
        thresholds=tuple(args.thresholds),
        topk_fractions=tuple(args.topk_fractions),
        calibration_validation_fraction=args.calibration_validation_fraction,
        max_pred_points=args.max_pred_points,
        max_gt_points=args.max_gt_points,
        apply_alignment=args.apply_alignment,
        invert_alignment=args.invert_alignment,
        use_crop=args.use_crop,
        epsilon=args.epsilon,
        gate_floor=args.gate_floor,
        batch_size=args.batch_size,
        distance_chunk_size=args.distance_chunk_size,
    )
    print(f"Wrote {result['generated_splats_path']}")
    print(f"Wrote {result['candidate_metadata_path']}")
    print(f"Wrote {result['field_scores_path']}")
    print(f"Wrote {result['calibration_summary_path']}")
    print(f"Wrote {result['calibrated_scores_path']}")
    print(f"Wrote {result['calibrated_confidence_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    for row in result["status"]["best_calibrators"]:
        print(
            f"best_calibrator@{row['geometry_threshold']:.6g}: "
            f"{row['method_name']} brier={row['brier']:.6g} auc={row['auc']} "
            f"top={row['best_topk_fraction']:.6g} f={row['best_f_score']:.6g}"
        )


if __name__ == "__main__":
    main()
