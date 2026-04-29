from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.real_geometry import evaluate_tanks_temples_geometry


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate splat geometry against Tanks and Temples ground-truth geometry.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--splats-path", default=None, help="Defaults to <scene-dir>/real_splats.npz.")
    parser.add_argument("--ground-truth-path", default=None, help="Defaults to the Tanks and Temples path stored in real_scene.json.")
    parser.add_argument("--alignment-path", default=None, help="Defaults to the Tanks and Temples alignment path stored in real_scene.json.")
    parser.add_argument("--crop-path", default=None, help="Defaults to the Tanks and Temples crop path stored in real_scene.json.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/evaluations.")
    parser.add_argument("--method-name", default=None, help="Defaults to the splat file stem.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--max-pred-points", type=optional_positive_int, default=100000, help="Prediction subsample cap. Use 0 for all.")
    parser.add_argument("--max-gt-points", type=optional_positive_int, default=100000, help="Ground-truth subsample cap. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--apply-alignment", type=optional_bool, default=None, help="true, false, or auto. Auto applies a resolved Tanks and Temples alignment.")
    parser.add_argument("--invert-alignment", type=str_to_bool, default=False)
    parser.add_argument("--use-crop", type=str_to_bool, default=True)
    parser.add_argument("--gate-path", default=None, help="Optional .npz with gate or raw_gate arrays for stratified metrics.")
    parser.add_argument("--model-path", default=None, help="Optional GPIS model used to compute splat gates for stratified metrics.")
    parser.add_argument("--epsilon", type=float, default=0.24)
    parser.add_argument("--gate-floor", type=float, default=0.0)
    parser.add_argument("--gate-thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--gate-batch-size", type=int, default=4096)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    status = evaluate_tanks_temples_geometry(
        scene_dir=scene_dir,
        splats_path=args.splats_path,
        ground_truth_path=args.ground_truth_path,
        alignment_path=args.alignment_path,
        crop_path=args.crop_path,
        output_dir=args.output_dir,
        method_name=args.method_name,
        thresholds=tuple(args.thresholds),
        max_pred_points=args.max_pred_points,
        max_gt_points=args.max_gt_points,
        seed=args.seed,
        apply_alignment=args.apply_alignment,
        invert_alignment=args.invert_alignment,
        use_crop=args.use_crop,
        gate_path=args.gate_path,
        model_path=args.model_path,
        epsilon=args.epsilon,
        gate_floor=args.gate_floor,
        gate_thresholds=tuple(args.gate_thresholds),
        gate_batch_size=args.gate_batch_size,
        distance_chunk_size=args.distance_chunk_size,
    )
    first = status["summary"][0]
    print(f"Wrote {status['summary_path']}")
    print(f"Wrote {status['threshold_metrics_path']}")
    print(f"Wrote {status['status_path']}")
    print(f"Wrote {status['report_path']}")
    print(f"chamfer_l1: {first['chamfer_l1']:.6g}")
    print(f"chamfer_l2: {first['chamfer_l2']:.6g}")


if __name__ == "__main__":
    main()
