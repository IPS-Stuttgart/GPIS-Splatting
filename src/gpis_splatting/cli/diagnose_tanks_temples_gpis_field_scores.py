from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.real_field_scores import default_score_lambdas, run_tanks_temples_gpis_field_score_diagnostics
from gpis_splatting.real_gate_diagnostics import default_topk_fractions


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
    parser = argparse.ArgumentParser(description="Diagnose GPIS field components and alternative scores against Tanks and Temples geometry error.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--splats-path", default=None, help="Defaults to <scene-dir>/real_splats.npz.")
    parser.add_argument("--model-path", required=True, help="GPIS model used to evaluate field components at splat centers.")
    parser.add_argument("--ground-truth-path", default=None, help="Defaults to the Tanks and Temples path stored in real_scene.json.")
    parser.add_argument("--alignment-path", default=None, help="Defaults to the Tanks and Temples alignment path stored in real_scene.json.")
    parser.add_argument("--crop-path", default=None, help="Defaults to the Tanks and Temples crop path stored in real_scene.json.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/evaluations.")
    parser.add_argument("--method-name", default=None, help="Defaults to the splat file stem.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--topk-fractions", type=float, nargs="+", default=list(default_topk_fractions()))
    parser.add_argument("--score-lambdas", type=float, nargs="+", default=list(default_score_lambdas()))
    parser.add_argument("--max-pred-points", type=optional_positive_int, default=100000, help="Prediction subsample cap. Use 0 for all.")
    parser.add_argument("--max-gt-points", type=optional_positive_int, default=100000, help="Ground-truth subsample cap. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=13)
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
    result = run_tanks_temples_gpis_field_score_diagnostics(
        scene_dir=scene_dir,
        splats_path=args.splats_path,
        model_path=args.model_path,
        ground_truth_path=args.ground_truth_path,
        alignment_path=args.alignment_path,
        crop_path=args.crop_path,
        output_dir=args.output_dir,
        method_name=args.method_name,
        thresholds=tuple(args.thresholds),
        topk_fractions=tuple(args.topk_fractions),
        score_lambdas=tuple(args.score_lambdas),
        max_pred_points=args.max_pred_points,
        max_gt_points=args.max_gt_points,
        seed=args.seed,
        apply_alignment=args.apply_alignment,
        invert_alignment=args.invert_alignment,
        use_crop=args.use_crop,
        epsilon=args.epsilon,
        gate_floor=args.gate_floor,
        batch_size=args.batch_size,
        distance_chunk_size=args.distance_chunk_size,
    )
    print(f"Wrote {result['field_scores_path']}")
    print(f"Wrote {result['score_summary_path']}")
    print(f"Wrote {result['score_ranked_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    for row in result["status"]["best_by_spearman"]:
        print(
            f"best_spearman@{row['geometry_threshold']:.6g}: "
            f"{row['score_name']} rho={row['spearman_score_vs_negative_distance']} "
            f"top={row['best_topk_fraction']:.6g} f={row['best_f_score']:.6g}"
        )


if __name__ == "__main__":
    main()
