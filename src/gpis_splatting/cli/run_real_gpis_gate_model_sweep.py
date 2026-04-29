from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.real_gate_diagnostics import default_topk_fractions
from gpis_splatting.real_gate_model_sweep import CONSTRUCTION_MODES, run_real_gpis_gate_model_sweep


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    return None if parsed <= 0 else parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep real GPIS construction and hyperparameters against gate-quality diagnostics.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--sweep-name", default="real_gate_model_sweep")
    parser.add_argument("--construction-modes", nargs="+", choices=CONSTRUCTION_MODES, default=["surface_free", "behind_surface", "normal_offsets"])
    parser.add_argument("--samples-path", default=None, help="Existing samples path used by construction mode `existing`.")
    parser.add_argument("--splats-path", default=None, help="Existing splats path used by construction mode `existing`.")
    parser.add_argument("--point-source", default="auto", choices=["auto", "colmap", "ply"])
    parser.add_argument("--point-path", default=None)
    parser.add_argument("--lengthscales", type=float, nargs="+", default=[0.15, 0.25, 0.4])
    parser.add_argument("--noise-stds", type=float, nargs="+", default=[0.03, 0.06])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.08, 0.16, 0.24])
    parser.add_argument("--gate-floors", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--use-observation-noise", type=str_to_bool, default=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--topk-fractions", type=float, nargs="+", default=list(default_topk_fractions()))
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--max-bootstrap-points", type=optional_positive_int, default=5000, help="Bootstrap point cap. Use 0 for all.")
    parser.add_argument("--max-train-points", type=optional_positive_int, default=1200, help="Dense GPIS training cap. Use 0 for all.")
    parser.add_argument("--max-pred-points", type=optional_positive_int, default=100000, help="Diagnostic prediction cap. Use 0 for all.")
    parser.add_argument("--max-gt-points", type=optional_positive_int, default=100000, help="Diagnostic ground-truth cap. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--gate-batch-size", type=int, default=4096)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    parser.add_argument("--normal-offset-distance", type=float, default=0.04)
    parser.add_argument("--normal-offset-noise-std", type=float, default=0.05)
    parser.add_argument("--normal-offset-neighbors", type=int, default=12)
    parser.add_argument("--max-normal-offset-points", type=optional_positive_int, default=3000, help="Normal-offset augmentation cap. Use 0 for all.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/model_sweeps/<sweep-name>.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    result = run_real_gpis_gate_model_sweep(
        scene_dir=scene_dir,
        sweep_name=args.sweep_name,
        construction_modes=tuple(args.construction_modes),
        samples_path=args.samples_path,
        splats_path=args.splats_path,
        point_source=args.point_source,
        point_path=args.point_path,
        lengthscales=tuple(args.lengthscales),
        noise_stds=tuple(args.noise_stds),
        epsilons=tuple(args.epsilons),
        gate_floors=tuple(args.gate_floors),
        variance=args.variance,
        jitter=args.jitter,
        use_observation_noise=args.use_observation_noise,
        thresholds=tuple(args.thresholds),
        topk_fractions=tuple(args.topk_fractions),
        num_bins=args.num_bins,
        max_bootstrap_points=args.max_bootstrap_points,
        max_train_points=args.max_train_points,
        max_pred_points=args.max_pred_points,
        max_gt_points=args.max_gt_points,
        seed=args.seed,
        gate_batch_size=args.gate_batch_size,
        distance_chunk_size=args.distance_chunk_size,
        normal_offset_distance=args.normal_offset_distance,
        normal_offset_noise_std=args.normal_offset_noise_std,
        normal_offset_neighbors=args.normal_offset_neighbors,
        max_normal_offset_points=args.max_normal_offset_points,
        output_dir=args.output_dir,
    )
    print(f"Wrote {result['summary_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"rows: {result['status']['row_count']}")
    print(f"failures: {result['status']['failure_count']}")
    for row in result["status"]["best_by_gate_error_spearman"]:
        print(
            f"best_spearman@{row['geometry_threshold']:.6g}: "
            f"{row['construction_mode']} ls={row['lengthscale']:.6g} n={row['noise_std']:.6g} "
            f"eps={row['epsilon']:.6g} floor={row['gate_floor']:.6g} "
            f"rho={row['spearman_gate_vs_negative_distance']}"
        )


if __name__ == "__main__":
    main()
