from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.gpis_initialization import GPISAwareInitializationConfig, run_gpis_aware_initialization


def positive_int_or_none(value: str) -> int | None:
    parsed = int(value)
    return None if parsed <= 0 else parsed


def optional_positive_float(value: str) -> float | None:
    parsed = float(value)
    return None if parsed <= 0.0 else parsed


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize GPIS-aware anisotropic Gaussians from a fitted real-scene GPIS model and seed splats.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--splats-path", default=None)
    parser.add_argument("--output-prefix", default="gpis_init")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-count", type=positive_int_or_none, default=None)
    parser.add_argument("--proposals-per-seed", type=int, default=3)
    parser.add_argument("--include-seed-points", type=str_to_bool, default=True)
    parser.add_argument("--jitter-std", type=optional_positive_float, default=None)
    parser.add_argument("--projection-iterations", type=int, default=4)
    parser.add_argument("--max-projection-step", type=optional_positive_float, default=None)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--max-abs-distance", type=optional_positive_float, default=None)
    parser.add_argument("--max-distance-std", type=optional_positive_float, default=None)
    parser.add_argument("--min-surface-probability", type=float, default=0.0)
    parser.add_argument("--min-grad-norm", type=float, default=1e-5)
    parser.add_argument("--min-view-count", type=int, default=0)
    parser.add_argument("--min-separation", type=optional_positive_float, default=None)
    parser.add_argument("--normal-scale", type=optional_positive_float, default=None)
    parser.add_argument("--tangent-scale", type=optional_positive_float, default=None)
    parser.add_argument("--normal-scale-factor", type=float, default=0.20)
    parser.add_argument("--tangent-scale-factor", type=float, default=0.80)
    parser.add_argument("--scale-from-uncertainty", type=str_to_bool, default=True)
    parser.add_argument("--max-uncertainty-scale-multiplier", type=float, default=3.0)
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument("--opacity-confidence-power", type=float, default=1.0)
    parser.add_argument("--min-opacity", type=float, default=0.02)
    parser.add_argument("--max-opacity", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--confidence-model-path", default=None)
    parser.add_argument("--confidence-threshold", type=optional_positive_float, default=None)
    parser.add_argument("--projection-convention", choices=("auto", "opencv", "opengl"), default="auto")
    parser.add_argument("--near-plane", type=float, default=1e-4)
    parser.add_argument("--sh-degree", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    config = GPISAwareInitializationConfig(
        target_count=args.target_count,
        proposals_per_seed=args.proposals_per_seed,
        include_seed_points=args.include_seed_points,
        jitter_std=args.jitter_std,
        projection_iterations=args.projection_iterations,
        max_projection_step=args.max_projection_step,
        epsilon=args.epsilon,
        max_abs_distance=args.max_abs_distance,
        max_distance_std=args.max_distance_std,
        min_surface_probability=args.min_surface_probability,
        min_grad_norm=args.min_grad_norm,
        min_view_count=args.min_view_count,
        min_separation=args.min_separation,
        normal_scale=args.normal_scale,
        tangent_scale=args.tangent_scale,
        normal_scale_factor=args.normal_scale_factor,
        tangent_scale_factor=args.tangent_scale_factor,
        scale_from_uncertainty=args.scale_from_uncertainty,
        max_uncertainty_scale_multiplier=args.max_uncertainty_scale_multiplier,
        opacity=args.opacity,
        opacity_confidence_power=args.opacity_confidence_power,
        min_opacity=args.min_opacity,
        max_opacity=args.max_opacity,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    result = run_gpis_aware_initialization(
        scene_dir=scene_dir,
        model_path=args.model_path,
        splats_path=args.splats_path,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        confidence_model_path=args.confidence_model_path,
        confidence_threshold=args.confidence_threshold,
        projection_convention=args.projection_convention,
        near_plane=args.near_plane,
        sh_degree=args.sh_degree,
        config=config,
    )
    print(f"Wrote {result['arrays_path']}")
    print(f"Wrote {result['splats_path']}")
    print(f"Wrote {result['ply_path']}")
    print(f"Wrote {result['field_scores_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"selected_gaussians: {result['status']['selected_count']}")


if __name__ == "__main__":
    main()
