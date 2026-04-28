from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.real_bootstrap import POINT_SOURCES, bootstrap_real_gpis


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap real-scene GPIS observations and initial splats from a sparse point cloud.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--point-source", choices=POINT_SOURCES, default="auto")
    parser.add_argument("--point-path", default=None)
    parser.add_argument("--output-prefix", default="real")
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--free-space-samples-per-point", type=int, default=2)
    parser.add_argument("--free-space-min-fraction", type=float, default=0.2)
    parser.add_argument("--free-space-max-fraction", type=float, default=0.85)
    parser.add_argument("--add-behind-surface-samples", type=str_to_bool, default=True)
    parser.add_argument("--behind-surface-fraction", type=float, default=1.08)
    parser.add_argument("--max-sample-distance", type=float, default=0.35)
    parser.add_argument("--surface-noise-std", type=float, default=0.03)
    parser.add_argument("--free-space-noise-std", type=float, default=0.08)
    parser.add_argument("--behind-surface-noise-std", type=float, default=0.12)
    parser.add_argument("--splat-tau", type=float, default=0.45)
    parser.add_argument("--splat-sigma", type=float, default=0.025)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    max_points = None if args.max_points <= 0 else args.max_points
    result = bootstrap_real_gpis(
        scene_dir=scene_dir,
        point_source=args.point_source,
        point_path=args.point_path,
        output_prefix=args.output_prefix,
        max_points=max_points,
        seed=args.seed,
        free_space_samples_per_point=args.free_space_samples_per_point,
        free_space_min_fraction=args.free_space_min_fraction,
        free_space_max_fraction=args.free_space_max_fraction,
        add_behind_surface_samples=args.add_behind_surface_samples,
        behind_surface_fraction=args.behind_surface_fraction,
        max_sample_distance=args.max_sample_distance,
        surface_noise_std=args.surface_noise_std,
        free_space_noise_std=args.free_space_noise_std,
        behind_surface_noise_std=args.behind_surface_noise_std,
        splat_tau=args.splat_tau,
        splat_sigma=args.splat_sigma,
    )
    print(f"Wrote {result['samples_path']}")
    print(f"Wrote {result['splats_path']}")
    print(f"Wrote {result['config_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"surface_points: {result['report']['surface_point_count']}")
    print(f"samples: {result['report']['sample_count']}")
    print(f"splats: {result['report']['splat_count']}")


if __name__ == "__main__":
    main()
