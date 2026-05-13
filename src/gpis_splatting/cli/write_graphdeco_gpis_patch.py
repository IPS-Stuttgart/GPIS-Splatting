from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.graphdeco_patch import GraphdecoGpisPatchConfig, write_graphdeco_patch_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a reviewable Graphdeco 3DGS train.py patch fragment for GPIS training-time regularization.")
    parser.add_argument("--output", required=True, help="Path for the generated train.py patch fragment.")
    parser.add_argument("--guide", default=None, help="Optional Markdown guide path to write next to the patch.")
    parser.add_argument("--gpis-epsilon", type=float, default=0.08)
    parser.add_argument("--surface-weight", type=float, default=0.01)
    parser.add_argument("--opacity-weight", type=float, default=0.001)
    parser.add_argument("--normal-weight", type=float, default=0.001)
    parser.add_argument("--start-iteration", type=int, default=500)
    parser.add_argument("--ramp-iterations", type=int, default=1000)
    parser.add_argument("--max-gaussians", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = GraphdecoGpisPatchConfig(
        default_gpis_epsilon=args.gpis_epsilon,
        default_surface_weight=args.surface_weight,
        default_opacity_weight=args.opacity_weight,
        default_normal_weight=args.normal_weight,
        default_start_iteration=args.start_iteration,
        default_ramp_iterations=args.ramp_iterations,
        default_max_gaussians=args.max_gaussians,
        default_batch_size=args.batch_size,
    )
    result = write_graphdeco_patch_bundle(
        Path(args.output),
        None if args.guide is None else Path(args.guide),
        config=config,
    )
    print(f"Wrote {result['patch_path']}")
    if "guide_path" in result:
        print(f"Wrote {result['guide_path']}")


if __name__ == "__main__":
    main()
