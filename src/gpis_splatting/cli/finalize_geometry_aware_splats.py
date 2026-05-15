from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.geometry_aware_splat_export import finalize_gpis_aware_splat_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize GPIS-aware initialization splats with preserved anisotropic geometry fields.")
    parser.add_argument("--gaussians-path", required=True, help="Path to the rich *_gaussians.npz written by initialize_gpis_splats.")
    parser.add_argument("--output-splats", default=None, help="Defaults to replacing *_gaussians.npz with *_splats.npz.")
    parser.add_argument("--output-status", default=None, help="Defaults to <output-splats-stem>_geometry_status.json.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = finalize_gpis_aware_splat_export(
        gaussians_path=Path(args.gaussians_path),
        output_splats_path=Path(args.output_splats) if args.output_splats is not None else None,
        output_status_path=Path(args.output_status) if args.output_status is not None else None,
    )
    print(f"Wrote {result['splats_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"splats: {result['status']['splat_count']}")


if __name__ == "__main__":
    main()
