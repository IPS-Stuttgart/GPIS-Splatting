from __future__ import annotations

import argparse

from gpis_splatting.external_3dgs import convert_3dgs_ply_to_splats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a trained 3DGS Gaussian PLY into internal splats for GPIS scoring/calibration.")
    parser.add_argument("--input-ply", required=True, help="Path to a trained 3DGS point_cloud.ply.")
    parser.add_argument("--output-splats", required=True, help="Output .npz splats path for GPIS scoring.")
    parser.add_argument("--opacity-mode", choices=["logit", "linear"], default="logit", help="Interpretation of the 3DGS opacity property.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = convert_3dgs_ply_to_splats(
        ply_path=args.input_ply,
        output_splats_path=args.output_splats,
        opacity_mode=args.opacity_mode,
    )
    print(f"Wrote {result['splats_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"splats: {result['status']['splat_count']}")


if __name__ == "__main__":
    main()
