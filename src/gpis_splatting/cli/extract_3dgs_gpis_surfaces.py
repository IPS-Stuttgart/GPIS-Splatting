from __future__ import annotations

import argparse

from gpis_splatting.cli.evaluate_tanks_temples_geometry import optional_positive_int, str_to_bool
from gpis_splatting.gaussian_surface import SURFACE_EXTRACTION_MODES, extract_gpis_gated_gaussian_surfaces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract 2DGS/GOF-style surface proxies from GPIS-gated trained 3DGS Gaussians.")
    parser.add_argument("--input-ply", required=True, help="Path to a trained 3DGS point_cloud.ply.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-path", default=None, help="Optional .npz containing gate or raw_gate aligned to the PLY Gaussian order.")
    parser.add_argument("--method-name", default="gpis_gated_3dgs_surface")
    parser.add_argument("--gate-thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--extraction-modes", nargs="+", choices=list(SURFACE_EXTRACTION_MODES), default=list(SURFACE_EXTRACTION_MODES))
    parser.add_argument("--include-baseline", type=str_to_bool, default=True)
    parser.add_argument("--opacity-mode", choices=["logit", "linear"], default="logit")
    parser.add_argument("--surfel-scale", type=float, default=1.0)
    parser.add_argument("--min-surfel-radius", type=float, default=1e-5)
    parser.add_argument("--max-surfel-radius", type=float, default=None)
    parser.add_argument("--opacity-field-resolution", type=int, default=48)
    parser.add_argument("--opacity-field-threshold", type=float, default=0.15)
    parser.add_argument("--opacity-field-sigma-scale", type=float, default=1.0)
    parser.add_argument("--opacity-field-margin-sigma", type=float, default=3.0)
    parser.add_argument("--max-field-gaussians", type=optional_positive_int, default=20000, help="Subsample cap for opacity-field extraction. Use 0 for all selected Gaussians.")
    parser.add_argument("--field-query-chunk-size", type=int, default=4096)
    parser.add_argument("--field-gaussian-chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = extract_gpis_gated_gaussian_surfaces(
        input_ply_path=args.input_ply,
        output_dir=args.output_dir,
        gate_path=args.gate_path,
        method_name=args.method_name,
        gate_thresholds=tuple(args.gate_thresholds),
        extraction_modes=tuple(args.extraction_modes),
        include_baseline=args.include_baseline,
        opacity_mode=args.opacity_mode,
        surfel_scale=args.surfel_scale,
        min_surfel_radius=args.min_surfel_radius,
        max_surfel_radius=args.max_surfel_radius,
        opacity_field_resolution=args.opacity_field_resolution,
        opacity_field_threshold=args.opacity_field_threshold,
        opacity_field_sigma_scale=args.opacity_field_sigma_scale,
        opacity_field_margin_sigma=args.opacity_field_margin_sigma,
        max_field_gaussians=args.max_field_gaussians,
        field_query_chunk_size=args.field_query_chunk_size,
        field_gaussian_chunk_size=args.field_gaussian_chunk_size,
        seed=args.seed,
    )
    print(f"Wrote {result['manifest_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"surfaces: {result['status']['surface_count']}")


if __name__ == "__main__":
    main()
