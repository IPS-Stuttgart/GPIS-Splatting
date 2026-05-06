from __future__ import annotations

import argparse

from gpis_splatting.cli.evaluate_tanks_temples_geometry import str_to_bool
from gpis_splatting.external_3dgs import export_3dgs_gpis_variants


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trained 3DGS point_cloud.ply variants from aligned GPIS confidence gates.")
    parser.add_argument("--input-ply", required=True, help="Path to a trained 3DGS point_cloud.ply.")
    parser.add_argument("--gate-path", required=True, help="Gate .npz containing gate or raw_gate aligned to the PLY Gaussian order.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="gpis_confidence_3dgs")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--gate-thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--include-baseline", type=str_to_bool, default=True)
    parser.add_argument("--write-scaled", type=str_to_bool, default=True)
    parser.add_argument("--write-filtered", type=str_to_bool, default=True)
    parser.add_argument("--opacity-mode", choices=["logit", "linear"], default="logit")
    parser.add_argument("--opacity-scale-floor", type=float, default=0.0)
    parser.add_argument(
        "--template-model-dir",
        default=None,
        help="Optional source 3DGS model directory whose cfg_args/cameras.json/exposure.json are copied into each variant. Inferred from standard input PLY paths when omitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = export_3dgs_gpis_variants(
        input_ply_path=args.input_ply,
        gate_path=args.gate_path,
        output_dir=args.output_dir,
        method_name=args.method_name,
        iteration=args.iteration,
        gate_thresholds=tuple(args.gate_thresholds),
        include_baseline=args.include_baseline,
        write_scaled=args.write_scaled,
        write_filtered=args.write_filtered,
        opacity_mode=args.opacity_mode,
        opacity_scale_floor=args.opacity_scale_floor,
        template_model_dir=args.template_model_dir,
    )
    print(f"Wrote {result['manifest_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"variants: {result['status']['variant_count']}")


if __name__ == "__main__":
    main()
