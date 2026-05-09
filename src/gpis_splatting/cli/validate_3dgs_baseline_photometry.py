from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.baseline_photometry import validate_3dgs_baseline_photometry
from gpis_splatting.cli.evaluate_real_renders import str_to_bool


def parse_sh_degree(value: str) -> int | str:
    return "auto" if value.strip().lower() == "auto" else int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate gsplat rendering of a trained 3DGS baseline before GPIS comparisons.")
    parser.add_argument("--input-ply", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reference-renders-dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--color-mode", choices=("auto", "sh", "rgb"), default="auto")
    parser.add_argument("--sh-degree", type=parse_sh_degree, default="auto")
    parser.add_argument("--strict-3dgs-fidelity", type=str_to_bool, default=True)
    parser.add_argument("--background-mode", choices=("auto", "black", "white", "rgb"), default="auto")
    parser.add_argument("--rasterize-mode", choices=("classic", "antialiased"), default="classic")
    parser.add_argument("--compute-lpips", type=str_to_bool, default=False)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-reference-psnr", type=float, default=55.0)
    parser.add_argument("--max-reference-l1", type=float, default=0.002)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    output_dir = Path(args.output_dir) if args.output_dir is not None else scene_dir / "baseline_3dgs_photometry"
    result = validate_3dgs_baseline_photometry(
        input_ply_path=args.input_ply,
        scene_dir=scene_dir,
        output_dir=output_dir,
        reference_predictions_dir=args.reference_renders_dir,
        split=args.split,
        device=args.device,
        color_mode=args.color_mode,
        sh_degree=args.sh_degree,
        strict_3dgs_fidelity=args.strict_3dgs_fidelity,
        background_mode=args.background_mode,
        rasterize_mode=args.rasterize_mode,
        compute_lpips=args.compute_lpips,
        max_frames=args.max_frames,
        min_reference_psnr=args.min_reference_psnr,
        max_reference_l1=args.max_reference_l1,
    )
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"passed: {result['passed']}")
    for reason in result["pass_reasons"]:
        print(f"- {reason}")


if __name__ == "__main__":
    main()
