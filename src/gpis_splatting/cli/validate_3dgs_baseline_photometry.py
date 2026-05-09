from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.baseline_photometry import validate_3dgs_baseline_photometry
from gpis_splatting.cli.evaluate_real_renders import str_to_bool
from gpis_splatting.gsplat_adapter import BACKGROUND_MODES, RASTERIZE_MODES, SH_COLOR_MODES
from gpis_splatting.real_pipeline import parse_rgb_triplet


def parse_sh_degree(value: str) -> int | str:
    if value.strip().lower() == "auto":
        return "auto"
    degree = int(value)
    if degree < 0:
        raise argparse.ArgumentTypeError("SH degree must be non-negative or 'auto'.")
    return degree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render and validate a trained 3DGS baseline before running GPIS-gated comparisons.")
    parser.add_argument("--input-ply", required=True, help="Trained Graphdeco-style point_cloud.ply.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/baseline_3dgs_photometry.")
    parser.add_argument("--reference-renders-dir", default=None, help="Optional canonical renderer output directory for pass/fail agreement checks.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--projection-convention", choices=["auto", "opencv", "opengl"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "fp32", "float64", "fp64"], default="float32")
    parser.add_argument("--opacity-mode", choices=["logit", "linear"], default="logit")
    parser.add_argument("--color-mode", choices=SH_COLOR_MODES, default="auto")
    parser.add_argument("--sh-degree", type=parse_sh_degree, default="auto")
    parser.add_argument("--strict-3dgs-fidelity", type=str_to_bool, default=True)
    parser.add_argument("--background-mode", choices=BACKGROUND_MODES, default="auto")
    parser.add_argument("--background-color", type=parse_rgb_triplet, default=(0.0, 0.0, 0.0))
    parser.add_argument("--near-plane", type=float, default=1e-2)
    parser.add_argument("--far-plane", type=float, default=1.0e10)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--eps2d", type=float, default=0.3)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--packed", type=str_to_bool, default=True)
    parser.add_argument("--render-mode", default="RGB")
    parser.add_argument("--rasterize-mode", choices=RASTERIZE_MODES, default="classic")
    parser.add_argument("--channel-chunk", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--compute-lpips", type=str_to_bool, default=False)
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
        projection_convention=args.projection_convention,
        device=args.device,
        dtype=args.dtype,
        opacity_mode=args.opacity_mode,
        color_mode=args.color_mode,
        sh_degree=args.sh_degree,
        strict_3dgs_fidelity=args.strict_3dgs_fidelity,
        background_mode=args.background_mode,
        background_color=args.background_color,
        near_plane=args.near_plane,
        far_plane=args.far_plane,
        radius_clip=args.radius_clip,
        eps2d=args.eps2d,
        tile_size=args.tile_size,
        packed=args.packed,
        render_mode=args.render_mode,
        rasterize_mode=args.rasterize_mode,
        channel_chunk=args.channel_chunk,
        max_frames=args.max_frames,
        max_gaussians=args.max_gaussians,
        compute_lpips=args.compute_lpips,
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
