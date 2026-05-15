from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.cli.evaluate_real_renders import str_to_bool
from gpis_splatting.real_pipeline import parse_rgb_triplet
from gpis_splatting.render_consistency import DEFAULT_AA_DOWNSAMPLE_FACTORS
from gpis_splatting.render_scale_diagnostics import run_render_scale_diagnostics


def parse_sh_degree(value: str) -> int | str:
    if value.strip().lower() == "auto":
        return "auto"
    degree = int(value)
    if degree < 0:
        raise argparse.ArgumentTypeError("SH degree must be non-negative or 'auto'.")
    return degree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render controlled gsplat scale/AA variants and evaluate scale/anti-aliasing consistency.")
    parser.add_argument("--input-ply", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--method-name", default="trained_3dgs_scale_aa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--render-scale-factor", action="append", type=float, default=None)
    parser.add_argument("--include-gsplat-antialiased", type=str_to_bool, default=True)
    parser.add_argument("--output-resolution", choices=["target", "render"], default="target")
    parser.add_argument("--aa-downsample-factor", action="append", type=int, default=None)
    parser.add_argument("--projection-convention", choices=["auto", "opencv", "opengl"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "fp32", "float64", "fp64"], default="float32")
    parser.add_argument("--opacity-mode", choices=["logit", "linear"], default="logit")
    parser.add_argument("--color-mode", choices=["auto", "sh", "rgb"], default="auto")
    parser.add_argument("--sh-degree", type=parse_sh_degree, default="auto")
    parser.add_argument("--strict-3dgs-fidelity", type=str_to_bool, default=True)
    parser.add_argument("--background-mode", choices=["auto", "black", "white", "rgb"], default="auto")
    parser.add_argument("--background-color", type=parse_rgb_triplet, default=(0.0, 0.0, 0.0))
    parser.add_argument("--near-plane", type=float, default=1e-2)
    parser.add_argument("--far-plane", type=float, default=1.0e10)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--eps2d", type=float, default=0.3)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--packed", type=str_to_bool, default=True)
    parser.add_argument("--render-mode", default="RGB")
    parser.add_argument("--channel-chunk", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--require-all", type=str_to_bool, default=True)
    parser.add_argument("--max-temporal-pairs", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    output_dir = Path(args.output_dir) if args.output_dir is not None else scene_dir / "scale_aa_diagnostics" / args.method_name
    result = run_render_scale_diagnostics(
        scene_dir=scene_dir,
        input_ply_path=args.input_ply,
        output_dir=output_dir,
        method_name=args.method_name,
        split=args.split,
        render_scale_factors=tuple(args.render_scale_factor or (0.5, 1.0, 2.0)),
        include_gsplat_antialiased=args.include_gsplat_antialiased,
        output_resolution=args.output_resolution,
        aa_downsample_factors=tuple(args.aa_downsample_factor or DEFAULT_AA_DOWNSAMPLE_FACTORS),
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
        channel_chunk=args.channel_chunk,
        max_frames=args.max_frames,
        max_gaussians=args.max_gaussians,
        require_all=args.require_all,
        max_temporal_pairs=args.max_temporal_pairs,
    )
    summary = result["status"]["summary"]
    print(f"Wrote {result['manifest_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"scale_comparisons: {summary['scale_image_count']}")
    print(f"aa_comparisons: {summary['aa_image_count']}")


if __name__ == "__main__":
    main()
