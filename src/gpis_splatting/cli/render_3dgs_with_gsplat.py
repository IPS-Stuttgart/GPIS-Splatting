from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.cli.evaluate_real_renders import str_to_bool
from gpis_splatting.gsplat_fidelity_adapter import BACKGROUND_MODES, RASTERIZE_MODES, SH_COLOR_MODES, render_3dgs_manifest_with_gsplat, render_3dgs_ply_with_gsplat
from gpis_splatting.real_pipeline import parse_rgb_triplet


def parse_sh_degree(value: str) -> int | str:
    if value.strip().lower() == "auto":
        return "auto"
    degree = int(value)
    if degree < 0:
        raise argparse.ArgumentTypeError("SH degree must be non-negative or 'auto'.")
    return degree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render trained 3DGS PLYs or GPIS-gated 3DGS variant manifests with the optional gsplat backend.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-ply", default=None, help="Single trained 3DGS point_cloud.ply to render.")
    source.add_argument("--manifest-path", default=None, help="CSV written by export_3dgs_gpis_variants; every variant is rendered.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="Single-PLY render directory or manifest variant prediction root. Defaults under <scene-dir>/renders.")
    parser.add_argument("--method-name", default="trained_3dgs_gsplat", help="Used for manifest summary artifact names.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--projection-convention", choices=["auto", "opencv", "opengl"], default="auto")
    parser.add_argument("--device", default="auto", help="Torch device for gsplat, e.g. auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--dtype", choices=["float32", "fp32", "float64", "fp64"], default="float32")
    parser.add_argument("--opacity-mode", choices=["logit", "linear"], default="logit")
    parser.add_argument("--color-mode", choices=SH_COLOR_MODES, default="auto", help="auto uses SH when f_rest_* coefficients are present; rgb uses post-activation RGB/DC fallback.")
    parser.add_argument("--sh-degree", type=parse_sh_degree, default="auto", help="SH degree to activate when --color-mode auto/sh. Defaults to the maximum degree stored in the PLY.")
    parser.add_argument("--strict-3dgs-fidelity", type=str_to_bool, default=True, help="Require trained-3DGS opacity, scale, rotation, and requested SH properties.")
    parser.add_argument("--background-mode", choices=BACKGROUND_MODES, default="auto", help="auto uses --background-color unless overridden by a fixed black/white mode.")
    parser.add_argument("--background-color", type=parse_rgb_triplet, default=(0.0, 0.0, 0.0), help="RGB triplet in [0,1], for example 0,0,0.")
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    default_output = scene_dir / "renders" / ("gsplat_3dgs_variants" if args.manifest_path is not None else "gsplat_3dgs")
    output_dir = Path(args.output_dir) if args.output_dir is not None else default_output
    common = {
        "scene_dir": scene_dir,
        "split": args.split,
        "projection_convention": args.projection_convention,
        "device": args.device,
        "dtype": args.dtype,
        "opacity_mode": args.opacity_mode,
        "color_mode": args.color_mode,
        "sh_degree": args.sh_degree,
        "strict_3dgs_fidelity": args.strict_3dgs_fidelity,
        "background_mode": args.background_mode,
        "background_color": args.background_color,
        "near_plane": args.near_plane,
        "far_plane": args.far_plane,
        "radius_clip": args.radius_clip,
        "eps2d": args.eps2d,
        "tile_size": args.tile_size,
        "packed": args.packed,
        "render_mode": args.render_mode,
        "rasterize_mode": args.rasterize_mode,
        "channel_chunk": args.channel_chunk,
        "max_frames": args.max_frames,
        "max_gaussians": args.max_gaussians,
    }
    if args.manifest_path is not None:
        result = render_3dgs_manifest_with_gsplat(manifest_path=args.manifest_path, output_root=output_dir, method_name=args.method_name, **common)
        print(f"Wrote {result['render_manifest_path']}")
        print(f"Wrote {result['status_path']}")
        print(f"Wrote {result['report_path']}")
        print(f"variants: {result['status']['variant_count']}")
    else:
        result = render_3dgs_ply_with_gsplat(input_ply_path=args.input_ply, output_dir=output_dir, **common)
        print(f"Wrote {result['report_path']}")
        print(f"images: {result['report']['image_count']}")
        print(f"color: {result['report']['color']['effective_color_mode']} sh_degree={result['report']['color']['effective_sh_degree']}")


if __name__ == "__main__":
    main()
