from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.cli.evaluate_real_renders import str_to_bool
from gpis_splatting.cli.render_3dgs_with_gsplat import parse_sh_degree
from gpis_splatting.gsplat_fidelity_adapter import BACKGROUND_MODES, RASTERIZE_MODES, SH_COLOR_MODES
from gpis_splatting.real_pipeline import parse_rgb_triplet
from gpis_splatting.scale_robust_rendering import DEFAULT_RASTERIZE_MODES, DEFAULT_SCALE_FACTORS, run_scale_robust_3dgs_experiment


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive scales.")
    return values


def parse_mode_list(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modes:
        raise argparse.ArgumentTypeError("Expected at least one rasterize mode.")
    unsupported = sorted(set(modes) - set(RASTERIZE_MODES))
    if unsupported:
        raise argparse.ArgumentTypeError(f"Unsupported rasterize modes: {', '.join(unsupported)}")
    return modes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scale-robust trained-3DGS rendering experiments with classic and antialiased gsplat modes.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-ply", default=None, help="Single trained 3DGS point_cloud.ply to evaluate as the baseline variant.")
    source.add_argument("--manifest-path", default=None, help="CSV written by export_3dgs_gpis_variants; every variant is evaluated.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="scale_robust_3dgs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--scales", type=parse_float_list, default=DEFAULT_SCALE_FACTORS, help="Comma-separated test scales, e.g. 0.5,1.0,2.0.")
    parser.add_argument("--rasterize-modes", type=parse_mode_list, default=DEFAULT_RASTERIZE_MODES, help="Comma-separated gsplat rasterize modes: classic,antialiased.")
    parser.add_argument("--projection-convention", choices=["auto", "opencv", "opengl"], default="auto")
    parser.add_argument("--device", default="auto", help="Torch device for gsplat, e.g. auto, cuda, cuda:0, or cpu.")
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
    parser.add_argument("--channel-chunk", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--compute-lpips", type=str_to_bool, default=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    result = run_scale_robust_3dgs_experiment(
        scene_dir=scene_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        input_ply_path=args.input_ply,
        method_name=args.method_name,
        split=args.split,
        scales=args.scales,
        rasterize_modes=args.rasterize_modes,
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
        compute_lpips=args.compute_lpips,
    )
    print(f"Wrote {result['render_manifest_path']}")
    print(f"Wrote {result['metrics_path']}")
    print(f"Wrote {result['summary_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"render cells: {result['status']['render_count']}")


if __name__ == "__main__":
    main()
