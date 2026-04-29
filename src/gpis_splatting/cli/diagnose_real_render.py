from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.real_diagnostics import diagnose_real_render
from gpis_splatting.real_pipeline import PROJECTION_CONVENTIONS, parse_rgb_triplet


def background_color(value: str) -> tuple[float, float, float]:
    try:
        return parse_rgb_triplet(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write diagnostic panels and per-frame stats for real-scene plain and GPIS-gated splat renders.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--splats-path", default=None, help="Defaults to <scene-dir>/real_splats.npz.")
    parser.add_argument("--model-path", default=None, help="Defaults to <scene-dir>/real_gpis_model.npz.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/diagnostics/real_render.")
    parser.add_argument("--plain-renders-dir", default=None, help="Optional existing plain-render directory to diagnose instead of rendering plain images.")
    parser.add_argument("--gated-renders-dir", default=None, help="Optional existing GPIS-gated render directory to diagnose instead of rendering gated images.")
    parser.add_argument("--split", default="test", help="Scene split to diagnose, or all.")
    parser.add_argument("--max-frames", type=int, default=4, help="Frame cap for diagnostics. Use 0 for the whole split.")
    parser.add_argument("--epsilon", type=float, default=0.16)
    parser.add_argument("--gate-floor", type=float, default=0.0)
    parser.add_argument("--projection-convention", choices=PROJECTION_CONVENTIONS, default="auto")
    parser.add_argument("--near-plane", type=float, default=1e-4)
    parser.add_argument("--kernel-radius", type=float, default=3.0)
    parser.add_argument("--min-sigma-px", type=float, default=0.8)
    parser.add_argument("--background-color", type=background_color, default=(0.0, 0.0, 0.0), help="RGB triplet in [0, 1], for example 0,0,0.")
    parser.add_argument("--gate-batch-size", type=int, default=4096)
    parser.add_argument("--max-overlay-splats", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    max_frames = None if args.max_frames <= 0 else args.max_frames
    result = diagnose_real_render(
        scene_dir=scene_dir,
        splats_path=args.splats_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        plain_renders_dir=args.plain_renders_dir,
        gated_renders_dir=args.gated_renders_dir,
        split=args.split,
        max_frames=max_frames,
        epsilon=args.epsilon,
        gate_floor=args.gate_floor,
        projection_convention=args.projection_convention,
        near_plane=args.near_plane,
        kernel_radius=args.kernel_radius,
        min_sigma_px=args.min_sigma_px,
        background_color=args.background_color,
        gate_batch_size=args.gate_batch_size,
        max_overlay_splats=args.max_overlay_splats,
    )
    print(f"Wrote {result['frame_metrics_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"Wrote diagnostics to {result['output_dir']}")


if __name__ == "__main__":
    main()
