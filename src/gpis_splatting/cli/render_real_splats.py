from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.real_pipeline import PROJECTION_CONVENTIONS, parse_rgb_triplet, render_real_splats


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def background_color(value: str) -> tuple[float, float, float]:
    try:
        return parse_rgb_triplet(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render prepared real-scene splats through real cameras with optional GPIS optical-depth gating.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--splats-path", default=None, help="Defaults to <scene-dir>/real_splats.npz.")
    parser.add_argument("--model-path", default=None, help="Defaults to <scene-dir>/real_gpis_model.npz when GPIS gating is enabled.")
    parser.add_argument(
        "--gate-path",
        default=None,
        help="Optional external gate/confidence .npz with gate or raw_gate. Overrides GPIS model gating.",
    )
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/renders/<method-name>.")
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--split", default="test", help="Scene split to render, or all.")
    parser.add_argument("--use-gpis-gate", type=str_to_bool, default=True)
    parser.add_argument("--epsilon", type=float, default=0.09)
    parser.add_argument("--gate-floor", type=float, default=0.0, help="Minimum multiplicative gate applied before optical thickness.")
    parser.add_argument("--projection-convention", choices=PROJECTION_CONVENTIONS, default="auto")
    parser.add_argument("--near-plane", type=float, default=1e-4)
    parser.add_argument("--kernel-radius", type=float, default=3.0)
    parser.add_argument("--min-sigma-px", type=float, default=0.6)
    parser.add_argument("--background-color", type=background_color, default=(0.0, 0.0, 0.0), help="RGB triplet in [0, 1], for example 0,0,0.")
    parser.add_argument("--gate-batch-size", type=int, default=4096)
    parser.add_argument("--max-frames", type=int, default=0, help="Optional render cap for debugging. Use 0 to render the whole split.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    max_frames = None if args.max_frames <= 0 else args.max_frames
    result = render_real_splats(
        scene_dir=scene_dir,
        splats_path=args.splats_path,
        model_path=args.model_path,
        gate_path=args.gate_path,
        output_dir=args.output_dir,
        method_name=args.method_name,
        split=args.split,
        use_gpis_gate=args.use_gpis_gate,
        epsilon=args.epsilon,
        gate_floor=args.gate_floor,
        projection_convention=args.projection_convention,
        near_plane=args.near_plane,
        kernel_radius=args.kernel_radius,
        min_sigma_px=args.min_sigma_px,
        background_color=args.background_color,
        gate_batch_size=args.gate_batch_size,
        max_frames=max_frames,
    )
    print(f"Wrote {result['report_path']}")
    if result["gate_path"] is not None:
        print(f"Wrote {result['gate_path']}")
    print(f"Wrote renders to {result['output_dir']}")
    print(f"images: {result['report']['image_count']}")


if __name__ == "__main__":
    main()
