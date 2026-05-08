from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.depth_normal_supervision import augment_samples_with_depth_normal_confidence


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Augment real-scene GPIS samples with confidence-weighted depth and normal pseudo-SDF observations."
    )
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--depth-confidence-dir", default=None)
    parser.add_argument("--normal-dir", default=None)
    parser.add_argument("--normal-confidence-dir", default=None)
    parser.add_argument("--base-samples-path", default=None)
    parser.add_argument("--include-base-samples", type=str_to_bool, default=True)
    parser.add_argument("--output-samples-path", default="real_depth_normal_samples.npz")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-pixels-per-frame", type=int, default=2048)
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--projection-convention", choices=("auto", "opencv", "opengl"), default="auto")
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--depth-min", type=float, default=1e-4)
    parser.add_argument("--depth-max", type=float, default=None)
    parser.add_argument("--normal-space", choices=("camera", "world"), default="camera")
    parser.add_argument("--add-free-space-samples", type=str_to_bool, default=True)
    parser.add_argument("--free-space-samples-per-depth", type=int, default=1)
    parser.add_argument("--free-space-min-fraction", type=float, default=0.25)
    parser.add_argument("--free-space-max-fraction", type=float, default=0.85)
    parser.add_argument("--max-free-space-sdf", type=float, default=0.35)
    parser.add_argument("--add-normal-offset-samples", type=str_to_bool, default=True)
    parser.add_argument("--normal-offset-distance", type=float, default=0.04)
    parser.add_argument("--surface-noise-min", type=float, default=0.015)
    parser.add_argument("--surface-noise-max", type=float, default=0.12)
    parser.add_argument("--free-space-noise-min", type=float, default=0.04)
    parser.add_argument("--free-space-noise-max", type=float, default=0.18)
    parser.add_argument("--normal-noise-min", type=float, default=0.02)
    parser.add_argument("--normal-noise-max", type=float, default=0.14)
    parser.add_argument("--default-depth-confidence", type=float, default=1.0)
    parser.add_argument("--default-normal-confidence", type=float, default=0.7)
    parser.add_argument("--min-depth-confidence", type=float, default=0.0)
    parser.add_argument("--min-normal-confidence", type=float, default=0.0)
    parser.add_argument("--confidence-power", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    max_pixels = None if args.max_pixels_per_frame is not None and args.max_pixels_per_frame <= 0 else args.max_pixels_per_frame
    result = augment_samples_with_depth_normal_confidence(
        scene_dir=scene_dir,
        depth_dir=args.depth_dir,
        depth_confidence_dir=args.depth_confidence_dir,
        normal_dir=args.normal_dir,
        normal_confidence_dir=args.normal_confidence_dir,
        base_samples_path=args.base_samples_path,
        include_base_samples=args.include_base_samples,
        output_samples_path=args.output_samples_path,
        report_path=args.report_path,
        split=args.split,
        max_frames=args.max_frames,
        max_pixels_per_frame=max_pixels,
        pixel_stride=args.pixel_stride,
        seed=args.seed,
        projection_convention=args.projection_convention,
        depth_scale=args.depth_scale,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        normal_space=args.normal_space,
        add_free_space_samples=args.add_free_space_samples,
        free_space_samples_per_depth=args.free_space_samples_per_depth,
        free_space_min_fraction=args.free_space_min_fraction,
        free_space_max_fraction=args.free_space_max_fraction,
        max_free_space_sdf=args.max_free_space_sdf,
        add_normal_offset_samples=args.add_normal_offset_samples,
        normal_offset_distance=args.normal_offset_distance,
        surface_noise_min=args.surface_noise_min,
        surface_noise_max=args.surface_noise_max,
        free_space_noise_min=args.free_space_noise_min,
        free_space_noise_max=args.free_space_noise_max,
        normal_noise_min=args.normal_noise_min,
        normal_noise_max=args.normal_noise_max,
        default_depth_confidence=args.default_depth_confidence,
        default_normal_confidence=args.default_normal_confidence,
        min_depth_confidence=args.min_depth_confidence,
        min_normal_confidence=args.min_normal_confidence,
        confidence_power=args.confidence_power,
    )
    print(f"Wrote {result['samples_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"samples: {result['report']['sample_count']}")
    print(f"depth_normal_samples: {result['report']['depth_normal_sample_count']}")


if __name__ == "__main__":
    main()
