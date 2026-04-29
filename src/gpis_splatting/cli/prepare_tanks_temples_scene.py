from __future__ import annotations

import argparse

from gpis_splatting.cli.render_real_splats import str_to_bool
from gpis_splatting.tanks_temples import SUPPORTED_TANKS_TEMPLES_SCENES, prepare_tanks_temples_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a Tanks and Temples scene from images plus Redwood .log camera poses.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--scene", choices=SUPPORTED_TANKS_TEMPLES_SCENES, default="Ignatius")
    parser.add_argument("--prepared-scene", default=None, help="Prepared scene name. Defaults to <scene>_tanks_temples in lowercase.")
    parser.add_argument("--output-root", default="real_scenes")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--reconstruction-path", default=None)
    parser.add_argument("--ground-truth-path", default=None)
    parser.add_argument("--alignment-path", default=None)
    parser.add_argument("--crop-path", default=None)
    parser.add_argument("--train-view-count", type=int, default=12)
    parser.add_argument("--copy-images", type=str_to_bool, default=True)
    parser.add_argument("--focal-length-factor", type=float, default=0.7)
    parser.add_argument("--bounds-scale", type=float, default=1.1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = prepare_tanks_temples_scene(
        input_dir=args.input_dir,
        output_root=args.output_root,
        scene=args.scene,
        prepared_scene=args.prepared_scene,
        image_dir=args.image_dir,
        log_path=args.log_path,
        reconstruction_path=args.reconstruction_path,
        ground_truth_path=args.ground_truth_path,
        alignment_path=args.alignment_path,
        crop_path=args.crop_path,
        train_view_count=args.train_view_count,
        copy_images=args.copy_images,
        focal_length_factor=args.focal_length_factor,
        bounds_scale=args.bounds_scale,
    )
    print(f"Wrote {out_dir / 'real_scene.json'}")
    print(f"Wrote {out_dir / 'cameras.json'}")
    print(f"Wrote {out_dir / 'splits.json'}")
    print(f"Wrote {out_dir / 'validation.json'}")


if __name__ == "__main__":
    main()
