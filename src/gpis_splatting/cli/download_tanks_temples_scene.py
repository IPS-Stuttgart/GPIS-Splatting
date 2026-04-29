from __future__ import annotations

import argparse

from gpis_splatting.cli.render_real_splats import str_to_bool
from gpis_splatting.tanks_temples import SUPPORTED_TANKS_TEMPLES_SCENES, TANKS_TEMPLES_RESOURCE_NAMES, download_tanks_temples_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download official Tanks and Temples scene assets for GPIS-splatting real-scene evaluation.")
    parser.add_argument("--scene", choices=SUPPORTED_TANKS_TEMPLES_SCENES, default="Ignatius")
    parser.add_argument("--output-root", default="real_scenes/_downloads")
    parser.add_argument("--include-images", type=str_to_bool, default=True)
    parser.add_argument("--include-auxiliary", type=str_to_bool, default=True, help="Download reconstruction, camera log, alignment, and crop assets.")
    parser.add_argument("--include-ground-truth", type=str_to_bool, default=True)
    parser.add_argument("--resources", nargs="+", choices=TANKS_TEMPLES_RESOURCE_NAMES, default=None, help="Optional exact resources to download.")
    parser.add_argument("--unpack-images", type=str_to_bool, default=True)
    parser.add_argument("--max-images", type=int, default=0, help="Optional extraction cap for smoke runs. Use 0 for all images.")
    parser.add_argument("--force", type=str_to_bool, default=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    max_images = None if args.max_images <= 0 else args.max_images
    result = download_tanks_temples_scene(
        scene=args.scene,
        output_root=args.output_root,
        include_images=args.include_images,
        include_auxiliary=args.include_auxiliary,
        include_ground_truth=args.include_ground_truth,
        unpack_images=args.unpack_images,
        max_images=max_images,
        resources=tuple(args.resources) if args.resources is not None else None,
        force=args.force,
    )
    print(f"Wrote {result['report_path']}")
    print(f"Dataset directory: {result['output_dir']}")
    print(f"downloaded: {result['report']['downloaded_count']}")
    print(f"skipped: {result['report']['skipped_count']}")
    print(f"extracted images: {result['report']['extracted_image_count']}")


if __name__ == "__main__":
    main()
