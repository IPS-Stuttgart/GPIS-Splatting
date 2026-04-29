from __future__ import annotations

import argparse

from gpis_splatting.real_download import REAL_DOWNLOAD_PRESETS, download_real_scene


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a small real scene dataset in the format expected by the GPIS-splatting real pipeline.")
    parser.add_argument("--dataset", choices=REAL_DOWNLOAD_PRESETS, default="nerfstudio_poster")
    parser.add_argument("--output-root", default="real_scenes/_downloads")
    parser.add_argument("--image-scale", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap for laptop smoke runs. Use 0 to download all available scaled images.")
    parser.add_argument("--force", type=str_to_bool, default=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    max_images = None if args.max_images <= 0 else args.max_images
    result = download_real_scene(
        dataset=args.dataset,
        output_root=args.output_root,
        image_scale=args.image_scale,
        max_images=max_images,
        force=args.force,
    )
    print(f"Wrote {result['report_path']}")
    print(f"Dataset directory: {result['output_dir']}")
    print(f"images: {result['report']['image_count']}")
    print(f"downloaded: {result['report']['downloaded_count']}")
    print(f"skipped: {result['report']['skipped_count']}")


if __name__ == "__main__":
    main()
