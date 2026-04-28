from __future__ import annotations

import argparse

from gpis_splatting.real_scene import SUPPORTED_INPUT_FORMATS, prepare_real_scene


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a real image/camera dataset in the GPIS-splatting scene format.")
    parser.add_argument("--input-dir", required=True, help="Dataset directory containing transforms.json or COLMAP text files.")
    parser.add_argument("--scene", required=True, help="Prepared scene name.")
    parser.add_argument("--output-root", default="real_scenes")
    parser.add_argument("--dataset", default="mipnerf360_sparse")
    parser.add_argument("--input-format", choices=SUPPORTED_INPUT_FORMATS, default="auto")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--train-view-count", type=int, default=12)
    parser.add_argument("--copy-images", type=str_to_bool, default=True)
    parser.add_argument("--bounds-scale", type=float, default=1.1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = prepare_real_scene(
        input_dir=args.input_dir,
        output_root=args.output_root,
        scene=args.scene,
        dataset=args.dataset,
        input_format=args.input_format,
        image_dir=args.image_dir,
        train_view_count=args.train_view_count,
        copy_images=args.copy_images,
        bounds_scale=args.bounds_scale,
    )
    print(f"Wrote {out_dir / 'real_scene.json'}")
    print(f"Wrote {out_dir / 'cameras.json'}")
    print(f"Wrote {out_dir / 'splits.json'}")
    print(f"Wrote {out_dir / 'validation.json'}")


if __name__ == "__main__":
    main()
