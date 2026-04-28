from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.real_scene import validate_real_scene_dir
from gpis_splatting.serialization import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a prepared real GPIS-splatting scene.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--output", default=None, help="Defaults to <scene-dir>/validation.json.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    validation = validate_real_scene_dir(scene_dir)
    output_path = Path(args.output) if args.output is not None else scene_dir / "validation.json"
    write_json(output_path, validation)
    print(f"Wrote {output_path}")
    print(f"passed: {validation['passed']}")
    print(f"images: {validation['image_count']}")
    print(f"train/test: {validation['train_view_count']}/{validation['test_view_count']}")
    if not validation["passed"]:
        raise SystemExit("Prepared real scene validation failed.")


if __name__ == "__main__":
    main()
