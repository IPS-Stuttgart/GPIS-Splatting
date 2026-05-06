from __future__ import annotations

import argparse

from gpis_splatting.cli.prepare_real_scene import str_to_bool
from gpis_splatting.colmap_render_mapping import LINK_MODES, map_3dgs_renders_to_prepared_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Map standard 3DGS render outputs back to prepared-scene image paths for render evaluation.")
    parser.add_argument("--map-path", required=True, help="render_name_map.csv written by export_prepared_scene_to_colmap_3dgs.")
    parser.add_argument("--renders-dir", required=True, help="Directory containing standard 3DGS rendered images, e.g. 00000.png.")
    parser.add_argument("--output-dir", required=True, help="Output prediction directory in prepared-scene layout.")
    parser.add_argument("--link-mode", choices=LINK_MODES, default="copy", help="How to materialize mapped images.")
    parser.add_argument("--require-all", type=str_to_bool, default=True, help="Fail if any mapped render is missing.")
    parser.add_argument("--overwrite", type=str_to_bool, default=True, help="Overwrite existing mapped images.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = map_3dgs_renders_to_prepared_scene(
        map_path=args.map_path,
        renders_dir=args.renders_dir,
        output_dir=args.output_dir,
        link_mode=args.link_mode,
        require_all=args.require_all,
        overwrite=args.overwrite,
    )
    print(f"Wrote {result['mapping_csv_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"mapped: {result['status']['mapped_count']}")
    print(f"missing: {result['status']['missing_count']}")


if __name__ == "__main__":
    main()
