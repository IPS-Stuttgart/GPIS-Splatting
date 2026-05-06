from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.cli.prepare_real_scene import str_to_bool
from gpis_splatting.prepared_colmap_export import COLMAP_SPLITS, export_prepared_scene_to_colmap_3dgs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a prepared GPIS-Splatting real scene as COLMAP text files for standard 3DGS training.")
    parser.add_argument("--scene", default=None, help="Prepared scene name under --prepared-root.")
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None, help="Prepared scene directory. Overrides --scene/--prepared-root.")
    parser.add_argument("--output-dir", required=True, help="Output directory containing images/ and sparse/0/.")
    parser.add_argument("--split", choices=COLMAP_SPLITS, default="train", help="Prepared-scene split to export.")
    parser.add_argument("--points-path", default=None, help="Optional point source relative to the scene directory or absolute path (.ply, points3D.txt, or internal splats .npz).")
    parser.add_argument("--max-points", type=int, default=100000, help="Maximum exported sparse points. Use 0 for all points.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--copy-images", type=str_to_bool, default=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    max_points = None if args.max_points <= 0 else args.max_points
    result = export_prepared_scene_to_colmap_3dgs(
        scene_dir=scene_dir,
        output_dir=args.output_dir,
        split=args.split,
        points_path=args.points_path,
        max_points=max_points,
        seed=args.seed,
        copy_images=args.copy_images,
    )
    print(f"Wrote {result['status']['cameras_path']}")
    print(f"Wrote {result['status']['images_path']}")
    print(f"Wrote {result['status']['points3d_path']}")
    print(f"Wrote {result['status']['render_name_map_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"frames: {result['status']['frame_count']}")
    print(f"points: {result['status']['point_count']}")


if __name__ == "__main__":
    main()
