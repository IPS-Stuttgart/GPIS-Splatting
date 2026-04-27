from __future__ import annotations

import argparse

import numpy as np

from gpis_splatting.paths import ensure_scene_dir
from gpis_splatting.scenes import SCENES, available_shapes, sample_scene
from gpis_splatting.serialization import tensors_to_numpy, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic noisy SDF samples.")
    parser.add_argument("--shape", choices=available_shapes(), required=True)
    parser.add_argument("--scene", default=None, help="Experiment scene name. Defaults to the shape name.")
    parser.add_argument("--num-points", type=int, default=180)
    parser.add_argument("--noise-std", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-root", default="experiments")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    scene_name = args.scene or args.shape
    out_dir = ensure_scene_dir(scene_name, args.output_root)

    data = sample_scene(args.shape, args.num_points, args.seed, args.noise_std)
    np.savez_compressed(out_dir / "samples.npz", **tensors_to_numpy(data))

    write_json(
        out_dir / "config.json",
        {
            "scene": scene_name,
            "shape": args.shape,
            "bounds": list(SCENES[args.shape].bounds),
            "num_points": args.num_points,
            "noise_std": args.noise_std,
            "seed": args.seed,
        },
    )
    print(f"Wrote {out_dir / 'samples.npz'}")
    print(f"Wrote {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
