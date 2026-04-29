from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.real_pipeline import fit_real_gpis


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a dense GPIS model from bootstrapped real-scene pseudo-SDF samples.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--samples-path", default=None, help="Defaults to <scene-dir>/real_samples.npz.")
    parser.add_argument("--output-model", default=None, help="Defaults to <scene-dir>/real_gpis_model.npz.")
    parser.add_argument("--lengthscale", type=float, default=0.25)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--max-train-points", type=int, default=1200, help="Dense GPIS cap. Use 0 or a negative value to fit all samples.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--use-observation-noise", type=str_to_bool, default=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    max_train_points = None if args.max_train_points <= 0 else args.max_train_points
    result = fit_real_gpis(
        scene_dir=scene_dir,
        samples_path=args.samples_path,
        output_model=args.output_model,
        lengthscale=args.lengthscale,
        variance=args.variance,
        noise_std=args.noise_std,
        jitter=args.jitter,
        max_train_points=max_train_points,
        seed=args.seed,
        use_observation_noise=args.use_observation_noise,
    )
    print(f"Wrote {result['model_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"train_samples: {result['report']['train_sample_count']}")
    print(f"available_samples: {result['report']['available_sample_count']}")


if __name__ == "__main__":
    main()
