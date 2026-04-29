from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.real_workflow import run_real_evaluation


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def positive_or_none(value: int) -> int | None:
    return None if value <= 0 else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible real-data plain-vs-GPIS-gated splat evaluation workflow.")
    parser.add_argument("--scene", default="nerfstudio_poster8_eval")
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--source-dir", default=None, help="Existing Nerfstudio-style source directory. If omitted, the dataset can be downloaded.")
    parser.add_argument("--download-dataset", type=str_to_bool, default=True)
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--dataset", default="nerfstudio_poster")
    parser.add_argument("--image-scale", type=int, default=8)
    parser.add_argument("--max-download-images", type=int, default=0)
    parser.add_argument("--train-view-count", type=int, default=12)
    parser.add_argument("--copy-images", type=str_to_bool, default=True)
    parser.add_argument("--max-points", type=int, default=800)
    parser.add_argument("--max-train-points", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lengthscale", type=float, default=0.35)
    parser.add_argument("--noise-std", type=float, default=0.06)
    parser.add_argument("--splat-sigmas", type=float, nargs="+", default=[0.025])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.16])
    parser.add_argument("--gate-floors", type=float, nargs="+", default=[0.0])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--require-all", type=str_to_bool, default=False)
    parser.add_argument("--min-sigma-px", type=float, default=0.8)
    parser.add_argument("--benchmark-target", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_real_evaluation(
        scene=args.scene,
        prepared_root=args.prepared_root,
        source_dir=Path(args.source_dir) if args.source_dir is not None else None,
        download_dataset=args.download_dataset,
        download_root=args.download_root,
        dataset=args.dataset,
        image_scale=args.image_scale,
        max_download_images=positive_or_none(args.max_download_images),
        train_view_count=args.train_view_count,
        copy_images=args.copy_images,
        max_points=positive_or_none(args.max_points),
        max_train_points=positive_or_none(args.max_train_points),
        seed=args.seed,
        lengthscale=args.lengthscale,
        noise_std=args.noise_std,
        splat_sigmas=tuple(args.splat_sigmas),
        epsilons=tuple(args.epsilons),
        gate_floors=tuple(args.gate_floors),
        split=args.split,
        max_frames=positive_or_none(args.max_frames),
        require_all=args.require_all,
        min_sigma_px=args.min_sigma_px,
        benchmark_target=args.benchmark_target,
    )
    print(f"Wrote {result['comparison_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    best_psnr = result["status"].get("best_psnr")
    if best_psnr is not None:
        print(f"best_psnr: {best_psnr['method']} {best_psnr['mean_psnr']:.6g}")
    best_ssim = result["status"].get("best_ssim")
    if best_ssim is not None:
        print(f"best_ssim: {best_ssim['method']} {best_ssim['mean_ssim']:.6g}")


if __name__ == "__main__":
    main()
