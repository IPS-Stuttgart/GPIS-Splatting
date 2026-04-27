from __future__ import annotations

import argparse

import numpy as np
import torch

from gpis_splatting.gpis import GPISPrediction, load_model, predict_gpis
from gpis_splatting.metrics import gpis_metric_row, save_metrics_csv
from gpis_splatting.paths import scene_dir
from gpis_splatting.scenes import sdf
from gpis_splatting.serialization import read_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GPIS geometry/calibration and rendered image PSNR.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = scene_dir(args.scene, args.output_root)
    config_path = out_dir / "config.json"
    model_path = out_dir / "gpis_model.npz"
    posterior_path = out_dir / "posterior_grid.npz"
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing scene/model artifacts in {out_dir}. Run generate_scene and fit_gpis first.")

    config = read_json(config_path)
    if posterior_path.exists():
        posterior = np.load(posterior_path)
        prediction = GPISPrediction(
            mean=torch.from_numpy(posterior["mean"]).to(dtype=torch.float64),
            variance=torch.from_numpy(posterior["variance"]).to(dtype=torch.float64),
            gradient=torch.zeros((posterior["mean"].shape[0], 3), dtype=torch.float64),
        )
        if "distance" in posterior.files and "distance_std" in posterior.files:
            true_sdf = torch.from_numpy(posterior["true_sdf"]).to(dtype=torch.float64)
            row = {
                "rmse_sdf": float(np.sqrt(np.mean((posterior["mean"] - posterior["true_sdf"]) ** 2))),
                "iou_inside": float(
                    np.logical_and(posterior["mean"] <= 0.0, posterior["true_sdf"] <= 0.0).sum()
                    / max(np.logical_or(posterior["mean"] <= 0.0, posterior["true_sdf"] <= 0.0).sum(), 1)
                ),
            }
            full_prediction = GPISPrediction(
                mean=torch.from_numpy(posterior["mean"]).to(dtype=torch.float64),
                variance=torch.from_numpy(posterior["variance"]).to(dtype=torch.float64),
                gradient=torch.from_numpy(_gradient_from_distance_arrays(posterior)).to(dtype=torch.float64),
            )
            row = gpis_metric_row(full_prediction, true_sdf, render_dir=out_dir)
        else:
            model, _ = load_model(str(model_path))
            grid = torch.from_numpy(posterior["grid_xyz"]).to(dtype=torch.float64)
            full_prediction = predict_gpis(model, grid, batch_size=args.batch_size)
            true_sdf = torch.from_numpy(posterior["true_sdf"]).to(dtype=torch.float64)
            row = gpis_metric_row(full_prediction, true_sdf, render_dir=out_dir)
    else:
        model, _ = load_model(str(model_path))
        samples = np.load(out_dir / "samples.npz")
        points = torch.from_numpy(samples["points"]).to(dtype=torch.float64)
        prediction = predict_gpis(model, points, batch_size=args.batch_size)
        true_sdf = sdf(points, config["shape"])
        row = gpis_metric_row(prediction, true_sdf, render_dir=out_dir)

    metrics_path = out_dir / "metrics.csv"
    save_metrics_csv(metrics_path, row)
    print(f"Wrote {metrics_path}")
    for key, value in row.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")


def _gradient_from_distance_arrays(posterior: np.lib.npyio.NpzFile) -> np.ndarray:
    """Reconstruct a compatible gradient magnitude for metrics saved in posterior_grid.npz."""
    mean = posterior["mean"]
    distance = posterior["distance"]
    distance_std = np.clip(posterior["distance_std"], 1e-6, None)
    std = np.sqrt(np.clip(posterior["variance"], 1e-12, None))
    grad_norm_from_distance = np.abs(mean) / np.clip(np.abs(distance), 1e-6, None)
    grad_norm_from_std = std / distance_std
    grad_norm = np.where(np.isfinite(grad_norm_from_distance), grad_norm_from_distance, grad_norm_from_std)
    grad_norm = np.clip(grad_norm, 1e-6, None)
    gradient = np.zeros((mean.shape[0], 3), dtype=np.float64)
    gradient[:, 0] = grad_norm
    return gradient


if __name__ == "__main__":
    main()

