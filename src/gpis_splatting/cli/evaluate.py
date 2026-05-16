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
        eval_points = torch.from_numpy(posterior["grid_xyz"]).to(dtype=torch.float64)
        true_sdf = torch.from_numpy(posterior["true_sdf"]).to(dtype=torch.float64)
        if "gradient" in posterior.files:
            prediction = _prediction_from_posterior_arrays(posterior)
        else:
            model, _ = load_model(str(model_path))
            prediction = predict_gpis(model, eval_points, batch_size=args.batch_size)
    else:
        model, _ = load_model(str(model_path))
        samples = np.load(out_dir / "samples.npz")
        eval_points = torch.from_numpy(samples["points"]).to(dtype=torch.float64)
        prediction = predict_gpis(model, eval_points, batch_size=args.batch_size)
        true_sdf = sdf(eval_points, config["shape"])

    row = gpis_metric_row(prediction, true_sdf, render_dir=out_dir)
    feedback_model_path = out_dir / "feedback_gpis_model.npz"
    if feedback_model_path.exists():
        feedback_model, _ = load_model(str(feedback_model_path))
        feedback_prediction = predict_gpis(feedback_model, eval_points, batch_size=args.batch_size)
        feedback_row = gpis_metric_row(feedback_prediction, true_sdf)
        for key, value in feedback_row.items():
            row[f"feedback_{key}"] = value

    metrics_path = out_dir / "metrics.csv"
    save_metrics_csv(metrics_path, row)
    print(f"Wrote {metrics_path}")
    for key, value in row.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")


def _prediction_from_posterior_arrays(posterior: np.lib.npyio.NpzFile) -> GPISPrediction:
    mean = torch.from_numpy(posterior["mean"]).to(dtype=torch.float64)
    variance = torch.from_numpy(posterior["variance"]).to(dtype=torch.float64)
    gradient = torch.from_numpy(posterior["gradient"]).to(dtype=torch.float64)
    if variance.shape != mean.shape:
        raise ValueError("posterior_grid.npz fields 'mean' and 'variance' must have matching shapes.")
    if gradient.ndim != 2 or gradient.shape != (mean.shape[0], 3):
        raise ValueError("posterior_grid.npz field 'gradient' must have shape (N, 3) matching 'mean'.")
    return GPISPrediction(mean=mean, variance=variance, gradient=gradient)


if __name__ == "__main__":
    main()
