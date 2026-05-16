from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gpis_splatting.cli.evaluate import main as evaluate_main
from gpis_splatting.gpis import fit_dense_gpis, predict_gpis, save_model
from gpis_splatting.serialization import write_json


def test_evaluate_uses_saved_posterior_gradient_not_legacy_distance_arrays(tmp_path: Path) -> None:
    out_dir = tmp_path / "posterior_with_gradient"
    out_dir.mkdir()
    write_json(out_dir / "config.json", {"shape": "sphere"})
    _write_toy_model(out_dir / "gpis_model.npz")

    np.savez_compressed(
        out_dir / "posterior_grid.npz",
        grid_xyz=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        true_sdf=np.asarray([1.0], dtype=np.float64),
        mean=np.asarray([2.0], dtype=np.float64),
        variance=np.asarray([4.0], dtype=np.float64),
        gradient=np.asarray([[2.0, 0.0, 0.0]], dtype=np.float64),
        distance=np.asarray([100.0], dtype=np.float64),
        distance_std=np.asarray([100.0], dtype=np.float64),
    )

    evaluate_main(["--scene", "posterior_with_gradient", "--output-root", str(tmp_path)])

    metrics = pd.read_csv(out_dir / "metrics.csv").iloc[0]
    assert math.isclose(metrics["nll_distance"], 0.5 * math.log(2.0 * math.pi), rel_tol=1e-12)


def test_evaluate_recomputes_model_prediction_for_legacy_posterior_without_gradient(tmp_path: Path) -> None:
    out_dir = tmp_path / "legacy_posterior"
    out_dir.mkdir()
    write_json(out_dir / "config.json", {"shape": "sphere"})

    model = _write_toy_model(out_dir / "gpis_model.npz")
    eval_points = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=torch.float64)
    prediction = predict_gpis(model, eval_points, batch_size=2)
    true_sdf = prediction.mean.detach().cpu().numpy()

    np.savez_compressed(
        out_dir / "posterior_grid.npz",
        grid_xyz=eval_points.detach().cpu().numpy(),
        true_sdf=true_sdf,
        mean=true_sdf + 10.0,
        variance=np.ones_like(true_sdf),
        distance=np.ones_like(true_sdf),
        distance_std=np.ones_like(true_sdf),
    )

    evaluate_main(["--scene", "legacy_posterior", "--output-root", str(tmp_path), "--batch-size", "2"])

    metrics = pd.read_csv(out_dir / "metrics.csv").iloc[0]
    assert metrics["rmse_sdf"] < 1e-12


def _write_toy_model(path: Path):
    x_train = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64)
    y_train = torch.tensor([-0.25, 0.25], dtype=torch.float64)
    model = fit_dense_gpis(x_train, y_train, lengthscale=0.7, variance=1.0, noise_std=0.02)
    save_model(str(path), model)
    return model
