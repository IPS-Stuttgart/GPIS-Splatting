from __future__ import annotations

import argparse

import numpy as np
import torch

from gpis_splatting.gpis import fit_dense_gpis, predict_gpis, save_model
from gpis_splatting.paths import scene_dir
from gpis_splatting.scenes import make_grid, sdf
from gpis_splatting.serialization import read_json, write_json
from gpis_splatting.visualization import save_surface_scatter, save_uncertainty_slice


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit dense GPIS and save posterior grid artifacts.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--lengthscale", type=float, default=0.8)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=None, help="Defaults to config noise_std.")
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = scene_dir(args.scene, args.output_root)
    config_path = out_dir / "config.json"
    samples_path = out_dir / "samples.npz"
    if not config_path.exists() or not samples_path.exists():
        raise FileNotFoundError(f"Missing generated scene artifacts in {out_dir}. Run generate_scene first.")

    config = read_json(config_path)
    samples = np.load(samples_path)
    x_train = torch.from_numpy(samples["points"]).to(dtype=torch.float64)
    y_train = torch.from_numpy(samples["observed_sdf"]).to(dtype=torch.float64)
    noise_std = float(args.noise_std if args.noise_std is not None else config["noise_std"])

    model = fit_dense_gpis(
        x_train,
        y_train,
        lengthscale=args.lengthscale,
        variance=args.variance,
        noise_std=noise_std,
        jitter=args.jitter,
    )
    save_model(
        str(out_dir / "gpis_model.npz"),
        model,
        metadata={
            "shape": config["shape"],
            "scene": args.scene,
            "grid_size": args.grid_size,
        },
    )

    bounds = tuple(config["bounds"])
    grid = make_grid(bounds, args.grid_size)
    prediction = predict_gpis(model, grid, batch_size=args.batch_size)
    true_sdf = sdf(grid, config["shape"])

    grid_np = grid.detach().cpu().numpy()
    mean_np = prediction.mean.detach().cpu().numpy()
    var_np = prediction.variance.detach().cpu().numpy()
    std_np = np.sqrt(np.clip(var_np, 1e-12, None))
    inside_np = prediction.inside_probability.detach().cpu().numpy()
    dist_np = prediction.distance.detach().cpu().numpy()
    dist_std_np = prediction.distance_std.detach().cpu().numpy()

    np.savez_compressed(
        out_dir / "posterior_grid.npz",
        grid_xyz=grid_np,
        mean=mean_np,
        variance=var_np,
        std=std_np,
        inside_probability=inside_np,
        distance=dist_np,
        distance_std=dist_std_np,
        true_sdf=true_sdf.detach().cpu().numpy(),
        grid_size=np.array(args.grid_size),
    )

    save_surface_scatter(out_dir / "gpis_surface.png", grid_np, mean_np, std_np)
    save_uncertainty_slice(out_dir / "uncertainty_slice.png", grid_np, mean_np, std_np, args.grid_size)

    config["gpis"] = {
        "lengthscale": args.lengthscale,
        "variance": args.variance,
        "noise_std": noise_std,
        "jitter": args.jitter,
        "grid_size": args.grid_size,
    }
    write_json(config_path, config)

    print(f"Wrote {out_dir / 'gpis_model.npz'}")
    print(f"Wrote {out_dir / 'posterior_grid.npz'}")
    print(f"Wrote {out_dir / 'gpis_surface.png'}")
    print(f"Wrote {out_dir / 'uncertainty_slice.png'}")


if __name__ == "__main__":
    main()
