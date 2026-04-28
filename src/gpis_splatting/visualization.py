from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_surface_scatter(
    path: str | Path,
    grid_xyz: np.ndarray,
    mean: np.ndarray,
    uncertainty: np.ndarray,
    *,
    max_points: int = 2200,
) -> None:
    band = np.abs(mean) <= np.quantile(np.abs(mean), 0.06)
    points = grid_xyz[band]
    colors = uncertainty[band]
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(int)
        points = points[idx]
        colors = colors[idx]

    fig = plt.figure(figsize=(6.0, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    if points.size:
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=6, cmap="viridis", alpha=0.9)
        fig.colorbar(scatter, ax=ax, shrink=0.72, label="posterior std")
    ax.set_title("GPIS zero-level band")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_uncertainty_slice(
    path: str | Path,
    grid_xyz: np.ndarray,
    mean: np.ndarray,
    uncertainty: np.ndarray,
    grid_size: int,
) -> None:
    z_value = np.unique(grid_xyz[:, 2])[np.argmin(np.abs(np.unique(grid_xyz[:, 2])))]
    mask = np.isclose(grid_xyz[:, 2], z_value)
    xy = grid_xyz[mask][:, :2]
    unc = uncertainty[mask]
    mu = mean[mask]
    order = np.lexsort((xy[:, 1], xy[:, 0]))
    unc_img = unc[order].reshape(grid_size, grid_size).T
    mu_img = mu[order].reshape(grid_size, grid_size).T

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(unc_img, origin="lower", cmap="magma")
    ax.contour(mu_img, levels=[0.0], colors="cyan", linewidths=1.2)
    fig.colorbar(im, ax=ax, label="posterior std")
    ax.set_title("Uncertainty slice with zero contour")
    ax.set_xlabel("x grid")
    ax.set_ylabel("y grid")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
