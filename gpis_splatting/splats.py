from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .gpis import GPISModel, predict_gpis, surface_band_probability
from .scenes import SCENES, sdf, sdf_normals


Tensor = torch.Tensor


@dataclass
class SplatCloud:
    centers: Tensor
    colors: Tensor
    tau: Tensor
    sigma: Tensor
    is_surface: Tensor


def make_candidate_splats(
    shape: str,
    *,
    num_splats: int = 700,
    offsurface_fraction: float = 0.28,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> SplatCloud:
    """Create true surface splats plus off-surface distractors for the GPIS gate to suppress."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 101)
    lo, hi = SCENES[shape].bounds

    n_off = int(round(num_splats * offsurface_fraction))
    n_surface = num_splats - n_off

    candidates = torch.empty(max(12000, num_splats * 120), 3, dtype=dtype).uniform_(lo, hi, generator=generator)
    values = sdf(candidates, shape)
    surface_idx = torch.topk(values.abs(), k=n_surface, largest=False).indices
    centers_surface = candidates[surface_idx]
    normals = sdf_normals(centers_surface, shape)
    centers_surface = centers_surface + normals * (torch.randn(centers_surface.shape, dtype=dtype, generator=generator) * 0.015)

    off_candidates = torch.empty(max(6000, n_off * 80), 3, dtype=dtype).uniform_(lo, hi, generator=generator)
    off_values = sdf(off_candidates, shape).abs()
    mask = off_values > 0.18
    if int(mask.sum()) < n_off:
        off_idx = torch.topk(off_values, k=n_off, largest=True).indices
        centers_off = off_candidates[off_idx]
    else:
        centers_off = off_candidates[mask][:n_off]

    centers = torch.cat((centers_surface, centers_off), dim=0)
    is_surface = torch.cat(
        (
            torch.ones(n_surface, dtype=torch.bool),
            torch.zeros(centers_off.shape[0], dtype=torch.bool),
        ),
        dim=0,
    )
    perm = torch.randperm(centers.shape[0], generator=generator)
    centers = centers[perm]
    is_surface = is_surface[perm]

    span = hi - lo
    colors = torch.stack(
        (
            (centers[:, 0] - lo) / span,
            0.35 + 0.5 * (centers[:, 1] - lo) / span,
            1.0 - 0.65 * (centers[:, 2] - lo) / span,
        ),
        dim=-1,
    ).clamp(0.0, 1.0)
    tau = torch.full((centers.shape[0],), 0.45, dtype=dtype)
    sigma = torch.full((centers.shape[0],), 0.045, dtype=dtype)
    return SplatCloud(centers, colors, tau, sigma, is_surface)


def save_splats(path: str, splats: SplatCloud) -> None:
    np.savez_compressed(
        path,
        centers=splats.centers.detach().cpu().numpy(),
        colors=splats.colors.detach().cpu().numpy(),
        tau=splats.tau.detach().cpu().numpy(),
        sigma=splats.sigma.detach().cpu().numpy(),
        is_surface=splats.is_surface.detach().cpu().numpy(),
    )


def load_splats(path: str) -> SplatCloud:
    npz = np.load(path)
    return SplatCloud(
        centers=torch.from_numpy(npz["centers"]).to(dtype=torch.float64),
        colors=torch.from_numpy(npz["colors"]).to(dtype=torch.float64),
        tau=torch.from_numpy(npz["tau"]).to(dtype=torch.float64),
        sigma=torch.from_numpy(npz["sigma"]).to(dtype=torch.float64),
        is_surface=torch.from_numpy(npz["is_surface"]).to(dtype=torch.bool),
    )


def gpis_gate_for_splats(
    splats: SplatCloud,
    model: GPISModel,
    epsilon: float,
    *,
    batch_size: int = 4096,
) -> Tensor:
    prediction = predict_gpis(model, splats.centers, batch_size=batch_size)
    return surface_band_probability(prediction, epsilon)

