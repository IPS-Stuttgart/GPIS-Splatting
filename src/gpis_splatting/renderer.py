from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .splats import SplatCloud


Tensor = torch.Tensor


@dataclass(frozen=True)
class Camera:
    name: str
    axes: tuple[int, int]
    depth_axis: int
    depth_sign: float


CAMERAS: dict[str, Camera] = {
    "front": Camera("front", (0, 1), 2, -1.0),
    "side": Camera("side", (1, 2), 0, -1.0),
    "top": Camera("top", (0, 2), 1, -1.0),
}


def selected_views(view: str) -> list[str]:
    if view == "all":
        return list(CAMERAS)
    if view not in CAMERAS:
        raise ValueError(f"Unknown view '{view}'. Expected one of all, {', '.join(CAMERAS)}.")
    return [view]


def render_splats(
    splats: SplatCloud,
    *,
    gate: Tensor | None = None,
    image_size: int = 128,
    bounds: tuple[float, float] = (-1.6, 1.6),
    view: str = "front",
    surface_only: bool = False,
) -> Tensor:
    """Render splats with front-to-back optical-thickness compositing."""
    camera = CAMERAS[view]
    centers = splats.centers
    colors = splats.colors
    tau = splats.tau
    sigma = splats.sigma

    if surface_only:
        mask = splats.is_surface
        centers = centers[mask]
        colors = colors[mask]
        tau = tau[mask]
        sigma = sigma[mask]
        gate = torch.ones_like(tau)
    elif gate is None:
        gate = torch.ones_like(tau)
    else:
        gate = gate.to(dtype=tau.dtype)

    depth = camera.depth_sign * centers[:, camera.depth_axis]
    order = torch.argsort(depth, descending=False)
    centers = centers[order]
    colors = colors[order]
    tau = tau[order]
    sigma = sigma[order]
    gate = gate[order]

    lo, hi = bounds
    span = hi - lo
    coords = centers[:, list(camera.axes)]
    pixels = (coords - lo) / span * (image_size - 1)
    sigma_px = torch.clamp(sigma / span * image_size, min=0.8)

    image = torch.zeros((image_size, image_size, 3), dtype=torch.float64)
    transmittance = torch.ones((image_size, image_size), dtype=torch.float64)

    for i in range(centers.shape[0]):
        cx = float(pixels[i, 0])
        cy = float(pixels[i, 1])
        spx = float(sigma_px[i])
        radius = max(1, int(np.ceil(3.0 * spx)))
        x0 = max(0, int(np.floor(cx - radius)))
        x1 = min(image_size, int(np.ceil(cx + radius + 1)))
        y0 = max(0, int(np.floor(cy - radius)))
        y1 = min(image_size, int(np.ceil(cy + radius + 1)))
        if x0 >= x1 or y0 >= y1:
            continue

        xs = torch.arange(x0, x1, dtype=torch.float64)
        ys = torch.arange(y0, y1, dtype=torch.float64)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        weight = torch.exp(-0.5 * (((xx - cx) / spx) ** 2 + ((yy - cy) / spx) ** 2))
        optical = torch.clamp(tau[i] * gate[i], min=0.0) * weight
        alpha = 1.0 - torch.exp(-optical)
        patch_trans = transmittance[y0:y1, x0:x1]
        image[y0:y1, x0:x1, :] += patch_trans[..., None] * alpha[..., None] * colors[i]
        transmittance[y0:y1, x0:x1] = patch_trans * torch.exp(-optical)

    return image.clamp(0.0, 1.0)


def save_image(path: str | Path, image: Tensor) -> None:
    arr = (image.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def load_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0

