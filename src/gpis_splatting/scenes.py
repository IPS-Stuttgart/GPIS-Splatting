from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class SceneSpec:
    name: str
    bounds: tuple[float, float]
    color: tuple[float, float, float]


SCENES: dict[str, SceneSpec] = {
    "sphere": SceneSpec("sphere", (-1.35, 1.35), (0.35, 0.62, 1.0)),
    "torus": SceneSpec("torus", (-1.6, 1.6), (1.0, 0.55, 0.25)),
    "two_objects": SceneSpec("two_objects", (-1.6, 1.6), (0.4, 0.9, 0.55)),
    "non_star_convex": SceneSpec("non_star_convex", (-1.6, 1.6), (0.92, 0.45, 0.78)),
}


def available_shapes() -> tuple[str, ...]:
    return tuple(SCENES)


def sdf(points: Tensor, shape: str) -> Tensor:
    """Signed distance fields; negative values are inside the shape."""
    if shape == "sphere":
        return torch.linalg.norm(points, dim=-1) - 0.72

    if shape == "torus":
        major_radius = 0.78
        minor_radius = 0.28
        qx = torch.linalg.norm(points[..., :2], dim=-1) - major_radius
        qy = points[..., 2]
        return torch.linalg.norm(torch.stack((qx, qy), dim=-1), dim=-1) - minor_radius

    if shape == "two_objects":
        c1 = torch.tensor([-0.52, 0.0, 0.0], dtype=points.dtype, device=points.device)
        c2 = torch.tensor([0.52, 0.0, 0.0], dtype=points.dtype, device=points.device)
        s1 = torch.linalg.norm(points - c1, dim=-1) - 0.47
        s2 = torch.linalg.norm(points - c2, dim=-1) - 0.42
        return torch.minimum(s1, s2)

    if shape == "non_star_convex":
        outer = torch.linalg.norm(points, dim=-1) - 0.82
        cutter_center = torch.tensor([0.38, 0.0, 0.0], dtype=points.dtype, device=points.device)
        cutter = torch.linalg.norm(points - cutter_center, dim=-1) - 0.62
        notch = -cutter
        return torch.maximum(outer, notch)

    raise ValueError(f"Unknown shape '{shape}'. Expected one of {', '.join(available_shapes())}.")


def sdf_normals(points: Tensor, shape: str, eps: float = 1e-3) -> Tensor:
    """Central-difference SDF normals for diagnostics and optional supervision artifacts."""
    basis = torch.eye(3, dtype=points.dtype, device=points.device)
    grads = []
    for axis in range(3):
        delta = eps * basis[axis]
        grads.append((sdf(points + delta, shape) - sdf(points - delta, shape)) / (2.0 * eps))
    grad = torch.stack(grads, dim=-1)
    return grad / torch.clamp(torch.linalg.norm(grad, dim=-1, keepdim=True), min=1e-8)


def make_grid(
    bounds: tuple[float, float],
    grid_size: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> Tensor:
    lo, hi = bounds
    axis = torch.linspace(lo, hi, grid_size, dtype=dtype, device=device)
    xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)


def sample_scene(
    shape: str,
    num_points: int,
    seed: int,
    noise_std: float,
    near_surface_fraction: float = 0.72,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor]:
    """Sample noisy SDF observations with a bias toward the zero-level set."""
    if shape not in SCENES:
        raise ValueError(f"Unknown shape '{shape}'. Expected one of {', '.join(available_shapes())}.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    lo, hi = SCENES[shape].bounds

    n_surface = int(round(num_points * near_surface_fraction))
    n_volume = num_points - n_surface

    candidates = torch.empty(max(5000, num_points * 80), 3, dtype=dtype).uniform_(lo, hi, generator=generator)
    candidate_sdf = sdf(candidates, shape)
    near_count = min(n_surface, candidates.shape[0])
    near_idx = torch.topk(candidate_sdf.abs(), k=near_count, largest=False).indices
    near_points = candidates[near_idx]
    normals = sdf_normals(near_points, shape)
    jitter = torch.randn((near_points.shape[0], 1), dtype=dtype, generator=generator) * 0.035
    near_points = near_points + normals * jitter

    volume_points = torch.empty(n_volume, 3, dtype=dtype).uniform_(lo, hi, generator=generator)
    points = torch.cat((near_points, volume_points), dim=0)
    perm = torch.randperm(points.shape[0], generator=generator)
    points = points[perm]

    true_values = sdf(points, shape)
    noise = torch.randn(true_values.shape, dtype=dtype, generator=generator) * noise_std
    observed = true_values + noise
    normals = sdf_normals(points, shape)

    return {
        "points": points,
        "observed_sdf": observed,
        "true_sdf": true_values,
        "normals": normals,
    }
