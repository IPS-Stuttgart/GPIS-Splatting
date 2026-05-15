from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from gpis_splatting.serialization import write_json
from gpis_splatting.splats import SplatCloud, save_splats

EPS = 1e-12


def finalize_gpis_aware_splat_export(
    *,
    gaussians_path: str | Path,
    output_splats_path: str | Path | None = None,
    output_status_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a geometry-preserving internal SplatCloud from GPIS-aware Gaussian arrays.

    ``initialize_gpis_splats`` writes a rich ``*_gaussians.npz`` file and a
    legacy renderer-oriented ``*_splats.npz`` file. This finalizer converts the
    rich Gaussian arrays into an internal ``SplatCloud`` that keeps the geometry
    fields needed downstream: normals, anisotropic scales, rotations,
    covariances, and confidence. The scalar renderer fallback uses the tangent
    footprint, while alpha/opacity is converted to Beer-Lambert optical
    thickness for the CPU renderer.
    """
    source = Path(gaussians_path)
    output_splats = Path(output_splats_path) if output_splats_path is not None else source.with_name(source.name.replace("_gaussians.npz", "_splats.npz"))
    if output_splats == source:
        raise ValueError("output_splats_path must differ from gaussians_path.")

    with np.load(source, allow_pickle=False) as data:
        centers = _array(data, "centers", ndim=2, columns=3)
        colors = _array(data, "colors", ndim=2, columns=3)
        scales = _array(data, "scales", ndim=2, columns=3)
        rotations = _array(data, "rotations", ndim=2, columns=4)
        normals = _array(data, "normals", ndim=2, columns=3)
        confidence = _array(data, "confidence", ndim=1)
        opacity = _array(data, "opacity", ndim=1) if "opacity" in data.files else _array(data, "tau", ndim=1)

    count = int(centers.shape[0])
    for name, values in {
        "colors": colors,
        "scales": scales,
        "rotations": rotations,
        "normals": normals,
        "confidence": confidence,
        "opacity": opacity,
    }.items():
        if int(values.shape[0]) != count:
            raise ValueError(f"{name} has {values.shape[0]} rows, expected {count}.")

    covariances = covariances_from_scales_and_rotations(scales, rotations)
    tangent_sigma = np.mean(scales[:, 1:3], axis=1)
    tau = alpha_to_optical_thickness(opacity)
    splats = SplatCloud(
        centers=torch.from_numpy(centers).to(dtype=torch.float64),
        colors=torch.from_numpy(np.clip(colors, 0.0, 1.0)).to(dtype=torch.float64),
        tau=torch.from_numpy(tau).to(dtype=torch.float64),
        sigma=torch.from_numpy(tangent_sigma).to(dtype=torch.float64),
        is_surface=torch.ones((count,), dtype=torch.bool),
        normals=torch.from_numpy(normals).to(dtype=torch.float64),
        scales=torch.from_numpy(scales).to(dtype=torch.float64),
        rotations=torch.from_numpy(normalize_quaternions(rotations)).to(dtype=torch.float64),
        covariances=torch.from_numpy(covariances).to(dtype=torch.float64),
        confidence=torch.from_numpy(np.clip(confidence, 0.0, 1.0)).to(dtype=torch.float64),
    )
    output_splats.parent.mkdir(parents=True, exist_ok=True)
    save_splats(str(output_splats), splats)

    status = {
        "schema_version": 1,
        "source_gaussians_path": str(source),
        "output_splats_path": str(output_splats),
        "splat_count": count,
        "preserved_fields": ["normals", "scales", "rotations", "covariances", "confidence"],
        "sigma_convention": "mean_tangent_scale",
        "tau_convention": "beer_lambert_optical_thickness_from_alpha",
        "normal_axis_variance_mean": float(np.mean(normal_axis_variance(normals, covariances))) if count else None,
    }
    status_path = Path(output_status_path) if output_status_path is not None else output_splats.with_name(f"{output_splats.stem}_geometry_status.json")
    write_json(status_path, status)
    return {"splats_path": output_splats, "status_path": status_path, "status": status}


def _array(data: np.lib.npyio.NpzFile, name: str, *, ndim: int, columns: int | None = None) -> np.ndarray:
    if name not in data.files:
        raise ValueError(f"{data.filename} is missing required array {name!r}.")
    values = np.asarray(data[name], dtype=np.float64)
    if values.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {values.ndim}.")
    if columns is not None and values.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")
    return values


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), EPS)


def rotation_matrices_from_quaternions(quaternions: np.ndarray) -> np.ndarray:
    q = normalize_quaternions(quaternions)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrices = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def covariances_from_scales_and_rotations(scales: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    axes = np.asarray(scales, dtype=np.float64)
    if axes.ndim != 2 or axes.shape[1] != 3:
        raise ValueError("scales must have shape (N, 3).")
    matrices = rotation_matrices_from_quaternions(rotations)
    if matrices.shape[0] != axes.shape[0]:
        raise ValueError("scales and rotations must contain the same number of rows.")
    variances = np.square(np.clip(axes, EPS, None))
    return (matrices * variances[:, None, :]) @ np.transpose(matrices, (0, 2, 1))


def normal_axis_variance(normals: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    unit_normals = np.asarray(normals, dtype=np.float64)
    unit_normals = unit_normals / np.maximum(np.linalg.norm(unit_normals, axis=1, keepdims=True), EPS)
    return np.einsum("ni,nij,nj->n", unit_normals, covariances, unit_normals)


def alpha_to_optical_thickness(alpha: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, 1.0 - 1e-6)
    return -np.log1p(-clipped)
