from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gpis_splatting.real_scene import load_prepared_scene
from gpis_splatting.serialization import write_json
from gpis_splatting.splats import SplatCloud, save_splats

POINT_SOURCES = ("auto", "colmap", "ply")
SAMPLE_TYPE_IDS = {
    "surface": 0,
    "free_space": 1,
    "behind_surface": 2,
}


@dataclass(frozen=True)
class SparsePointCloud:
    points: np.ndarray
    colors: np.ndarray
    errors: np.ndarray | None = None
    point_ids: np.ndarray | None = None


def bootstrap_real_gpis(
    *,
    scene_dir: str | Path,
    point_source: str = "auto",
    point_path: str | Path | None = None,
    output_prefix: str = "real",
    max_points: int | None = 5000,
    seed: int = 7,
    free_space_samples_per_point: int = 2,
    free_space_min_fraction: float = 0.2,
    free_space_max_fraction: float = 0.85,
    add_behind_surface_samples: bool = True,
    behind_surface_fraction: float = 1.08,
    max_sample_distance: float = 0.35,
    surface_noise_std: float = 0.03,
    free_space_noise_std: float = 0.08,
    behind_surface_noise_std: float = 0.12,
    splat_tau: float = 0.45,
    splat_sigma: float = 0.025,
) -> dict[str, Any]:
    scene_root = Path(scene_dir)
    scene_meta, frames, splits = load_prepared_scene(scene_root)
    cloud, resolved_source, resolved_path = load_sparse_point_cloud(
        scene_root=scene_root,
        scene_meta=scene_meta,
        point_source=point_source,
        point_path=point_path,
    )
    cloud = subsample_point_cloud(cloud, max_points=max_points, seed=seed)
    train_frames = [frames[index] for index in splits.get("train", [])]
    if not train_frames:
        raise ValueError("Prepared scene has no training frames.")

    samples = build_ray_bootstrap_samples(
        cloud,
        train_frames=train_frames,
        free_space_samples_per_point=free_space_samples_per_point,
        free_space_min_fraction=free_space_min_fraction,
        free_space_max_fraction=free_space_max_fraction,
        add_behind_surface_samples=add_behind_surface_samples,
        behind_surface_fraction=behind_surface_fraction,
        max_sample_distance=max_sample_distance,
        surface_noise_std=surface_noise_std,
        free_space_noise_std=free_space_noise_std,
        behind_surface_noise_std=behind_surface_noise_std,
    )
    splats = splats_from_point_cloud(cloud, tau=splat_tau, sigma=splat_sigma)

    samples_path = scene_root / f"{output_prefix}_samples.npz"
    splats_path = scene_root / f"{output_prefix}_splats.npz"
    config_path = scene_root / f"{output_prefix}_gpis_config.json"
    report_path = scene_root / f"{output_prefix}_bootstrap_report.json"
    np.savez_compressed(samples_path, **samples)
    save_splats(str(splats_path), splats)
    config = {
        "schema_version": 1,
        "scene": scene_meta["scene"],
        "point_source": resolved_source,
        "point_path": str(resolved_path),
        "output_prefix": output_prefix,
        "max_points": max_points,
        "seed": seed,
        "free_space_samples_per_point": free_space_samples_per_point,
        "free_space_min_fraction": free_space_min_fraction,
        "free_space_max_fraction": free_space_max_fraction,
        "add_behind_surface_samples": add_behind_surface_samples,
        "behind_surface_fraction": behind_surface_fraction,
        "max_sample_distance": max_sample_distance,
        "surface_noise_std": surface_noise_std,
        "free_space_noise_std": free_space_noise_std,
        "behind_surface_noise_std": behind_surface_noise_std,
        "splat_tau": splat_tau,
        "splat_sigma": splat_sigma,
    }
    report = {
        **config,
        "surface_point_count": int(cloud.points.shape[0]),
        "sample_count": int(samples["points"].shape[0]),
        "free_space_sample_count": int((samples["sample_type"] == SAMPLE_TYPE_IDS["free_space"]).sum()),
        "behind_surface_sample_count": int((samples["sample_type"] == SAMPLE_TYPE_IDS["behind_surface"]).sum()),
        "splat_count": int(cloud.points.shape[0]),
        "samples_path": str(samples_path),
        "splats_path": str(splats_path),
    }
    write_json(config_path, config)
    write_json(report_path, report)
    return {
        "samples_path": samples_path,
        "splats_path": splats_path,
        "config_path": config_path,
        "report_path": report_path,
        "report": report,
    }


def load_sparse_point_cloud(
    *,
    scene_root: Path,
    scene_meta: dict[str, Any],
    point_source: str = "auto",
    point_path: str | Path | None = None,
) -> tuple[SparsePointCloud, str, Path]:
    if point_source not in POINT_SOURCES:
        raise ValueError(f"Unsupported point source {point_source!r}. Expected one of {', '.join(POINT_SOURCES)}.")
    resolved_source, resolved_path = resolve_point_source(scene_root=scene_root, scene_meta=scene_meta, point_source=point_source, point_path=point_path)
    if resolved_source == "colmap":
        return load_colmap_points3d(resolved_path), resolved_source, resolved_path
    if resolved_source == "ply":
        return load_ascii_ply_point_cloud(resolved_path), resolved_source, resolved_path
    raise ValueError(f"Unsupported resolved point source {resolved_source!r}.")


def resolve_point_source(
    *,
    scene_root: Path,
    scene_meta: dict[str, Any],
    point_source: str,
    point_path: str | Path | None,
) -> tuple[str, Path]:
    if point_path is not None:
        path = Path(point_path)
        if not path.is_absolute():
            path = scene_root / path
        if not path.exists():
            raise FileNotFoundError(f"Missing point source file: {path}")
        if point_source == "auto":
            return _source_from_suffix(path), path
        return point_source, path

    source_dir = Path(scene_meta.get("source_dir", scene_root))
    tanks_temples_meta = scene_meta.get("tanks_temples") or {}
    tanks_temples_reconstruction = tanks_temples_meta.get("reconstruction_path")
    candidates = []
    if tanks_temples_reconstruction:
        candidates.append((Path(tanks_temples_reconstruction), "ply"))
    candidates.extend([
        (source_dir / "sparse" / "0" / "points3D.txt", "colmap"),
        (source_dir / "sparse" / "points3D.txt", "colmap"),
        (source_dir / "points3D.txt", "colmap"),
        (scene_root / "points3D.txt", "colmap"),
        (source_dir / "points3D.ply", "ply"),
        (source_dir / "sparse" / "0" / "points3D.ply", "ply"),
        (source_dir / "point_cloud.ply", "ply"),
        (scene_root / "point_cloud.ply", "ply"),
    ])
    for candidate, candidate_source in candidates:
        if candidate.exists() and point_source in {"auto", candidate_source}:
            return candidate_source, candidate
    raise FileNotFoundError("Could not find points3D.txt or .ply point cloud. Pass --point-path explicitly.")


def load_colmap_points3d(path: str | Path) -> SparsePointCloud:
    points = []
    colors = []
    errors = []
    point_ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        point_ids.append(int(parts[0]))
        points.append([float(value) for value in parts[1:4]])
        colors.append([int(value) / 255.0 for value in parts[4:7]])
        errors.append(float(parts[7]))
    if not points:
        raise ValueError(f"No COLMAP points were found in {path}.")
    return SparsePointCloud(
        points=np.asarray(points, dtype=np.float64),
        colors=np.asarray(colors, dtype=np.float64),
        errors=np.asarray(errors, dtype=np.float64),
        point_ids=np.asarray(point_ids, dtype=np.int64),
    )


def load_ascii_ply_point_cloud(path: str | Path) -> SparsePointCloud:
    ply_path = Path(path)
    lines = ply_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{ply_path} is not an ASCII PLY file.")
    vertex_count: int | None = None
    properties: list[str] = []
    header_end = None
    in_vertex = False
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("format") and "ascii" not in stripped:
            raise ValueError("Only ASCII PLY point clouds are supported.")
        if stripped.startswith("element vertex"):
            vertex_count = int(stripped.split()[-1])
            in_vertex = True
            continue
        if stripped.startswith("element ") and not stripped.startswith("element vertex"):
            in_vertex = False
        if in_vertex and stripped.startswith("property"):
            properties.append(stripped.split()[-1])
        if stripped == "end_header":
            header_end = line_index + 1
            break
    if vertex_count is None or header_end is None:
        raise ValueError(f"{ply_path} is missing a vertex header.")

    rows = [line.split() for line in lines[header_end : header_end + vertex_count]]
    prop_index = {name: index for index, name in enumerate(properties)}
    for required in ("x", "y", "z"):
        if required not in prop_index:
            raise ValueError(f"{ply_path} is missing vertex property {required!r}.")
    points = np.asarray([[float(row[prop_index["x"]]), float(row[prop_index["y"]]), float(row[prop_index["z"]])] for row in rows], dtype=np.float64)
    if {"red", "green", "blue"}.issubset(prop_index):
        colors = np.asarray([[float(row[prop_index["red"]]), float(row[prop_index["green"]]), float(row[prop_index["blue"]])] for row in rows], dtype=np.float64)
        colors = np.clip(colors / 255.0, 0.0, 1.0)
    else:
        colors = np.full((points.shape[0], 3), 0.7, dtype=np.float64)
    return SparsePointCloud(points=points, colors=colors)


def subsample_point_cloud(cloud: SparsePointCloud, *, max_points: int | None, seed: int) -> SparsePointCloud:
    if max_points is None or cloud.points.shape[0] <= max_points:
        return cloud
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(cloud.points.shape[0], size=max_points, replace=False))
    return SparsePointCloud(
        points=cloud.points[indices],
        colors=cloud.colors[indices],
        errors=cloud.errors[indices] if cloud.errors is not None else None,
        point_ids=cloud.point_ids[indices] if cloud.point_ids is not None else None,
    )


def build_ray_bootstrap_samples(
    cloud: SparsePointCloud,
    *,
    train_frames: list[dict[str, Any]],
    free_space_samples_per_point: int,
    free_space_min_fraction: float,
    free_space_max_fraction: float,
    add_behind_surface_samples: bool,
    behind_surface_fraction: float,
    max_sample_distance: float,
    surface_noise_std: float,
    free_space_noise_std: float,
    behind_surface_noise_std: float,
) -> dict[str, np.ndarray]:
    if free_space_samples_per_point < 0:
        raise ValueError("free_space_samples_per_point must be non-negative.")
    if not 0.0 < free_space_min_fraction <= free_space_max_fraction < 1.0:
        raise ValueError("free-space fractions must satisfy 0 < min <= max < 1.")
    if behind_surface_fraction <= 1.0:
        raise ValueError("behind_surface_fraction must be greater than 1.")

    camera_centers = _camera_centers(train_frames)
    nearest_camera_index = _nearest_camera_indices(cloud.points, camera_centers)
    rows = []
    sdf = []
    noise = []
    sample_type = []
    source_index = []
    camera_index = []
    ray_distance = []

    free_fractions = np.linspace(free_space_min_fraction, free_space_max_fraction, max(free_space_samples_per_point, 1))
    for point_index, point in enumerate(cloud.points):
        camera_id = int(nearest_camera_index[point_index])
        camera = camera_centers[camera_id]
        vector = point - camera
        distance = float(np.linalg.norm(vector))
        if distance <= 1e-9:
            continue

        rows.append(point)
        sdf.append(0.0)
        noise.append(surface_noise_std)
        sample_type.append(SAMPLE_TYPE_IDS["surface"])
        source_index.append(point_index)
        camera_index.append(camera_id)
        ray_distance.append(distance)

        for fraction in free_fractions[:free_space_samples_per_point]:
            sample = camera + vector * float(fraction)
            rows.append(sample)
            sdf.append(min(distance * (1.0 - float(fraction)), max_sample_distance))
            noise.append(free_space_noise_std)
            sample_type.append(SAMPLE_TYPE_IDS["free_space"])
            source_index.append(point_index)
            camera_index.append(camera_id)
            ray_distance.append(distance * float(fraction))

        if add_behind_surface_samples:
            sample = camera + vector * behind_surface_fraction
            rows.append(sample)
            sdf.append(-min(distance * (behind_surface_fraction - 1.0), max_sample_distance))
            noise.append(behind_surface_noise_std)
            sample_type.append(SAMPLE_TYPE_IDS["behind_surface"])
            source_index.append(point_index)
            camera_index.append(camera_id)
            ray_distance.append(distance * behind_surface_fraction)

    if not rows:
        raise ValueError("No bootstrap samples were generated.")
    return {
        "points": np.asarray(rows, dtype=np.float64),
        "sdf": np.asarray(sdf, dtype=np.float64),
        "observation_noise_std": np.asarray(noise, dtype=np.float64),
        "sample_type": np.asarray(sample_type, dtype=np.int64),
        "source_point_index": np.asarray(source_index, dtype=np.int64),
        "camera_index": np.asarray(camera_index, dtype=np.int64),
        "ray_distance": np.asarray(ray_distance, dtype=np.float64),
        "sample_type_names": np.asarray(["surface", "free_space", "behind_surface"]),
    }


def splats_from_point_cloud(cloud: SparsePointCloud, *, tau: float, sigma: float) -> SplatCloud:
    centers = torch.from_numpy(cloud.points).to(dtype=torch.float64)
    colors = torch.from_numpy(np.clip(cloud.colors, 0.0, 1.0)).to(dtype=torch.float64)
    return SplatCloud(
        centers=centers,
        colors=colors,
        tau=torch.full((centers.shape[0],), tau, dtype=torch.float64),
        sigma=torch.full((centers.shape[0],), sigma, dtype=torch.float64),
        is_surface=torch.ones((centers.shape[0],), dtype=torch.bool),
    )


def _source_from_suffix(path: Path) -> str:
    if path.suffix.lower() == ".ply":
        return "ply"
    if path.name.lower() == "points3d.txt":
        return "colmap"
    raise ValueError(f"Could not infer point source from {path}. Pass --point-source explicitly.")


def _camera_centers(frames: list[dict[str, Any]]) -> np.ndarray:
    centers = []
    for frame in frames:
        camera_to_world = frame.get("camera_to_world")
        if camera_to_world is None:
            continue
        centers.append(np.asarray(camera_to_world, dtype=np.float64)[:3, 3])
    if not centers:
        raise ValueError("Training frames do not contain camera_to_world matrices.")
    return np.stack(centers)


def _nearest_camera_indices(points: np.ndarray, camera_centers: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(points[:, None, :] - camera_centers[None, :, :], axis=-1)
    return np.argmin(distances, axis=1)
