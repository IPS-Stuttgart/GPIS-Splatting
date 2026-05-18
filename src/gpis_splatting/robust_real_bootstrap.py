from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gpis_splatting.real_bootstrap import SAMPLE_TYPE_IDS, POINT_SOURCES, load_ply_point_cloud, resolve_point_source, splats_from_point_cloud
from gpis_splatting.real_scene import load_prepared_scene
from gpis_splatting.serialization import write_json

if TYPE_CHECKING:
    from gpis_splatting.splats import SplatCloud


@dataclass(frozen=True)
class RobustSparsePointCloud:
    points: np.ndarray
    colors: np.ndarray
    errors: np.ndarray | None = None
    point_ids: np.ndarray | None = None
    track_image_ids: tuple[tuple[int, ...], ...] | None = None


def bootstrap_robust_real_gpis(
    *,
    scene_dir: str | Path,
    point_source: str = "auto",
    point_path: str | Path | None = None,
    output_prefix: str = "real_robust",
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
    max_point_error: float | None = None,
    point_error_percentile: float | None = 95.0,
    use_point_error_noise: bool = True,
    max_surface_noise_multiplier: float = 4.0,
    max_views_per_point: int = 3,
    visibility_distance_factor: float = 1.25,
) -> dict[str, Any]:
    scene_root = Path(scene_dir)
    scene_meta, frames, splits = load_prepared_scene(scene_root)
    cloud, resolved_source, resolved_path = load_robust_sparse_point_cloud(
        scene_root=scene_root,
        scene_meta=scene_meta,
        point_source=point_source,
        point_path=point_path,
    )
    cloud, filter_report = filter_point_cloud_by_error(cloud, max_error=max_point_error, error_percentile=point_error_percentile)
    cloud = subsample_point_cloud(cloud, max_points=max_points, seed=seed)
    train_frames = [frames[index] for index in splits.get("train", [])]
    if not train_frames:
        raise ValueError("Prepared scene has no training frames.")

    samples = build_multiview_ray_bootstrap_samples(
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
        use_point_error_noise=use_point_error_noise,
        max_surface_noise_multiplier=max_surface_noise_multiplier,
        max_views_per_point=max_views_per_point,
        visibility_distance_factor=visibility_distance_factor,
    )
    splats = splats_from_point_cloud(cloud, tau=splat_tau, sigma=splat_sigma)

    samples_path = scene_root / f"{output_prefix}_samples.npz"
    splats_path = scene_root / f"{output_prefix}_splats.npz"
    config_path = scene_root / f"{output_prefix}_gpis_config.json"
    report_path = scene_root / f"{output_prefix}_bootstrap_report.json"
    np.savez_compressed(samples_path, **samples)
    from gpis_splatting.splats import save_splats

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
        "max_point_error": max_point_error,
        "point_error_percentile": point_error_percentile,
        "use_point_error_noise": use_point_error_noise,
        "max_surface_noise_multiplier": max_surface_noise_multiplier,
        "max_views_per_point": max_views_per_point,
        "visibility_distance_factor": visibility_distance_factor,
    }
    report = {
        **config,
        "point_filter": filter_report,
        "surface_point_count": int(cloud.points.shape[0]),
        "sample_count": int(samples["points"].shape[0]),
        "free_space_sample_count": int((samples["sample_type"] == SAMPLE_TYPE_IDS["free_space"]).sum()),
        "behind_surface_sample_count": int((samples["sample_type"] == SAMPLE_TYPE_IDS["behind_surface"]).sum()),
        "mean_ray_view_count": float(samples["ray_view_count"].mean()) if samples["ray_view_count"].size else 0.0,
        "max_ray_view_count": int(samples["ray_view_count"].max()) if samples["ray_view_count"].size else 0,
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


def load_robust_sparse_point_cloud(
    *,
    scene_root: Path,
    scene_meta: dict[str, Any],
    point_source: str = "auto",
    point_path: str | Path | None = None,
) -> tuple[RobustSparsePointCloud, str, Path]:
    if point_source not in POINT_SOURCES:
        raise ValueError(f"Unsupported point source {point_source!r}. Expected one of {', '.join(POINT_SOURCES)}.")
    resolved_source, resolved_path = resolve_point_source(scene_root=scene_root, scene_meta=scene_meta, point_source=point_source, point_path=point_path)
    if resolved_source == "colmap":
        return load_colmap_points3d_with_tracks(resolved_path), resolved_source, resolved_path
    if resolved_source == "ply":
        cloud = load_ply_point_cloud(resolved_path)
        return RobustSparsePointCloud(points=cloud.points, colors=cloud.colors, errors=cloud.errors, point_ids=cloud.point_ids), resolved_source, resolved_path
    raise ValueError(f"Unsupported resolved point source {resolved_source!r}.")


def load_colmap_points3d_with_tracks(path: str | Path) -> RobustSparsePointCloud:
    points = []
    colors = []
    errors = []
    point_ids = []
    track_image_ids = []
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
        track_image_ids.append(tuple(int(value) for value in parts[8::2]))
    if not points:
        raise ValueError(f"No COLMAP points were found in {path}.")
    return RobustSparsePointCloud(
        points=np.asarray(points, dtype=np.float64),
        colors=np.asarray(colors, dtype=np.float64),
        errors=np.asarray(errors, dtype=np.float64),
        point_ids=np.asarray(point_ids, dtype=np.int64),
        track_image_ids=tuple(track_image_ids),
    )


def filter_point_cloud_by_error(
    cloud: RobustSparsePointCloud,
    *,
    max_error: float | None,
    error_percentile: float | None,
) -> tuple[RobustSparsePointCloud, dict[str, Any]]:
    if cloud.errors is None:
        return cloud, {
            "enabled": False,
            "reason": "point cloud has no per-point reconstruction errors",
            "input_count": int(cloud.points.shape[0]),
            "kept_count": int(cloud.points.shape[0]),
            "dropped_count": 0,
            "threshold": None,
        }
    if max_error is not None and max_error <= 0.0:
        raise ValueError("max_point_error must be positive when provided.")
    if error_percentile is not None and not 0.0 < error_percentile <= 100.0:
        raise ValueError("point_error_percentile must be in (0, 100].")

    errors = np.asarray(cloud.errors, dtype=np.float64)
    finite = np.isfinite(errors)
    thresholds = []
    if max_error is not None:
        thresholds.append(float(max_error))
    if error_percentile is not None and error_percentile < 100.0 and finite.any():
        thresholds.append(float(np.percentile(errors[finite], error_percentile)))
    if not thresholds:
        return cloud, {
            "enabled": False,
            "reason": "no error threshold configured",
            "input_count": int(cloud.points.shape[0]),
            "kept_count": int(cloud.points.shape[0]),
            "dropped_count": 0,
            "threshold": None,
        }

    threshold = min(thresholds)
    keep = finite & (errors <= threshold)
    kept_count = int(keep.sum())
    if kept_count == 0:
        raise ValueError("Point-error filtering removed all sparse points. Increase --max-point-error or --point-error-percentile.")
    filtered = select_point_cloud(cloud, np.flatnonzero(keep))
    return filtered, {
        "enabled": True,
        "input_count": int(cloud.points.shape[0]),
        "kept_count": kept_count,
        "dropped_count": int(cloud.points.shape[0] - kept_count),
        "threshold": float(threshold),
        "max_point_error": max_error,
        "point_error_percentile": error_percentile,
        "mean_error_before": float(errors[finite].mean()) if finite.any() else None,
        "mean_error_after": float(filtered.errors.mean()) if filtered.errors is not None and filtered.errors.size else None,
    }


def subsample_point_cloud(cloud: RobustSparsePointCloud, *, max_points: int | None, seed: int) -> RobustSparsePointCloud:
    if max_points is None or cloud.points.shape[0] <= max_points:
        return cloud
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(cloud.points.shape[0], size=max_points, replace=False))
    return select_point_cloud(cloud, indices)


def select_point_cloud(cloud: RobustSparsePointCloud, indices: np.ndarray) -> RobustSparsePointCloud:
    return RobustSparsePointCloud(
        points=cloud.points[indices],
        colors=cloud.colors[indices],
        errors=cloud.errors[indices] if cloud.errors is not None else None,
        point_ids=cloud.point_ids[indices] if cloud.point_ids is not None else None,
        track_image_ids=tuple(cloud.track_image_ids[int(index)] for index in indices) if cloud.track_image_ids is not None else None,
    )


def build_multiview_ray_bootstrap_samples(
    cloud: RobustSparsePointCloud,
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
    use_point_error_noise: bool = True,
    max_surface_noise_multiplier: float = 4.0,
    max_views_per_point: int = 3,
    visibility_distance_factor: float = 1.25,
) -> dict[str, np.ndarray]:
    if free_space_samples_per_point < 0:
        raise ValueError("free_space_samples_per_point must be non-negative.")
    if not 0.0 < free_space_min_fraction <= free_space_max_fraction < 1.0:
        raise ValueError("free-space fractions must satisfy 0 < min <= max < 1.")
    if behind_surface_fraction <= 1.0:
        raise ValueError("behind_surface_fraction must be greater than 1.")
    if max_views_per_point < 1:
        raise ValueError("max_views_per_point must be positive.")
    if visibility_distance_factor < 1.0:
        raise ValueError("visibility_distance_factor must be at least 1.")
    if max_surface_noise_multiplier < 1.0:
        raise ValueError("max_surface_noise_multiplier must be at least 1.")

    camera_centers = _camera_centers(train_frames)
    camera_frame_indices = _camera_frame_indices(train_frames)
    image_id_to_train_index = _image_id_to_train_index(train_frames)
    point_error_normalizer = _positive_error_median(cloud.errors)
    rows = []
    sdf = []
    noise = []
    sample_type = []
    source_index = []
    camera_index = []
    camera_frame_index = []
    ray_distance = []
    ray_view_rank = []
    ray_view_count = []
    track_length = []

    free_fractions = np.linspace(free_space_min_fraction, free_space_max_fraction, max(free_space_samples_per_point, 1))
    for point_index, point in enumerate(cloud.points):
        selected_cameras = _ray_camera_indices_for_point(
            point_index=point_index,
            point=point,
            camera_centers=camera_centers,
            image_id_to_train_index=image_id_to_train_index,
            track_image_ids=cloud.track_image_ids,
            max_views_per_point=max_views_per_point,
            visibility_distance_factor=visibility_distance_factor,
        )
        if not selected_cameras:
            continue

        first_camera_id, first_distance = selected_cameras[0]
        if first_distance <= 1e-9:
            continue
        point_track_length = _track_length(cloud.track_image_ids, point_index)
        rows.append(point)
        sdf.append(0.0)
        noise.append(
            _surface_noise_for_point(
                base_noise_std=surface_noise_std,
                errors=cloud.errors,
                point_index=point_index,
                error_normalizer=point_error_normalizer,
                use_point_error_noise=use_point_error_noise,
                max_surface_noise_multiplier=max_surface_noise_multiplier,
            )
        )
        sample_type.append(SAMPLE_TYPE_IDS["surface"])
        source_index.append(point_index)
        camera_index.append(first_camera_id)
        camera_frame_index.append(camera_frame_indices[first_camera_id])
        ray_distance.append(first_distance)
        ray_view_rank.append(0)
        ray_view_count.append(len(selected_cameras))
        track_length.append(point_track_length)

        for view_rank, (camera_id, distance) in enumerate(selected_cameras):
            camera = camera_centers[camera_id]
            vector = point - camera
            if distance <= 1e-9:
                continue
            for fraction in free_fractions[:free_space_samples_per_point]:
                sample = camera + vector * float(fraction)
                rows.append(sample)
                sdf.append(min(distance * (1.0 - float(fraction)), max_sample_distance))
                noise.append(free_space_noise_std)
                sample_type.append(SAMPLE_TYPE_IDS["free_space"])
                source_index.append(point_index)
                camera_index.append(camera_id)
                camera_frame_index.append(camera_frame_indices[camera_id])
                ray_distance.append(distance * float(fraction))
                ray_view_rank.append(view_rank)
                ray_view_count.append(len(selected_cameras))
                track_length.append(point_track_length)

            if add_behind_surface_samples:
                sample = camera + vector * behind_surface_fraction
                rows.append(sample)
                sdf.append(-min(distance * (behind_surface_fraction - 1.0), max_sample_distance))
                noise.append(behind_surface_noise_std)
                sample_type.append(SAMPLE_TYPE_IDS["behind_surface"])
                source_index.append(point_index)
                camera_index.append(camera_id)
                camera_frame_index.append(camera_frame_indices[camera_id])
                ray_distance.append(distance * behind_surface_fraction)
                ray_view_rank.append(view_rank)
                ray_view_count.append(len(selected_cameras))
                track_length.append(point_track_length)

    if not rows:
        raise ValueError("No bootstrap samples were generated.")
    return {
        "points": np.asarray(rows, dtype=np.float64),
        "sdf": np.asarray(sdf, dtype=np.float64),
        "observation_noise_std": np.asarray(noise, dtype=np.float64),
        "sample_type": np.asarray(sample_type, dtype=np.int64),
        "source_point_index": np.asarray(source_index, dtype=np.int64),
        "camera_index": np.asarray(camera_index, dtype=np.int64),
        "camera_frame_index": np.asarray(camera_frame_index, dtype=np.int64),
        "ray_distance": np.asarray(ray_distance, dtype=np.float64),
        "ray_view_rank": np.asarray(ray_view_rank, dtype=np.int64),
        "ray_view_count": np.asarray(ray_view_count, dtype=np.int64),
        "track_length": np.asarray(track_length, dtype=np.int64),
        "sample_type_names": np.asarray(["surface", "free_space", "behind_surface"]),
    }


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


def _camera_frame_indices(frames: list[dict[str, Any]]) -> np.ndarray:
    indices = []
    for local_index, frame in enumerate(frames):
        try:
            indices.append(int(frame.get("index", local_index)))
        except (TypeError, ValueError):
            indices.append(local_index)
    return np.asarray(indices, dtype=np.int64)


def _image_id_to_train_index(frames: list[dict[str, Any]]) -> dict[int, int]:
    mapping = {}
    for local_index, frame in enumerate(frames):
        try:
            mapping[int(frame.get("image_id"))] = local_index
        except (TypeError, ValueError):
            continue
    return mapping


def _ray_camera_indices_for_point(
    *,
    point_index: int,
    point: np.ndarray,
    camera_centers: np.ndarray,
    image_id_to_train_index: dict[int, int],
    track_image_ids: tuple[tuple[int, ...], ...] | None,
    max_views_per_point: int,
    visibility_distance_factor: float,
) -> list[tuple[int, float]]:
    distances = np.linalg.norm(camera_centers - point[None, :], axis=1)
    track_candidates: list[int] = []
    if track_image_ids is not None:
        track_candidates = [image_id_to_train_index[image_id] for image_id in track_image_ids[point_index] if image_id in image_id_to_train_index]
    if track_candidates:
        unique = sorted(set(track_candidates), key=lambda camera_id: distances[camera_id])
        selected = unique[:max_views_per_point]
    else:
        order = np.argsort(distances)
        nearest = float(distances[order[0]])
        if visibility_distance_factor > 1.0:
            visible = [int(index) for index in order if distances[index] <= nearest * visibility_distance_factor]
        else:
            visible = [int(index) for index in order]
        selected = visible[:max_views_per_point]
    return [(int(camera_id), float(distances[camera_id])) for camera_id in selected]


def _positive_error_median(errors: np.ndarray | None) -> float | None:
    if errors is None:
        return None
    finite_positive = np.asarray(errors, dtype=np.float64)
    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0.0)]
    if finite_positive.size == 0:
        return None
    return float(np.median(finite_positive))


def _surface_noise_for_point(
    *,
    base_noise_std: float,
    errors: np.ndarray | None,
    point_index: int,
    error_normalizer: float | None,
    use_point_error_noise: bool,
    max_surface_noise_multiplier: float,
) -> float:
    if not use_point_error_noise or errors is None or error_normalizer is None:
        return float(base_noise_std)
    error = float(errors[point_index])
    if not np.isfinite(error) or error <= 0.0:
        return float(base_noise_std)
    multiplier = min(max(error / error_normalizer, 1.0), max_surface_noise_multiplier)
    return float(base_noise_std * multiplier)


def _track_length(track_image_ids: tuple[tuple[int, ...], ...] | None, point_index: int) -> int:
    if track_image_ids is None:
        return 0
    return len(track_image_ids[point_index])
