from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from gpis_splatting.real_scene import load_prepared_scene
from gpis_splatting.serialization import write_json

MAP_SUFFIXES = (".npy", ".npz", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
PROJECTION_CONVENTIONS = ("auto", "opencv", "opengl")
NORMAL_SPACES = ("camera", "world")

SAMPLE_TYPE_IDS = {
    "surface": 0,
    "free_space": 1,
    "behind_surface": 2,
    "normal_positive": 3,
    "normal_negative": 4,
    "depth_surface": 5,
    "depth_free_space": 6,
    "depth_normal_positive": 7,
    "depth_normal_negative": 8,
}
SAMPLE_TYPE_NAMES = tuple(name for name, _id in sorted(SAMPLE_TYPE_IDS.items(), key=lambda item: item[1]))


def augment_samples_with_depth_normal_confidence(
    *,
    scene_dir: str | Path,
    depth_dir: str | Path,
    depth_confidence_dir: str | Path | None = None,
    normal_dir: str | Path | None = None,
    normal_confidence_dir: str | Path | None = None,
    base_samples_path: str | Path | None = None,
    include_base_samples: bool = True,
    output_samples_path: str | Path | None = None,
    report_path: str | Path | None = None,
    split: str = "train",
    max_frames: int | None = None,
    max_pixels_per_frame: int | None = 2048,
    pixel_stride: int = 1,
    seed: int = 29,
    projection_convention: str = "auto",
    depth_scale: float = 1.0,
    depth_min: float = 1e-4,
    depth_max: float | None = None,
    normal_space: str = "camera",
    add_free_space_samples: bool = True,
    free_space_samples_per_depth: int = 1,
    free_space_min_fraction: float = 0.25,
    free_space_max_fraction: float = 0.85,
    max_free_space_sdf: float = 0.35,
    add_normal_offset_samples: bool = True,
    normal_offset_distance: float = 0.04,
    surface_noise_min: float = 0.015,
    surface_noise_max: float = 0.12,
    free_space_noise_min: float = 0.04,
    free_space_noise_max: float = 0.18,
    normal_noise_min: float = 0.02,
    normal_noise_max: float = 0.14,
    default_depth_confidence: float = 1.0,
    default_normal_confidence: float = 0.7,
    min_depth_confidence: float = 0.0,
    min_normal_confidence: float = 0.0,
    confidence_power: float = 1.0,
) -> dict[str, Any]:
    """Append confidence-weighted depth/normal pseudo-SDF observations to a real-scene sample file."""
    validate_config(
        projection_convention=projection_convention,
        normal_space=normal_space,
        pixel_stride=pixel_stride,
        depth_scale=depth_scale,
        depth_min=depth_min,
        depth_max=depth_max,
        max_pixels_per_frame=max_pixels_per_frame,
        free_space_samples_per_depth=free_space_samples_per_depth,
        free_space_min_fraction=free_space_min_fraction,
        free_space_max_fraction=free_space_max_fraction,
        max_free_space_sdf=max_free_space_sdf,
        normal_offset_distance=normal_offset_distance,
        noise_ranges=((surface_noise_min, surface_noise_max), (free_space_noise_min, free_space_noise_max), (normal_noise_min, normal_noise_max)),
        default_depth_confidence=default_depth_confidence,
        default_normal_confidence=default_normal_confidence,
        min_depth_confidence=min_depth_confidence,
        min_normal_confidence=min_normal_confidence,
        confidence_power=confidence_power,
    )

    scene_root = Path(scene_dir)
    scene_meta, frames, splits = load_prepared_scene(scene_root)
    convention = resolve_projection_convention(scene_meta, projection_convention)
    frame_indices = resolve_frame_indices(splits, frame_count=len(frames), split=split)
    if max_frames is not None and max_frames > 0:
        frame_indices = frame_indices[:max_frames]
    if not frame_indices:
        raise ValueError(f"Split {split!r} did not resolve to any frames.")

    rng = np.random.default_rng(seed)
    depth_root = resolve_scene_path(scene_root, depth_dir)
    depth_conf_root = resolve_scene_path(scene_root, depth_confidence_dir) if depth_confidence_dir is not None else None
    normal_root = resolve_scene_path(scene_root, normal_dir) if normal_dir is not None else None
    normal_conf_root = resolve_scene_path(scene_root, normal_confidence_dir) if normal_confidence_dir is not None else None

    frame_samples = []
    frame_reports = []
    for frame_index in frame_indices:
        sample, report = build_frame_depth_normal_samples(
            frame=frames[int(frame_index)],
            frame_index=int(frame_index),
            depth_root=depth_root,
            depth_conf_root=depth_conf_root,
            normal_root=normal_root,
            normal_conf_root=normal_conf_root,
            rng=rng,
            convention=convention,
            normal_space=normal_space,
            max_pixels_per_frame=max_pixels_per_frame,
            pixel_stride=pixel_stride,
            depth_scale=depth_scale,
            depth_min=depth_min,
            depth_max=depth_max,
            add_free_space_samples=add_free_space_samples,
            free_space_samples_per_depth=free_space_samples_per_depth,
            free_space_min_fraction=free_space_min_fraction,
            free_space_max_fraction=free_space_max_fraction,
            max_free_space_sdf=max_free_space_sdf,
            add_normal_offset_samples=add_normal_offset_samples,
            normal_offset_distance=normal_offset_distance,
            surface_noise_min=surface_noise_min,
            surface_noise_max=surface_noise_max,
            free_space_noise_min=free_space_noise_min,
            free_space_noise_max=free_space_noise_max,
            normal_noise_min=normal_noise_min,
            normal_noise_max=normal_noise_max,
            default_depth_confidence=default_depth_confidence,
            default_normal_confidence=default_normal_confidence,
            min_depth_confidence=min_depth_confidence,
            min_normal_confidence=min_normal_confidence,
            confidence_power=confidence_power,
        )
        frame_samples.append(sample)
        frame_reports.append(report)

    depth_normal_samples = concatenate_sample_dicts(frame_samples)
    base = None
    if include_base_samples:
        resolved_base = resolve_scene_path(scene_root, base_samples_path or "real_samples.npz")
        if resolved_base.exists():
            base = load_npz_dict(resolved_base)
    merged = merge_sample_sets(base, depth_normal_samples)
    merged["sample_type_names"] = np.asarray(SAMPLE_TYPE_NAMES)

    output_path = resolve_scene_path(scene_root, output_samples_path or "real_depth_normal_samples.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **merged)

    counts = sample_type_counts(merged["sample_type"])
    report = {
        "schema_version": 1,
        "scene": scene_meta["scene"],
        "split": split,
        "projection_convention": convention,
        "normal_space": normal_space,
        "frame_count": len(frame_reports),
        "depth_normal_sample_count": int(depth_normal_samples["points"].shape[0]),
        "base_sample_count": int(base["points"].shape[0]) if base is not None and "points" in base else 0,
        "sample_count": int(merged["points"].shape[0]),
        "sample_type_counts": counts,
        "output_samples_path": str(output_path),
        "frame_reports": frame_reports,
        "config": {
            "max_pixels_per_frame": max_pixels_per_frame,
            "pixel_stride": pixel_stride,
            "depth_scale": depth_scale,
            "depth_min": depth_min,
            "depth_max": depth_max,
            "add_free_space_samples": add_free_space_samples,
            "free_space_samples_per_depth": free_space_samples_per_depth,
            "add_normal_offset_samples": add_normal_offset_samples,
            "normal_offset_distance": normal_offset_distance,
            "confidence_power": confidence_power,
        },
    }
    resolved_report = resolve_scene_path(scene_root, report_path or output_path.with_suffix(".json"))
    write_json(resolved_report, report)
    return {"samples_path": output_path, "report_path": resolved_report, "report": report}


def build_frame_depth_normal_samples(
    *,
    frame: dict[str, Any],
    frame_index: int,
    depth_root: Path,
    depth_conf_root: Path | None,
    normal_root: Path | None,
    normal_conf_root: Path | None,
    rng: np.random.Generator,
    convention: str,
    normal_space: str,
    max_pixels_per_frame: int | None,
    pixel_stride: int,
    depth_scale: float,
    depth_min: float,
    depth_max: float | None,
    add_free_space_samples: bool,
    free_space_samples_per_depth: int,
    free_space_min_fraction: float,
    free_space_max_fraction: float,
    max_free_space_sdf: float,
    add_normal_offset_samples: bool,
    normal_offset_distance: float,
    surface_noise_min: float,
    surface_noise_max: float,
    free_space_noise_min: float,
    free_space_noise_max: float,
    normal_noise_min: float,
    normal_noise_max: float,
    default_depth_confidence: float,
    default_normal_confidence: float,
    min_depth_confidence: float,
    min_normal_confidence: float,
    confidence_power: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    depth_path = find_map_path(depth_root, frame)
    depth = load_map(depth_path).astype(np.float64) * depth_scale
    height, width = frame_size(frame)
    depth = resize_nearest_if_needed(depth, height=height, width=width)

    valid = np.isfinite(depth) & (depth >= depth_min)
    if depth_max is not None:
        valid &= depth <= depth_max
    yy, xx = np.nonzero(valid[::pixel_stride, ::pixel_stride])
    yy = yy * pixel_stride
    xx = xx * pixel_stride
    if xx.size == 0:
        raise ValueError(f"Depth map {depth_path} produced no valid pixels.")
    if max_pixels_per_frame is not None and max_pixels_per_frame > 0 and xx.size > max_pixels_per_frame:
        chosen = np.sort(rng.choice(xx.size, size=max_pixels_per_frame, replace=False))
        xx = xx[chosen]
        yy = yy[chosen]

    selected_depth = depth[yy, xx]
    depth_confidence = load_optional_confidence(depth_conf_root, frame, height=height, width=width, default=default_depth_confidence)[yy, xx]
    depth_confidence = normalize_confidence(depth_confidence, minimum=min_depth_confidence, power=confidence_power)
    world_points, camera_centers, ray_distance = unproject_pixels(frame, xx=xx, yy=yy, depth=selected_depth, convention=convention)

    parts = [
        make_samples(
            points=world_points,
            sdf=np.zeros(world_points.shape[0], dtype=np.float64),
            noise=confidence_to_noise(depth_confidence, min_noise=surface_noise_min, max_noise=surface_noise_max),
            sample_type=np.full(world_points.shape[0], SAMPLE_TYPE_IDS["depth_surface"], dtype=np.int64),
            source_index=np.arange(world_points.shape[0], dtype=np.int64),
            camera_index=np.full(world_points.shape[0], frame_index, dtype=np.int64),
            ray_distance=ray_distance,
            depth_confidence=depth_confidence,
            normal_confidence=np.zeros(world_points.shape[0], dtype=np.float64),
            sample_confidence=depth_confidence,
        )
    ]

    if add_free_space_samples and free_space_samples_per_depth > 0:
        fractions = np.linspace(free_space_min_fraction, free_space_max_fraction, free_space_samples_per_depth)
        for fraction in fractions:
            free_points = camera_centers + (world_points - camera_centers) * float(fraction)
            free_sdf = np.minimum(ray_distance * (1.0 - float(fraction)), max_free_space_sdf)
            parts.append(
                make_samples(
                    points=free_points,
                    sdf=free_sdf,
                    noise=confidence_to_noise(depth_confidence, min_noise=free_space_noise_min, max_noise=free_space_noise_max),
                    sample_type=np.full(world_points.shape[0], SAMPLE_TYPE_IDS["depth_free_space"], dtype=np.int64),
                    source_index=np.arange(world_points.shape[0], dtype=np.int64),
                    camera_index=np.full(world_points.shape[0], frame_index, dtype=np.int64),
                    ray_distance=ray_distance * float(fraction),
                    depth_confidence=depth_confidence,
                    normal_confidence=np.zeros(world_points.shape[0], dtype=np.float64),
                    sample_confidence=depth_confidence,
                )
            )

    normal_count = 0
    if add_normal_offset_samples and normal_root is not None:
        normal_path = find_map_path(normal_root, frame)
        normals = load_normals(normal_path, height=height, width=width)[yy, xx]
        normal_confidence = load_optional_confidence(normal_conf_root, frame, height=height, width=width, default=default_normal_confidence)[yy, xx]
        normal_confidence = normalize_confidence(normal_confidence, minimum=min_normal_confidence, power=confidence_power)
        world_normals = orient_normals_to_camera(transform_normals(frame, normals, normal_space=normal_space), world_points=world_points, camera_centers=camera_centers)
        combined_confidence = np.minimum(depth_confidence, normal_confidence)
        noise = confidence_to_noise(combined_confidence, min_noise=normal_noise_min, max_noise=normal_noise_max)
        for direction, key in ((1.0, "depth_normal_positive"), (-1.0, "depth_normal_negative")):
            parts.append(
                make_samples(
                    points=world_points + direction * normal_offset_distance * world_normals,
                    sdf=np.full(world_points.shape[0], direction * normal_offset_distance, dtype=np.float64),
                    noise=noise,
                    sample_type=np.full(world_points.shape[0], SAMPLE_TYPE_IDS[key], dtype=np.int64),
                    source_index=np.arange(world_points.shape[0], dtype=np.int64),
                    camera_index=np.full(world_points.shape[0], frame_index, dtype=np.int64),
                    ray_distance=ray_distance,
                    depth_confidence=depth_confidence,
                    normal_confidence=normal_confidence,
                    sample_confidence=combined_confidence,
                )
            )
        normal_count = int(2 * world_points.shape[0])

    merged = concatenate_sample_dicts(parts)
    report = {
        "frame_index": frame_index,
        "file_name": frame.get("file_name"),
        "depth_path": str(depth_path),
        "selected_depth_pixels": int(world_points.shape[0]),
        "sample_count": int(merged["points"].shape[0]),
        "normal_offset_sample_count": normal_count,
        "depth_confidence_mean": float(depth_confidence.mean()) if depth_confidence.size else None,
    }
    return merged, report


def confidence_to_noise(confidence: np.ndarray, *, min_noise: float, max_noise: float) -> np.ndarray:
    confidence = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
    return max_noise - confidence * (max_noise - min_noise)


def normalize_confidence(values: np.ndarray, *, minimum: float, power: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size and values.max() > 1.0:
        values = values / 255.0
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, minimum, 1.0)
    if power != 1.0:
        values = values**power
    return np.clip(values, 0.0, 1.0)


def unproject_pixels(frame: dict[str, Any], *, xx: np.ndarray, yy: np.ndarray, depth: np.ndarray, convention: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intrinsics = frame["intrinsics"]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    z = np.asarray(depth, dtype=np.float64)
    if convention == "opencv":
        camera_xyz = np.stack(((xx - cx) * z / fx, (yy - cy) * z / fy, z), axis=1)
    elif convention == "opengl":
        camera_xyz = np.stack(((xx - cx) * z / fx, -(yy - cy) * z / fy, -z), axis=1)
    else:
        raise ValueError("convention must be 'opencv' or 'opengl'.")
    camera_to_world = np.asarray(frame["camera_to_world"], dtype=np.float64)
    homogeneous = np.concatenate((camera_xyz, np.ones((camera_xyz.shape[0], 1), dtype=np.float64)), axis=1)
    world_points = (homogeneous @ camera_to_world.T)[:, :3]
    center = camera_to_world[:3, 3]
    centers = np.repeat(center[None, :], world_points.shape[0], axis=0)
    ray_distance = np.linalg.norm(world_points - centers, axis=1)
    return world_points, centers, ray_distance


def transform_normals(frame: dict[str, Any], normals: np.ndarray, *, normal_space: str) -> np.ndarray:
    if normal_space == "world":
        world = normals
    elif normal_space == "camera":
        rotation = np.asarray(frame["camera_to_world"], dtype=np.float64)[:3, :3]
        world = normals @ rotation.T
    else:
        raise ValueError("normal_space must be 'camera' or 'world'.")
    return normalize_vectors(world)


def orient_normals_to_camera(normals: np.ndarray, *, world_points: np.ndarray, camera_centers: np.ndarray) -> np.ndarray:
    camera_vectors = camera_centers - world_points
    flip = np.sum(normals * camera_vectors, axis=1) < 0.0
    normals = normals.copy()
    normals[flip] *= -1.0
    return normalize_vectors(normals)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    fallback = np.zeros_like(vectors)
    fallback[:, 2] = 1.0
    return np.where(norms > 1e-12, vectors / np.clip(norms, 1e-12, None), fallback)


def make_samples(**arrays: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "points": np.asarray(arrays["points"], dtype=np.float64),
        "sdf": np.asarray(arrays["sdf"], dtype=np.float64).reshape(-1),
        "observation_noise_std": np.asarray(arrays["noise"], dtype=np.float64).reshape(-1),
        "sample_type": np.asarray(arrays["sample_type"], dtype=np.int64).reshape(-1),
        "source_point_index": np.asarray(arrays["source_index"], dtype=np.int64).reshape(-1),
        "camera_index": np.asarray(arrays["camera_index"], dtype=np.int64).reshape(-1),
        "ray_distance": np.asarray(arrays["ray_distance"], dtype=np.float64).reshape(-1),
        "depth_confidence": np.asarray(arrays["depth_confidence"], dtype=np.float64).reshape(-1),
        "normal_confidence": np.asarray(arrays["normal_confidence"], dtype=np.float64).reshape(-1),
        "sample_confidence": np.asarray(arrays["sample_confidence"], dtype=np.float64).reshape(-1),
    }


def concatenate_sample_dicts(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not items:
        raise ValueError("At least one sample dictionary is required.")
    keys = sorted(set().union(*(item.keys() for item in items)) - {"sample_type_names"})
    count_by_item = [int(item["points"].shape[0]) for item in items]
    merged = {}
    for key in keys:
        arrays = []
        for item, count in zip(items, count_by_item, strict=True):
            if key in item:
                arrays.append(np.asarray(item[key]))
            else:
                arrays.append(default_array_for_key(key, count))
        merged[key] = np.concatenate(arrays, axis=0)
    return merged


def merge_sample_sets(base: dict[str, np.ndarray] | None, extra: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return concatenate_sample_dicts([base, extra]) if base is not None else dict(extra)


def default_array_for_key(key: str, count: int) -> np.ndarray:
    if key == "points":
        return np.zeros((count, 3), dtype=np.float64)
    if key in {"sample_type", "source_point_index", "camera_index"}:
        return np.full(count, -1, dtype=np.int64)
    if key in {"depth_confidence", "normal_confidence", "sample_confidence"}:
        return np.zeros(count, dtype=np.float64)
    return np.zeros(count, dtype=np.float64)


def load_npz_dict(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files if key != "sample_type_names"}


def load_optional_confidence(root: Path | None, frame: dict[str, Any], *, height: int, width: int, default: float) -> np.ndarray:
    if root is None:
        return np.full((height, width), default, dtype=np.float64)
    path = find_map_path(root, frame)
    values = resize_nearest_if_needed(load_map(path), height=height, width=width)
    return np.asarray(values, dtype=np.float64)


def load_map(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = "arr_0" if "arr_0" in data.files else data.files[0]
            return np.asarray(data[key], dtype=np.float64)
    image = np.asarray(Image.open(path), dtype=np.float64)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    return image


def load_normals(path: Path, *, height: int, width: int) -> np.ndarray:
    normals = resize_nearest_if_needed(load_map(path), height=height, width=width)
    if normals.ndim != 3 or normals.shape[2] < 3:
        raise ValueError(f"Normal map {path} must have shape HxWx3.")
    normals = normals[..., :3].astype(np.float64)
    if normals.max(initial=0.0) > 1.5:
        normals = normals / 127.5 - 1.0
    elif normals.min(initial=0.0) >= 0.0:
        normals = normals * 2.0 - 1.0
    return normalize_vectors(normals.reshape(-1, 3)).reshape(normals.shape)


def resize_nearest_if_needed(array: np.ndarray, *, height: int, width: int) -> np.ndarray:
    array = np.asarray(array)
    if array.shape[:2] == (height, width):
        return array
    if array.ndim == 2:
        image = Image.fromarray(array.astype(np.float32))
        return np.asarray(image.resize((width, height), resample=Image.Resampling.NEAREST), dtype=np.float64)
    channels = []
    for channel in range(array.shape[2]):
        image = Image.fromarray(array[..., channel].astype(np.float32))
        channels.append(np.asarray(image.resize((width, height), resample=Image.Resampling.NEAREST), dtype=np.float64))
    return np.stack(channels, axis=2)


def find_map_path(root: Path, frame: dict[str, Any]) -> Path:
    names = []
    file_name = Path(str(frame.get("file_name") or frame.get("image_path", ""))).name
    if file_name:
        names.append(file_name)
        names.append(Path(file_name).stem)
    index = frame.get("index")
    if index is not None:
        names.extend([f"{int(index):06d}", str(index)])
    image_id = frame.get("image_id")
    if image_id is not None:
        names.append(str(image_id))
    seen = []
    for name in names:
        if name and name not in seen:
            seen.append(name)
    for name in seen:
        candidate = root / name
        if candidate.exists():
            return candidate
        stem = candidate.with_suffix("")
        for suffix in MAP_SUFFIXES:
            path = stem.with_suffix(suffix)
            if path.exists():
                return path
    raise FileNotFoundError(f"No map file for frame {file_name!r} under {root}.")


def frame_size(frame: dict[str, Any]) -> tuple[int, int]:
    intrinsics = frame.get("intrinsics", {})
    width = int(frame.get("width") or intrinsics.get("width"))
    height = int(frame.get("height") or intrinsics.get("height"))
    return height, width


def resolve_projection_convention(scene_meta: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    return "opengl" if scene_meta.get("source_format") == "transforms" else "opencv"


def resolve_frame_indices(splits: dict[str, Any], *, frame_count: int, split: str) -> list[int]:
    if split == "all":
        return list(range(frame_count))
    if split not in splits:
        raise ValueError(f"Split {split!r} does not exist.")
    return [int(index) for index in splits[split] if 0 <= int(index) < frame_count]


def resolve_scene_path(scene_root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else scene_root / resolved


def sample_type_counts(sample_type: np.ndarray) -> dict[str, int]:
    inverse = {value: key for key, value in SAMPLE_TYPE_IDS.items()}
    labels, counts = np.unique(sample_type, return_counts=True)
    return {inverse.get(int(label), str(int(label))): int(count) for label, count in zip(labels, counts, strict=True)}


def validate_config(
    *,
    projection_convention: str,
    normal_space: str,
    pixel_stride: int,
    depth_scale: float,
    depth_min: float,
    depth_max: float | None,
    max_pixels_per_frame: int | None,
    free_space_samples_per_depth: int,
    free_space_min_fraction: float,
    free_space_max_fraction: float,
    max_free_space_sdf: float,
    normal_offset_distance: float,
    noise_ranges: tuple[tuple[float, float], ...],
    default_depth_confidence: float,
    default_normal_confidence: float,
    min_depth_confidence: float,
    min_normal_confidence: float,
    confidence_power: float,
) -> None:
    if projection_convention not in PROJECTION_CONVENTIONS:
        raise ValueError(f"Unsupported projection convention {projection_convention!r}.")
    if normal_space not in NORMAL_SPACES:
        raise ValueError(f"Unsupported normal space {normal_space!r}.")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be positive.")
    if depth_scale <= 0.0 or depth_min <= 0.0:
        raise ValueError("depth_scale and depth_min must be positive.")
    if depth_max is not None and depth_max <= depth_min:
        raise ValueError("depth_max must exceed depth_min.")
    if max_pixels_per_frame is not None and max_pixels_per_frame <= 0:
        raise ValueError("max_pixels_per_frame must be positive when set.")
    if free_space_samples_per_depth < 0:
        raise ValueError("free_space_samples_per_depth must be non-negative.")
    if not 0.0 < free_space_min_fraction <= free_space_max_fraction < 1.0:
        raise ValueError("free-space fractions must satisfy 0 < min <= max < 1.")
    if max_free_space_sdf <= 0.0 or normal_offset_distance <= 0.0:
        raise ValueError("max_free_space_sdf and normal_offset_distance must be positive.")
    for low, high in noise_ranges:
        if low <= 0.0 or high < low:
            raise ValueError("Noise ranges must satisfy 0 < min <= max.")
    if not 0.0 <= default_depth_confidence <= 1.0 or not 0.0 <= default_normal_confidence <= 1.0:
        raise ValueError("Default confidences must be in [0, 1].")
    if not 0.0 <= min_depth_confidence <= 1.0 or not 0.0 <= min_normal_confidence <= 1.0:
        raise ValueError("Minimum confidences must be in [0, 1].")
    if confidence_power <= 0.0:
        raise ValueError("confidence_power must be positive.")
