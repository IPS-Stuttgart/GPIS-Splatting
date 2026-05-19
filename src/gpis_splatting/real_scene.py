from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from gpis_splatting.serialization import read_json, write_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SUPPORTED_INPUT_FORMATS = ("auto", "transforms", "colmap_text")
PINHOLE_MODELS = {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}
SIMPLE_PINHOLE_MODELS = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}


def prepare_real_scene(
    *,
    input_dir: str | Path,
    output_root: str | Path,
    scene: str,
    dataset: str = "mipnerf360_sparse",
    input_format: str = "auto",
    image_dir: str = "images",
    train_view_count: int = 12,
    copy_images: bool = True,
    bounds_scale: float = 1.1,
) -> Path:
    if input_format not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(f"Unsupported input format {input_format!r}. Expected one of {', '.join(SUPPORTED_INPUT_FORMATS)}.")
    if train_view_count < 1:
        raise ValueError("train_view_count must be positive.")

    source = Path(input_dir)
    if not source.exists():
        raise FileNotFoundError(f"Missing input directory: {source}")
    resolved_format = detect_input_format(source, input_format)
    frames = _load_frames(source, resolved_format, image_dir=image_dir)
    if not frames:
        raise ValueError(f"No images were found in {source}.")

    out_dir = Path(output_root) / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_frames = _materialize_images(frames, out_dir, copy_images=copy_images)
    splits = build_sparse_split(len(prepared_frames), train_view_count)
    scene_meta = {
        "schema_version": 1,
        "scene": scene,
        "dataset": dataset,
        "source_dir": str(source.resolve()),
        "source_format": resolved_format,
        "image_count": len(prepared_frames),
        "train_view_count": len(splits["train"]),
        "test_view_count": len(splits["test"]),
        "bounds": estimate_bounds(prepared_frames, scale=bounds_scale),
    }

    write_json(out_dir / "real_scene.json", scene_meta)
    write_json(out_dir / "cameras.json", {"schema_version": 1, "frames": prepared_frames})
    write_json(out_dir / "splits.json", splits)
    validation = validate_real_scene_dir(out_dir)
    write_json(out_dir / "validation.json", validation)
    return out_dir


def detect_input_format(input_dir: Path, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if (input_dir / "transforms.json").exists():
        return "transforms"
    if _colmap_text_dir(input_dir) is not None:
        return "colmap_text"
    raise FileNotFoundError(
        f"Could not auto-detect dataset format under {input_dir}. Expected transforms.json or COLMAP text files cameras.txt/images.txt."
    )


def build_sparse_split(image_count: int, train_view_count: int) -> dict[str, Any]:
    if image_count < 1:
        raise ValueError("image_count must be positive.")
    train_count = min(train_view_count, image_count)
    if train_count == image_count:
        train_indices = list(range(image_count))
    else:
        train_indices = sorted({int(round(index)) for index in np.linspace(0, image_count - 1, train_count)})
        candidate = 0
        while len(train_indices) < train_count:
            if candidate not in train_indices:
                train_indices.append(candidate)
            candidate += 1
        train_indices = sorted(train_indices)
    train_set = set(train_indices)
    test_indices = [index for index in range(image_count) if index not in train_set]
    return {
        "schema_version": 1,
        "train": train_indices,
        "test": test_indices,
        "split_policy": {
            "name": "even_sparse_train_views",
            "requested_train_view_count": train_view_count,
        },
    }


def validate_real_scene_dir(scene_dir: str | Path) -> dict[str, Any]:
    root = Path(scene_dir)
    required = ["real_scene.json", "cameras.json", "splits.json"]
    missing_files = [name for name in required if not (root / name).exists()]
    if missing_files:
        return {
            "passed": False,
            "scene_dir": str(root),
            "missing_files": missing_files,
            "image_count": 0,
            "missing_images": [],
            "split_errors": ["missing metadata files"],
        }

    scene_meta = read_json(root / "real_scene.json")
    cameras = read_json(root / "cameras.json")
    splits = read_json(root / "splits.json")
    frames = cameras.get("frames", [])
    missing_images = []
    for frame in frames:
        image_path = resolve_scene_image_path(root, frame["image_path"])
        if not image_path.exists():
            missing_images.append(frame["image_path"])

    split_errors = _validate_splits(splits, len(frames))
    passed = not missing_files and not missing_images and not split_errors and scene_meta.get("image_count") == len(frames)
    return {
        "passed": passed,
        "scene_dir": str(root),
        "scene": scene_meta.get("scene"),
        "source_format": scene_meta.get("source_format"),
        "image_count": len(frames),
        "train_view_count": len(splits.get("train", [])),
        "test_view_count": len(splits.get("test", [])),
        "missing_files": missing_files,
        "missing_images": missing_images,
        "split_errors": split_errors,
    }


def resolve_scene_image_path(scene_dir: str | Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return Path(scene_dir) / path


def load_prepared_scene(scene_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = Path(scene_dir)
    return read_json(root / "real_scene.json"), read_json(root / "cameras.json")["frames"], read_json(root / "splits.json")


def estimate_bounds(frames: list[dict[str, Any]], *, scale: float = 1.1) -> dict[str, Any] | None:
    centers = []
    for frame in frames:
        camera_to_world = frame.get("camera_to_world")
        if camera_to_world is None:
            continue
        centers.append(np.asarray(camera_to_world, dtype=np.float64)[:3, 3])
    if not centers:
        return None
    points = np.stack(centers)
    center = points.mean(axis=0)
    radius = float(np.linalg.norm(points - center, axis=1).max())
    radius = max(radius * scale, 1e-3)
    return {
        "center": center.tolist(),
        "radius": radius,
        "min": (center - radius).tolist(),
        "max": (center + radius).tolist(),
        "source": "camera_centers",
    }


def _load_frames(input_dir: Path, input_format: str, *, image_dir: str) -> list[dict[str, Any]]:
    if input_format == "transforms":
        return _load_transforms_frames(input_dir, image_dir=image_dir)
    if input_format == "colmap_text":
        return _load_colmap_text_frames(input_dir, image_dir=image_dir)
    raise ValueError(f"Unsupported input format {input_format!r}.")


def _load_transforms_frames(input_dir: Path, *, image_dir: str) -> list[dict[str, Any]]:
    transforms = read_json(input_dir / "transforms.json")
    frames = []
    for index, frame in enumerate(transforms.get("frames", [])):
        source_path = _resolve_source_image(input_dir, image_dir, frame["file_path"])
        width, height = _image_size(source_path)
        intrinsics = _intrinsics_from_transforms(transforms, width=width, height=height)
        camera_to_world = frame.get("transform_matrix")
        frames.append(
            {
                "index": index,
                "image_id": str(index),
                "file_name": source_path.name,
                "source_path": str(source_path.resolve()),
                "width": width,
                "height": height,
                "camera_id": str(frame.get("camera_id", "0")),
                "intrinsics": intrinsics,
                "camera_to_world": camera_to_world,
                "world_to_camera": _invert_matrix(camera_to_world) if camera_to_world is not None else None,
            }
        )
    return frames


def _load_colmap_text_frames(input_dir: Path, *, image_dir: str) -> list[dict[str, Any]]:
    colmap_dir = _colmap_text_dir(input_dir)
    if colmap_dir is None:
        raise FileNotFoundError(f"Could not find COLMAP text files under {input_dir}.")
    cameras = _parse_colmap_cameras(colmap_dir / "cameras.txt")
    images = _parse_colmap_images(colmap_dir / "images.txt")
    frames = []
    for index, image in enumerate(sorted(images, key=lambda item: item["name"])):
        source_path = _resolve_source_image(input_dir, image_dir, image["name"])
        width, height = _image_size(source_path)
        intrinsics = dict(cameras[image["camera_id"]])
        intrinsics["width"] = width
        intrinsics["height"] = height
        world_to_camera = _colmap_world_to_camera(image)
        camera_to_world = np.linalg.inv(np.asarray(world_to_camera, dtype=np.float64)).tolist()
        frames.append(
            {
                "index": index,
                "image_id": str(image["image_id"]),
                "file_name": source_path.name,
                "source_path": str(source_path.resolve()),
                "width": width,
                "height": height,
                "camera_id": str(image["camera_id"]),
                "intrinsics": intrinsics,
                "camera_to_world": camera_to_world,
                "world_to_camera": world_to_camera,
            }
        )
    return frames


def _materialize_images(frames: list[dict[str, Any]], scene_dir: Path, *, copy_images: bool) -> list[dict[str, Any]]:
    image_out = scene_dir / "images"
    if copy_images:
        image_out.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    prepared = []
    for frame in frames:
        source_path = Path(frame["source_path"])
        dest_name = _unique_image_name(source_path.name, used_names, index=int(frame["index"]))
        next_frame = dict(frame)
        next_frame["file_name"] = dest_name
        if copy_images:
            dest = image_out / dest_name
            shutil.copy2(source_path, dest)
            next_frame["image_path"] = Path("images", dest_name).as_posix()
        else:
            next_frame["image_path"] = str(source_path.resolve())
        next_frame.pop("source_path", None)
        prepared.append(next_frame)
    return prepared


def _unique_image_name(name: str, used_names: set[str], *, index: int) -> str:
    if name not in used_names:
        used_names.add(name)
        return name
    path = Path(name)
    suffix = 0
    while True:
        prefix = f"{index:06d}" if suffix == 0 else f"{index:06d}_{suffix}"
        candidate = f"{prefix}_{path.name}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        suffix += 1


def _resolve_source_image(input_dir: Path, image_dir: str, image_name: str) -> Path:
    requested = Path(image_name)
    candidates = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.extend([input_dir / requested, input_dir / image_dir / requested])
    if requested.suffix:
        for candidate in candidates:
            if candidate.exists():
                return candidate
    else:
        for candidate in candidates:
            for suffix in IMAGE_EXTENSIONS:
                path = candidate.with_suffix(suffix)
                if path.exists():
                    return path
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image {image_name!r} under {input_dir}.")


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _intrinsics_from_transforms(transforms: dict[str, Any], *, width: int, height: int) -> dict[str, Any]:
    camera_angle_x = transforms.get("camera_angle_x")
    camera_angle_y = transforms.get("camera_angle_y")
    fx = transforms.get("fl_x")
    fy = transforms.get("fl_y")
    if fx is None and camera_angle_x is not None:
        fx = 0.5 * width / math.tan(0.5 * float(camera_angle_x))
    if fy is None and camera_angle_y is not None:
        fy = 0.5 * height / math.tan(0.5 * float(camera_angle_y))
    if fx is None and fy is not None:
        fx = fy
    if fy is None and fx is not None:
        fy = fx
    return {
        "model": "PINHOLE",
        "width": width,
        "height": height,
        "fx": float(fx) if fx is not None else None,
        "fy": float(fy) if fy is not None else None,
        "cx": float(transforms.get("cx", width / 2.0)),
        "cy": float(transforms.get("cy", height / 2.0)),
        "params": [],
    }


def _colmap_text_dir(input_dir: Path) -> Path | None:
    candidates = [input_dir, input_dir / "sparse", input_dir / "sparse" / "0"]
    for candidate in candidates:
        if (candidate / "cameras.txt").exists() and (candidate / "images.txt").exists():
            return candidate
    return None


def _parse_colmap_cameras(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    for line in _iter_colmap_data_lines(path):
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(value) for value in parts[4:]]
        fx, fy, cx, cy = _pinhole_intrinsics_from_colmap(model, params)
        cameras[camera_id] = {
            "model": model,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "params": params,
        }
    return cameras


def _parse_colmap_images(path: Path) -> list[dict[str, Any]]:
    lines = list(_iter_colmap_data_lines(path))
    images = []
    index = 0
    while index < len(lines):
        line = lines[index]
        parts = line.split()
        if not _looks_like_colmap_image_header(parts):
            index += 1
            continue
        images.append(
            {
                "image_id": int(parts[0]),
                "qvec": [float(value) for value in parts[1:5]],
                "tvec": [float(value) for value in parts[5:8]],
                "camera_id": int(parts[8]),
                "name": " ".join(parts[9:]),
            }
        )
        if index + 1 < len(lines) and not _looks_like_colmap_image_header(lines[index + 1].split()):
            index += 2
        else:
            index += 1
    return images


def _iter_colmap_data_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def _looks_like_colmap_image_header(parts: list[str]) -> bool:
    if len(parts) < 10:
        return False
    try:
        int(parts[0])
        [float(value) for value in parts[1:8]]
        int(parts[8])
    except ValueError:
        return False
    return True


def _pinhole_intrinsics_from_colmap(model: str, params: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    if model in PINHOLE_MODELS and len(params) >= 4:
        return params[0], params[1], params[2], params[3]
    if model in SIMPLE_PINHOLE_MODELS and len(params) >= 3:
        return params[0], params[0], params[1], params[2]
    return None, None, None, None


def _colmap_world_to_camera(image: dict[str, Any]) -> list[list[float]]:
    rotation = _quaternion_to_rotation(image["qvec"])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(image["tvec"], dtype=np.float64)
    return transform.tolist()


def _quaternion_to_rotation(qvec: list[float]) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.asarray(
        [
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw],
            [2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw],
            [2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ],
        dtype=np.float64,
    )


def _invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return np.linalg.inv(np.asarray(matrix, dtype=np.float64)).tolist()


def _validate_splits(splits: dict[str, Any], image_count: int) -> list[str]:
    errors = []
    seen: set[int] = set()
    for split_name in ("train", "test"):
        for index in splits.get(split_name, []):
            if not isinstance(index, int):
                errors.append(f"{split_name} contains non-integer index {index!r}")
                continue
            if index < 0 or index >= image_count:
                errors.append(f"{split_name} index {index} is outside [0, {image_count})")
            if index in seen:
                errors.append(f"index {index} appears in multiple splits")
            seen.add(index)
    return errors
