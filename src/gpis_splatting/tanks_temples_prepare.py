from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from gpis_splatting.real_scene import IMAGE_EXTENSIONS, build_sparse_split, estimate_bounds, validate_real_scene_dir
from gpis_splatting.serialization import write_json
from gpis_splatting.tanks_temples_common import image_size, natural_sort_key, unique_image_name
from gpis_splatting.tanks_temples_resources import (
    TANKS_TEMPLES_LICENSE_URL,
    TANKS_TEMPLES_SOURCE_URL,
    TANKS_TEMPLES_TUTORIAL_URL,
)


def prepare_tanks_temples_scene(
    *,
    input_dir: str | Path,
    output_root: str | Path = "real_scenes",
    scene: str = "Ignatius",
    prepared_scene: str | None = None,
    image_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    reconstruction_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
    crop_path: str | Path | None = None,
    train_view_count: int = 12,
    copy_images: bool = True,
    focal_length_factor: float = 0.7,
    bounds_scale: float = 1.1,
) -> Path:
    if train_view_count < 1:
        raise ValueError("train_view_count must be positive.")
    if focal_length_factor <= 0.0:
        raise ValueError("focal_length_factor must be positive.")

    source = Path(input_dir)
    if not source.exists():
        raise FileNotFoundError(f"Missing Tanks and Temples input directory: {source}")
    scene_name = prepared_scene or f"{scene.lower()}_tanks_temples"
    images = find_tanks_temples_images(source, scene=scene, image_dir=image_dir)
    log = resolve_tanks_temples_file(source, log_path, candidates=[Path("camera_poses") / f"{scene}.log", Path(f"{scene}.log")], required=True)
    poses = read_tanks_temples_log(log)
    if len(poses) != len(images):
        raise ValueError(f"Tanks and Temples image/log count mismatch: found {len(images)} images and {len(poses)} poses.")

    out_dir = Path(output_root) / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_frames = materialize_tanks_temples_frames(
        images,
        poses,
        out_dir,
        copy_images=copy_images,
        focal_length_factor=focal_length_factor,
    )
    splits = build_sparse_split(len(prepared_frames), train_view_count)
    auxiliary = resolve_tanks_temples_auxiliary(
        source,
        scene=scene,
        reconstruction_path=reconstruction_path,
        ground_truth_path=ground_truth_path,
        alignment_path=alignment_path,
        crop_path=crop_path,
    )
    scene_meta = {
        "schema_version": 1,
        "scene": scene_name,
        "dataset": "tanks_temples",
        "source_scene": scene,
        "source_dir": str(source.resolve()),
        "source_format": "tanks_temples_log",
        "image_count": len(prepared_frames),
        "train_view_count": len(splits["train"]),
        "test_view_count": len(splits["test"]),
        "bounds": estimate_bounds(prepared_frames, scale=bounds_scale),
        "tanks_temples": {
            "source_url": TANKS_TEMPLES_SOURCE_URL,
            "tutorial_url": TANKS_TEMPLES_TUTORIAL_URL,
            "license_url": TANKS_TEMPLES_LICENSE_URL,
            "camera_log_path": str(log),
            "reconstruction_path": str(auxiliary["reconstruction"]) if auxiliary["reconstruction"] is not None else None,
            "ground_truth_path": str(auxiliary["ground_truth"]) if auxiliary["ground_truth"] is not None else None,
            "alignment_path": str(auxiliary["alignment"]) if auxiliary["alignment"] is not None else None,
            "crop_path": str(auxiliary["crop"]) if auxiliary["crop"] is not None else None,
            "focal_length_factor": focal_length_factor,
            "intrinsics_source": "tanks_temples_download_page_recommended_pinhole",
        },
    }
    write_json(out_dir / "real_scene.json", scene_meta)
    write_json(out_dir / "cameras.json", {"schema_version": 1, "frames": prepared_frames})
    write_json(out_dir / "splits.json", splits)
    validation = validate_real_scene_dir(out_dir)
    validation["tanks_temples_assets"] = {key: str(value) if value is not None else None for key, value in auxiliary.items()}
    write_json(out_dir / "validation.json", validation)
    return out_dir


def find_tanks_temples_images(source: Path, *, scene: str, image_dir: str | Path | None) -> list[Path]:
    candidates = []
    if image_dir is not None:
        requested = Path(image_dir)
        candidates.append(requested if requested.is_absolute() else source / requested)
    candidates.extend([source / "image_sets" / scene, source / scene, source / "images", source])
    for candidate in candidates:
        if candidate.exists():
            images = [path for path in candidate.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
            images = sorted(images, key=lambda path: natural_sort_key(str(path.relative_to(candidate))))
            if images:
                return images
    raise FileNotFoundError(f"Could not find Tanks and Temples images for {scene!r} under {source}.")


def read_tanks_temples_log(path: str | Path) -> list[dict[str, Any]]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 5 != 0:
        raise ValueError(f"Tanks and Temples log file {path} must contain five lines per pose.")
    poses = []
    for index in range(0, len(lines), 5):
        metadata = [int(value) for value in lines[index].split()]
        if len(metadata) != 3:
            raise ValueError(f"Tanks and Temples log metadata line must contain three integers: {lines[index]!r}")
        matrix = np.asarray([[float(value) for value in lines[index + row + 1].split()] for row in range(4)], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Tanks and Temples log pose at item {index // 5} is not 4x4.")
        poses.append({"metadata": metadata, "camera_to_world": matrix})
    return poses


def materialize_tanks_temples_frames(
    images: list[Path],
    poses: list[dict[str, Any]],
    scene_dir: Path,
    *,
    copy_images: bool,
    focal_length_factor: float,
) -> list[dict[str, Any]]:
    image_out = scene_dir / "images"
    if copy_images:
        image_out.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    frames = []
    for index, (image_path, pose) in enumerate(zip(images, poses, strict=True)):
        width, height = image_size(image_path)
        dest_name = unique_image_name(image_path.name, used_names, index=index)
        if copy_images:
            destination = image_out / dest_name
            shutil.copy2(image_path, destination)
            frame_image_path = Path("images", dest_name).as_posix()
        else:
            frame_image_path = str(image_path.resolve())
        camera_to_world = np.asarray(pose["camera_to_world"], dtype=np.float64)
        frames.append(
            {
                "index": index,
                "image_id": str(pose["metadata"][0]),
                "file_name": dest_name,
                "source_path": str(image_path.resolve()),
                "image_path": frame_image_path,
                "width": width,
                "height": height,
                "camera_id": str(pose["metadata"][1]),
                "tanks_temples_metadata": pose["metadata"],
                "intrinsics": {
                    "model": "PINHOLE",
                    "width": width,
                    "height": height,
                    "fx": float(focal_length_factor * width),
                    "fy": float(focal_length_factor * width),
                    "cx": float(width / 2.0),
                    "cy": float(height / 2.0),
                    "params": [],
                },
                "camera_to_world": camera_to_world.tolist(),
                "world_to_camera": np.linalg.inv(camera_to_world).tolist(),
            }
        )
    return frames


def resolve_tanks_temples_auxiliary(
    source: Path,
    *,
    scene: str,
    reconstruction_path: str | Path | None,
    ground_truth_path: str | Path | None,
    alignment_path: str | Path | None,
    crop_path: str | Path | None,
) -> dict[str, Path | None]:
    return {
        "reconstruction": resolve_tanks_temples_file(source, reconstruction_path, candidates=[Path("reconstruction") / f"{scene}.ply", Path(f"{scene}.ply")], required=False),
        "ground_truth": resolve_tanks_temples_file(source, ground_truth_path, candidates=[Path("ground_truth") / f"{scene}.ply"], required=False),
        "alignment": resolve_tanks_temples_file(source, alignment_path, candidates=[Path("alignment") / f"{scene}.txt"], required=False),
        "crop": resolve_tanks_temples_file(source, crop_path, candidates=[Path("crop") / f"{scene}.json"], required=False),
    }


def resolve_tanks_temples_file(source: Path, path: str | Path | None, *, candidates: list[Path], required: bool) -> Path | None:
    if path is not None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = source / resolved
        if not resolved.exists():
            raise FileNotFoundError(f"Missing Tanks and Temples file: {resolved}")
        return resolved.resolve()
    for candidate in candidates:
        resolved = source / candidate
        if resolved.exists():
            return resolved.resolve()
    if required:
        formatted = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not find required Tanks and Temples file under {source}. Tried: {formatted}")
    return None
