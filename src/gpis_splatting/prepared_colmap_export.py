from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from gpis_splatting.colmap_render_mapping import write_render_name_map
from gpis_splatting.real_bootstrap import SparsePointCloud, load_sparse_point_cloud, subsample_point_cloud
from gpis_splatting.real_scene import load_prepared_scene, resolve_scene_image_path
from gpis_splatting.serialization import write_json

COLMAP_SPLITS = ("train", "test", "all")


def export_prepared_scene_to_colmap_3dgs(
    *,
    scene_dir: str | Path,
    output_dir: str | Path,
    split: str = "train",
    points_path: str | Path | None = None,
    max_points: int | None = 100000,
    seed: int = 13,
    copy_images: bool = True,
) -> dict[str, Any]:
    if split not in COLMAP_SPLITS:
        raise ValueError(f"Unsupported split {split!r}. Expected one of {', '.join(COLMAP_SPLITS)}.")
    if max_points is not None and max_points < 1:
        raise ValueError("max_points must be positive or None.")

    scene_root = Path(scene_dir)
    scene_meta, frames, splits = load_prepared_scene(scene_root)
    selected_frames = select_split_frames(frames, splits, split=split)
    if not selected_frames:
        raise ValueError(f"Prepared scene {scene_root} has no frames for split {split!r}.")

    out_dir = Path(output_dir)
    images_dir = out_dir / "images"
    sparse_dir = out_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    export_frames = materialize_export_images(scene_root=scene_root, frames=selected_frames, images_dir=images_dir, copy_images=copy_images)
    camera_rows, frame_camera_ids = build_colmap_cameras(export_frames)
    cloud, point_source, resolved_points_path = load_export_point_cloud(
        scene_root=scene_root,
        scene_meta=scene_meta,
        points_path=points_path,
        max_points=max_points,
        seed=seed,
    )

    cameras_path = sparse_dir / "cameras.txt"
    images_path = sparse_dir / "images.txt"
    points_path_out = sparse_dir / "points3D.txt"
    render_name_map_path = out_dir / "render_name_map.csv"
    write_colmap_cameras(cameras_path, camera_rows)
    write_colmap_images(images_path, export_frames, frame_camera_ids)
    write_colmap_points(points_path_out, cloud)
    write_render_name_map(render_name_map_path, export_frames, split=split)

    status_path = out_dir / "export_status.json"
    report_path = out_dir / "export_report.md"
    status = {
        "schema_version": 1,
        "scene_dir": str(scene_root),
        "scene": scene_meta.get("scene"),
        "dataset": scene_meta.get("dataset"),
        "output_dir": str(out_dir),
        "split": split,
        "frame_count": len(export_frames),
        "camera_count": len(camera_rows),
        "point_count": int(cloud.points.shape[0]),
        "point_source": point_source,
        "points_path": str(resolved_points_path) if resolved_points_path is not None else None,
        "max_points": max_points,
        "seed": seed,
        "copy_images": copy_images,
        "images_dir": str(images_dir),
        "sparse_dir": str(sparse_dir),
        "cameras_path": str(cameras_path),
        "images_path": str(images_path),
        "points3d_path": str(points_path_out),
        "render_name_map_path": str(render_name_map_path),
    }
    write_json(status_path, status)
    report_path.write_text(format_export_report(status), encoding="utf-8")
    return {
        "output_dir": out_dir,
        "status_path": status_path,
        "report_path": report_path,
        "status": status,
    }


def select_split_frames(frames: list[dict[str, Any]], splits: dict[str, Any], *, split: str) -> list[dict[str, Any]]:
    if split == "all":
        indices = list(range(len(frames)))
    else:
        indices = list(splits.get(split, []))
    selected = []
    for index in indices:
        if not isinstance(index, int) or index < 0 or index >= len(frames):
            raise ValueError(f"Split {split!r} contains invalid frame index {index!r}.")
        selected.append(frames[index])
    return selected


def materialize_export_images(
    *,
    scene_root: Path,
    frames: list[dict[str, Any]],
    images_dir: Path,
    copy_images: bool,
) -> list[dict[str, Any]]:
    used_names: set[str] = set()
    export_frames = []
    for output_index, frame in enumerate(frames, start=1):
        source_path = resolve_scene_image_path(scene_root, frame["image_path"])
        if not source_path.exists():
            raise FileNotFoundError(f"Missing prepared-scene image: {source_path}")
        image_name = unique_image_name(str(frame.get("file_name") or source_path.name), used_names, index=output_index)
        if copy_images:
            shutil.copy2(source_path, images_dir / image_name)
        export_frame = dict(frame)
        export_frame["colmap_image_id"] = output_index
        export_frame["colmap_image_name"] = image_name
        export_frames.append(export_frame)
    return export_frames


def build_colmap_cameras(frames: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    camera_ids_by_key: dict[tuple[float | int, ...], int] = {}
    camera_rows: list[dict[str, Any]] = []
    frame_camera_ids = []
    for frame in frames:
        intrinsics = frame.get("intrinsics") or {}
        width = require_int_dimension(frame, intrinsics, "width")
        height = require_int_dimension(frame, intrinsics, "height")
        fx = require_float_intrinsic(intrinsics, "fx")
        fy = require_float_intrinsic(intrinsics, "fy")
        cx = require_float_intrinsic(intrinsics, "cx")
        cy = require_float_intrinsic(intrinsics, "cy")
        key = (width, height, fx, fy, cx, cy)
        camera_id = camera_ids_by_key.get(key)
        if camera_id is None:
            camera_id = len(camera_rows) + 1
            camera_ids_by_key[key] = camera_id
            camera_rows.append(
                {
                    "camera_id": camera_id,
                    "model": "PINHOLE",
                    "width": width,
                    "height": height,
                    "params": (fx, fy, cx, cy),
                }
            )
        frame_camera_ids.append(camera_id)
    return camera_rows, frame_camera_ids


def require_int_dimension(frame: dict[str, Any], intrinsics: dict[str, Any], name: str) -> int:
    value = frame.get(name) or intrinsics.get(name)
    if value is None:
        raise ValueError(f"Prepared-scene frame is missing {name}.")
    return int(value)


def require_float_intrinsic(intrinsics: dict[str, Any], name: str) -> float:
    value = intrinsics.get(name)
    if value is None:
        raise ValueError(f"Prepared-scene frame is missing intrinsics.{name}.")
    return float(value)


def load_export_point_cloud(
    *,
    scene_root: Path,
    scene_meta: dict[str, Any],
    points_path: str | Path | None,
    max_points: int | None,
    seed: int,
) -> tuple[SparsePointCloud, str, Path | None]:
    if points_path is not None:
        resolved = resolve_points_path(scene_root, points_path)
        cloud, source = load_point_cloud_by_path(resolved)
        return subsample_point_cloud(cloud, max_points=max_points, seed=seed), source, resolved

    try:
        cloud, source, resolved = load_sparse_point_cloud(scene_root=scene_root, scene_meta=scene_meta, point_source="auto", point_path=None)
        return subsample_point_cloud(cloud, max_points=max_points, seed=seed), source, resolved
    except FileNotFoundError:
        splats_path = scene_root / "real_splats.npz"
        if splats_path.exists():
            cloud = load_splat_npz_point_cloud(splats_path)
            return subsample_point_cloud(cloud, max_points=max_points, seed=seed), "splat_npz", splats_path
        raise


def resolve_points_path(scene_root: Path, points_path: str | Path) -> Path:
    path = Path(points_path)
    if not path.is_absolute():
        path = scene_root / path
    if not path.exists():
        raise FileNotFoundError(f"Missing point source file: {path}")
    return path


def load_point_cloud_by_path(path: Path) -> tuple[SparsePointCloud, str]:
    if path.suffix.lower() == ".npz":
        return load_splat_npz_point_cloud(path), "splat_npz"
    cloud, source, _resolved = load_sparse_point_cloud(scene_root=path.parent, scene_meta={}, point_source="auto", point_path=path)
    return cloud, source


def load_splat_npz_point_cloud(path: str | Path) -> SparsePointCloud:
    with np.load(path, allow_pickle=False) as data:
        if "centers" not in data.files:
            raise ValueError(f"{path} is missing the internal splat array 'centers'.")
        points = np.asarray(data["centers"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{path} has invalid centers shape {points.shape}; expected Nx3.")
        if "colors" in data.files:
            colors = np.asarray(data["colors"], dtype=np.float64)
            if colors.shape != points.shape:
                raise ValueError(f"{path} has colors shape {colors.shape}, expected {points.shape}.")
            colors = np.clip(colors, 0.0, 1.0)
        else:
            colors = np.full((points.shape[0], 3), 0.7, dtype=np.float64)
    if points.shape[0] == 0:
        raise ValueError(f"{path} contains no splat centers.")
    return SparsePointCloud(points=points, colors=colors)


def write_colmap_cameras(path: str | Path, cameras: list[dict[str, Any]]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    for camera in cameras:
        params = " ".join(format_float(value) for value in camera["params"])
        lines.append(f"{camera['camera_id']} {camera['model']} {camera['width']} {camera['height']} {params}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_colmap_images(path: str | Path, frames: list[dict[str, Any]], camera_ids: list[int]) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(frames)}",
    ]
    for frame, camera_id in zip(frames, camera_ids, strict=True):
        world_to_camera = frame_world_to_camera(frame)
        qvec = rotation_matrix_to_quaternion(world_to_camera[:3, :3])
        tvec = world_to_camera[:3, 3]
        pose_values = " ".join(format_float(value) for value in [*qvec, *tvec])
        lines.append(f"{frame['colmap_image_id']} {pose_values} {camera_id} {frame['colmap_image_name']}")
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_colmap_points(path: str | Path, cloud: SparsePointCloud) -> None:
    colors = np.clip(np.rint(np.clip(cloud.colors, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    errors = cloud.errors if cloud.errors is not None else np.ones((cloud.points.shape[0],), dtype=np.float64)
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {cloud.points.shape[0]}",
    ]
    for point_id, (point, color, error) in enumerate(zip(cloud.points, colors, errors, strict=True), start=1):
        xyz = " ".join(format_float(value) for value in point)
        rgb = " ".join(str(int(value)) for value in color)
        lines.append(f"{point_id} {xyz} {rgb} {format_float(float(error))}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def frame_world_to_camera(frame: dict[str, Any]) -> np.ndarray:
    world_to_camera = frame.get("world_to_camera")
    if world_to_camera is None:
        camera_to_world = frame.get("camera_to_world")
        if camera_to_world is None:
            raise ValueError(f"Frame {frame.get('index')} has neither world_to_camera nor camera_to_world.")
        world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    matrix = np.asarray(world_to_camera, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Frame {frame.get('index')} world_to_camera has shape {matrix.shape}, expected 4x4.")
    return matrix


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Rotation matrix has shape {matrix.shape}, expected 3x3.")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    qvec = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(qvec))
    if norm <= 1e-12:
        raise ValueError("Rotation matrix produced a zero quaternion.")
    qvec = qvec / norm
    if qvec[0] < 0.0:
        qvec = -qvec
    return tuple(float(value) for value in qvec)


def unique_image_name(name: str, used_names: set[str], *, index: int) -> str:
    path = Path(name)
    candidate = path.name
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    deduplicated = f"{index:06d}_{path.name}"
    used_names.add(deduplicated)
    return deduplicated


def format_float(value: float) -> str:
    return f"{float(value):.17g}"


def format_export_report(status: dict[str, Any]) -> str:
    lines = [
        "# Prepared Scene COLMAP/3DGS Export",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Dataset: `{status['dataset']}`",
        f"- Split: `{status['split']}`",
        f"- Frames: `{status['frame_count']}`",
        f"- Cameras: `{status['camera_count']}`",
        f"- Points: `{status['point_count']}` from `{status['point_source']}`",
        f"- Output: `{status['output_dir']}`",
        f"- Render-name map: `{status['render_name_map_path']}`",
        "",
        "The directory is laid out as `images/` plus `sparse/0/{cameras.txt,images.txt,points3D.txt}` for standard 3DGS training tools.",
    ]
    return "\n".join(lines) + "\n"
