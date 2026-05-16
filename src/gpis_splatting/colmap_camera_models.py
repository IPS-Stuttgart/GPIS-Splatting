from __future__ import annotations

from typing import Any

import numpy as np

COLMAP_CAMERA_MODEL_PARAM_COUNTS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "SIMPLE_FISHEYE": 3,
    "FISHEYE": 4,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
}

LINEAR_COLMAP_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}
FISHEYE_COLMAP_MODELS = {
    "SIMPLE_FISHEYE",
    "FISHEYE",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
    "OPENCV_FISHEYE",
}
DISTORTED_COLMAP_PARAM_INDICES = {
    "SIMPLE_RADIAL": (3,),
    "RADIAL": (3, 4),
    "OPENCV": (4, 5, 6, 7),
    "FULL_OPENCV": (4, 5, 6, 7, 8, 9, 10, 11),
    "SIMPLE_RADIAL_FISHEYE": (3,),
    "RADIAL_FISHEYE": (3, 4),
    "OPENCV_FISHEYE": (4, 5, 6, 7),
}

_PATCHED = False


def colmap_intrinsics_from_params(model: str, params: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    """Return fx, fy, cx, cy for a supported COLMAP camera model."""
    model = model.upper()
    validate_colmap_camera_params(model, params)
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        return params[0], params[0], params[1], params[2]
    if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FISHEYE"}:
        return params[0], params[1], params[2], params[3]
    return None, None, None, None


def validate_colmap_camera_params(model: str, params: list[float] | tuple[float, ...] | np.ndarray) -> None:
    model = model.upper()
    expected_count = COLMAP_CAMERA_MODEL_PARAM_COUNTS.get(model)
    if expected_count is None:
        supported = ", ".join(sorted(COLMAP_CAMERA_MODEL_PARAM_COUNTS))
        raise ValueError(f"Unsupported COLMAP camera model {model!r}. Supported models: {supported}.")
    if len(params) != expected_count:
        raise ValueError(f"COLMAP camera model {model!r} expects {expected_count} params, got {len(params)}.")


def project_normalized_points_with_intrinsics(
    x: np.ndarray,
    y: np.ndarray,
    intrinsics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Project normalized camera coordinates with supported COLMAP camera models.

    The equations follow COLMAP's camera-model parameter order. They are used for
    diagnostic CPU splat rendering and projected-coverage diagnostics, where
    silently treating distorted COLMAP cameras as pinhole cameras changes the
    reported render/alignment results.
    """
    model = str(intrinsics.get("model") or "PINHOLE").upper()
    params = _intrinsic_params(intrinsics)
    if model == "PINHOLE":
        return _project_pinhole_fields(x, y, intrinsics)
    if model == "SIMPLE_PINHOLE":
        if params.shape[0] == 3:
            f, cx, cy = _require_colmap_params(model, params, 3)
            return f * x + cx, f * y + cy
        return _project_pinhole_fields(x, y, intrinsics)
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k = _require_colmap_params(model, params, 4)
        r2 = x * x + y * y
        radial = k * r2
        return f * (x + x * radial) + cx, f * (y + y * radial) + cy
    if model == "RADIAL":
        f, cx, cy, k1, k2 = _require_colmap_params(model, params, 5)
        r2 = x * x + y * y
        radial = k1 * r2 + k2 * r2 * r2
        return f * (x + x * radial) + cx, f * (y + y * radial) + cy
    if model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = _require_colmap_params(model, params, 8)
        x_d, y_d = _opencv_distort(x, y, k1, k2, p1, p2)
        return fx * x_d + cx, fy * y_d + cy
    if model == "FULL_OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = _require_colmap_params(model, params, 12)
        x_d, y_d = _full_opencv_distort(x, y, k1, k2, p1, p2, k3, k4, k5, k6)
        return fx * x_d + cx, fy * y_d + cy
    if model == "SIMPLE_FISHEYE":
        f, cx, cy = _require_colmap_params(model, params, 3)
        uu, vv = _normal_to_fisheye(x, y)
        return f * uu + cx, f * vv + cy
    if model == "FISHEYE":
        fx, fy, cx, cy = _require_colmap_params(model, params, 4)
        uu, vv = _normal_to_fisheye(x, y)
        return fx * uu + cx, fy * vv + cy
    if model == "SIMPLE_RADIAL_FISHEYE":
        f, cx, cy, k = _require_colmap_params(model, params, 4)
        uu, vv = _normal_to_fisheye(x, y)
        theta2 = uu * uu + vv * vv
        radial = k * theta2
        return f * (uu + uu * radial) + cx, f * (vv + vv * radial) + cy
    if model == "RADIAL_FISHEYE":
        f, cx, cy, k1, k2 = _require_colmap_params(model, params, 5)
        uu, vv = _normal_to_fisheye(x, y)
        theta2 = uu * uu + vv * vv
        radial = k1 * theta2 + k2 * theta2 * theta2
        return f * (uu + uu * radial) + cx, f * (vv + vv * radial) + cy
    if model == "OPENCV_FISHEYE":
        fx, fy, cx, cy, k1, k2, k3, k4 = _require_colmap_params(model, params, 8)
        uu, vv = _normal_to_fisheye(x, y)
        theta2 = uu * uu + vv * vv
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        radial = k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8
        return fx * (uu + uu * radial) + cx, fy * (vv + vv * radial) + cy
    raise ValueError(
        f"Unsupported COLMAP camera model {model!r}; undistort the images/cameras first or add projection support for this model."
    )


def intrinsics_need_nonlinear_projection(intrinsics: dict[str, Any], *, atol: float = 0.0) -> bool:
    """Return True when pinhole-only rendering would drop COLMAP projection effects."""
    model = str(intrinsics.get("model") or "PINHOLE").upper()
    if model in LINEAR_COLMAP_MODELS:
        return False
    if model in FISHEYE_COLMAP_MODELS:
        return True
    indices = DISTORTED_COLMAP_PARAM_INDICES.get(model)
    if indices is None:
        return True
    params = _intrinsic_params(intrinsics)
    if params.shape[0] <= max(indices):
        return True
    return bool(np.any(np.abs(params[list(indices)]) > atol))


def colmap_model_and_params(
    intrinsics: dict[str, Any], *, fx: float, fy: float, cx: float, cy: float
) -> tuple[str, tuple[float, ...]]:
    """Return the COLMAP camera model/params to export without dropping distortion."""
    model = str(intrinsics.get("model") or "PINHOLE").upper()
    raw_params = intrinsics.get("params") or []
    if raw_params:
        params = tuple(float(value) for value in raw_params)
        validate_colmap_camera_params(model, params)
        return model, params
    if model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
        raise ValueError(f"Prepared camera model {model!r} has no params; exporting as PINHOLE would drop distortion.")
    return "PINHOLE", (float(fx), float(fy), float(cx), float(cy))


def install_colmap_camera_model_patches() -> None:
    """Patch legacy real-data paths so COLMAP distortion is not silently ignored."""
    global _PATCHED
    if _PATCHED:
        return

    from gpis_splatting import real_pipeline, real_scene

    real_scene._pinhole_intrinsics_from_colmap = colmap_intrinsics_from_params
    real_pipeline.project_splats_to_frame = _project_splats_to_frame

    try:
        from gpis_splatting import prepared_colmap_export

        prepared_colmap_export.build_colmap_cameras = _build_colmap_cameras
    except Exception:
        pass

    try:
        from gpis_splatting import gsplat_adapter

        if not getattr(gsplat_adapter.frame_to_gsplat_camera, "_colmap_distortion_guard", False):
            original = gsplat_adapter.frame_to_gsplat_camera

            def guarded_frame_to_gsplat_camera(frame: dict[str, Any], *, projection_convention: str, device: str | Any = "auto", dtype: str | Any = "float32"):
                intrinsics = frame["intrinsics"]
                if intrinsics_need_nonlinear_projection(intrinsics):
                    model = str(intrinsics.get("model") or "PINHOLE").upper()
                    raise ValueError(
                        f"gsplat_adapter currently renders with camera_model='pinhole'; camera model {model!r} "
                        "has COLMAP distortion/fisheye projection that would otherwise be ignored. "
                        "Use undistorted cameras/images or a renderer with this camera model."
                    )
                return original(frame, projection_convention=projection_convention, device=device, dtype=dtype)

            guarded_frame_to_gsplat_camera._colmap_distortion_guard = True
            gsplat_adapter.frame_to_gsplat_camera = guarded_frame_to_gsplat_camera
    except Exception:
        pass

    _PATCHED = True


def _project_splats_to_frame(
    splats: Any,
    frame: dict[str, Any],
    *,
    projection_convention: str,
    near_plane: float,
) -> dict[str, np.ndarray]:
    centers = splats.centers.detach().cpu().numpy()
    world_to_camera = np.asarray(frame["world_to_camera"], dtype=np.float64)
    if world_to_camera.shape != (4, 4):
        raise ValueError("Prepared frames must contain a 4x4 world_to_camera matrix.")
    homogeneous = np.concatenate((centers, np.ones((centers.shape[0], 1), dtype=np.float64)), axis=1)
    camera_xyz = homogeneous @ world_to_camera.T
    camera_xyz = camera_xyz[:, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        if projection_convention == "opencv":
            depth = camera_xyz[:, 2]
            x_norm = camera_xyz[:, 0] / depth
            y_norm = camera_xyz[:, 1] / depth
        elif projection_convention == "opengl":
            depth = -camera_xyz[:, 2]
            x_norm = camera_xyz[:, 0] / depth
            y_norm = -camera_xyz[:, 1] / depth
        else:
            raise ValueError("projection_convention must be 'opencv' or 'opengl'.")
        u, v = project_normalized_points_with_intrinsics(x_norm, y_norm, frame["intrinsics"])
    centers_px = np.stack((u, v), axis=1)
    valid = (depth > near_plane) & np.isfinite(depth) & np.isfinite(centers_px).all(axis=1)
    return {"centers_px": centers_px, "depth": depth, "valid": valid, "camera_xyz": camera_xyz}


def _build_colmap_cameras(frames: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    from gpis_splatting import prepared_colmap_export

    camera_ids_by_key: dict[tuple[float | int | str, ...], int] = {}
    camera_rows: list[dict[str, Any]] = []
    frame_camera_ids = []
    for frame in frames:
        intrinsics = frame.get("intrinsics") or {}
        width = prepared_colmap_export.require_int_dimension(frame, intrinsics, "width")
        height = prepared_colmap_export.require_int_dimension(frame, intrinsics, "height")
        fx = prepared_colmap_export.require_float_intrinsic(intrinsics, "fx")
        fy = prepared_colmap_export.require_float_intrinsic(intrinsics, "fy")
        cx = prepared_colmap_export.require_float_intrinsic(intrinsics, "cx")
        cy = prepared_colmap_export.require_float_intrinsic(intrinsics, "cy")
        model, params = colmap_model_and_params(intrinsics, fx=fx, fy=fy, cx=cx, cy=cy)
        key = (width, height, model, *params)
        camera_id = camera_ids_by_key.get(key)
        if camera_id is None:
            camera_id = len(camera_rows) + 1
            camera_ids_by_key[key] = camera_id
            camera_rows.append({"camera_id": camera_id, "model": model, "width": width, "height": height, "params": params})
        frame_camera_ids.append(camera_id)
    return camera_rows, frame_camera_ids


def _project_pinhole_fields(x: np.ndarray, y: np.ndarray, intrinsics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    fx = _required_intrinsic(intrinsics, "fx")
    fy = _required_intrinsic(intrinsics, "fy")
    cx = _required_intrinsic(intrinsics, "cx")
    cy = _required_intrinsic(intrinsics, "cy")
    return fx * x + cx, fy * y + cy


def _intrinsic_params(intrinsics: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(value) for value in intrinsics.get("params") or []], dtype=np.float64)


def _require_colmap_params(model: str, params: np.ndarray, expected_count: int) -> tuple[float, ...]:
    if params.shape[0] != expected_count:
        raise ValueError(f"COLMAP camera model {model!r} expects {expected_count} params, got {params.shape[0]}.")
    return tuple(float(value) for value in params)


def _required_intrinsic(intrinsics: dict[str, Any], key: str) -> float:
    value = intrinsics.get(key)
    if value is None:
        raise ValueError(f"Prepared camera is missing required intrinsic {key!r}.")
    return float(value)


def _opencv_distort(x: np.ndarray, y: np.ndarray, k1: float, k2: float, p1: float, p2: float) -> tuple[np.ndarray, np.ndarray]:
    x2 = x * x
    xy = x * y
    y2 = y * y
    r2 = x2 + y2
    radial = k1 * r2 + k2 * r2 * r2
    x_d = x + x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
    y_d = y + y * radial + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2)
    return x_d, y_d


def _full_opencv_distort(
    x: np.ndarray,
    y: np.ndarray,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    k3: float,
    k4: float,
    k5: float,
    k6: float,
) -> tuple[np.ndarray, np.ndarray]:
    x2 = x * x
    xy = x * y
    y2 = y * y
    r2 = x2 + y2
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6)
    x_d = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
    y_d = y * radial + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2)
    return x_d, y_d


def _normal_to_fisheye(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.sqrt(x * x + y * y)
    scale = np.ones_like(radius, dtype=np.float64)
    np.divide(np.arctan(radius), radius, out=scale, where=radius > np.finfo(np.float64).eps)
    return x * scale, y * scale
