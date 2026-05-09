from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gpis_splatting.external_3dgs import load_3dgs_ply, opacity_to_alpha, vertex_colors
from gpis_splatting.gsplat_adapter import (
    GsplatGaussianTensors,
    format_gsplat_manifest_report,
    frame_to_gsplat_camera,
    infer_iteration_from_point_cloud_path,
    resolve_device,
    resolve_gsplat_rasterization,
    resolve_torch_dtype,
    validate_gsplat_manifest,
)
from gpis_splatting.real_pipeline import resolve_frame_indices, resolve_projection_convention
from gpis_splatting.real_scene import load_prepared_scene
from gpis_splatting.renderer import save_image
from gpis_splatting.serialization import write_json

SH_COLOR_MODES = ("auto", "sh", "rgb")
BACKGROUND_MODES = ("auto", "black", "white", "rgb")
RASTERIZE_MODES = ("classic", "antialiased")


def render_3dgs_ply_with_gsplat(
    *,
    input_ply_path: str | Path,
    scene_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
    projection_convention: str = "auto",
    device: str = "auto",
    dtype: str | torch.dtype = "float32",
    opacity_mode: str = "logit",
    color_mode: str = "auto",
    sh_degree: int | str | None = "auto",
    strict_3dgs_fidelity: bool = True,
    background_mode: str = "auto",
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    near_plane: float = 1e-2,
    far_plane: float = 1.0e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    tile_size: int = 16,
    packed: bool = True,
    render_mode: str = "RGB",
    rasterize_mode: str = "classic",
    channel_chunk: int = 32,
    max_frames: int | None = None,
    max_gaussians: int | None = None,
    rasterization_fn: Any | None = None,
) -> dict[str, Any]:
    scene_root = Path(scene_dir)
    scene_meta, frames, splits = load_prepared_scene(scene_root)
    indices = resolve_frame_indices(splits, frame_count=len(frames), split=split)
    if max_frames is not None:
        indices = indices[:max_frames]
    if not indices:
        raise ValueError(f"Split {split!r} did not resolve to any frames.")
    dev = resolve_device(device)
    dt = resolve_torch_dtype(dtype)
    gaussians, color_info = gaussian_ply_to_gsplat_tensors(
        load_3dgs_ply(input_ply_path),
        device=dev,
        dtype=dt,
        opacity_mode=opacity_mode,
        color_mode=color_mode,
        sh_degree=sh_degree,
        strict_3dgs_fidelity=strict_3dgs_fidelity,
        max_gaussians=max_gaussians,
    )
    rasterizer = rasterization_fn or resolve_gsplat_rasterization()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    convention = resolve_projection_convention(scene_meta, projection_convention)
    bg = resolve_background(background_mode=background_mode, background_color=background_color)
    background = torch.tensor(bg, dtype=dt, device=dev).reshape(1, 3)
    outputs = []
    for frame_index in indices:
        frame = frames[int(frame_index)]
        camera = frame_to_gsplat_camera(frame, projection_convention=convention, device=dev, dtype=dt)
        kwargs = {
            "means": gaussians.means,
            "quats": gaussians.quats,
            "scales": gaussians.scales,
            "opacities": gaussians.opacities,
            "colors": gaussians.colors,
            "viewmats": camera.viewmat.unsqueeze(0),
            "Ks": camera.K.unsqueeze(0),
            "width": camera.width,
            "height": camera.height,
            "packed": packed,
            "near_plane": near_plane,
            "far_plane": far_plane,
            "radius_clip": radius_clip,
            "eps2d": eps2d,
            "tile_size": tile_size,
            "render_mode": render_mode,
            "rasterize_mode": rasterize_mode,
            "channel_chunk": channel_chunk,
            "camera_model": "pinhole",
            "backgrounds": background,
        }
        if color_info["effective_sh_degree"] is not None:
            kwargs["sh_degree"] = int(color_info["effective_sh_degree"])
        try:
            rendered = rasterizer(**kwargs)
        except TypeError:
            for key in ("radius_clip", "rasterize_mode", "channel_chunk"):
                kwargs.pop(key, None)
            rendered = rasterizer(**kwargs)
        image = extract_rgb_image(rendered)
        file_name = str(frame.get("file_name") or Path(str(frame["image_path"])).name)
        image_path = out_dir / file_name
        save_image(image_path, image)
        outputs.append({"frame_index": int(frame_index), "prediction_path": str(image_path), "file_name": file_name, "width": camera.width, "height": camera.height})
    report_path = out_dir / "gsplat_render_report.json"
    report = {
        "schema_version": 2,
        "backend": "gsplat",
        "input_ply_path": str(Path(input_ply_path)),
        "scene_dir": str(scene_root),
        "output_dir": str(out_dir),
        "split": split,
        "projection_convention": convention,
        "source_gaussian_count": gaussians.source_gaussian_count,
        "rendered_gaussian_count": gaussians.gaussian_count,
        "background_color": bg,
        "color": color_info,
        "image_count": len(outputs),
        "outputs": outputs,
    }
    write_json(report_path, report)
    return {"output_dir": out_dir, "report_path": report_path, "report": report}


def render_3dgs_manifest_with_gsplat(
    *,
    manifest_path: str | Path,
    scene_dir: str | Path,
    output_root: str | Path,
    method_name: str = "trained_3dgs_gsplat",
    split: str = "test",
    projection_convention: str = "auto",
    device: str = "auto",
    dtype: str | torch.dtype = "float32",
    opacity_mode: str = "logit",
    color_mode: str = "auto",
    sh_degree: int | str | None = "auto",
    strict_3dgs_fidelity: bool = True,
    background_mode: str = "auto",
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    near_plane: float = 1e-2,
    far_plane: float = 1.0e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    tile_size: int = 16,
    packed: bool = True,
    render_mode: str = "RGB",
    rasterize_mode: str = "classic",
    channel_chunk: int = 32,
    max_frames: int | None = None,
    max_gaussians: int | None = None,
    rasterization_fn: Any | None = None,
) -> dict[str, Any]:
    manifest = pd.read_csv(manifest_path)
    validate_gsplat_manifest(manifest)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rasterizer = rasterization_fn or resolve_gsplat_rasterization()
    rows = []
    for row in manifest.itertuples(index=False):
        variant = str(row.variant)
        point_cloud_path = Path(str(row.point_cloud_path))
        iteration = infer_iteration_from_point_cloud_path(point_cloud_path)
        prediction_subdir = Path(split) / f"ours_{iteration}" / "renders"
        result = render_3dgs_ply_with_gsplat(
            input_ply_path=point_cloud_path,
            scene_dir=scene_dir,
            output_dir=out_root / variant / prediction_subdir,
            split=split,
            projection_convention=projection_convention,
            device=device,
            dtype=dtype,
            opacity_mode=opacity_mode,
            color_mode=color_mode,
            sh_degree=sh_degree,
            strict_3dgs_fidelity=strict_3dgs_fidelity,
            background_mode=background_mode,
            background_color=background_color,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            eps2d=eps2d,
            tile_size=tile_size,
            packed=packed,
            render_mode=render_mode,
            rasterize_mode=rasterize_mode,
            channel_chunk=channel_chunk,
            max_frames=max_frames,
            max_gaussians=max_gaussians,
            rasterization_fn=rasterizer,
        )
        rows.append({
            "method": method_name,
            "variant": variant,
            "variant_kind": getattr(row, "variant_kind", None),
            "point_cloud_path": str(point_cloud_path),
            "predictions_dir": str(result["output_dir"]),
            "prediction_subdir": str(prediction_subdir),
            "iteration": iteration,
            "image_count": int(result["report"]["image_count"]),
            "rendered_gaussian_count": int(result["report"]["rendered_gaussian_count"]),
            "color_mode": result["report"]["color"]["effective_color_mode"],
            "sh_degree": result["report"]["color"]["effective_sh_degree"],
            "report_path": str(result["report_path"]),
        })
    render_manifest = pd.DataFrame(rows)
    render_manifest_path = out_root / f"{method_name}_gsplat_render_manifest.csv"
    status_path = out_root / f"{method_name}_gsplat_render_status.json"
    report_path = out_root / f"{method_name}_gsplat_render_report.md"
    render_manifest.to_csv(render_manifest_path, index=False)
    status = {
        "schema_version": 2,
        "backend": "gsplat",
        "method": method_name,
        "manifest_path": str(Path(manifest_path)),
        "scene_dir": str(Path(scene_dir)),
        "output_root": str(out_root),
        "split": split,
        "variant_count": int(len(render_manifest)),
        "render_manifest_path": str(render_manifest_path),
        "report_path": str(report_path),
    }
    write_json(status_path, status)
    report_path.write_text(format_gsplat_manifest_report(status, render_manifest), encoding="utf-8")
    return {"render_manifest_path": render_manifest_path, "status_path": status_path, "report_path": report_path, "manifest": render_manifest, "status": status}


def gaussian_ply_to_gsplat_tensors(
    ply: Any,
    *,
    device: str | torch.device,
    dtype: str | torch.dtype,
    opacity_mode: str,
    color_mode: str = "auto",
    sh_degree: int | str | None = "auto",
    strict_3dgs_fidelity: bool = True,
    max_gaussians: int | None = None,
) -> tuple[GsplatGaussianTensors, dict[str, Any]]:
    v = ply.vertices if max_gaussians is None else ply.vertices[:max_gaussians]
    names = set(v.dtype.names or ())
    missing = sorted({"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"} - names)
    if strict_3dgs_fidelity and missing:
        raise ValueError(f"Strict 3DGS fidelity requested, but {ply.path} is missing: {', '.join(missing)}.")
    n = int(v.shape[0])
    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)) if {"scale_0", "scale_1", "scale_2"}.issubset(names) else np.full((n, 3), 0.01, dtype=np.float32)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32) if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names) else np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1))
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
    opacities = opacity_to_alpha(v["opacity"].astype(np.float64), opacity_mode=opacity_mode).astype(np.float32) if "opacity" in names else np.ones((n,), dtype=np.float32)
    colors, info = resolve_colors(v, color_mode=color_mode, requested_sh_degree=sh_degree, strict=strict_3dgs_fidelity)
    dt = resolve_torch_dtype(dtype)
    dev = resolve_device(device)
    return GsplatGaussianTensors(
        torch.as_tensor(means, dtype=dt, device=dev),
        torch.as_tensor(quats, dtype=dt, device=dev),
        torch.as_tensor(scales, dtype=dt, device=dev),
        torch.as_tensor(opacities, dtype=dt, device=dev),
        torch.as_tensor(colors, dtype=dt, device=dev),
        int(ply.vertex_count),
    ), info


def resolve_colors(v: np.ndarray, *, color_mode: str, requested_sh_degree: int | str | None, strict: bool) -> tuple[np.ndarray, dict[str, Any]]:
    names = set(v.dtype.names or ())
    rest_names = sorted((name for name in names if name.startswith("f_rest_")), key=lambda name: int(name.split("_")[-1]))
    has_dc = {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names)
    has_sh = has_dc and bool(rest_names)
    mode = "sh" if color_mode == "auto" and has_sh else ("rgb" if color_mode == "auto" else color_mode)
    max_degree = infer_max_sh_degree(rest_names)
    if mode == "rgb" or not has_dc:
        if strict and mode == "sh":
            raise ValueError("Strict SH rendering requested, but f_dc_0..2 are missing.")
        return vertex_colors(v).astype(np.float32), {"requested_color_mode": color_mode, "effective_color_mode": "rgb", "requested_sh_degree": requested_sh_degree, "effective_sh_degree": None, "max_sh_degree": max_degree, "color_layout": "RGB [N,3]"}
    degree = max_degree if requested_sh_degree in (None, "auto") else int(requested_sh_degree)
    if degree > max_degree:
        if strict:
            raise ValueError(f"Requested SH degree {degree}, but PLY stores only degree {max_degree}.")
        degree = max_degree
    coeff_count = (degree + 1) ** 2
    colors = np.zeros((v.shape[0], coeff_count, 3), dtype=np.float32)
    colors[:, 0, :] = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    if coeff_count > 1:
        required_rest = 3 * (coeff_count - 1)
        if len(rest_names) < required_rest:
            if strict:
                raise ValueError(f"Requested SH degree {degree}, but only {len(rest_names)} f_rest coefficients are available.")
            required_rest = len(rest_names) - (len(rest_names) % 3)
        rest = np.stack([v[name] for name in rest_names[:required_rest]], axis=1).astype(np.float32)
        colors[:, 1 : 1 + required_rest // 3, :] = np.transpose(rest.reshape(v.shape[0], 3, required_rest // 3), (0, 2, 1))
    return colors, {"requested_color_mode": color_mode, "effective_color_mode": "sh", "requested_sh_degree": requested_sh_degree, "effective_sh_degree": degree, "max_sh_degree": max_degree, "color_layout": "SH coefficients [N,K,3]"}


def infer_max_sh_degree(rest_names: list[str]) -> int:
    if not rest_names:
        return 0
    total = len(rest_names) // 3 + 1
    for degree in range(8):
        if (degree + 1) ** 2 == total:
            return degree
    return max(0, int(np.floor(np.sqrt(total))) - 1)


def resolve_background(*, background_mode: str, background_color: tuple[float, float, float]) -> tuple[float, float, float]:
    if background_mode == "white":
        return (1.0, 1.0, 1.0)
    if background_mode == "black":
        return (0.0, 0.0, 0.0)
    return tuple(float(v) for v in background_color)


def extract_rgb_image(output: Any) -> torch.Tensor:
    rendered = output[0] if isinstance(output, tuple) else output
    if isinstance(rendered, dict):
        rendered = next((rendered[key] for key in ("render_colors", "colors", "rgb", "image") if key in rendered), None)
    if rendered is None:
        raise ValueError("gsplat rasterization did not return an RGB image.")
    image = rendered if isinstance(rendered, torch.Tensor) else torch.as_tensor(rendered)
    image = image[0] if image.ndim == 4 else image
    return image[..., :3].clamp(0.0, 1.0).detach().cpu()
