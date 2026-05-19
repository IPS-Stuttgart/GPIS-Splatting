from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from gpis_splatting.external_3dgs import load_3dgs_ply, opacity_to_alpha, vertex_colors
from gpis_splatting.real_pipeline import PROJECTION_CONVENTIONS, resolve_frame_indices, resolve_projection_convention
from gpis_splatting.real_scene import load_prepared_scene, resolve_frame_output_names
from gpis_splatting.renderer import save_image
from gpis_splatting.serialization import write_json

RasterizationFn = Callable[..., Any]
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
TORCH_DTYPES = {"float32": torch.float32, "fp32": torch.float32, "float64": torch.float64, "fp64": torch.float64}


class GsplatUnavailableError(RuntimeError):
    """Raised when the optional gsplat backend is requested but unavailable."""


@dataclass(frozen=True)
class GsplatCamera:
    viewmat: torch.Tensor
    K: torch.Tensor
    width: int
    height: int
    projection_convention: str


@dataclass(frozen=True)
class GsplatGaussianTensors:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    source_gaussian_count: int

    @property
    def gaussian_count(self) -> int:
        return int(self.means.shape[0])


def resolve_gsplat_rasterization() -> RasterizationFn:
    try:
        from gsplat.rendering import rasterization  # type: ignore[import-not-found]

        return rasterization
    except (ImportError, AttributeError):
        try:
            from gsplat import rasterization  # type: ignore[import-not-found]

            return rasterization
        except (ImportError, AttributeError) as exc:
            raise GsplatUnavailableError("Install the optional gsplat extra with `pip install -e .[gsplat]` to render trained 3DGS models.") from exc


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
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    near_plane: float = 1e-2,
    far_plane: float = 1.0e10,
    eps2d: float = 0.3,
    tile_size: int = 16,
    packed: bool = True,
    render_mode: str = "RGB",
    max_frames: int | None = None,
    max_gaussians: int | None = None,
    rasterization_fn: RasterizationFn | None = None,
) -> dict[str, Any]:
    validate_render_options(split=split, projection_convention=projection_convention, near_plane=near_plane, far_plane=far_plane, eps2d=eps2d, tile_size=tile_size, render_mode=render_mode)
    scene_root = Path(scene_dir)
    scene_meta, frames, splits = load_prepared_scene(scene_root)
    frame_indices = resolve_frame_indices(splits, frame_count=len(frames), split=split)
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    if not frame_indices:
        raise ValueError(f"Split {split!r} did not resolve to any frames.")

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_torch_dtype(dtype)
    gaussians = gaussian_ply_to_gsplat_tensors(load_3dgs_ply(input_ply_path), device=resolved_device, dtype=resolved_dtype, opacity_mode=opacity_mode, max_gaussians=max_gaussians)
    rasterizer = rasterization_fn or resolve_gsplat_rasterization()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    convention = resolve_projection_convention(scene_meta, projection_convention)
    background = torch.tensor(background_color, dtype=resolved_dtype, device=resolved_device).reshape(1, 3)

    output_file_names = resolve_frame_output_names(frames, frame_indices)
    outputs = []
    for frame_index in frame_indices:
        frame = frames[int(frame_index)]
        camera = frame_to_gsplat_camera(frame, projection_convention=convention, device=resolved_device, dtype=resolved_dtype)
        image, render_stats = render_frame_with_gsplat(
            gaussians=gaussians,
            camera=camera,
            rasterization_fn=rasterizer,
            background=background,
            near_plane=near_plane,
            far_plane=far_plane,
            eps2d=eps2d,
            tile_size=tile_size,
            packed=packed,
            render_mode=render_mode,
        )
        file_name = output_file_names[int(frame_index)]
        image_path = out_dir / file_name
        save_image(image_path, image)
        outputs.append({"frame_index": int(frame_index), "image_path": frame.get("image_path"), "prediction_path": str(image_path), "file_name": file_name, "width": camera.width, "height": camera.height, **render_stats})

    report_path = out_dir / "gsplat_render_report.json"
    report = {
        "schema_version": 1,
        "backend": "gsplat",
        "scene": scene_meta.get("scene"),
        "scene_dir": str(scene_root),
        "input_ply_path": str(Path(input_ply_path)),
        "output_dir": str(out_dir),
        "split": split,
        "projection_convention": convention,
        "device": str(resolved_device),
        "dtype": str(resolved_dtype).replace("torch.", ""),
        "source_gaussian_count": gaussians.source_gaussian_count,
        "rendered_gaussian_count": gaussians.gaussian_count,
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
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    near_plane: float = 1e-2,
    far_plane: float = 1.0e10,
    eps2d: float = 0.3,
    tile_size: int = 16,
    packed: bool = True,
    render_mode: str = "RGB",
    max_frames: int | None = None,
    max_gaussians: int | None = None,
    rasterization_fn: RasterizationFn | None = None,
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
            background_color=background_color,
            near_plane=near_plane,
            far_plane=far_plane,
            eps2d=eps2d,
            tile_size=tile_size,
            packed=packed,
            render_mode=render_mode,
            max_frames=max_frames,
            max_gaussians=max_gaussians,
            rasterization_fn=rasterizer,
        )
        rows.append({"method": method_name, "variant": variant, "variant_kind": getattr(row, "variant_kind", None), "point_cloud_path": str(point_cloud_path), "predictions_dir": str(result["output_dir"]), "prediction_subdir": str(prediction_subdir), "iteration": iteration, "image_count": int(result["report"]["image_count"]), "rendered_gaussian_count": int(result["report"]["rendered_gaussian_count"]), "report_path": str(result["report_path"])})

    render_manifest = pd.DataFrame(rows)
    render_manifest_path = out_root / f"{method_name}_gsplat_render_manifest.csv"
    status_path = out_root / f"{method_name}_gsplat_render_status.json"
    report_path = out_root / f"{method_name}_gsplat_render_report.md"
    render_manifest.to_csv(render_manifest_path, index=False)
    status = {"schema_version": 1, "backend": "gsplat", "method": method_name, "manifest_path": str(Path(manifest_path)), "scene_dir": str(Path(scene_dir)), "output_root": str(out_root), "split": split, "variant_count": int(len(render_manifest)), "render_manifest_path": str(render_manifest_path), "report_path": str(report_path)}
    write_json(status_path, status)
    report_path.write_text(format_gsplat_manifest_report(status, render_manifest), encoding="utf-8")
    return {"render_manifest_path": render_manifest_path, "status_path": status_path, "report_path": report_path, "manifest": render_manifest, "status": status}


def gaussian_ply_to_gsplat_tensors(ply: Any, *, device: str | torch.device, dtype: str | torch.dtype, opacity_mode: str, max_gaussians: int | None = None) -> GsplatGaussianTensors:
    vertices = ply.vertices if max_gaussians is None else ply.vertices[:max_gaussians]
    names = set(vertices.dtype.names or ())
    count = int(vertices.shape[0])
    means = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    scales = np.exp(np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1).astype(np.float32)) if {"scale_0", "scale_1", "scale_2"}.issubset(names) else np.full((count, 3), 0.01, dtype=np.float32)
    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names):
        quats = np.stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=1).astype(np.float32)
        quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
    else:
        quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1))
    opacities = opacity_to_alpha(vertices["opacity"].astype(np.float64), opacity_mode=opacity_mode).astype(np.float32) if "opacity" in names else np.ones((count,), dtype=np.float32)
    colors = vertex_colors(vertices).astype(np.float32)
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_torch_dtype(dtype)
    return GsplatGaussianTensors(
        means=torch.as_tensor(means, dtype=resolved_dtype, device=resolved_device),
        quats=torch.as_tensor(quats, dtype=resolved_dtype, device=resolved_device),
        scales=torch.as_tensor(scales, dtype=resolved_dtype, device=resolved_device),
        opacities=torch.as_tensor(opacities, dtype=resolved_dtype, device=resolved_device),
        colors=torch.as_tensor(colors, dtype=resolved_dtype, device=resolved_device),
        source_gaussian_count=int(ply.vertex_count),
    )


def frame_to_gsplat_camera(frame: dict[str, Any], *, projection_convention: str, device: str | torch.device = "auto", dtype: str | torch.dtype = "float32") -> GsplatCamera:
    if projection_convention not in {"opencv", "opengl"}:
        raise ValueError("projection_convention must be resolved to 'opencv' or 'opengl'.")
    intrinsics = frame["intrinsics"]
    width = int(frame.get("width") or intrinsics.get("width"))
    height = int(frame.get("height") or intrinsics.get("height"))
    world_to_camera = np.asarray(frame["world_to_camera"], dtype=np.float64)
    if projection_convention == "opengl":
        world_to_camera = OPENGL_TO_OPENCV @ world_to_camera
    K = np.asarray([[required_intrinsic(intrinsics, "fx"), 0.0, required_intrinsic(intrinsics, "cx")], [0.0, required_intrinsic(intrinsics, "fy"), required_intrinsic(intrinsics, "cy")], [0.0, 0.0, 1.0]], dtype=np.float64)
    return GsplatCamera(torch.as_tensor(world_to_camera, dtype=resolve_torch_dtype(dtype), device=resolve_device(device)), torch.as_tensor(K, dtype=resolve_torch_dtype(dtype), device=resolve_device(device)), width, height, projection_convention)


def render_frame_with_gsplat(*, gaussians: GsplatGaussianTensors, camera: GsplatCamera, rasterization_fn: RasterizationFn, background: torch.Tensor | None = None, near_plane: float = 1e-2, far_plane: float = 1.0e10, eps2d: float = 0.3, tile_size: int = 16, packed: bool = True, render_mode: str = "RGB") -> tuple[torch.Tensor, dict[str, Any]]:
    kwargs = {"means": gaussians.means, "quats": gaussians.quats, "scales": gaussians.scales, "opacities": gaussians.opacities, "colors": gaussians.colors, "viewmats": camera.viewmat.unsqueeze(0), "Ks": camera.K.unsqueeze(0), "width": int(camera.width), "height": int(camera.height), "packed": packed, "near_plane": near_plane, "far_plane": far_plane, "eps2d": eps2d, "tile_size": tile_size, "render_mode": render_mode, "camera_model": "pinhole"}
    if background is not None:
        kwargs["backgrounds"] = background
    try:
        output = rasterization_fn(**kwargs)
    except TypeError:
        core_keys = ("means", "quats", "scales", "opacities", "colors", "viewmats", "Ks", "width", "height")
        output = rasterization_fn(**{key: kwargs[key] for key in core_keys})
    image = extract_rgb_image(output)
    return image, {"rendered_splat_count": gaussians.gaussian_count, "output_channels": int(image.shape[-1])}


def extract_rgb_image(output: Any) -> torch.Tensor:
    rendered = output[0] if isinstance(output, tuple) else output
    if isinstance(rendered, dict):
        rendered = next((rendered[key] for key in ("render_colors", "colors", "rgb", "image") if key in rendered), None)
    if rendered is None:
        raise ValueError("gsplat rasterization did not return an RGB image.")
    image = rendered if isinstance(rendered, torch.Tensor) else torch.as_tensor(rendered)
    image = image[0] if image.ndim == 4 else image
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected HxWxC image, got {tuple(image.shape)}.")
    return image[..., :3].clamp(0.0, 1.0).detach().cpu()


def infer_iteration_from_point_cloud_path(path: str | Path) -> int:
    match = re.search(r"iteration_(\d+)", str(path))
    return int(match.group(1)) if match else 0


def validate_render_options(*, split: str, projection_convention: str, near_plane: float, far_plane: float, eps2d: float, tile_size: int, render_mode: str) -> None:
    if not split or projection_convention not in PROJECTION_CONVENTIONS or near_plane <= 0.0 or far_plane <= near_plane or eps2d < 0.0 or tile_size <= 0 or not render_mode:
        raise ValueError("Invalid gsplat rendering options.")


def validate_gsplat_manifest(manifest: pd.DataFrame) -> None:
    missing = sorted({"variant", "point_cloud_path"} - set(manifest.columns))
    if missing:
        raise ValueError(f"3DGS variant manifest is missing columns: {', '.join(missing)}.")
    if manifest.empty:
        raise ValueError("3DGS variant manifest is empty.")


def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device("cuda" if device.strip().lower() == "auto" and torch.cuda.is_available() else ("cpu" if device.strip().lower() == "auto" else device))


def resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    key = dtype.strip().lower()
    if key not in TORCH_DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}.")
    return TORCH_DTYPES[key]


def required_intrinsic(intrinsics: dict[str, Any], key: str) -> float:
    value = intrinsics.get(key)
    if value is None:
        raise ValueError(f"Prepared camera is missing required pinhole intrinsic {key!r}.")
    return float(value)


def format_gsplat_manifest_report(status: dict[str, Any], manifest: pd.DataFrame) -> str:
    lines = ["# gsplat 3DGS Render Report", "", f"- Backend: `{status['backend']}`", f"- Method: `{status['method']}`", f"- Source manifest: `{status['manifest_path']}`", f"- Scene: `{status['scene_dir']}`", f"- Output root: `{status['output_root']}`", f"- Split: `{status['split']}`", f"- Variants rendered: `{status['variant_count']}`", f"- Render manifest: `{status['render_manifest_path']}`", "", "## Variants", "", "| variant | kind | iteration | images | gaussians | predictions_dir |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for row in manifest.itertuples(index=False):
        lines.append(f"| `{row.variant}` | `{row.variant_kind}` | {row.iteration} | {row.image_count} | {row.rendered_gaussian_count} | `{row.predictions_dir}` |")
    return "\n".join(lines) + "\n"
