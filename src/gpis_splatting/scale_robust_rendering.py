from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image

from gpis_splatting.external_3dgs import format_optional_number, load_3dgs_ply
from gpis_splatting.gsplat_adapter import frame_to_gsplat_camera, infer_iteration_from_point_cloud_path, resolve_device, resolve_gsplat_rasterization, resolve_torch_dtype, validate_gsplat_manifest
from gpis_splatting.gsplat_fidelity_adapter import RASTERIZE_MODES, gaussian_ply_to_gsplat_tensors, render_gsplat_frame_image, resolve_background
from gpis_splatting.real_benchmark import _load_lpips_model, lpips_arrays, psnr_arrays, ssim_arrays
from gpis_splatting.real_pipeline import resolve_frame_indices, resolve_projection_convention
from gpis_splatting.real_scene import load_prepared_scene, resolve_scene_image_path
from gpis_splatting.renderer import load_image, save_image
from gpis_splatting.serialization import write_json

DEFAULT_SCALE_FACTORS = (0.5, 1.0, 2.0)
DEFAULT_RASTERIZE_MODES = ("classic", "antialiased")


def run_scale_robust_3dgs_experiment(
    *,
    scene_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    input_ply_path: str | Path | None = None,
    method_name: str = "scale_robust_3dgs",
    split: str = "test",
    scales: Iterable[float] = DEFAULT_SCALE_FACTORS,
    rasterize_modes: Iterable[str] = DEFAULT_RASTERIZE_MODES,
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
    channel_chunk: int = 32,
    max_frames: int | None = None,
    max_gaussians: int | None = None,
    compute_lpips: bool = False,
    rasterization_fn: Any | None = None,
) -> dict[str, Any]:
    if (manifest_path is None) == (input_ply_path is None):
        raise ValueError("Pass exactly one of manifest_path or input_ply_path.")
    scale_values = _scales(scales)
    mode_values = _modes(rasterize_modes)
    manifest = _manifest(manifest_path, input_ply_path, method_name)
    validate_gsplat_manifest(manifest)
    scene_root = Path(scene_dir)
    out_dir = Path(output_dir)
    render_root = out_dir / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_root.mkdir(parents=True, exist_ok=True)
    rasterizer = rasterization_fn or resolve_gsplat_rasterization()

    render_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for rasterize_mode in mode_values:
        for scale in scale_values:
            label = scale_label(scale)
            for row in manifest.itertuples(index=False):
                point_cloud_path = Path(str(row.point_cloud_path))
                variant = str(row.variant)
                iteration = infer_iteration_from_point_cloud_path(point_cloud_path)
                rel_dir = Path(rasterize_mode) / label / variant / split / f"ours_{iteration}" / "renders"
                result = render_scaled_3dgs_ply_with_gsplat(
                    input_ply_path=point_cloud_path,
                    scene_dir=scene_root,
                    output_dir=render_root / rel_dir,
                    scale=scale,
                    scale_label_value=label,
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
                render_rows.append(_render_row(row, method_name, point_cloud_path, rel_dir, rasterize_mode, scale, label, result))
                metric_rows.extend(
                    evaluate_scaled_render_report(
                        scene_dir=scene_root,
                        render_report=result["report"],
                        method_name=method_name,
                        variant=variant,
                        variant_kind=str(getattr(row, "variant_kind", "baseline")),
                        rasterize_mode=rasterize_mode,
                        scale=scale,
                        scale_label=label,
                        compute_lpips=compute_lpips,
                    )
                )

    render_manifest = pd.DataFrame(render_rows)
    metrics = pd.DataFrame(metric_rows)
    summary = summarize_scale_robust_metrics(metrics, render_manifest=render_manifest)
    paths = {
        "render_manifest_path": out_dir / f"{method_name}_scale_robust_render_manifest.csv",
        "metrics_path": out_dir / f"{method_name}_scale_robust_image_metrics.csv",
        "summary_path": out_dir / f"{method_name}_scale_robust_summary.csv",
        "status_path": out_dir / f"{method_name}_scale_robust_status.json",
        "report_path": out_dir / f"{method_name}_scale_robust_report.md",
    }
    render_manifest.to_csv(paths["render_manifest_path"], index=False)
    metrics.to_csv(paths["metrics_path"], index=False)
    summary.to_csv(paths["summary_path"], index=False)
    status = {
        "schema_version": 1,
        "method": method_name,
        "scene_dir": str(scene_root),
        "manifest_path": str(Path(manifest_path)) if manifest_path is not None else None,
        "input_ply_path": str(Path(input_ply_path)) if input_ply_path is not None else None,
        "output_dir": str(out_dir),
        "render_root": str(render_root),
        "split": split,
        "scales": list(scale_values),
        "rasterize_modes": list(mode_values),
        "compute_lpips": bool(compute_lpips),
        "variant_count": int(len(manifest)),
        "render_count": int(len(render_manifest)),
        "image_metric_count": int(len(metrics)),
        **{k: str(v) for k, v in paths.items() if k.endswith("_path")},
    }
    write_json(paths["status_path"], status)
    paths["report_path"].write_text(format_scale_robust_report(status, summary), encoding="utf-8")
    return {**paths, "render_manifest": render_manifest, "metrics": metrics, "summary": summary, "status": status}


def render_scaled_3dgs_ply_with_gsplat(
    *,
    input_ply_path: str | Path,
    scene_dir: str | Path,
    output_dir: str | Path,
    scale: float,
    scale_label_value: str | None = None,
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
    if scale <= 0.0 or rasterize_mode not in RASTERIZE_MODES:
        raise ValueError("Invalid scale or rasterize_mode.")
    scene_root = Path(scene_dir)
    scene_meta, frames, splits = load_prepared_scene(scene_root)
    indices = resolve_frame_indices(splits, frame_count=len(frames), split=split)
    indices = indices if max_frames is None else indices[:max_frames]
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
    convention = resolve_projection_convention(scene_meta, projection_convention)
    background = torch.tensor(resolve_background(background_mode=background_mode, background_color=background_color), dtype=dt, device=dev).reshape(1, 3)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for frame_index in indices:
        frame = scale_prepared_frame(frames[int(frame_index)], scale)
        camera = frame_to_gsplat_camera(frame, projection_convention=convention, device=dev, dtype=dt)
        image = render_gsplat_frame_image(
            gaussians=gaussians,
            camera=camera,
            color_info=color_info,
            rasterizer=rasterizer,
            background=background,
            packed=packed,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            eps2d=eps2d,
            tile_size=tile_size,
            render_mode=render_mode,
            rasterize_mode=rasterize_mode,
            channel_chunk=channel_chunk,
        )
        file_name = str(frame.get("file_name") or Path(str(frame["image_path"])).name)
        image_path = out_dir / file_name
        save_image(image_path, image)
        outputs.append({"frame_index": int(frame_index), "image_path": frame.get("image_path"), "file_name": file_name, "prediction_path": str(image_path), "width": camera.width, "height": camera.height})
    report = {
        "schema_version": 1,
        "backend": "gsplat",
        "input_ply_path": str(Path(input_ply_path)),
        "scene_dir": str(scene_root),
        "output_dir": str(out_dir),
        "split": split,
        "scale": float(scale),
        "scale_label": scale_label_value or scale_label(scale),
        "rasterize_mode": rasterize_mode,
        "projection_convention": convention,
        "source_gaussian_count": gaussians.source_gaussian_count,
        "rendered_gaussian_count": gaussians.gaussian_count,
        "color": color_info,
        "image_count": len(outputs),
        "outputs": outputs,
    }
    report_path = out_dir / "scale_robust_gsplat_render_report.json"
    write_json(report_path, report)
    return {"output_dir": out_dir, "report_path": report_path, "report": report}


def evaluate_scaled_render_report(*, scene_dir: str | Path, render_report: dict[str, Any], method_name: str, variant: str, variant_kind: str, rasterize_mode: str, scale: float, scale_label: str, compute_lpips: bool = False) -> list[dict[str, Any]]:
    lpips_model, lpips_status = _load_lpips_model(compute_lpips)
    rows = []
    for output in render_report["outputs"]:
        prediction = load_image(output["prediction_path"])
        target = resize_target_image(resolve_scene_image_path(scene_dir, str(output["image_path"])), width=int(output["width"]), height=int(output["height"]))
        rows.append({"method": method_name, "variant": variant, "variant_kind": variant_kind, "rasterize_mode": rasterize_mode, "scale": float(scale), "scale_label": scale_label, "frame_index": int(output["frame_index"]), "image_path": output["image_path"], "prediction_path": output["prediction_path"], "width": int(output["width"]), "height": int(output["height"]), "psnr": psnr_arrays(prediction, target), "ssim": ssim_arrays(prediction, target), "lpips_vgg": lpips_arrays(lpips_model, prediction, target) if lpips_model is not None else np.nan, "lpips_status": lpips_status})
    return rows


def summarize_scale_robust_metrics(metrics: pd.DataFrame, *, render_manifest: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["method", "variant", "variant_kind", "rasterize_mode", "scale", "scale_label"], dropna=False).agg(image_count=("frame_index", "count"), mean_psnr=("psnr", "mean"), mean_ssim=("ssim", "mean"), mean_lpips_vgg=("lpips_vgg", _mean_or_nan)).reset_index()
    meta = [col for col in ["method", "variant", "variant_kind", "rasterize_mode", "scale", "scale_label", "rendered_gaussian_count", "source_gaussian_count", "retained_count", "retention_fraction", "gate_threshold", "opacity_scaled"] if col in render_manifest.columns]
    summary = grouped.merge(render_manifest[meta].drop_duplicates(), on=["method", "variant", "variant_kind", "rasterize_mode", "scale", "scale_label"], how="left")
    base = summary[summary["rasterize_mode"] == "classic"][["variant", "scale", "mean_psnr", "mean_ssim", "mean_lpips_vgg"]].rename(columns={"mean_psnr": "classic_mean_psnr", "mean_ssim": "classic_mean_ssim", "mean_lpips_vgg": "classic_mean_lpips_vgg"})
    summary = summary.merge(base, on=["variant", "scale"], how="left")
    summary["delta_psnr_vs_classic"] = summary["mean_psnr"] - summary["classic_mean_psnr"]
    summary["delta_ssim_vs_classic"] = summary["mean_ssim"] - summary["classic_mean_ssim"]
    summary["delta_lpips_vgg_vs_classic"] = summary["mean_lpips_vgg"] - summary["classic_mean_lpips_vgg"]
    return summary


def scale_prepared_frame(frame: dict[str, Any], scale: float) -> dict[str, Any]:
    scaled = copy.deepcopy(frame)
    intrinsics = dict(scaled.get("intrinsics") or {})
    width = max(1, int(round(float(scaled.get("width") or intrinsics.get("width")) * scale)))
    height = max(1, int(round(float(scaled.get("height") or intrinsics.get("height")) * scale)))
    scaled["width"] = width
    scaled["height"] = height
    for key in ("fx", "fy", "cx", "cy"):
        if key in intrinsics and intrinsics[key] is not None:
            intrinsics[key] = float(intrinsics[key]) * scale
    intrinsics["width"] = width
    intrinsics["height"] = height
    scaled["intrinsics"] = intrinsics
    return scaled


def resize_target_image(path: str | Path, *, width: int, height: int) -> np.ndarray:
    target = load_image(path)
    if target.shape[1] == width and target.shape[0] == height:
        return target
    arr = (np.clip(target, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    resample = getattr(Image, "Resampling", Image).BICUBIC
    return np.asarray(Image.fromarray(arr, mode="RGB").resize((width, height), resample=resample).convert("RGB"), dtype=np.float64) / 255.0


def scale_label(scale: float) -> str:
    return f"scale_{float(scale):.6g}".replace(".", "p").replace("-", "m")


def format_scale_robust_report(status: dict[str, Any], summary: pd.DataFrame) -> str:
    lines = ["# Scale-Robust 3DGS Rendering", "", f"- Method: `{status['method']}`", f"- Scene: `{status['scene_dir']}`", f"- Split: `{status['split']}`", f"- Scales: `{', '.join(str(v) for v in status['scales'])}`", f"- Rasterize modes: `{', '.join(status['rasterize_modes'])}`", f"- Variants: `{status['variant_count']}`", f"- Render cells: `{status['render_count']}`", f"- Summary CSV: `{status['summary_path']}`"]
    if not summary.empty:
        lines.extend(["", "## Summary", "", _summary_table(summary)])
    return "\n".join(lines) + "\n"


def _manifest(manifest_path: str | Path | None, input_ply_path: str | Path | None, method_name: str) -> pd.DataFrame:
    if manifest_path is not None:
        return pd.read_csv(manifest_path)
    path = Path(str(input_ply_path))
    ply = load_3dgs_ply(path)
    return pd.DataFrame([{"variant": "baseline", "variant_kind": "baseline", "model_dir": str(path.parent.parent.parent) if path.parent.name.startswith("iteration_") else str(path.parent), "point_cloud_path": str(path), "retained_count": int(ply.vertex_count), "retention_fraction": 1.0, "gate_threshold": np.nan, "opacity_scaled": False, "gate_min": np.nan, "gate_max": np.nan, "gate_mean": np.nan, "method": method_name}])


def _render_row(row: Any, method_name: str, point_cloud_path: Path, prediction_subdir: Path, rasterize_mode: str, scale: float, label: str, result: dict[str, Any]) -> dict[str, Any]:
    report = result["report"]
    return {"method": method_name, "variant": str(row.variant), "variant_kind": str(getattr(row, "variant_kind", "baseline")), "rasterize_mode": rasterize_mode, "scale": float(scale), "scale_label": label, "point_cloud_path": str(point_cloud_path), "predictions_dir": str(result["output_dir"]), "prediction_subdir": str(prediction_subdir), "iteration": infer_iteration_from_point_cloud_path(point_cloud_path), "image_count": int(report["image_count"]), "source_gaussian_count": int(report["source_gaussian_count"]), "rendered_gaussian_count": int(report["rendered_gaussian_count"]), "retained_count": _attr(row, "retained_count"), "retention_fraction": _attr(row, "retention_fraction"), "gate_threshold": _attr(row, "gate_threshold"), "opacity_scaled": _attr(row, "opacity_scaled"), "report_path": str(result["report_path"])}


def _scales(scales: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(v) for v in scales)
    if not values or any(v <= 0.0 for v in values):
        raise ValueError("scales must contain positive values.")
    return values


def _modes(modes: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(mode).strip() for mode in modes if str(mode).strip())
    bad = sorted(set(values) - set(RASTERIZE_MODES))
    if not values or bad:
        raise ValueError(f"Unsupported rasterize modes: {bad}.")
    return values


def _attr(row: Any, name: str) -> Any:
    value = getattr(row, name, np.nan)
    return None if pd.isna(value) else value


def _mean_or_nan(values: pd.Series) -> float:
    valid = values.dropna()
    return float("nan") if valid.empty else float(valid.mean())


def _summary_table(summary: pd.DataFrame) -> str:
    lines = ["| variant | rasterize | scale | gaussians | retention | psnr | ssim | lpips | delta_psnr_vs_classic | delta_ssim_vs_classic |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| `{row.variant}` | `{row.rasterize_mode}` | {row.scale:.6g} | {format_optional_number(getattr(row, 'rendered_gaussian_count', None))} | {format_optional_number(getattr(row, 'retention_fraction', None))} | {format_optional_number(row.mean_psnr)} | {format_optional_number(row.mean_ssim)} | {format_optional_number(row.mean_lpips_vgg)} | {format_optional_number(row.delta_psnr_vs_classic)} | {format_optional_number(row.delta_ssim_vs_classic)} |")
    return "\n".join(lines)
