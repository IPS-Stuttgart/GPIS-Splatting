from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from gpis_splatting.gpis import load_model
from gpis_splatting.real_benchmark import find_prediction_image, psnr_arrays, ssim_arrays
from gpis_splatting.real_pipeline import (
    PROJECTION_CONVENTIONS,
    _resolve_scene_file,
    project_splats_to_frame,
    render_real_splat_image,
    resolve_frame_indices,
    resolve_projection_convention,
)
from gpis_splatting.real_scene import load_prepared_scene, resolve_scene_image_path
from gpis_splatting.renderer import load_image, save_image
from gpis_splatting.serialization import write_json
from gpis_splatting.splats import gpis_gate_for_splats, load_splats


def diagnose_real_render(
    *,
    scene_dir: str | Path,
    splats_path: str | Path | None = None,
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    plain_renders_dir: str | Path | None = None,
    gated_renders_dir: str | Path | None = None,
    split: str = "test",
    max_frames: int | None = 4,
    epsilon: float = 0.16,
    gate_floor: float = 0.0,
    projection_convention: str = "auto",
    near_plane: float = 1e-4,
    kernel_radius: float = 3.0,
    min_sigma_px: float = 0.8,
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gate_batch_size: int = 4096,
    max_overlay_splats: int = 2000,
) -> dict[str, Any]:
    if projection_convention not in PROJECTION_CONVENTIONS:
        raise ValueError(f"Unsupported projection convention {projection_convention!r}. Expected one of {', '.join(PROJECTION_CONVENTIONS)}.")
    if not 0.0 <= gate_floor <= 1.0:
        raise ValueError("gate_floor must be in [0, 1].")
    if max_overlay_splats < 1:
        raise ValueError("max_overlay_splats must be positive.")

    scene_root = Path(scene_dir)
    scene_meta, frames, splits = load_prepared_scene(scene_root)
    resolved_splats = _resolve_scene_file(scene_root, splats_path, "real_splats.npz")
    resolved_model = _resolve_scene_file(scene_root, model_path, "real_gpis_model.npz")
    diagnostics_dir = Path(output_dir) if output_dir is not None else scene_root / "diagnostics" / "real_render"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = diagnostics_dir / "panels"
    overlay_dir = diagnostics_dir / "overlays"
    render_dir = diagnostics_dir / "renders"
    for directory in (panel_dir, overlay_dir, render_dir):
        directory.mkdir(parents=True, exist_ok=True)

    splats = load_splats(str(resolved_splats))
    model, _ = load_model(str(resolved_model))
    raw_gate = gpis_gate_for_splats(splats, model, epsilon, batch_size=gate_batch_size)
    gate = torch.clamp(gate_floor + (1.0 - gate_floor) * raw_gate, min=0.0, max=1.0)
    raw_gate_np = raw_gate.detach().cpu().numpy()
    gate_np = gate.detach().cpu().numpy()
    gate_histogram_path = diagnostics_dir / "gate_histogram.png"
    save_histogram(gate_histogram_path, gate_np, title="GPIS gate")

    frame_indices = resolve_frame_indices(splits, frame_count=len(frames), split=split)
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    if not frame_indices:
        raise ValueError(f"Split {split!r} did not resolve to any frames.")

    convention = resolve_projection_convention(scene_meta, projection_convention)
    plain_root = Path(plain_renders_dir) if plain_renders_dir is not None else None
    gated_root = Path(gated_renders_dir) if gated_renders_dir is not None else None
    rows = []
    artifacts = []
    plain_gate = torch.ones_like(gate)
    for frame_index in frame_indices:
        frame = frames[int(frame_index)]
        target = load_image(resolve_scene_image_path(scene_root, frame["image_path"]))
        plain, plain_stats, plain_path = resolve_diagnostic_render(
            splats=splats,
            frame=frame,
            gate=plain_gate,
            predictions_dir=plain_root,
            output_dir=render_dir / "plain",
            projection_convention=convention,
            near_plane=near_plane,
            kernel_radius=kernel_radius,
            min_sigma_px=min_sigma_px,
            background_color=background_color,
        )
        gated, gated_stats, gated_path = resolve_diagnostic_render(
            splats=splats,
            frame=frame,
            gate=gate,
            predictions_dir=gated_root,
            output_dir=render_dir / "gated",
            projection_convention=convention,
            near_plane=near_plane,
            kernel_radius=kernel_radius,
            min_sigma_px=min_sigma_px,
            background_color=background_color,
        )

        projected = project_splats_to_frame(splats, frame, projection_convention=convention, near_plane=near_plane)
        centers_px = projected["centers_px"]
        depth = projected["depth"]
        projected_mask = projected["valid"]
        height, width = target.shape[:2]
        in_frame_mask = projected_mask & (centers_px[:, 0] >= 0.0) & (centers_px[:, 0] < width) & (centers_px[:, 1] >= 0.0) & (centers_px[:, 1] < height)
        overlay_mask = select_overlay_mask(in_frame_mask, depth, max_overlay_splats=max_overlay_splats)

        frame_tag = f"frame_{int(frame_index):06d}"
        panel_path = panel_dir / f"{frame_tag}_target_plain_gated.png"
        overlay_path = overlay_dir / f"{frame_tag}_projected_splats.png"
        depth_path = overlay_dir / f"{frame_tag}_depth.png"
        gate_path = overlay_dir / f"{frame_tag}_gate_overlay.png"
        gate_heatmap_path = overlay_dir / f"{frame_tag}_gate_heatmap.png"
        frame_histogram_path = overlay_dir / f"{frame_tag}_gate_histogram.png"

        save_panel(panel_path, [("target", target), ("plain", plain), ("gated", gated)])
        save_projection_overlay(overlay_path, target, centers_px, overlay_mask, colors=splats.colors.detach().cpu().numpy())
        save_value_projection(depth_path, target.shape[:2], centers_px, overlay_mask, depth, title="depth")
        save_gate_overlay(gate_path, target, centers_px, overlay_mask, gate_np)
        save_value_projection(gate_heatmap_path, target.shape[:2], centers_px, overlay_mask, gate_np, title="gate")
        save_histogram(frame_histogram_path, gate_np[in_frame_mask], title="visible GPIS gate")

        visible_depth = depth[in_frame_mask]
        visible_gate = gate_np[in_frame_mask]
        visible_raw_gate = raw_gate_np[in_frame_mask]
        row = {
            "scene": scene_meta["scene"],
            "split": split,
            "frame_index": int(frame_index),
            "image_path": frame["image_path"],
            "projection_convention": convention,
            "width": width,
            "height": height,
            "projected_splat_count": int(projected_mask.sum()),
            "in_frame_splat_count": int(in_frame_mask.sum()),
            "visible_splat_count": int(in_frame_mask.sum()),
            "plain_drawn_splat_count": int(plain_stats["drawn_splat_count"]),
            "gated_drawn_splat_count": int(gated_stats["drawn_splat_count"]),
            "min_depth": finite_min(visible_depth),
            "max_depth": finite_max(visible_depth),
            "mean_depth": finite_mean(visible_depth),
            "gate_min": finite_min(visible_gate),
            "gate_mean": finite_mean(visible_gate),
            "gate_max": finite_max(visible_gate),
            "raw_gate_min": finite_min(visible_raw_gate),
            "raw_gate_mean": finite_mean(visible_raw_gate),
            "raw_gate_max": finite_max(visible_raw_gate),
            "plain_psnr": psnr_arrays(plain, target),
            "plain_ssim": ssim_arrays(plain, target),
            "gated_psnr": psnr_arrays(gated, target),
            "gated_ssim": ssim_arrays(gated, target),
            "target_plain_gated_panel": str(panel_path),
            "projected_splat_overlay": str(overlay_path),
            "depth_visualization": str(depth_path),
            "gate_overlay": str(gate_path),
            "gate_heatmap": str(gate_heatmap_path),
            "gate_histogram": str(frame_histogram_path),
            "plain_prediction_path": str(plain_path),
            "gated_prediction_path": str(gated_path),
        }
        rows.append(row)
        artifacts.append(
            {
                "frame_index": int(frame_index),
                "panel": str(panel_path),
                "projected_splats": str(overlay_path),
                "depth": str(depth_path),
                "gate_overlay": str(gate_path),
                "gate_heatmap": str(gate_heatmap_path),
                "gate_histogram": str(frame_histogram_path),
            }
        )

    frame_metrics = pd.DataFrame(rows)
    frame_metrics_path = diagnostics_dir / "real_render_diagnostics.csv"
    status_path = diagnostics_dir / "real_render_diagnostics.json"
    report_path = diagnostics_dir / "real_render_diagnostics.md"
    frame_metrics.to_csv(frame_metrics_path, index=False)
    status = {
        "schema_version": 1,
        "scene": scene_meta["scene"],
        "scene_dir": str(scene_root),
        "split": split,
        "frame_count": len(rows),
        "splats_path": str(resolved_splats),
        "model_path": str(resolved_model),
        "diagnostics_dir": str(diagnostics_dir),
        "frame_metrics_path": str(frame_metrics_path),
        "report_path": str(report_path),
        "gate_histogram_path": str(gate_histogram_path),
        "epsilon": epsilon,
        "gate_floor": gate_floor,
        "projection_convention": convention,
        "splat_count": int(splats.centers.shape[0]),
        "gate_summary": summarize_values(gate_np),
        "raw_gate_summary": summarize_values(raw_gate_np),
        "metric_summary": summarize_frame_metrics(frame_metrics),
        "artifacts": artifacts,
    }
    write_json(status_path, status)
    report_path.write_text(format_diagnostics_report(status, frame_metrics), encoding="utf-8")
    return {
        "output_dir": diagnostics_dir,
        "frame_metrics_path": frame_metrics_path,
        "status_path": status_path,
        "report_path": report_path,
        "gate_histogram_path": gate_histogram_path,
        "status": status,
    }


def resolve_diagnostic_render(
    *,
    splats: Any,
    frame: dict[str, Any],
    gate: torch.Tensor,
    predictions_dir: Path | None,
    output_dir: Path,
    projection_convention: str,
    near_plane: float,
    kernel_radius: float,
    min_sigma_px: float,
    background_color: tuple[float, float, float],
) -> tuple[np.ndarray, dict[str, Any], Path]:
    if predictions_dir is not None:
        prediction_path = find_prediction_image(predictions_dir, frame)
        if prediction_path is None:
            raise FileNotFoundError(f"Could not find prediction for frame {frame['image_path']!r} under {predictions_dir}.")
        return load_image(prediction_path), rendered_stats_from_projection(splats, frame, projection_convention=projection_convention, near_plane=near_plane), prediction_path

    output_dir.mkdir(parents=True, exist_ok=True)
    image, stats = render_real_splat_image(
        splats,
        frame,
        gate=gate,
        projection_convention=projection_convention,
        near_plane=near_plane,
        kernel_radius=kernel_radius,
        min_sigma_px=min_sigma_px,
        background_color=background_color,
    )
    output_path = output_dir / frame["file_name"]
    save_image(output_path, image)
    return image.detach().cpu().numpy(), stats, output_path


def rendered_stats_from_projection(
    splats: Any,
    frame: dict[str, Any],
    *,
    projection_convention: str,
    near_plane: float,
) -> dict[str, Any]:
    projected = project_splats_to_frame(splats, frame, projection_convention=projection_convention, near_plane=near_plane)
    valid = projected["valid"]
    depth = projected["depth"][valid]
    return {
        "drawn_splat_count": int(valid.sum()),
        "projected_splat_count": int(valid.sum()),
        "min_depth": float(depth.min()) if depth.size else None,
        "max_depth": float(depth.max()) if depth.size else None,
    }


def select_overlay_mask(mask: np.ndarray, depth: np.ndarray, *, max_overlay_splats: int) -> np.ndarray:
    selected = np.flatnonzero(mask)
    if selected.shape[0] <= max_overlay_splats:
        return mask
    finite_depth = np.where(np.isfinite(depth[selected]), depth[selected], np.inf)
    keep = selected[np.argsort(finite_depth)[:max_overlay_splats]]
    overlay = np.zeros_like(mask, dtype=bool)
    overlay[keep] = True
    return overlay


def save_panel(path: str | Path, images: list[tuple[str, np.ndarray]]) -> None:
    prepared = [(label, to_uint8(image)) for label, image in images]
    heights = [image.shape[0] for _, image in prepared]
    widths = [image.shape[1] for _, image in prepared]
    header = 22
    canvas = Image.new("RGB", (sum(widths), max(heights) + header), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in prepared:
        canvas.paste(Image.fromarray(image, mode="RGB"), (x, header))
        draw.text((x + 4, 4), label, fill=(235, 235, 235))
        x += image.shape[1]
    canvas.save(path)


def save_projection_overlay(
    path: str | Path,
    target: np.ndarray,
    centers_px: np.ndarray,
    mask: np.ndarray,
    *,
    colors: np.ndarray,
    radius: int = 2,
) -> None:
    base = Image.fromarray((0.55 * to_uint8(target)).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(base)
    for index in np.flatnonzero(mask):
        x, y = centers_px[index]
        rgb = tuple(int(channel) for channel in np.clip(colors[index], 0.0, 1.0) * 255.0)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rgb, outline=(255, 255, 255))
    base.save(path)


def save_gate_overlay(
    path: str | Path,
    target: np.ndarray,
    centers_px: np.ndarray,
    mask: np.ndarray,
    gate: np.ndarray,
    radius: int = 2,
) -> None:
    base = Image.fromarray((0.45 * to_uint8(target)).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(base)
    for index in np.flatnonzero(mask):
        x, y = centers_px[index]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value_color(float(gate[index])), outline=(255, 255, 255))
    base.save(path)


def save_value_projection(
    path: str | Path,
    image_shape: tuple[int, int],
    centers_px: np.ndarray,
    mask: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    radius: int = 2,
) -> None:
    height, width = image_shape
    canvas = Image.new("RGB", (width, height + 22), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill=(235, 235, 235))
    selected_values = values[mask]
    value_min = float(np.nanmin(selected_values)) if selected_values.size else 0.0
    value_max = float(np.nanmax(selected_values)) if selected_values.size else 1.0
    if abs(value_max - value_min) < 1e-12:
        value_max = value_min + 1.0
    for index in np.flatnonzero(mask):
        normalized = (float(values[index]) - value_min) / (value_max - value_min)
        x, y = centers_px[index]
        draw.ellipse((x - radius, y + 22 - radius, x + radius, y + 22 + radius), fill=value_color(normalized))
    canvas.save(path)


def save_histogram(path: str | Path, values: np.ndarray, *, title: str, bins: int = 20) -> None:
    width, height = 360, 220
    margin_left, margin_top, margin_right, margin_bottom = 42, 30, 14, 32
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_left, 8), title, fill=(20, 20, 20))
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    cleaned = np.asarray(values, dtype=np.float64)
    cleaned = cleaned[np.isfinite(cleaned)]
    counts = np.histogram(cleaned, bins=bins, range=(0.0, 1.0))[0] if cleaned.size else np.zeros(bins, dtype=np.int64)
    max_count = max(int(counts.max()), 1)
    draw.rectangle((margin_left, margin_top, width - margin_right, margin_top + plot_height), outline=(80, 80, 80))
    bar_width = plot_width / bins
    for index, count in enumerate(counts):
        bar_height = int(plot_height * int(count) / max_count)
        x0 = margin_left + int(index * bar_width)
        x1 = margin_left + int((index + 1) * bar_width) - 1
        y0 = margin_top + plot_height - bar_height
        y1 = margin_top + plot_height
        draw.rectangle((x0, y0, x1, y1), fill=value_color((index + 0.5) / bins))
    draw.text((margin_left, height - 24), "0", fill=(20, 20, 20))
    draw.text((width - margin_right - 12, height - 24), "1", fill=(20, 20, 20))
    draw.text((margin_left, height - 12), f"n={cleaned.size}", fill=(20, 20, 20))
    canvas.save(path)


def value_color(value: float) -> tuple[int, int, int]:
    clipped = float(np.clip(value, 0.0, 1.0))
    red = int(255.0 * clipped)
    green = int(255.0 * (1.0 - abs(clipped - 0.5) * 2.0))
    blue = int(255.0 * (1.0 - clipped))
    return red, green, blue


def to_uint8(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = image
    return (np.asarray(array).clip(0.0, 1.0) * 255.0).round().astype(np.uint8)


def summarize_values(values: np.ndarray) -> dict[str, float | int | None]:
    cleaned = np.asarray(values, dtype=np.float64)
    cleaned = cleaned[np.isfinite(cleaned)]
    if not cleaned.size:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": int(cleaned.size),
        "min": float(cleaned.min()),
        "mean": float(cleaned.mean()),
        "max": float(cleaned.max()),
    }


def summarize_frame_metrics(frame_metrics: pd.DataFrame) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {"frame_count": int(len(frame_metrics))}
    for column in ("projected_splat_count", "in_frame_splat_count", "visible_splat_count", "plain_drawn_splat_count", "gated_drawn_splat_count"):
        summary[f"mean_{column}"] = float(frame_metrics[column].mean()) if column in frame_metrics else None
    for column in ("plain_psnr", "plain_ssim", "gated_psnr", "gated_ssim", "gate_mean", "raw_gate_mean"):
        summary[f"mean_{column}"] = float(frame_metrics[column].mean()) if column in frame_metrics else None
    return summary


def format_diagnostics_report(status: dict[str, Any], frame_metrics: pd.DataFrame) -> str:
    metric_summary = status["metric_summary"]
    lines = [
        "# Real Render Diagnostics",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Split: `{status['split']}`",
        f"- Frames: `{status['frame_count']}`",
        f"- Splats: `{status['splat_count']}`",
        f"- Projection convention: `{status['projection_convention']}`",
        f"- Epsilon: `{status['epsilon']}`",
        f"- Gate floor: `{status['gate_floor']}`",
        f"- Frame diagnostics CSV: `{status['frame_metrics_path']}`",
        f"- Gate histogram: `{status['gate_histogram_path']}`",
        f"- Mean plain PSNR: `{format_optional(metric_summary.get('mean_plain_psnr'))}`",
        f"- Mean gated PSNR: `{format_optional(metric_summary.get('mean_gated_psnr'))}`",
        f"- Mean plain SSIM: `{format_optional(metric_summary.get('mean_plain_ssim'))}`",
        f"- Mean gated SSIM: `{format_optional(metric_summary.get('mean_gated_ssim'))}`",
        "",
        "## Frames",
        "",
    ]
    columns = ["frame_index", "projected_splat_count", "visible_splat_count", "plain_psnr", "gated_psnr", "plain_ssim", "gated_ssim", "gate_mean"]
    lines.extend(markdown_table(frame_metrics[columns]))
    return "\n".join(lines) + "\n"


def markdown_table(dataframe: pd.DataFrame) -> list[str]:
    columns = list(dataframe.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in dataframe.iterrows():
        values = [format_optional(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def format_optional(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def finite_min(values: np.ndarray) -> float | None:
    cleaned = np.asarray(values, dtype=np.float64)
    cleaned = cleaned[np.isfinite(cleaned)]
    return float(cleaned.min()) if cleaned.size else None


def finite_mean(values: np.ndarray) -> float | None:
    cleaned = np.asarray(values, dtype=np.float64)
    cleaned = cleaned[np.isfinite(cleaned)]
    return float(cleaned.mean()) if cleaned.size else None


def finite_max(values: np.ndarray) -> float | None:
    cleaned = np.asarray(values, dtype=np.float64)
    cleaned = cleaned[np.isfinite(cleaned)]
    return float(cleaned.max()) if cleaned.size else None
