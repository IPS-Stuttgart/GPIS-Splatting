from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from gpis_splatting.real_benchmark import find_prediction_image, psnr_arrays, ssim_arrays
from gpis_splatting.real_diagnostics import (
    finite_max,
    finite_mean,
    finite_min,
    format_optional,
    markdown_table,
    save_projection_overlay,
    save_value_projection,
    select_overlay_mask,
    to_uint8,
)
from gpis_splatting.real_pipeline import PROJECTION_CONVENTIONS, project_splats_to_frame, resolve_frame_indices, resolve_projection_convention
from gpis_splatting.real_render_audit import image_pair_stats, nonblack_fraction
from gpis_splatting.real_scene import load_prepared_scene, resolve_scene_image_path
from gpis_splatting.renderer import load_image
from gpis_splatting.serialization import read_json, write_json
from gpis_splatting.splats import SplatCloud, load_splats


def diagnose_real_alignment(
    *,
    scene_dir: str | Path,
    render_dir: str | Path,
    splats_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    split: str = "test",
    max_frames: int | None = None,
    projection_convention: str = "auto",
    near_plane: float = 1e-4,
    kernel_radius: float = 3.0,
    min_sigma_px: float = 0.8,
    coverage_downsample: int = 4,
    max_overlay_splats: int = 2000,
    require_predictions: bool = False,
    low_psnr_threshold: float = 12.0,
    min_valid_depth_fraction: float = 0.05,
    min_in_frame_fraction: float = 0.01,
    min_coverage_fraction: float = 0.01,
    min_prediction_nonblack_fraction: float = 0.001,
) -> dict[str, Any]:
    if projection_convention not in PROJECTION_CONVENTIONS:
        raise ValueError(f"Unsupported projection convention {projection_convention!r}. Expected one of {', '.join(PROJECTION_CONVENTIONS)}.")
    if coverage_downsample < 1:
        raise ValueError("coverage_downsample must be positive.")
    if max_overlay_splats < 1:
        raise ValueError("max_overlay_splats must be positive.")

    scene_root = Path(scene_dir)
    render_root = Path(render_dir)
    scene_meta, frames, splits = load_prepared_scene(scene_root)
    render_report = load_render_report(render_root)
    resolved_splats = resolve_alignment_splats_path(scene_root=scene_root, render_report=render_report, splats_path=splats_path)
    splats = load_splats(str(resolved_splats))

    resolved_output = Path(output_dir) if output_dir is not None else scene_root / "diagnostics" / "real_alignment" / render_root.name
    overlay_dir = resolved_output / "overlays"
    depth_dir = resolved_output / "depth"
    histogram_dir = resolved_output / "depth_histograms"
    panel_dir = resolved_output / "panels"
    for directory in (resolved_output, overlay_dir, depth_dir, histogram_dir, panel_dir):
        directory.mkdir(parents=True, exist_ok=True)

    convention = resolve_projection_convention(scene_meta, projection_convention)
    frame_indices = resolve_frame_indices(splits, frame_count=len(frames), split=split)
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    if not frame_indices:
        raise ValueError(f"Split {split!r} did not resolve to any frames.")

    render_outputs = render_outputs_by_frame(render_report)
    rows = []
    missing_predictions = []
    for frame_index in frame_indices:
        frame = frames[int(frame_index)]
        target_path = resolve_scene_image_path(scene_root, frame["image_path"])
        target = load_image(target_path)
        height, width = target.shape[:2]
        prediction_path = find_prediction_image(render_root, frame)
        frame_tag = f"frame_{int(frame_index):06d}"

        projected = project_splats_to_frame(splats, frame, projection_convention=convention, near_plane=near_plane)
        projection = projection_stats(
            splats=splats,
            frame=frame,
            target_shape=(height, width),
            projected=projected,
            near_plane=near_plane,
            kernel_radius=kernel_radius,
            min_sigma_px=min_sigma_px,
            coverage_downsample=coverage_downsample,
        )
        overlay_mask = select_overlay_mask(projection["in_frame_mask"], projected["depth"], max_overlay_splats=max_overlay_splats)
        overlay_path = overlay_dir / f"{frame_tag}_projected_splats.png"
        depth_path = depth_dir / f"{frame_tag}_depth_projection.png"
        histogram_path = histogram_dir / f"{frame_tag}_depth_histogram.png"
        save_projection_overlay(overlay_path, target, projected["centers_px"], overlay_mask, colors=splats.colors.detach().cpu().numpy())
        save_value_projection(depth_path, target.shape[:2], projected["centers_px"], overlay_mask, projected["depth"], title="depth")
        save_depth_histogram(histogram_path, projected["depth"][projection["in_frame_mask"]], title=f"frame {int(frame_index)} depth")

        row: dict[str, Any] = {
            "scene": scene_meta["scene"],
            "split": split,
            "frame_index": int(frame_index),
            "image_path": frame["image_path"],
            "target_path": str(target_path),
            "prediction_path": str(prediction_path) if prediction_path is not None else None,
            "missing_prediction": prediction_path is None,
            "panel_path": None,
            "psnr": np.nan,
            "ssim": np.nan,
            "prediction_nonblack_fraction": np.nan,
            "projection_convention": convention,
            "width": width,
            "height": height,
            "splat_count": int(splats.centers.shape[0]),
            "overlay_path": str(overlay_path),
            "depth_projection_path": str(depth_path),
            "depth_histogram_path": str(histogram_path),
            **projection["summary"],
            **render_report_fields(render_outputs.get(int(frame_index), {})),
        }
        if prediction_path is None:
            missing_predictions.append(frame["image_path"])
        else:
            prediction = load_image(prediction_path)
            if prediction.shape != target.shape:
                raise ValueError(f"Prediction shape {prediction.shape} for {prediction_path} does not match target shape {target.shape}.")
            panel_path = panel_dir / f"{frame_tag}_target_prediction_diff.png"
            save_alignment_panel(panel_path, target=target, prediction=prediction)
            row.update(
                {
                    "panel_path": str(panel_path),
                    "psnr": psnr_arrays(prediction, target),
                    "ssim": ssim_arrays(prediction, target),
                    "prediction_nonblack_fraction": nonblack_fraction(prediction),
                    **{f"image_{key}": value for key, value in image_pair_stats(target=target, prediction=prediction).items()},
                }
            )
        failure = classify_alignment_failure(
            row,
            low_psnr_threshold=low_psnr_threshold,
            min_valid_depth_fraction=min_valid_depth_fraction,
            min_in_frame_fraction=min_in_frame_fraction,
            min_coverage_fraction=min_coverage_fraction,
            min_prediction_nonblack_fraction=min_prediction_nonblack_fraction,
        )
        row.update(failure)
        rows.append(row)

    if missing_predictions and require_predictions:
        raise FileNotFoundError(f"Missing {len(missing_predictions)} prediction images under {render_root}: {missing_predictions[:5]}")

    frames_df = pd.DataFrame(rows)
    ranked_df = rank_alignment_rows(frames_df)
    summary = summarize_alignment(frames_df, scene_meta=scene_meta, split=split, render_dir=render_root)

    frames_path = resolved_output / "real_alignment_frames.csv"
    ranked_path = resolved_output / "real_alignment_ranked.csv"
    status_path = resolved_output / "real_alignment_status.json"
    report_path = resolved_output / "real_alignment_report.md"
    frames_df.to_csv(frames_path, index=False)
    ranked_df.to_csv(ranked_path, index=False)
    status = {
        "schema_version": 1,
        "scene": scene_meta["scene"],
        "scene_dir": str(scene_root),
        "render_dir": str(render_root),
        "splats_path": str(resolved_splats),
        "split": split,
        "projection_convention": convention,
        "output_dir": str(resolved_output),
        "frames_path": str(frames_path),
        "ranked_path": str(ranked_path),
        "report_path": str(report_path),
        "overlay_dir": str(overlay_dir),
        "depth_dir": str(depth_dir),
        "depth_histogram_dir": str(histogram_dir),
        "panel_dir": str(panel_dir),
        "parameters": {
            "near_plane": near_plane,
            "kernel_radius": kernel_radius,
            "min_sigma_px": min_sigma_px,
            "coverage_downsample": coverage_downsample,
            "max_overlay_splats": max_overlay_splats,
            "low_psnr_threshold": low_psnr_threshold,
            "min_valid_depth_fraction": min_valid_depth_fraction,
            "min_in_frame_fraction": min_in_frame_fraction,
            "min_coverage_fraction": min_coverage_fraction,
            "min_prediction_nonblack_fraction": min_prediction_nonblack_fraction,
        },
        "summary": summary,
    }
    write_json(status_path, status)
    report_path.write_text(format_alignment_report(status, ranked_df), encoding="utf-8")
    return {
        "output_dir": resolved_output,
        "frames_path": frames_path,
        "ranked_path": ranked_path,
        "status_path": status_path,
        "report_path": report_path,
        "status": status,
    }


def projection_stats(
    *,
    splats: SplatCloud,
    frame: dict[str, Any],
    target_shape: tuple[int, int],
    projected: dict[str, np.ndarray],
    near_plane: float,
    kernel_radius: float,
    min_sigma_px: float,
    coverage_downsample: int,
) -> dict[str, Any]:
    height, width = target_shape
    centers_px = projected["centers_px"]
    depth = projected["depth"]
    valid = projected["valid"]
    finite_projection = np.isfinite(depth) & np.isfinite(centers_px).all(axis=1)
    behind_camera = finite_projection & (depth <= near_plane)
    in_frame = valid & (centers_px[:, 0] >= 0.0) & (centers_px[:, 0] < width) & (centers_px[:, 1] >= 0.0) & (centers_px[:, 1] < height)
    off_frame = valid & ~in_frame
    sigma_px = projected_sigma_px(splats=splats, frame=frame, depth=depth, near_plane=near_plane, min_sigma_px=min_sigma_px)
    coverage_fraction = projected_coverage_fraction(
        image_shape=target_shape,
        centers_px=centers_px,
        sigma_px=sigma_px,
        mask=in_frame,
        kernel_radius=kernel_radius,
        downsample=coverage_downsample,
    )
    splat_count = int(splats.centers.shape[0])
    summary = {
        "finite_projection_count": int(finite_projection.sum()),
        "valid_depth_count": int(valid.sum()),
        "behind_camera_count": int(behind_camera.sum()),
        "invalid_projection_count": int((~finite_projection).sum()),
        "in_frame_splat_count": int(in_frame.sum()),
        "off_frame_splat_count": int(off_frame.sum()),
        "valid_depth_fraction": safe_fraction(int(valid.sum()), splat_count),
        "behind_camera_fraction": safe_fraction(int(behind_camera.sum()), splat_count),
        "in_frame_fraction": safe_fraction(int(in_frame.sum()), splat_count),
        "off_frame_fraction": safe_fraction(int(off_frame.sum()), splat_count),
        "projected_coverage_fraction": coverage_fraction,
        "min_valid_depth": finite_min(depth[valid]),
        "mean_valid_depth": finite_mean(depth[valid]),
        "max_valid_depth": finite_max(depth[valid]),
        "min_in_frame_depth": finite_min(depth[in_frame]),
        "mean_in_frame_depth": finite_mean(depth[in_frame]),
        "max_in_frame_depth": finite_max(depth[in_frame]),
        "mean_in_frame_sigma_px": finite_mean(sigma_px[in_frame]),
        "max_in_frame_sigma_px": finite_max(sigma_px[in_frame]),
    }
    return {"summary": summary, "in_frame_mask": in_frame, "valid_mask": valid, "sigma_px": sigma_px}


def projected_sigma_px(*, splats: SplatCloud, frame: dict[str, Any], depth: np.ndarray, near_plane: float, min_sigma_px: float) -> np.ndarray:
    intrinsics = frame["intrinsics"]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    focal = 0.5 * (fx + fy)
    sigma = splats.sigma.detach().cpu().numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_px = sigma * focal / np.clip(depth, near_plane, None)
    return np.maximum(sigma_px, min_sigma_px)


def projected_coverage_fraction(
    *,
    image_shape: tuple[int, int],
    centers_px: np.ndarray,
    sigma_px: np.ndarray,
    mask: np.ndarray,
    kernel_radius: float,
    downsample: int,
) -> float:
    height, width = image_shape
    coarse_height = max(1, int(np.ceil(height / downsample)))
    coarse_width = max(1, int(np.ceil(width / downsample)))
    coverage = np.zeros((coarse_height, coarse_width), dtype=bool)
    for index in np.flatnonzero(mask):
        cx = float(centers_px[index, 0]) / downsample
        cy = float(centers_px[index, 1]) / downsample
        radius = max(1, int(np.ceil(kernel_radius * float(sigma_px[index]) / downsample)))
        x0 = max(0, int(np.floor(cx - radius)))
        x1 = min(coarse_width, int(np.ceil(cx + radius + 1)))
        y0 = max(0, int(np.floor(cy - radius)))
        y1 = min(coarse_height, int(np.ceil(cy + radius + 1)))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        coverage[y0:y1, x0:x1] |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    return float(coverage.mean())


def classify_alignment_failure(
    row: dict[str, Any],
    *,
    low_psnr_threshold: float,
    min_valid_depth_fraction: float,
    min_in_frame_fraction: float,
    min_coverage_fraction: float,
    min_prediction_nonblack_fraction: float,
) -> dict[str, Any]:
    if row["missing_prediction"]:
        return {"failure_mode": "missing_prediction", "failure_score": 1000.0}
    valid_depth_fraction = float(row.get("valid_depth_fraction") or 0.0)
    in_frame_fraction = float(row.get("in_frame_fraction") or 0.0)
    coverage_fraction = float(row.get("projected_coverage_fraction") or 0.0)
    prediction_nonblack = float(row.get("prediction_nonblack_fraction") or 0.0)
    psnr = float(row.get("psnr") or np.nan)
    if valid_depth_fraction < min_valid_depth_fraction:
        return {
            "failure_mode": "mostly_behind_camera_or_bad_convention",
            "failure_score": 900.0 + min_valid_depth_fraction - valid_depth_fraction,
        }
    if in_frame_fraction < min_in_frame_fraction:
        return {"failure_mode": "mostly_out_of_frame", "failure_score": 800.0 + min_in_frame_fraction - in_frame_fraction}
    if coverage_fraction < min_coverage_fraction:
        return {"failure_mode": "low_projected_coverage", "failure_score": 700.0 + min_coverage_fraction - coverage_fraction}
    if prediction_nonblack < min_prediction_nonblack_fraction:
        return {"failure_mode": "render_opacity_or_compositing_failure", "failure_score": 600.0 + min_prediction_nonblack_fraction - prediction_nonblack}
    if np.isfinite(psnr) and psnr < low_psnr_threshold:
        return {"failure_mode": "appearance_or_alignment_error", "failure_score": 500.0 + low_psnr_threshold - psnr}
    return {"failure_mode": "no_obvious_alignment_failure", "failure_score": 0.0}


def rank_alignment_rows(frames: pd.DataFrame) -> pd.DataFrame:
    if frames.empty:
        return frames
    ranked = frames.copy()
    ranked["_rank_psnr"] = pd.to_numeric(ranked.get("psnr"), errors="coerce").fillna(-np.inf)
    ranked["_rank_coverage"] = pd.to_numeric(ranked.get("projected_coverage_fraction"), errors="coerce").fillna(-np.inf)
    ranked = ranked.sort_values(["failure_score", "_rank_psnr", "_rank_coverage"], ascending=[False, True, True])
    return ranked.drop(columns=["_rank_psnr", "_rank_coverage"])


def summarize_alignment(frames: pd.DataFrame, *, scene_meta: dict[str, Any], split: str, render_dir: Path) -> dict[str, Any]:
    evaluated = frames[~frames.get("missing_prediction", False).astype(bool)] if not frames.empty else frames
    failure_counts = {str(key): int(value) for key, value in frames.get("failure_mode", pd.Series(dtype=str)).value_counts().items()}
    summary: dict[str, Any] = {
        "scene": scene_meta["scene"],
        "dataset": scene_meta.get("dataset"),
        "split": split,
        "render_dir": str(render_dir),
        "frame_count": int(len(frames)),
        "evaluated_count": int(len(evaluated)),
        "missing_prediction_count": int(frames.get("missing_prediction", pd.Series(dtype=bool)).astype(bool).sum()) if not frames.empty else 0,
        "failure_counts": failure_counts,
        "mean_psnr": optional_mean(evaluated, "psnr"),
        "mean_ssim": optional_mean(evaluated, "ssim"),
        "mean_valid_depth_fraction": optional_mean(frames, "valid_depth_fraction"),
        "mean_in_frame_fraction": optional_mean(frames, "in_frame_fraction"),
        "mean_projected_coverage_fraction": optional_mean(frames, "projected_coverage_fraction"),
        "mean_prediction_nonblack_fraction": optional_mean(evaluated, "prediction_nonblack_fraction"),
    }
    return summary


def save_alignment_panel(path: str | Path, *, target: np.ndarray, prediction: np.ndarray) -> None:
    target_u8 = to_uint8(target)
    prediction_u8 = to_uint8(prediction)
    abs_diff = np.abs(prediction - target)
    scale = float(abs_diff.max())
    diff_u8 = to_uint8(abs_diff / scale) if scale > 1e-12 else np.zeros_like(target_u8)
    labels = [("target", target_u8), ("prediction", prediction_u8), ("abs diff", diff_u8)]
    header = 22
    spacer = 4
    width = sum(image.shape[1] for _, image in labels) + spacer * (len(labels) - 1)
    height = max(image.shape[0] for _, image in labels) + header
    canvas = Image.new("RGB", (width, height), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in labels:
        draw.text((x + 4, 4), label, fill=(235, 235, 235))
        canvas.paste(Image.fromarray(image, mode="RGB"), (x, header))
        x += image.shape[1] + spacer
    canvas.save(path)


def save_depth_histogram(path: str | Path, values: np.ndarray, *, title: str, bins: int = 24) -> None:
    width, height = 400, 230
    margin_left, margin_top, margin_right, margin_bottom = 48, 30, 18, 34
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_left, 8), title, fill=(20, 20, 20))
    cleaned = np.asarray(values, dtype=np.float64)
    cleaned = cleaned[np.isfinite(cleaned)]
    if cleaned.size:
        value_min = float(cleaned.min())
        value_max = float(cleaned.max())
        if abs(value_max - value_min) < 1e-12:
            value_max = value_min + 1.0
        counts, edges = np.histogram(cleaned, bins=bins, range=(value_min, value_max))
    else:
        counts = np.zeros(bins, dtype=np.int64)
        edges = np.asarray([0.0, 1.0], dtype=np.float64)
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_count = max(int(counts.max()), 1)
    draw.rectangle((margin_left, margin_top, width - margin_right, margin_top + plot_height), outline=(80, 80, 80))
    bar_width = plot_width / bins
    for index, count in enumerate(counts):
        bar_height = int(plot_height * int(count) / max_count)
        x0 = margin_left + int(index * bar_width)
        x1 = margin_left + int((index + 1) * bar_width) - 1
        y0 = margin_top + plot_height - bar_height
        y1 = margin_top + plot_height
        draw.rectangle((x0, y0, x1, y1), fill=(60, 120, 200))
    draw.text((margin_left, height - 25), f"{edges[0]:.3g}", fill=(20, 20, 20))
    draw.text((width - margin_right - 52, height - 25), f"{edges[-1]:.3g}", fill=(20, 20, 20))
    draw.text((margin_left, height - 12), f"n={cleaned.size}", fill=(20, 20, 20))
    canvas.save(path)


def format_alignment_report(status: dict[str, Any], ranked: pd.DataFrame) -> str:
    summary = status["summary"]
    lines = [
        "# Real Alignment Diagnostics",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Split: `{status['split']}`",
        f"- Render directory: `{status['render_dir']}`",
        f"- Projection convention: `{status['projection_convention']}`",
        f"- Frames: `{summary['frame_count']}`",
        f"- Evaluated predictions: `{summary['evaluated_count']}`",
        f"- Missing predictions: `{summary['missing_prediction_count']}`",
        f"- Mean PSNR: `{format_optional(summary['mean_psnr'])}`",
        f"- Mean SSIM: `{format_optional(summary['mean_ssim'])}`",
        f"- Mean valid-depth fraction: `{format_optional(summary['mean_valid_depth_fraction'])}`",
        f"- Mean in-frame fraction: `{format_optional(summary['mean_in_frame_fraction'])}`",
        f"- Mean projected coverage: `{format_optional(summary['mean_projected_coverage_fraction'])}`",
        f"- Frame CSV: `{status['frames_path']}`",
        f"- Ranked CSV: `{status['ranked_path']}`",
        "",
        "## Failure Counts",
        "",
    ]
    if summary["failure_counts"]:
        lines.extend(f"- `{key}`: `{value}`" for key, value in summary["failure_counts"].items())
    else:
        lines.append("- none")
    if not ranked.empty:
        columns = [
            "frame_index",
            "failure_mode",
            "psnr",
            "ssim",
            "valid_depth_fraction",
            "in_frame_fraction",
            "projected_coverage_fraction",
            "prediction_nonblack_fraction",
        ]
        lines.extend(["", "## Worst Frames", ""])
        lines.extend(markdown_table(ranked[columns].head(12)))
    return "\n".join(lines) + "\n"


def resolve_alignment_splats_path(*, scene_root: Path, render_report: dict[str, Any] | None, splats_path: str | Path | None) -> Path:
    if splats_path is not None:
        return resolve_scene_file(scene_root, splats_path)
    if render_report is not None and render_report.get("splats_path"):
        return resolve_scene_file(scene_root, render_report["splats_path"])
    return scene_root / "real_splats.npz"


def resolve_scene_file(scene_root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = scene_root / resolved
    return resolved


def load_render_report(render_dir: Path) -> dict[str, Any] | None:
    path = render_dir / "real_render_report.json"
    if not path.exists():
        return None
    return read_json(path)


def render_outputs_by_frame(report: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if report is None:
        return {}
    return {int(row["frame_index"]): row for row in report.get("outputs", [])}


def render_report_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "render_report_projected_splat_count": row.get("projected_splat_count"),
        "render_report_drawn_splat_count": row.get("drawn_splat_count"),
        "render_report_min_depth": row.get("min_depth"),
        "render_report_max_depth": row.get("max_depth"),
    }


def safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def optional_mean(table: pd.DataFrame, column: str) -> float | None:
    if table.empty or column not in table:
        return None
    values = pd.to_numeric(table[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())
