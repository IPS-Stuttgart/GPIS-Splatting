from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image

from gpis_splatting.real_benchmark import find_prediction_image, psnr_arrays, ssim_arrays
from gpis_splatting.real_scene import load_prepared_scene, resolve_scene_image_path
from gpis_splatting.renderer import load_image
from gpis_splatting.serialization import write_json

EPSILON = 1e-12


def evaluate_render_consistency(
    *,
    scene_dir: str | Path,
    predictions_dir: str | Path,
    output_dir: str | Path,
    method_name: str,
    split: str = "test",
    scale_prediction_dirs: Mapping[str, str | Path] | None = None,
    require_all: bool = True,
    max_temporal_pairs: int | None = None,
) -> dict[str, Any]:
    """Evaluate adjacent-view and optional scale consistency for real-scene render folders."""
    if max_temporal_pairs is not None and max_temporal_pairs < 0:
        raise ValueError("max_temporal_pairs must be non-negative or None.")

    scene_root = Path(scene_dir)
    pred_root = Path(predictions_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_meta, frames, splits = load_prepared_scene(scene_root)
    split_indices = splits.get(split)
    if split_indices is None:
        raise ValueError(f"Split {split!r} does not exist in {scene_root / 'splits.json'}.")

    loaded, missing = _load_frames(scene_root, pred_root, frames, split_indices)
    if missing and require_all:
        raise FileNotFoundError(f"Missing {len(missing)} prediction images under {pred_root}: {missing[:5]}")
    if not loaded:
        raise ValueError(f"No prediction images were available for split {split!r}.")

    temporal = _temporal_rows(loaded, scene_meta, method_name, split, max_temporal_pairs)
    scale, scale_missing = _scale_rows(scene_root, scale_prediction_dirs or {}, loaded, frames, scene_meta, method_name, split)
    if scale_missing and require_all:
        raise FileNotFoundError(f"Missing {len(scale_missing)} scale-variant prediction images: {scale_missing[:5]}")

    summary = _summary(scene_meta, method_name, split, loaded, missing, scale_missing, temporal, scale, scale_prediction_dirs or {})
    temporal_path = out_dir / f"{method_name}_{split}_temporal_consistency.csv"
    scale_path = out_dir / f"{method_name}_{split}_scale_consistency.csv"
    summary_path = out_dir / f"{method_name}_{split}_render_consistency_summary.csv"
    status_path = out_dir / f"{method_name}_{split}_render_consistency_status.json"
    report_path = out_dir / f"{method_name}_{split}_render_consistency_report.md"
    temporal.to_csv(temporal_path, index=False)
    scale.to_csv(scale_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    status = {
        "schema_version": 1,
        "scene": scene_meta["scene"],
        "method": method_name,
        "split": split,
        "scene_dir": str(scene_root),
        "predictions_dir": str(pred_root),
        "output_dir": str(out_dir),
        "temporal_path": str(temporal_path),
        "scale_path": str(scale_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "scale_prediction_dirs": {label: str(path) for label, path in (scale_prediction_dirs or {}).items()},
        "missing_predictions": missing,
        "scale_missing_predictions": scale_missing,
        "summary": summary,
    }
    write_json(status_path, status)
    report_path.write_text(_format_report(status, temporal, scale), encoding="utf-8")
    return status


def _load_frames(scene_dir: Path, predictions_dir: Path, frames: list[dict[str, Any]], split_indices: list[int]) -> tuple[list[dict[str, Any]], list[str]]:
    loaded: list[dict[str, Any]] = []
    missing: list[str] = []
    for index in split_indices:
        frame = frames[int(index)]
        pred_path = find_prediction_image(predictions_dir, frame)
        if pred_path is None:
            missing.append(frame["image_path"])
            continue
        target_path = resolve_scene_image_path(scene_dir, frame["image_path"])
        target = load_image(target_path)
        raw_prediction = load_image(pred_path)
        prediction = resize_like(raw_prediction, target) if raw_prediction.shape != target.shape else raw_prediction
        loaded.append(
            {
                "frame": frame,
                "frame_index": int(index),
                "target_path": target_path,
                "prediction_path": pred_path,
                "target": target,
                "prediction": prediction,
                "prediction_original_shape": tuple(int(value) for value in raw_prediction.shape),
                "prediction_resized_to_target": bool(raw_prediction.shape != target.shape),
            }
        )
    return loaded, missing


def _temporal_rows(
    loaded: list[dict[str, Any]], scene_meta: dict[str, Any], method_name: str, split: str, max_pairs: int | None
) -> pd.DataFrame:
    rows = []
    pairs = list(zip(loaded[:-1], loaded[1:]))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    for pair_index, (left, right) in enumerate(pairs):
        left_prediction, right_prediction = match_pair_shapes(left["prediction"], right["prediction"])
        left_target, right_target = match_pair_shapes(left["target"], right["target"])
        pred_stats = image_difference_stats(left_prediction, right_prediction)
        target_stats = image_difference_stats(left_target, right_target)
        pred_edge = edge_delta_mad(left_prediction, right_prediction)
        target_edge = edge_delta_mad(left_target, right_target)
        pred_nonblack_delta = abs(nonblack_fraction(left_prediction) - nonblack_fraction(right_prediction))
        target_nonblack_delta = abs(nonblack_fraction(left_target) - nonblack_fraction(right_target))
        rows.append(
            {
                "scene": scene_meta["scene"],
                "method": method_name,
                "split": split,
                "pair_index": pair_index,
                "left_frame_index": left["frame_index"],
                "right_frame_index": right["frame_index"],
                "left_image_path": left["frame"]["image_path"],
                "right_image_path": right["frame"]["image_path"],
                "left_prediction_path": str(left["prediction_path"]),
                "right_prediction_path": str(right["prediction_path"]),
                "prediction_resized_to_target": bool(left["prediction_resized_to_target"] or right["prediction_resized_to_target"]),
                "comparison_height": int(left_prediction.shape[0]),
                "comparison_width": int(left_prediction.shape[1]),
                "prediction_delta_mad": pred_stats["mad"],
                "prediction_delta_rmse": pred_stats["rmse"],
                "prediction_delta_mse": pred_stats["mse"],
                "prediction_delta_psnr": psnr_arrays(left_prediction, right_prediction),
                "prediction_delta_ssim": ssim_arrays(left_prediction, right_prediction),
                "target_delta_mad": target_stats["mad"],
                "target_delta_rmse": target_stats["rmse"],
                "target_delta_mse": target_stats["mse"],
                "target_delta_psnr": psnr_arrays(left_target, right_target),
                "target_delta_ssim": ssim_arrays(left_target, right_target),
                "delta_mad_excess": pred_stats["mad"] - target_stats["mad"],
                "delta_rmse_excess": pred_stats["rmse"] - target_stats["rmse"],
                "delta_mad_ratio": safe_ratio(pred_stats["mad"], target_stats["mad"]),
                "delta_rmse_ratio": safe_ratio(pred_stats["rmse"], target_stats["rmse"]),
                "prediction_edge_delta_mad": pred_edge,
                "target_edge_delta_mad": target_edge,
                "edge_delta_mad_excess": pred_edge - target_edge,
                "edge_delta_mad_ratio": safe_ratio(pred_edge, target_edge),
                "prediction_nonblack_fraction_delta": pred_nonblack_delta,
                "target_nonblack_fraction_delta": target_nonblack_delta,
                "nonblack_fraction_delta_excess": pred_nonblack_delta - target_nonblack_delta,
                "temporal_instability_score": max(0.0, pred_stats["mad"] - target_stats["mad"]) + max(0.0, pred_edge - target_edge),
            }
        )
    return pd.DataFrame(rows)


def _scale_rows(
    scene_dir: Path,
    variant_dirs: Mapping[str, str | Path],
    loaded: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    scene_meta: dict[str, Any],
    method_name: str,
    split: str,
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    for label, variant_dir in variant_dirs.items():
        variant_root = Path(variant_dir)
        for base in loaded:
            frame = frames[base["frame_index"]]
            variant_path = find_prediction_image(variant_root, frame)
            if variant_path is None:
                missing.append(f"{label}:{frame['image_path']}")
                continue
            raw_variant = load_image(variant_path)
            variant = resize_like(raw_variant, base["prediction"]) if raw_variant.shape != base["prediction"].shape else raw_variant
            diff_stats = image_difference_stats(base["prediction"], variant)
            edge_mad = edge_delta_mad(base["prediction"], variant)
            rows.append(
                {
                    "scene": scene_meta["scene"],
                    "method": method_name,
                    "split": split,
                    "scale_label": label,
                    "frame_index": base["frame_index"],
                    "image_path": frame["image_path"],
                    "base_prediction_path": str(base["prediction_path"]),
                    "variant_prediction_path": str(variant_path),
                    "base_height": int(base["prediction"].shape[0]),
                    "base_width": int(base["prediction"].shape[1]),
                    "variant_original_height": int(raw_variant.shape[0]),
                    "variant_original_width": int(raw_variant.shape[1]),
                    "variant_resized_to_base": bool(raw_variant.shape != base["prediction"].shape),
                    "scale_psnr": psnr_arrays(variant, base["prediction"]),
                    "scale_ssim": ssim_arrays(variant, base["prediction"]),
                    "scale_mse": diff_stats["mse"],
                    "scale_rmse": diff_stats["rmse"],
                    "scale_mad": diff_stats["mad"],
                    "scale_max_abs_diff": diff_stats["max_abs_diff"],
                    "scale_edge_mad": edge_mad,
                    "scale_nonblack_fraction_delta": abs(nonblack_fraction(variant) - nonblack_fraction(base["prediction"])),
                    "variant_target_psnr": psnr_arrays(variant, base["target"]),
                    "variant_target_ssim": ssim_arrays(variant, base["target"]),
                    "scale_instability_score": diff_stats["mad"] + edge_mad,
                }
            )
    return pd.DataFrame(rows), missing


def _summary(
    scene_meta: dict[str, Any],
    method_name: str,
    split: str,
    loaded: list[dict[str, Any]],
    missing: list[str],
    scale_missing: list[str],
    temporal: pd.DataFrame,
    scale: pd.DataFrame,
    scale_prediction_dirs: Mapping[str, str | Path],
) -> dict[str, Any]:
    return {
        "scene": scene_meta["scene"],
        "dataset": scene_meta.get("dataset"),
        "method": method_name,
        "split": split,
        "image_count": int(len(loaded)),
        "missing_count": int(len(missing)),
        "scale_missing_count": int(len(scale_missing)),
        "prediction_resized_to_target_count": int(sum(row["prediction_resized_to_target"] for row in loaded)),
        "temporal_pair_count": int(len(temporal)),
        "scale_variant_count": int(len(scale_prediction_dirs)),
        "scale_image_count": int(len(scale)),
        "mean_temporal_prediction_delta_mad": optional_mean(temporal, "prediction_delta_mad"),
        "mean_temporal_target_delta_mad": optional_mean(temporal, "target_delta_mad"),
        "mean_temporal_delta_mad_excess": optional_mean(temporal, "delta_mad_excess"),
        "mean_temporal_delta_mad_ratio": optional_mean(temporal, "delta_mad_ratio"),
        "max_temporal_delta_mad_ratio": optional_max(temporal, "delta_mad_ratio"),
        "mean_temporal_edge_delta_mad_excess": optional_mean(temporal, "edge_delta_mad_excess"),
        "mean_temporal_instability_score": optional_mean(temporal, "temporal_instability_score"),
        "max_temporal_instability_score": optional_max(temporal, "temporal_instability_score"),
        "mean_scale_psnr": optional_mean(scale, "scale_psnr"),
        "mean_scale_ssim": optional_mean(scale, "scale_ssim"),
        "mean_scale_mad": optional_mean(scale, "scale_mad"),
        "max_scale_mad": optional_max(scale, "scale_mad"),
        "mean_scale_edge_mad": optional_mean(scale, "scale_edge_mad"),
        "mean_scale_instability_score": optional_mean(scale, "scale_instability_score"),
        "max_scale_instability_score": optional_max(scale, "scale_instability_score"),
    }


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return resize_to_shape(image, reference.shape[:2])


def resize_to_shape(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    return np.asarray(Image.fromarray(to_uint8(image), mode="RGB").resize((width, height), resample=resampling).convert("RGB"), dtype=np.float64) / 255.0


def match_pair_shapes(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (left, right) if left.shape == right.shape else (left, resize_like(right, left))


def image_difference_stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    diff = right - left
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff**2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mad": float(abs_diff.mean()), "max_abs_diff": float(abs_diff.max())}


def edge_delta_mad(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(gradient_magnitude(right) - gradient_magnitude(left))))


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = image.mean(axis=2)
    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, 1:] = gray[:, 1:] - gray[:, :-1]
    grad_y[1:, :] = gray[1:, :] - gray[:-1, :]
    return np.sqrt(grad_x**2 + grad_y**2)


def nonblack_fraction(image: np.ndarray) -> float:
    return float(np.mean(np.any(image > (0.5 / 255.0), axis=2)))


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= EPSILON:
        return 0.0 if abs(numerator) <= EPSILON else float("inf")
    return float(numerator / denominator)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def optional_mean(table: pd.DataFrame, column: str) -> float | None:
    if table.empty or column not in table:
        return None
    values = pd.to_numeric(table[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return None if values.empty else float(values.mean())


def optional_max(table: pd.DataFrame, column: str) -> float | None:
    if table.empty or column not in table:
        return None
    values = pd.to_numeric(table[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return None if values.empty else float(values.max())


def _format_report(status: dict[str, Any], temporal: pd.DataFrame, scale: pd.DataFrame) -> str:
    summary = status["summary"]
    lines = [
        "# Render Consistency Evaluation",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Method: `{status['method']}`",
        f"- Split: `{status['split']}`",
        f"- Images evaluated: `{summary['image_count']}`",
        f"- Missing predictions: `{summary['missing_count']}`",
        f"- Missing scale predictions: `{summary['scale_missing_count']}`",
        f"- Prediction resized to target count: `{summary['prediction_resized_to_target_count']}`",
        f"- Temporal pairs: `{summary['temporal_pair_count']}`",
        f"- Mean temporal instability score: `{format_optional(summary['mean_temporal_instability_score'])}`",
        f"- Scale variants: `{summary['scale_variant_count']}`",
        f"- Scale comparisons: `{summary['scale_image_count']}`",
        f"- Mean scale MAD: `{format_optional(summary['mean_scale_mad'])}`",
        f"- Temporal CSV: `{status['temporal_path']}`",
        f"- Scale CSV: `{status['scale_path']}`",
        f"- Summary CSV: `{status['summary_path']}`",
    ]
    if status["missing_predictions"]:
        lines.extend(["", "## Missing Predictions", *[f"- `{item}`" for item in status["missing_predictions"]]])
    if status["scale_missing_predictions"]:
        lines.extend(["", "## Missing Scale Predictions", *[f"- `{item}`" for item in status["scale_missing_predictions"]]])
    if not temporal.empty:
        lines.extend(["", "## Worst Temporal Pairs", "", _format_temporal_table(temporal.sort_values("temporal_instability_score", ascending=False).head(8))])
    if not scale.empty:
        lines.extend(["", "## Worst Scale Comparisons", "", _format_scale_table(scale.sort_values("scale_instability_score", ascending=False).head(8))])
    return "\n".join(lines) + "\n"


def _format_temporal_table(rows: pd.DataFrame) -> str:
    lines = ["| pair | left | right | pred MAD | target MAD | excess MAD | edge excess | score |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.pair_index} | {row.left_frame_index} | {row.right_frame_index} | {format_optional(row.prediction_delta_mad)} | "
            f"{format_optional(row.target_delta_mad)} | {format_optional(row.delta_mad_excess)} | {format_optional(row.edge_delta_mad_excess)} | "
            f"{format_optional(row.temporal_instability_score)} |"
        )
    return "\n".join(lines)


def _format_scale_table(rows: pd.DataFrame) -> str:
    lines = ["| label | frame | PSNR | SSIM | MAD | edge MAD | score | resized |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| `{row.scale_label}` | {row.frame_index} | {format_optional(row.scale_psnr)} | {format_optional(row.scale_ssim)} | "
            f"{format_optional(row.scale_mad)} | {format_optional(row.scale_edge_mad)} | {format_optional(row.scale_instability_score)} | "
            f"`{bool(row.variant_resized_to_base)}` |"
        )
    return "\n".join(lines)


def format_optional(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value_float = float(value)
    if np.isinf(value_float):
        return "inf" if value_float > 0.0 else "-inf"
    return f"{value_float:.6g}"
