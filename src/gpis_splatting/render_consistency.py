from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from gpis_splatting.real_benchmark import find_prediction_image, psnr_arrays, ssim_arrays
from gpis_splatting.real_scene import load_prepared_scene, resolve_scene_image_path
from gpis_splatting.renderer import load_image
from gpis_splatting.serialization import write_json

EPSILON = 1e-12
DEFAULT_AA_DOWNSAMPLE_FACTORS = (2,)


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
    max_view_translation: float | None = None,
    max_view_rotation_deg: float | None = None,
    aa_downsample_factors: Sequence[int] | None = DEFAULT_AA_DOWNSAMPLE_FACTORS,
) -> dict[str, Any]:
    """Evaluate adjacent-view, scale-variant, and anti-aliasing consistency for render folders."""
    if max_temporal_pairs is not None and max_temporal_pairs < 0:
        raise ValueError("max_temporal_pairs must be non-negative or None.")
    if max_view_translation is not None and max_view_translation < 0.0:
        raise ValueError("max_view_translation must be non-negative or None.")
    if max_view_rotation_deg is not None and max_view_rotation_deg < 0.0:
        raise ValueError("max_view_rotation_deg must be non-negative or None.")
    aa_factors = validate_aa_downsample_factors(aa_downsample_factors)

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

    temporal, temporal_skipped = _temporal_rows(
        loaded,
        scene_meta,
        method_name,
        split,
        max_temporal_pairs,
        max_view_translation=max_view_translation,
        max_view_rotation_deg=max_view_rotation_deg,
    )
    scale, scale_missing = _scale_rows(scene_root, scale_prediction_dirs or {}, loaded, frames, scene_meta, method_name, split)
    if scale_missing and require_all:
        raise FileNotFoundError(f"Missing {len(scale_missing)} scale-variant prediction images: {scale_missing[:5]}")
    aa = _anti_aliasing_rows(loaded, scene_meta, method_name, split, aa_factors)

    summary = _summary(
        scene_meta,
        method_name,
        split,
        loaded,
        missing,
        scale_missing,
        temporal,
        scale,
        aa,
        scale_prediction_dirs or {},
        aa_factors,
        temporal_skipped,
    )
    temporal_path = out_dir / f"{method_name}_{split}_temporal_consistency.csv"
    scale_path = out_dir / f"{method_name}_{split}_scale_consistency.csv"
    aa_path = out_dir / f"{method_name}_{split}_antialiasing_consistency.csv"
    summary_path = out_dir / f"{method_name}_{split}_render_consistency_summary.csv"
    status_path = out_dir / f"{method_name}_{split}_render_consistency_status.json"
    report_path = out_dir / f"{method_name}_{split}_render_consistency_report.md"
    temporal.to_csv(temporal_path, index=False)
    scale.to_csv(scale_path, index=False)
    aa.to_csv(aa_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    status = {
        "schema_version": 2,
        "scene": scene_meta["scene"],
        "method": method_name,
        "split": split,
        "scene_dir": str(scene_root),
        "predictions_dir": str(pred_root),
        "output_dir": str(out_dir),
        "temporal_path": str(temporal_path),
        "scale_path": str(scale_path),
        "aa_path": str(aa_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "scale_prediction_dirs": {label: str(path) for label, path in (scale_prediction_dirs or {}).items()},
        "aa_downsample_factors": list(aa_factors),
        "max_view_translation": max_view_translation,
        "max_view_rotation_deg": max_view_rotation_deg,
        "temporal_skipped_view_filter_count": temporal_skipped,
        "missing_predictions": missing,
        "scale_missing_predictions": scale_missing,
        "summary": summary,
    }
    write_json(status_path, status)
    report_path.write_text(_format_report(status, temporal, scale, aa), encoding="utf-8")
    return status


def validate_aa_downsample_factors(factors: Sequence[int] | None) -> tuple[int, ...]:
    if factors is None:
        return ()
    validated: list[int] = []
    for factor in factors:
        factor_int = int(factor)
        if factor_int < 2:
            raise ValueError("aa_downsample_factors must contain integers >= 2.")
        if factor_int not in validated:
            validated.append(factor_int)
    return tuple(validated)


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
    loaded: list[dict[str, Any]],
    scene_meta: dict[str, Any],
    method_name: str,
    split: str,
    max_pairs: int | None,
    *,
    max_view_translation: float | None,
    max_view_rotation_deg: float | None,
) -> tuple[pd.DataFrame, int]:
    rows = []
    skipped_view_filter = 0
    for raw_pair_index, (left, right) in enumerate(zip(loaded[:-1], loaded[1:])):
        pose_delta = camera_pose_delta(left["frame"], right["frame"])
        if not view_delta_passes_filter(pose_delta, max_view_translation=max_view_translation, max_view_rotation_deg=max_view_rotation_deg):
            skipped_view_filter += 1
            continue
        if max_pairs is not None and len(rows) >= max_pairs:
            break
        left_prediction, right_prediction = match_pair_shapes(left["prediction"], right["prediction"])
        left_target, right_target = match_pair_shapes(left["target"], right["target"])
        pred_stats = image_difference_stats(left_prediction, right_prediction)
        target_stats = image_difference_stats(left_target, right_target)
        pred_edge = edge_delta_mad(left_prediction, right_prediction)
        target_edge = edge_delta_mad(left_target, right_target)
        pred_nonblack_delta = abs(nonblack_fraction(left_prediction) - nonblack_fraction(right_prediction))
        target_nonblack_delta = abs(nonblack_fraction(left_target) - nonblack_fraction(right_target))
        temporal_instability = max(0.0, pred_stats["mad"] - target_stats["mad"]) + max(0.0, pred_edge - target_edge)
        view_motion_score = pose_delta["view_motion_score"]
        rows.append(
            {
                "scene": scene_meta["scene"],
                "method": method_name,
                "split": split,
                "pair_index": len(rows),
                "raw_pair_index": raw_pair_index,
                "left_frame_index": left["frame_index"],
                "right_frame_index": right["frame_index"],
                "left_image_path": left["frame"]["image_path"],
                "right_image_path": right["frame"]["image_path"],
                "left_prediction_path": str(left["prediction_path"]),
                "right_prediction_path": str(right["prediction_path"]),
                "prediction_resized_to_target": bool(left["prediction_resized_to_target"] or right["prediction_resized_to_target"]),
                "comparison_height": int(left_prediction.shape[0]),
                "comparison_width": int(left_prediction.shape[1]),
                "camera_translation_delta": pose_delta["translation_delta"],
                "camera_rotation_delta_deg": pose_delta["rotation_delta_deg"],
                "view_motion_score": view_motion_score,
                "view_filter_max_translation": max_view_translation,
                "view_filter_max_rotation_deg": max_view_rotation_deg,
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
                "delta_mad_excess_per_view_motion": safe_ratio(pred_stats["mad"] - target_stats["mad"], view_motion_score),
                "edge_delta_mad_excess_per_view_motion": safe_ratio(pred_edge - target_edge, view_motion_score),
                "prediction_edge_delta_mad": pred_edge,
                "target_edge_delta_mad": target_edge,
                "edge_delta_mad_excess": pred_edge - target_edge,
                "edge_delta_mad_ratio": safe_ratio(pred_edge, target_edge),
                "prediction_nonblack_fraction_delta": pred_nonblack_delta,
                "target_nonblack_fraction_delta": target_nonblack_delta,
                "nonblack_fraction_delta_excess": pred_nonblack_delta - target_nonblack_delta,
                "temporal_instability_score": temporal_instability,
                "view_instability_score": safe_ratio(temporal_instability, view_motion_score),
            }
        )
    return pd.DataFrame(rows), skipped_view_filter


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


def _anti_aliasing_rows(loaded: list[dict[str, Any]], scene_meta: dict[str, Any], method_name: str, split: str, factors: Sequence[int]) -> pd.DataFrame:
    rows = []
    for factor in factors:
        for base in loaded:
            aa_prediction = antialias_roundtrip(base["prediction"], factor)
            aa_target = antialias_roundtrip(base["target"], factor)
            pred_stats = image_difference_stats(base["prediction"], aa_prediction)
            target_stats = image_difference_stats(base["target"], aa_target)
            pred_edge = edge_delta_mad(base["prediction"], aa_prediction)
            target_edge = edge_delta_mad(base["target"], aa_target)
            rows.append(
                {
                    "scene": scene_meta["scene"],
                    "method": method_name,
                    "split": split,
                    "aa_downsample_factor": int(factor),
                    "frame_index": base["frame_index"],
                    "image_path": base["frame"]["image_path"],
                    "prediction_path": str(base["prediction_path"]),
                    "height": int(base["prediction"].shape[0]),
                    "width": int(base["prediction"].shape[1]),
                    "aa_psnr": psnr_arrays(aa_prediction, base["prediction"]),
                    "aa_ssim": ssim_arrays(aa_prediction, base["prediction"]),
                    "aa_mse": pred_stats["mse"],
                    "aa_rmse": pred_stats["rmse"],
                    "aa_mad": pred_stats["mad"],
                    "aa_max_abs_diff": pred_stats["max_abs_diff"],
                    "aa_edge_mad": pred_edge,
                    "aa_nonblack_fraction_delta": abs(nonblack_fraction(aa_prediction) - nonblack_fraction(base["prediction"])),
                    "target_aa_mad": target_stats["mad"],
                    "target_aa_edge_mad": target_edge,
                    "aa_mad_excess": pred_stats["mad"] - target_stats["mad"],
                    "aa_edge_mad_excess": pred_edge - target_edge,
                    "aa_instability_score": max(0.0, pred_stats["mad"] - target_stats["mad"]) + max(0.0, pred_edge - target_edge),
                }
            )
    return pd.DataFrame(rows)


def _summary(
    scene_meta: dict[str, Any],
    method_name: str,
    split: str,
    loaded: list[dict[str, Any]],
    missing: list[str],
    scale_missing: list[str],
    temporal: pd.DataFrame,
    scale: pd.DataFrame,
    aa: pd.DataFrame,
    scale_prediction_dirs: Mapping[str, str | Path],
    aa_factors: Sequence[int],
    temporal_skipped_view_filter: int,
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
        "temporal_skipped_view_filter_count": int(temporal_skipped_view_filter),
        "mean_view_translation_delta": optional_mean(temporal, "camera_translation_delta"),
        "mean_view_rotation_delta_deg": optional_mean(temporal, "camera_rotation_delta_deg"),
        "mean_view_motion_score": optional_mean(temporal, "view_motion_score"),
        "mean_view_instability_score": optional_mean(temporal, "view_instability_score"),
        "scale_variant_count": int(len(scale_prediction_dirs)),
        "scale_image_count": int(len(scale)),
        "aa_factor_count": int(len(aa_factors)),
        "aa_image_count": int(len(aa)),
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
        "mean_aa_psnr": optional_mean(aa, "aa_psnr"),
        "mean_aa_ssim": optional_mean(aa, "aa_ssim"),
        "mean_aa_mad": optional_mean(aa, "aa_mad"),
        "max_aa_mad": optional_max(aa, "aa_mad"),
        "mean_aa_mad_excess": optional_mean(aa, "aa_mad_excess"),
        "mean_aa_edge_mad_excess": optional_mean(aa, "aa_edge_mad_excess"),
        "mean_aa_instability_score": optional_mean(aa, "aa_instability_score"),
        "max_aa_instability_score": optional_max(aa, "aa_instability_score"),
    }


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return resize_to_shape(image, reference.shape[:2])


def resize_to_shape(image: np.ndarray, shape_hw: tuple[int, int], *, resampling_name: str = "bicubic") -> np.ndarray:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    resampling_container = getattr(Image, "Resampling", Image)
    resampling_map = {"bicubic": resampling_container.BICUBIC, "lanczos": resampling_container.LANCZOS, "bilinear": resampling_container.BILINEAR}
    resampling = resampling_map[resampling_name]
    return np.asarray(Image.fromarray(to_uint8(image), mode="RGB").resize((width, height), resample=resampling).convert("RGB"), dtype=np.float64) / 255.0


def antialias_roundtrip(image: np.ndarray, downsample_factor: int) -> np.ndarray:
    factor = int(downsample_factor)
    if factor < 2:
        raise ValueError("downsample_factor must be >= 2.")
    height, width = image.shape[:2]
    low_shape = (max(1, int(round(height / factor))), max(1, int(round(width / factor))))
    low = resize_to_shape(image, low_shape, resampling_name="lanczos")
    return resize_to_shape(low, (height, width), resampling_name="bicubic")


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


def camera_pose_delta(left_frame: Mapping[str, Any], right_frame: Mapping[str, Any]) -> dict[str, float]:
    left_pose = camera_to_world_matrix(left_frame)
    right_pose = camera_to_world_matrix(right_frame)
    if left_pose is None or right_pose is None:
        return {"translation_delta": np.nan, "rotation_delta_deg": np.nan, "view_motion_score": np.nan}
    translation_delta = float(np.linalg.norm(right_pose[:3, 3] - left_pose[:3, 3]))
    relative_rotation = left_pose[:3, :3].T @ right_pose[:3, :3]
    cos_angle = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
    rotation_delta_rad = float(math.acos(cos_angle))
    return {"translation_delta": translation_delta, "rotation_delta_deg": float(math.degrees(rotation_delta_rad)), "view_motion_score": float(translation_delta + rotation_delta_rad)}


def camera_to_world_matrix(frame: Mapping[str, Any]) -> np.ndarray | None:
    camera_to_world = frame.get("camera_to_world")
    if camera_to_world is not None:
        matrix = _safe_matrix(camera_to_world)
        if matrix is not None:
            return matrix
    world_to_camera = frame.get("world_to_camera")
    if world_to_camera is not None:
        matrix = _safe_matrix(world_to_camera)
        if matrix is not None:
            try:
                return np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                return None
    return None


def _safe_matrix(value: Any) -> np.ndarray | None:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    return matrix


def view_delta_passes_filter(pose_delta: Mapping[str, float], *, max_view_translation: float | None, max_view_rotation_deg: float | None) -> bool:
    if max_view_translation is not None:
        translation = pose_delta["translation_delta"]
        if not np.isfinite(translation) or translation > max_view_translation:
            return False
    if max_view_rotation_deg is not None:
        rotation = pose_delta["rotation_delta_deg"]
        if not np.isfinite(rotation) or rotation > max_view_rotation_deg:
            return False
    return True


def nonblack_fraction(image: np.ndarray) -> float:
    return float(np.mean(np.any(image > (0.5 / 255.0), axis=2)))


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator):
        return np.nan
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


def _format_report(status: dict[str, Any], temporal: pd.DataFrame, scale: pd.DataFrame, aa: pd.DataFrame) -> str:
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
        f"- Temporal/view pairs: `{summary['temporal_pair_count']}`",
        f"- Skipped by view-motion filter: `{summary['temporal_skipped_view_filter_count']}`",
        f"- Mean view translation delta: `{format_optional(summary['mean_view_translation_delta'])}`",
        f"- Mean view rotation delta deg: `{format_optional(summary['mean_view_rotation_delta_deg'])}`",
        f"- Mean temporal instability score: `{format_optional(summary['mean_temporal_instability_score'])}`",
        f"- Mean view-normalized instability score: `{format_optional(summary['mean_view_instability_score'])}`",
        f"- Scale variants: `{summary['scale_variant_count']}`",
        f"- Scale comparisons: `{summary['scale_image_count']}`",
        f"- Mean scale MAD: `{format_optional(summary['mean_scale_mad'])}`",
        f"- AA downsample factors: `{status['aa_downsample_factors']}`",
        f"- AA comparisons: `{summary['aa_image_count']}`",
        f"- Mean AA MAD: `{format_optional(summary['mean_aa_mad'])}`",
        f"- Mean AA instability score: `{format_optional(summary['mean_aa_instability_score'])}`",
        f"- Temporal CSV: `{status['temporal_path']}`",
        f"- Scale CSV: `{status['scale_path']}`",
        f"- AA CSV: `{status['aa_path']}`",
        f"- Summary CSV: `{status['summary_path']}`",
    ]
    if status["missing_predictions"]:
        lines.extend(["", "## Missing Predictions", *[f"- `{item}`" for item in status["missing_predictions"]]])
    if status["scale_missing_predictions"]:
        lines.extend(["", "## Missing Scale Predictions", *[f"- `{item}`" for item in status["scale_missing_predictions"]]])
    if not temporal.empty:
        lines.extend(["", "## Worst View Pairs", "", _format_temporal_table(temporal.sort_values("temporal_instability_score", ascending=False).head(8))])
    if not scale.empty:
        lines.extend(["", "## Worst Scale Comparisons", "", _format_scale_table(scale.sort_values("scale_instability_score", ascending=False).head(8))])
    if not aa.empty:
        lines.extend(["", "## Worst Anti-Aliasing Round Trips", "", _format_aa_table(aa.sort_values("aa_instability_score", ascending=False).head(8))])
    return "\n".join(lines) + "\n"


def _format_temporal_table(rows: pd.DataFrame) -> str:
    lines = [
        "| pair | left | right | translation | rotation deg | pred MAD | target MAD | excess MAD | edge excess | score | view score |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.pair_index} | {row.left_frame_index} | {row.right_frame_index} | {format_optional(row.camera_translation_delta)} | "
            f"{format_optional(row.camera_rotation_delta_deg)} | {format_optional(row.prediction_delta_mad)} | {format_optional(row.target_delta_mad)} | "
            f"{format_optional(row.delta_mad_excess)} | {format_optional(row.edge_delta_mad_excess)} | {format_optional(row.temporal_instability_score)} | "
            f"{format_optional(row.view_instability_score)} |"
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


def _format_aa_table(rows: pd.DataFrame) -> str:
    lines = ["| factor | frame | PSNR | SSIM | MAD | target MAD | excess MAD | edge excess | score |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.aa_downsample_factor} | {row.frame_index} | {format_optional(row.aa_psnr)} | {format_optional(row.aa_ssim)} | "
            f"{format_optional(row.aa_mad)} | {format_optional(row.target_aa_mad)} | {format_optional(row.aa_mad_excess)} | "
            f"{format_optional(row.aa_edge_mad_excess)} | {format_optional(row.aa_instability_score)} |"
        )
    return "\n".join(lines)


def format_optional(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value_float = float(value)
    if np.isinf(value_float):
        return "inf" if value_float > 0.0 else "-inf"
    return f"{value_float:.6g}"
