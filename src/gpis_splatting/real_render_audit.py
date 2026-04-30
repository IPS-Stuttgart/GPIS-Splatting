from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from gpis_splatting.real_benchmark import find_prediction_image, psnr_arrays, ssim_arrays
from gpis_splatting.real_scene import load_prepared_scene, resolve_scene_image_path
from gpis_splatting.renderer import load_image
from gpis_splatting.serialization import read_json, write_json


def audit_real_renders(
    *,
    scene_dir: str | Path,
    predictions_dir: str | Path,
    output_dir: str | Path,
    method_name: str,
    split: str = "test",
    require_all: bool = True,
    max_panels: int = 16,
    fail_on_suspicious: bool = False,
) -> dict[str, Any]:
    if max_panels < 0:
        raise ValueError("max_panels must be non-negative.")
    scene_root = Path(scene_dir)
    pred_root = Path(predictions_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = out_dir / f"{method_name}_{split}_render_audit_panels"
    if max_panels > 0:
        panel_dir.mkdir(parents=True, exist_ok=True)

    scene_meta, frames, splits = load_prepared_scene(scene_root)
    split_indices = splits.get(split)
    if split_indices is None:
        raise ValueError(f"Split {split!r} does not exist in {scene_root / 'splits.json'}.")
    render_report = load_render_report(pred_root)
    render_outputs = render_outputs_by_frame(render_report)

    rows: list[dict[str, Any]] = []
    missing = []
    warnings = []
    panel_count = 0
    for index in split_indices:
        frame = frames[int(index)]
        target_path = resolve_scene_image_path(scene_root, frame["image_path"])
        prediction_path = find_prediction_image(pred_root, frame)
        if prediction_path is None:
            missing.append(frame["image_path"])
            rows.append(
                {
                    "scene": scene_meta["scene"],
                    "method": method_name,
                    "split": split,
                    "frame_index": int(index),
                    "image_path": frame["image_path"],
                    "target_path": str(target_path),
                    "prediction_path": None,
                    "missing_prediction": True,
                }
            )
            continue
        target = load_image(target_path)
        prediction = load_image(prediction_path)
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction shape {prediction.shape} for {prediction_path} does not match target shape {target.shape}.")
        stats = image_pair_stats(target=target, prediction=prediction)
        paths_identical = paths_refer_to_same_file(target_path, prediction_path)
        exact_pixels = bool(stats["mse"] <= 1e-12)
        infinite_psnr = bool(np.isinf(stats["psnr"]))
        frame_warnings = frame_audit_warnings(
            paths_identical=paths_identical,
            exact_pixels=exact_pixels,
            infinite_psnr=infinite_psnr,
            prediction_path=prediction_path,
            frame_index=int(index),
        )
        warnings.extend(frame_warnings)
        render_output = render_outputs.get(int(index), {})
        panel_path = None
        if panel_count < max_panels:
            panel_path = panel_dir / f"frame_{int(index):06d}_panel.png"
            save_audit_panel(panel_path, target=target, prediction=prediction)
            panel_count += 1
        rows.append(
            {
                "scene": scene_meta["scene"],
                "method": method_name,
                "split": split,
                "frame_index": int(index),
                "image_path": frame["image_path"],
                "target_path": str(target_path),
                "prediction_path": str(prediction_path),
                "panel_path": str(panel_path) if panel_path is not None else None,
                "missing_prediction": False,
                "paths_identical": paths_identical,
                "exact_pixels": exact_pixels,
                "infinite_psnr": infinite_psnr,
                "warning_count": len(frame_warnings),
                "projected_splat_count": render_output.get("projected_splat_count"),
                "drawn_splat_count": render_output.get("drawn_splat_count"),
                "min_depth": render_output.get("min_depth"),
                "max_depth": render_output.get("max_depth"),
                **stats,
            }
        )

    if missing and require_all:
        raise FileNotFoundError(f"Missing {len(missing)} prediction images under {pred_root}: {missing[:5]}")
    rows_df = pd.DataFrame(rows)
    summary = summarize_audit(rows_df=rows_df, scene_meta=scene_meta, method_name=method_name, split=split, warning_count=len(warnings))
    metrics_path = out_dir / f"{method_name}_{split}_render_audit.csv"
    summary_path = out_dir / f"{method_name}_{split}_render_audit_summary.csv"
    status_path = out_dir / f"{method_name}_{split}_render_audit_status.json"
    report_path = out_dir / f"{method_name}_{split}_render_audit_report.md"
    rows_df.to_csv(metrics_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    status = {
        "schema_version": 1,
        "scene": scene_meta["scene"],
        "method": method_name,
        "split": split,
        "scene_dir": str(scene_root),
        "predictions_dir": str(pred_root),
        "output_dir": str(out_dir),
        "render_report_path": str(pred_root / "real_render_report.json") if render_report is not None else None,
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "panel_dir": str(panel_dir) if max_panels > 0 else None,
        "warnings": warnings,
        "summary": summary,
    }
    write_json(status_path, status)
    report_path.write_text(format_audit_report(status, rows_df), encoding="utf-8")
    if fail_on_suspicious and summary["suspicious_infinite_psnr_count"] > 0:
        raise ValueError(f"Suspicious render audit found {summary['suspicious_infinite_psnr_count']} infinite-PSNR comparisons.")
    return status


def image_pair_stats(*, target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    diff = prediction - target
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff**2))
    return {
        "psnr": psnr_arrays(prediction, target),
        "ssim": ssim_arrays(prediction, target),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
        "min_signed_diff": float(diff.min()),
        "max_signed_diff": float(diff.max()),
        "target_mean": float(target.mean()),
        "prediction_mean": float(prediction.mean()),
        "target_std": float(target.std()),
        "prediction_std": float(prediction.std()),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "target_nonblack_fraction": nonblack_fraction(target),
        "prediction_nonblack_fraction": nonblack_fraction(prediction),
        "target_nonwhite_fraction": nonwhite_fraction(target),
        "prediction_nonwhite_fraction": nonwhite_fraction(prediction),
    }


def summarize_audit(*, rows_df: pd.DataFrame, scene_meta: dict[str, Any], method_name: str, split: str, warning_count: int) -> dict[str, Any]:
    evaluated = rows_df[~rows_df.get("missing_prediction", False).astype(bool)] if not rows_df.empty else rows_df
    summary: dict[str, Any] = {
        "scene": scene_meta["scene"],
        "dataset": scene_meta.get("dataset"),
        "method": method_name,
        "split": split,
        "image_count": int(len(evaluated)),
        "missing_count": int(rows_df.get("missing_prediction", pd.Series(dtype=bool)).astype(bool).sum()) if not rows_df.empty else 0,
        "warning_count": int(warning_count),
        "identical_path_count": count_true(evaluated, "paths_identical"),
        "exact_pixel_match_count": count_true(evaluated, "exact_pixels"),
        "infinite_psnr_count": count_true(evaluated, "infinite_psnr"),
        "suspicious_infinite_psnr_count": int(
            ((evaluated.get("infinite_psnr", False).astype(bool)) & (~evaluated.get("paths_identical", False).astype(bool))).sum()
        )
        if not evaluated.empty
        else 0,
        "mean_mse": optional_mean(evaluated, "mse"),
        "mean_abs_diff": optional_mean(evaluated, "mean_abs_diff"),
        "max_abs_diff": optional_max(evaluated, "max_abs_diff"),
        "mean_prediction_nonblack_fraction": optional_mean(evaluated, "prediction_nonblack_fraction"),
        "mean_target_nonblack_fraction": optional_mean(evaluated, "target_nonblack_fraction"),
        "mean_drawn_splat_count": optional_mean(evaluated, "drawn_splat_count"),
    }
    return summary


def frame_audit_warnings(
    *,
    paths_identical: bool,
    exact_pixels: bool,
    infinite_psnr: bool,
    prediction_path: Path,
    frame_index: int,
) -> list[str]:
    warnings = []
    if paths_identical:
        warnings.append(f"frame {frame_index}: prediction path is the target image path ({prediction_path}).")
    if infinite_psnr and not exact_pixels:
        warnings.append(f"frame {frame_index}: PSNR is infinite but pixel arrays are not exactly identical.")
    if infinite_psnr and exact_pixels and not paths_identical:
        warnings.append(f"frame {frame_index}: prediction pixels exactly match target pixels from a different path.")
    return warnings


def save_audit_panel(path: str | Path, *, target: np.ndarray, prediction: np.ndarray) -> None:
    target_u8 = to_uint8(target)
    prediction_u8 = to_uint8(prediction)
    abs_diff = np.abs(prediction - target)
    scale = float(abs_diff.max())
    diff_u8 = to_uint8(abs_diff / scale) if scale > 1e-12 else np.zeros_like(target_u8)
    spacer = np.full((target_u8.shape[0], 4, 3), 255, dtype=np.uint8)
    panel = np.concatenate((target_u8, spacer, prediction_u8, spacer, diff_u8), axis=1)
    Image.fromarray(panel, mode="RGB").save(path)


def load_render_report(predictions_dir: Path) -> dict[str, Any] | None:
    path = predictions_dir / "real_render_report.json"
    if not path.exists():
        return None
    return read_json(path)


def render_outputs_by_frame(report: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if report is None:
        return {}
    return {int(row["frame_index"]): row for row in report.get("outputs", [])}


def paths_refer_to_same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return left.resolve() == right.resolve()


def nonblack_fraction(image: np.ndarray) -> float:
    return float(np.mean(np.any(image > (0.5 / 255.0), axis=2)))


def nonwhite_fraction(image: np.ndarray) -> float:
    return float(np.mean(np.any(image < (254.5 / 255.0), axis=2)))


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def count_true(table: pd.DataFrame, column: str) -> int:
    if table.empty or column not in table:
        return 0
    return int(table[column].astype(bool).sum())


def optional_mean(table: pd.DataFrame, column: str) -> float | None:
    if table.empty or column not in table:
        return None
    values = pd.to_numeric(table[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def optional_max(table: pd.DataFrame, column: str) -> float | None:
    if table.empty or column not in table:
        return None
    values = pd.to_numeric(table[column], errors="coerce").dropna()
    return None if values.empty else float(values.max())


def format_audit_report(status: dict[str, Any], rows: pd.DataFrame) -> str:
    summary = status["summary"]
    lines = [
        "# Real Render Audit",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Method: `{status['method']}`",
        f"- Split: `{status['split']}`",
        f"- Images audited: `{summary['image_count']}`",
        f"- Missing predictions: `{summary['missing_count']}`",
        f"- Identical paths: `{summary['identical_path_count']}`",
        f"- Exact pixel matches: `{summary['exact_pixel_match_count']}`",
        f"- Infinite PSNR count: `{summary['infinite_psnr_count']}`",
        f"- Suspicious infinite PSNR count: `{summary['suspicious_infinite_psnr_count']}`",
        f"- Mean MSE: `{format_optional(summary['mean_mse'])}`",
        f"- Mean abs diff: `{format_optional(summary['mean_abs_diff'])}`",
        f"- Max abs diff: `{format_optional(summary['max_abs_diff'])}`",
        f"- Metrics CSV: `{status['metrics_path']}`",
        f"- Panel directory: `{status['panel_dir']}`",
    ]
    if status["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in status["warnings"])
    evaluated = rows[~rows.get("missing_prediction", False).astype(bool)] if not rows.empty else rows
    if not evaluated.empty:
        lines.extend(["", "## Per Image", "", format_rows_table(evaluated)])
    return "\n".join(lines) + "\n"


def format_rows_table(rows: pd.DataFrame) -> str:
    columns = ["frame_index", "paths_identical", "exact_pixels", "psnr", "mse", "mean_abs_diff", "prediction_nonblack_fraction", "drawn_splat_count"]
    lines = [
        "| frame | same path | exact pixels | psnr | mse | mean abs diff | pred nonblack | drawn splats |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows[columns].itertuples(index=False):
        drawn = "n/a" if pd.isna(row.drawn_splat_count) else f"{row.drawn_splat_count:.0f}"
        lines.append(
            f"| {row.frame_index} | `{bool(row.paths_identical)}` | `{bool(row.exact_pixels)}` | {format_optional(row.psnr)} | "
            f"{format_optional(row.mse)} | {format_optional(row.mean_abs_diff)} | {format_optional(row.prediction_nonblack_fraction)} | {drawn} |"
        )
    return "\n".join(lines)


def format_optional(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if np.isinf(float(value)):
        return "inf"
    return f"{float(value):.6g}"
