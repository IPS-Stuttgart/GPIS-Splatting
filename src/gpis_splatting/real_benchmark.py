from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.real_scene import load_prepared_scene, resolve_scene_image_path
from gpis_splatting.renderer import load_image
from gpis_splatting.serialization import read_json, write_json


def evaluate_real_renders(
    *,
    scene_dir: str | Path,
    predictions_dir: str | Path,
    output_dir: str | Path,
    method_name: str,
    split: str = "test",
    benchmark_target: str | Path | None = None,
    compute_lpips: bool = False,
    require_all: bool = True,
) -> dict[str, Any]:
    scene_root = Path(scene_dir)
    pred_root = Path(predictions_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_meta, frames, splits = load_prepared_scene(scene_root)
    split_indices = splits.get(split)
    if split_indices is None:
        raise ValueError(f"Split {split!r} does not exist in {scene_root / 'splits.json'}.")
    lpips_model, lpips_status = _load_lpips_model(compute_lpips)
    rows = []
    missing = []
    for index in split_indices:
        frame = frames[int(index)]
        target_path = resolve_scene_image_path(scene_root, frame["image_path"])
        pred_path = find_prediction_image(pred_root, frame)
        if pred_path is None:
            missing.append(frame["image_path"])
            if require_all:
                continue
            continue
        target = load_image(target_path)
        prediction = load_image(pred_path)
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction shape {prediction.shape} for {pred_path} does not match target shape {target.shape}.")
        rows.append(
            {
                "scene": scene_meta["scene"],
                "method": method_name,
                "split": split,
                "frame_index": int(index),
                "image_path": frame["image_path"],
                "prediction_path": str(pred_path),
                "psnr": psnr_arrays(prediction, target),
                "ssim": ssim_arrays(prediction, target),
                "lpips_vgg": lpips_arrays(lpips_model, prediction, target) if lpips_model is not None else np.nan,
                "lpips_status": lpips_status,
            }
        )
    if missing and require_all:
        raise FileNotFoundError(f"Missing {len(missing)} prediction images under {pred_root}: {missing[:5]}")
    if not rows:
        raise ValueError(f"No prediction images were evaluated for split {split!r}.")

    metrics = pd.DataFrame(rows)
    target = _load_target(benchmark_target)
    summary = build_real_summary(metrics, scene_meta=scene_meta, method_name=method_name, split=split, missing_count=len(missing), target=target)
    metrics_path = out_dir / f"{method_name}_{split}_image_metrics.csv"
    summary_path = out_dir / f"{method_name}_{split}_summary.csv"
    status_path = out_dir / f"{method_name}_{split}_status.json"
    report_path = out_dir / f"{method_name}_{split}_report.md"
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    status = {
        "scene": scene_meta["scene"],
        "method": method_name,
        "split": split,
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "image_count": int(len(metrics)),
        "missing_count": int(len(missing)),
        "lpips_status": lpips_status,
        "target": target,
        "summary": summary,
    }
    write_json(status_path, status)
    report_path.write_text(format_real_report(status), encoding="utf-8")
    return status


def find_prediction_image(predictions_dir: str | Path, frame: dict[str, Any]) -> Path | None:
    root = Path(predictions_dir)
    image_path = Path(frame["image_path"])
    candidates = [
        root / image_path,
        root / image_path.name,
        root / frame["file_name"],
        root / f"{frame['index']:06d}{image_path.suffix}",
        root / f"{frame['index']:03d}{image_path.suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_real_summary(
    metrics: pd.DataFrame,
    *,
    scene_meta: dict[str, Any],
    method_name: str,
    split: str,
    missing_count: int,
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "scene": scene_meta["scene"],
        "dataset": scene_meta.get("dataset"),
        "method": method_name,
        "split": split,
        "image_count": int(len(metrics)),
        "missing_count": int(missing_count),
        "mean_psnr": float(metrics["psnr"].mean()),
        "mean_ssim": float(metrics["ssim"].mean()),
        "mean_lpips_vgg": _mean_or_none(metrics["lpips_vgg"]),
    }
    if target is not None:
        baseline_name = target.get("primary_baseline")
        baseline = target.get("reference_baselines", {}).get(baseline_name, {})
        if "psnr" in baseline:
            summary["target_baseline_psnr"] = float(baseline["psnr"])
            summary["psnr_delta_vs_target_baseline"] = summary["mean_psnr"] - float(baseline["psnr"])
        if "ssim" in baseline:
            summary["target_baseline_ssim"] = float(baseline["ssim"])
            summary["ssim_delta_vs_target_baseline"] = summary["mean_ssim"] - float(baseline["ssim"])
        if "lpips_vgg" in baseline and summary["mean_lpips_vgg"] is not None:
            summary["target_baseline_lpips_vgg"] = float(baseline["lpips_vgg"])
            summary["lpips_delta_vs_target_baseline"] = summary["mean_lpips_vgg"] - float(baseline["lpips_vgg"])
    return summary


def format_real_report(status: dict[str, Any]) -> str:
    summary = status["summary"]
    lines = [
        "# Real Render Evaluation",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Method: `{status['method']}`",
        f"- Split: `{status['split']}`",
        f"- Evaluated images: `{status['image_count']}`",
        f"- Missing predictions: `{status['missing_count']}`",
        f"- Mean PSNR: `{summary['mean_psnr']:.6g}`",
        f"- Mean SSIM: `{summary['mean_ssim']:.6g}`",
        f"- Mean LPIPS VGG: `{_format_optional_float(summary['mean_lpips_vgg'])}`",
        f"- LPIPS status: `{status['lpips_status']}`",
        f"- Image metrics: `{status['metrics_path']}`",
        f"- Summary CSV: `{status['summary_path']}`",
    ]
    target = status.get("target")
    if target is not None:
        lines.extend(
            [
                "",
                "## Benchmark Target",
                "",
                f"- Name: `{target.get('name', '')}`",
                f"- Dataset: `{target.get('dataset', '')}`",
                f"- Protocol: {target.get('protocol_url', '')}",
                f"- Primary baseline: `{target.get('primary_baseline', '')}`",
            ]
        )
        for key in ("psnr_delta_vs_target_baseline", "ssim_delta_vs_target_baseline", "lpips_delta_vs_target_baseline"):
            if key in summary:
                lines.append(f"- `{key}`: `{summary[key]:.6g}`")
    return "\n".join(lines) + "\n"


def psnr_arrays(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def ssim_arrays(prediction: np.ndarray, target: np.ndarray) -> float:
    values = []
    c1 = 0.01**2
    c2 = 0.03**2
    for channel in range(prediction.shape[2]):
        pred_channel = prediction[..., channel]
        target_channel = target[..., channel]
        mu_x = float(pred_channel.mean())
        mu_y = float(target_channel.mean())
        var_x = float(pred_channel.var())
        var_y = float(target_channel.var())
        cov_xy = float(((pred_channel - mu_x) * (target_channel - mu_y)).mean())
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        values.append(numerator / denominator if denominator > 0.0 else 1.0)
    return float(np.clip(np.mean(values), -1.0, 1.0))


def lpips_arrays(model: Any, prediction: np.ndarray, target: np.ndarray) -> float:
    import torch

    pred_tensor = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) * 2.0 - 1.0
    target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) * 2.0 - 1.0
    with torch.no_grad():
        return float(model(pred_tensor, target_tensor).item())


def _load_lpips_model(compute_lpips: bool) -> tuple[Any | None, str]:
    if not compute_lpips:
        return None, "disabled"
    try:
        import lpips
    except ImportError:
        return None, "missing_lpips_package"
    model = lpips.LPIPS(net="vgg")
    model.eval()
    return model, "computed_vgg"


def _load_target(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return read_json(path)


def _mean_or_none(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.6g}"
