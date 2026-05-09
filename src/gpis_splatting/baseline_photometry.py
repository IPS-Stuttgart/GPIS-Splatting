from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.gsplat_adapter import render_3dgs_ply_with_gsplat
from gpis_splatting.real_benchmark import evaluate_real_renders, find_prediction_image, psnr_arrays, ssim_arrays
from gpis_splatting.real_scene import load_prepared_scene
from gpis_splatting.renderer import load_image
from gpis_splatting.serialization import write_json


def validate_3dgs_baseline_photometry(
    *,
    input_ply_path: str | Path,
    scene_dir: str | Path,
    output_dir: str | Path,
    reference_predictions_dir: str | Path | None = None,
    split: str = "test",
    projection_convention: str = "auto",
    device: str = "auto",
    dtype: str = "float32",
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
    compute_lpips: bool = False,
    min_reference_psnr: float | None = 55.0,
    max_reference_l1: float | None = 0.002,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_dir = out_dir / "renders"
    render_result = render_3dgs_ply_with_gsplat(
        input_ply_path=input_ply_path,
        scene_dir=scene_dir,
        output_dir=render_dir,
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
    )
    gt_status = evaluate_real_renders(
        scene_dir=scene_dir,
        predictions_dir=render_dir,
        output_dir=out_dir / "gt_evaluation",
        method_name="baseline_3dgs_gsplat",
        split=split,
        compute_lpips=compute_lpips,
        require_all=max_frames is None,
    )
    reference_status = None
    if reference_predictions_dir is not None:
        reference_status = compare_prediction_directories(
            scene_dir=scene_dir,
            candidate_predictions_dir=render_dir,
            reference_predictions_dir=reference_predictions_dir,
            output_dir=out_dir / "reference_comparison",
            split=split,
            max_frames=max_frames,
        )
    passed, reasons = baseline_pass_decision(reference_status, min_reference_psnr, max_reference_l1)
    status = {
        "schema_version": 1,
        "input_ply_path": str(Path(input_ply_path)),
        "scene_dir": str(Path(scene_dir)),
        "output_dir": str(out_dir),
        "render_report_path": str(render_result["report_path"]),
        "gt_evaluation_status": gt_status,
        "reference_comparison_status": reference_status,
        "min_reference_psnr": min_reference_psnr,
        "max_reference_l1": max_reference_l1,
        "passed": passed,
        "pass_reasons": reasons,
    }
    status_path = out_dir / "baseline_3dgs_photometry_status.json"
    report_path = out_dir / "baseline_3dgs_photometry_report.md"
    write_json(status_path, status)
    report_path.write_text(format_baseline_validation_report(status), encoding="utf-8")
    status["status_path"] = str(status_path)
    status["report_path"] = str(report_path)
    return status


def compare_prediction_directories(*, scene_dir: str | Path, candidate_predictions_dir: str | Path, reference_predictions_dir: str | Path, output_dir: str | Path, split: str = "test", max_frames: int | None = None) -> dict[str, Any]:
    scene_root = Path(scene_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_meta, frames, splits = load_prepared_scene(scene_root)
    indices = [int(i) for i in splits.get(split, [])]
    if max_frames is not None:
        indices = indices[:max_frames]
    rows = []
    for index in indices:
        frame = frames[index]
        candidate_path = find_prediction_image(candidate_predictions_dir, frame)
        reference_path = find_prediction_image(reference_predictions_dir, frame)
        if candidate_path is None or reference_path is None:
            continue
        candidate = load_image(candidate_path)
        reference = load_image(reference_path)
        if candidate.shape != reference.shape:
            raise ValueError(f"Candidate shape {candidate.shape} does not match reference shape {reference.shape}.")
        diff = np.abs(candidate - reference)
        rows.append({"scene": scene_meta["scene"], "split": split, "frame_index": index, "psnr_vs_reference": psnr_arrays(candidate, reference), "ssim_vs_reference": ssim_arrays(candidate, reference), "mean_l1_vs_reference": float(diff.mean()), "max_l1_vs_reference": float(diff.max()), "candidate_path": str(candidate_path), "reference_path": str(reference_path)})
    if not rows:
        raise ValueError("No candidate/reference image pairs were available for baseline validation.")
    metrics = pd.DataFrame(rows)
    metrics_path = out_dir / "baseline_3dgs_reference_image_metrics.csv"
    summary_path = out_dir / "baseline_3dgs_reference_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    summary = {"scene": scene_meta["scene"], "split": split, "image_count": int(len(metrics)), "mean_psnr_vs_reference": float(metrics["psnr_vs_reference"].mean()), "mean_ssim_vs_reference": float(metrics["ssim_vs_reference"].mean()), "mean_l1_vs_reference": float(metrics["mean_l1_vs_reference"].mean()), "max_l1_vs_reference": float(metrics["max_l1_vs_reference"].max())}
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    status = {"schema_version": 1, "metrics_path": str(metrics_path), "summary_path": str(summary_path), "summary": summary}
    write_json(out_dir / "baseline_3dgs_reference_status.json", status)
    return status


def baseline_pass_decision(reference_status: dict[str, Any] | None, min_reference_psnr: float | None, max_reference_l1: float | None) -> tuple[bool, list[str]]:
    if reference_status is None:
        return False, ["No reference renderer directory was provided; only GT metrics were computed."]
    summary = reference_status["summary"]
    reasons = []
    passed = True
    if min_reference_psnr is not None and summary["mean_psnr_vs_reference"] < min_reference_psnr:
        passed = False
        reasons.append(f"mean_psnr_vs_reference={summary['mean_psnr_vs_reference']:.6g} < {min_reference_psnr:.6g}")
    if max_reference_l1 is not None and summary["mean_l1_vs_reference"] > max_reference_l1:
        passed = False
        reasons.append(f"mean_l1_vs_reference={summary['mean_l1_vs_reference']:.6g} > {max_reference_l1:.6g}")
    if passed:
        reasons.append("Reference-render agreement thresholds passed.")
    return passed, reasons


def format_baseline_validation_report(status: dict[str, Any]) -> str:
    gt = status["gt_evaluation_status"]["summary"]
    lines = ["# Baseline 3DGS Photometry Validation", "", f"- Input PLY: `{status['input_ply_path']}`", f"- Scene: `{status['scene_dir']}`", f"- Passed: `{status['passed']}`", f"- Render report: `{status['render_report_path']}`", "", "## Ground-truth image metrics", "", f"- Mean PSNR: `{gt['mean_psnr']:.6g}`", f"- Mean SSIM: `{gt['mean_ssim']:.6g}`", f"- Mean LPIPS VGG: `{gt.get('mean_lpips_vgg')}`"]
    reference = status.get("reference_comparison_status")
    if reference is not None:
        summary = reference["summary"]
        lines += ["", "## Reference-render agreement", "", f"- Mean PSNR vs reference: `{summary['mean_psnr_vs_reference']:.6g}`", f"- Mean SSIM vs reference: `{summary['mean_ssim_vs_reference']:.6g}`", f"- Mean L1 vs reference: `{summary['mean_l1_vs_reference']:.6g}`", f"- Max L1 vs reference: `{summary['max_l1_vs_reference']:.6g}`", f"- Metrics CSV: `{reference['metrics_path']}`"]
    lines += ["", "## Decision", "", *[f"- {reason}" for reason in status["pass_reasons"]]]
    return "\n".join(lines) + "\n"
