from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.real_alignment import diagnose_real_alignment
from gpis_splatting.real_benchmark import evaluate_real_renders
from gpis_splatting.real_pipeline import PROJECTION_CONVENTIONS, render_real_splats
from gpis_splatting.real_render_audit import audit_real_renders
from gpis_splatting.real_scene import load_prepared_scene
from gpis_splatting.serialization import write_json
from gpis_splatting.splats import SplatCloud, load_splats, save_splats


@dataclass(frozen=True)
class RenderSweepVariant:
    name: str
    method_name: str
    splats_path: Path
    render_dir: Path
    sigma_scale: float
    tau_scale: float
    min_sigma_px: float
    kernel_radius: float
    background_color: tuple[float, float, float]
    projection_convention: str


def run_real_render_parameter_sweep(
    *,
    scene_dir: str | Path,
    splats_path: str | Path | None = None,
    method_name: str = "render_parameter_sweep",
    output_dir: str | Path | None = None,
    split: str = "test",
    max_frames: int | None = None,
    sigma_scales: tuple[float, ...] = (0.5, 1.0, 1.5),
    tau_scales: tuple[float, ...] = (0.5, 1.0, 2.0),
    min_sigma_pxs: tuple[float, ...] = (0.6, 1.0),
    kernel_radii: tuple[float, ...] = (2.0, 3.0),
    background_colors: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),),
    projection_conventions: tuple[str, ...] = ("auto",),
    near_plane: float = 1e-4,
    selection_metric: str = "mean_psnr",
    run_alignment: bool = True,
    alignment_coverage_downsample: int = 8,
    alignment_max_overlay_splats: int = 1000,
    audit_max_panels: int = 0,
) -> dict[str, Any]:
    validate_sweep_config(
        sigma_scales=sigma_scales,
        tau_scales=tau_scales,
        min_sigma_pxs=min_sigma_pxs,
        kernel_radii=kernel_radii,
        background_colors=background_colors,
        projection_conventions=projection_conventions,
        max_frames=max_frames,
        near_plane=near_plane,
        selection_metric=selection_metric,
        alignment_coverage_downsample=alignment_coverage_downsample,
        alignment_max_overlay_splats=alignment_max_overlay_splats,
        audit_max_panels=audit_max_panels,
    )
    scene_root = Path(scene_dir).resolve()
    scene_meta, _, _ = load_prepared_scene(scene_root)
    resolved_splats = resolve_scene_file(scene_root, splats_path, "real_splats.npz")
    out_dir = Path(output_dir).resolve() if output_dir is not None else scene_root / "evaluations" / method_name
    variants_dir = out_dir / "variant_splats"
    renders_dir = out_dir / "renders"
    alignment_dir = out_dir / "alignment"
    for directory in (out_dir, variants_dir, renders_dir, alignment_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_splats = load_splats(str(resolved_splats))
    variants = build_render_sweep_variants(
        source_splats=source_splats,
        variants_dir=variants_dir,
        renders_dir=renders_dir,
        method_name=method_name,
        sigma_scales=sigma_scales,
        tau_scales=tau_scales,
        min_sigma_pxs=min_sigma_pxs,
        kernel_radii=kernel_radii,
        background_colors=background_colors,
        projection_conventions=projection_conventions,
    )

    rows = []
    statuses = []
    for variant in variants:
        render_result = render_real_splats(
            scene_dir=scene_root,
            splats_path=variant.splats_path,
            output_dir=variant.render_dir,
            method_name=variant.method_name,
            split=split,
            use_gpis_gate=False,
            projection_convention=variant.projection_convention,
            near_plane=near_plane,
            kernel_radius=variant.kernel_radius,
            min_sigma_px=variant.min_sigma_px,
            background_color=variant.background_color,
            max_frames=max_frames,
        )
        eval_status = evaluate_real_renders(
            scene_dir=scene_root,
            predictions_dir=render_result["output_dir"],
            output_dir=out_dir,
            method_name=variant.method_name,
            split=split,
            require_all=False,
            allow_diagnostic_proxy=True,
        )
        audit_status = audit_real_renders(
            scene_dir=scene_root,
            predictions_dir=render_result["output_dir"],
            output_dir=out_dir,
            method_name=variant.method_name,
            split=split,
            require_all=False,
            max_panels=audit_max_panels,
        )
        alignment_status = None
        if run_alignment:
            alignment_result = diagnose_real_alignment(
                scene_dir=scene_root,
                render_dir=render_result["output_dir"],
                splats_path=variant.splats_path,
                output_dir=alignment_dir / variant.method_name,
                split=split,
                max_frames=max_frames,
                projection_convention=variant.projection_convention,
                near_plane=near_plane,
                kernel_radius=variant.kernel_radius,
                min_sigma_px=variant.min_sigma_px,
                coverage_downsample=alignment_coverage_downsample,
                max_overlay_splats=alignment_max_overlay_splats,
                require_predictions=False,
            )
            alignment_status = alignment_result["status"]

        row = build_variant_row(
            variant=variant,
            render_status=render_result["report"],
            eval_summary=eval_status["summary"],
            audit_summary=audit_status["summary"],
            alignment_summary=alignment_status["summary"] if alignment_status is not None else None,
        )
        rows.append(row)
        statuses.append(
            {
                "variant": variant.name,
                "method_name": variant.method_name,
                "splats_path": str(variant.splats_path),
                "render_dir": str(variant.render_dir),
                "render_report_path": str(render_result["report_path"]),
                "evaluation_status": eval_status,
                "audit_summary": audit_status["summary"],
                "alignment_summary": alignment_status["summary"] if alignment_status is not None else None,
            }
        )

    sweep = pd.DataFrame(rows)
    if sweep.empty:
        raise ValueError("Render parameter sweep produced no variants.")
    ranked = rank_sweep(sweep, selection_metric=selection_metric)
    best = ranked.iloc[0].to_dict()
    best_render_dir = copy_best_render_dir(Path(best["render_dir"]), out_dir / "best_render")

    sweep_path = out_dir / "render_parameter_sweep.csv"
    ranked_path = out_dir / "render_parameter_sweep_ranked.csv"
    best_path = out_dir / "best_render_parameters.json"
    status_path = out_dir / "render_parameter_sweep_status.json"
    report_path = out_dir / "render_parameter_sweep_report.md"
    sweep.to_csv(sweep_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    best_payload = {
        "selection_metric": selection_metric,
        "best": best,
        "best_render_dir": str(best_render_dir),
    }
    write_json(best_path, best_payload)
    status = {
        "schema_version": 1,
        "scene": scene_meta["scene"],
        "scene_dir": str(scene_root),
        "input_splats_path": str(resolved_splats),
        "method": method_name,
        "split": split,
        "max_frames": max_frames,
        "output_dir": str(out_dir),
        "variant_count": len(variants),
        "selection_metric": selection_metric,
        "best": best_payload,
        "sweep_path": str(sweep_path),
        "ranked_path": str(ranked_path),
        "best_path": str(best_path),
        "best_render_dir": str(best_render_dir),
        "report_path": str(report_path),
        "variant_statuses": statuses,
    }
    write_json(status_path, status)
    report_path.write_text(format_sweep_report(status, ranked), encoding="utf-8")
    return {
        "output_dir": out_dir,
        "sweep_path": sweep_path,
        "ranked_path": ranked_path,
        "best_path": best_path,
        "status_path": status_path,
        "report_path": report_path,
        "best_render_dir": best_render_dir,
        "status": status,
    }


def build_render_sweep_variants(
    *,
    source_splats: SplatCloud,
    variants_dir: Path,
    renders_dir: Path,
    method_name: str,
    sigma_scales: tuple[float, ...],
    tau_scales: tuple[float, ...],
    min_sigma_pxs: tuple[float, ...],
    kernel_radii: tuple[float, ...],
    background_colors: tuple[tuple[float, float, float], ...],
    projection_conventions: tuple[str, ...],
) -> list[RenderSweepVariant]:
    variants = []
    for sigma_scale, tau_scale, min_sigma_px, kernel_radius, background_color, projection_convention in product(
        sigma_scales,
        tau_scales,
        min_sigma_pxs,
        kernel_radii,
        background_colors,
        projection_conventions,
    ):
        name = variant_name(
            sigma_scale=sigma_scale,
            tau_scale=tau_scale,
            min_sigma_px=min_sigma_px,
            kernel_radius=kernel_radius,
            background_color=background_color,
            projection_convention=projection_convention,
        )
        variant_method = f"{method_name}_{name}"
        variant_splats = scale_splats(source_splats, sigma_scale=sigma_scale, tau_scale=tau_scale)
        variant_splats_path = variants_dir / f"{variant_method}_splats.npz"
        save_splats(str(variant_splats_path), variant_splats)
        variants.append(
            RenderSweepVariant(
                name=name,
                method_name=variant_method,
                splats_path=variant_splats_path,
                render_dir=renders_dir / variant_method,
                sigma_scale=sigma_scale,
                tau_scale=tau_scale,
                min_sigma_px=min_sigma_px,
                kernel_radius=kernel_radius,
                background_color=background_color,
                projection_convention=projection_convention,
            )
        )
    return variants


def scale_splats(splats: SplatCloud, *, sigma_scale: float, tau_scale: float) -> SplatCloud:
    return SplatCloud(
        centers=splats.centers.clone(),
        colors=splats.colors.clone(),
        tau=splats.tau.clone() * float(tau_scale),
        sigma=splats.sigma.clone() * float(sigma_scale),
        is_surface=splats.is_surface.clone(),
    )


def build_variant_row(
    *,
    variant: RenderSweepVariant,
    render_status: dict[str, Any],
    eval_summary: dict[str, Any],
    audit_summary: dict[str, Any],
    alignment_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "variant": variant.name,
        "method_name": variant.method_name,
        "splats_path": str(variant.splats_path),
        "render_dir": str(variant.render_dir),
        "sigma_scale": variant.sigma_scale,
        "tau_scale": variant.tau_scale,
        "min_sigma_px": variant.min_sigma_px,
        "kernel_radius": variant.kernel_radius,
        "background_color": ",".join(f"{value:.6g}" for value in variant.background_color),
        "projection_convention": variant.projection_convention,
        "rendered_image_count": render_status.get("image_count", 0),
        "mean_projected_splat_count": mean_render_output(render_status, "projected_splat_count"),
        "mean_drawn_splat_count": mean_render_output(render_status, "drawn_splat_count"),
        "mean_psnr": eval_summary.get("mean_psnr"),
        "mean_ssim": eval_summary.get("mean_ssim"),
        "evaluated_image_count": eval_summary.get("image_count"),
        "missing_prediction_count": eval_summary.get("missing_count"),
        "mean_mse": audit_summary.get("mean_mse"),
        "mean_abs_diff": audit_summary.get("mean_abs_diff"),
        "mean_prediction_nonblack_fraction": audit_summary.get("mean_prediction_nonblack_fraction"),
        "mean_target_nonblack_fraction": audit_summary.get("mean_target_nonblack_fraction"),
    }
    if alignment_summary is not None:
        row.update(
            {
                "alignment_failure_counts": json.dumps(alignment_summary.get("failure_counts", {}), sort_keys=True),
                "mean_valid_depth_fraction": alignment_summary.get("mean_valid_depth_fraction"),
                "mean_in_frame_fraction": alignment_summary.get("mean_in_frame_fraction"),
                "mean_projected_coverage_fraction": alignment_summary.get("mean_projected_coverage_fraction"),
                "alignment_mean_prediction_nonblack_fraction": alignment_summary.get("mean_prediction_nonblack_fraction"),
            }
        )
    return row


def rank_sweep(sweep: pd.DataFrame, *, selection_metric: str) -> pd.DataFrame:
    ranked = sweep.copy()
    ranked["_selection"] = numeric_rank_column(ranked, selection_metric)
    ranked["_ssim"] = numeric_rank_column(ranked, "mean_ssim")
    ranked["_coverage"] = numeric_rank_column(ranked, "mean_projected_coverage_fraction")
    ranked = ranked.sort_values(["_selection", "_ssim", "_coverage"], ascending=[False, False, False])
    return ranked.drop(columns=["_selection", "_ssim", "_coverage"])


def numeric_rank_column(dataframe: pd.DataFrame, column: str) -> pd.Series:
    if column not in dataframe:
        return pd.Series(-np.inf, index=dataframe.index, dtype=np.float64)
    return pd.to_numeric(dataframe[column], errors="coerce").fillna(-np.inf)


def copy_best_render_dir(source: Path, target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return target


def mean_render_output(report: dict[str, Any], key: str) -> float | None:
    values = [row.get(key) for row in report.get("outputs", []) if row.get(key) is not None]
    if not values:
        return None
    return float(np.asarray(values, dtype=np.float64).mean())


def validate_sweep_config(
    *,
    sigma_scales: tuple[float, ...],
    tau_scales: tuple[float, ...],
    min_sigma_pxs: tuple[float, ...],
    kernel_radii: tuple[float, ...],
    background_colors: tuple[tuple[float, float, float], ...],
    projection_conventions: tuple[str, ...],
    max_frames: int | None,
    near_plane: float,
    selection_metric: str,
    alignment_coverage_downsample: int,
    alignment_max_overlay_splats: int,
    audit_max_panels: int,
) -> None:
    for name, values in (
        ("sigma_scales", sigma_scales),
        ("tau_scales", tau_scales),
        ("min_sigma_pxs", min_sigma_pxs),
        ("kernel_radii", kernel_radii),
    ):
        if not values:
            raise ValueError(f"{name} must contain at least one value.")
        if any(value <= 0.0 for value in values):
            raise ValueError(f"{name} values must be positive.")
    if not background_colors:
        raise ValueError("background_colors must contain at least one value.")
    if any(len(color) != 3 or any(channel < 0.0 or channel > 1.0 for channel in color) for color in background_colors):
        raise ValueError("background colors must be RGB triplets in [0, 1].")
    if any(convention not in PROJECTION_CONVENTIONS for convention in projection_conventions):
        raise ValueError(f"projection_conventions must be in {', '.join(PROJECTION_CONVENTIONS)}.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive when provided.")
    if near_plane <= 0.0:
        raise ValueError("near_plane must be positive.")
    if selection_metric not in {"mean_psnr", "mean_ssim"}:
        raise ValueError("selection_metric must be 'mean_psnr' or 'mean_ssim'.")
    if alignment_coverage_downsample < 1:
        raise ValueError("alignment_coverage_downsample must be positive.")
    if alignment_max_overlay_splats < 1:
        raise ValueError("alignment_max_overlay_splats must be positive.")
    if audit_max_panels < 0:
        raise ValueError("audit_max_panels must be non-negative.")


def variant_name(
    *,
    sigma_scale: float,
    tau_scale: float,
    min_sigma_px: float,
    kernel_radius: float,
    background_color: tuple[float, float, float],
    projection_convention: str,
) -> str:
    bg = "_".join(format_float_label(channel) for channel in background_color)
    return (
        f"sig{format_float_label(sigma_scale)}"
        f"_tau{format_float_label(tau_scale)}"
        f"_minpx{format_float_label(min_sigma_px)}"
        f"_kr{format_float_label(kernel_radius)}"
        f"_bg{bg}"
        f"_{projection_convention}"
    )


def format_float_label(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text


def resolve_scene_file(scene_root: Path, path: str | Path | None, default_name: str) -> Path:
    if path is None:
        return scene_root / default_name
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = scene_root / resolved
    return resolved


def format_sweep_report(status: dict[str, Any], ranked: pd.DataFrame) -> str:
    best = status["best"]["best"]
    lines = [
        "# Real Render Parameter Sweep",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Method: `{status['method']}`",
        f"- Split: `{status['split']}`",
        f"- Variants: `{status['variant_count']}`",
        f"- Selection metric: `{status['selection_metric']}`",
        f"- Best variant: `{best['variant']}`",
        f"- Best render directory: `{status['best_render_dir']}`",
        f"- Mean PSNR: `{format_optional(best.get('mean_psnr'))}`",
        f"- Mean SSIM: `{format_optional(best.get('mean_ssim'))}`",
        f"- Sweep CSV: `{status['sweep_path']}`",
        f"- Ranked CSV: `{status['ranked_path']}`",
        f"- Best parameters JSON: `{status['best_path']}`",
    ]
    if "alignment_failure_counts" in best:
        lines.append(f"- Best alignment failures: `{best['alignment_failure_counts']}`")
    if not ranked.empty:
        lines.extend(["", "## Top Variants", "", format_ranked_table(ranked.head(12))])
    return "\n".join(lines) + "\n"


def format_ranked_table(ranked: pd.DataFrame) -> str:
    columns = [
        "variant",
        "mean_psnr",
        "mean_ssim",
        "sigma_scale",
        "tau_scale",
        "min_sigma_px",
        "kernel_radius",
        "background_color",
        "mean_prediction_nonblack_fraction",
        "mean_projected_coverage_fraction",
    ]
    available = [column for column in columns if column in ranked]
    lines = [
        "| " + " | ".join(available) + " |",
        "| " + " | ".join("---" for _ in available) + " |",
    ]
    for _, row in ranked[available].iterrows():
        lines.append("| " + " | ".join(format_optional(row[column]) for column in available) + " |")
    return "\n".join(lines)


def format_optional(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
