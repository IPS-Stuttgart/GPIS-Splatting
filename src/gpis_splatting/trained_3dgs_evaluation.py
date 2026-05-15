from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from gpis_splatting.colmap_render_mapping import LINK_MODES, map_3dgs_renders_to_prepared_scene
from gpis_splatting.external_3dgs import convert_3dgs_ply_to_splats, evaluate_3dgs_variant_renders, export_3dgs_gpis_variants, resolve_3dgs_variant_prediction_dir
from gpis_splatting.gsplat_fidelity_adapter import render_3dgs_manifest_with_gsplat
from gpis_splatting.primary_confidence import run_gpis_splat_score_calibration
from gpis_splatting.real_field_scores import default_score_lambdas, run_tanks_temples_gpis_field_score_diagnostics
from gpis_splatting.real_geometry import format_threshold_label
from gpis_splatting.serialization import write_json

TRAINED_3DGS_RENDERERS = ("none", "gsplat", "external", "precomputed")


def coerce_optional_positive_int(value: int | None) -> int | None:
    if value is None or int(value) <= 0:
        return None
    return int(value)


def run_trained_3dgs_gpis_experiment(
    *,
    scene_dir: str | Path,
    trained_ply_path: str | Path,
    method_name: str = "trained_3dgs",
    gpis_model_path: str | Path | None = None,
    gate_path: str | Path | None = None,
    splats_path: str | Path | None = None,
    evaluations_dir: str | Path | None = None,
    variants_dir: str | Path | None = None,
    render_output_root: str | Path | None = None,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.1),
    calibration_threshold: float = 0.05,
    gate_thresholds: tuple[float, ...] = (0.25, 0.5, 0.75),
    topk_fractions: tuple[float, ...] | None = None,
    feature_sets: tuple[str, ...] | None = None,
    score_lambdas: tuple[float, ...] | None = None,
    max_pred_points: int | None = None,
    max_gt_points: int | None = 150000,
    seed: int = 13,
    missing_gate_value: float = 1.0,
    iteration: int = 30000,
    opacity_mode: str = "logit",
    opacity_scale_floor: float = 0.0,
    include_baseline: bool = True,
    write_scaled: bool = True,
    write_filtered: bool = True,
    template_model_dir: str | Path | None = None,
    renderer: str = "none",
    render_command_template: str | None = None,
    rendered_predictions_root: str | Path | None = None,
    prediction_subdir: str = "",
    render_name_map_path: str | Path | None = None,
    render_mapping_link_mode: str = "copy",
    render_split: str = "test",
    compute_lpips: bool = False,
    require_all_images: bool = True,
    require_all_variants: bool = True,
    benchmark_target: str | Path | None = None,
    gsplat_projection_convention: str = "auto",
    gsplat_device: str = "auto",
    gsplat_dtype: str = "float32",
    gsplat_color_mode: str = "auto",
    gsplat_sh_degree: int | str | None = "auto",
    gsplat_strict_3dgs_fidelity: bool = True,
    gsplat_background_mode: str = "auto",
    gsplat_background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gsplat_near_plane: float = 1e-2,
    gsplat_far_plane: float = 1.0e10,
    gsplat_radius_clip: float = 0.0,
    gsplat_eps2d: float = 0.3,
    gsplat_tile_size: int = 16,
    gsplat_packed: bool = True,
    gsplat_render_mode: str = "RGB",
    gsplat_rasterize_mode: str = "classic",
    gsplat_channel_chunk: int = 32,
    gsplat_max_frames: int | None = None,
    gsplat_max_gaussians: int | None = None,
) -> dict[str, Any]:
    if renderer not in TRAINED_3DGS_RENDERERS:
        raise ValueError(f"renderer must be one of {TRAINED_3DGS_RENDERERS}.")
    if render_mapping_link_mode not in LINK_MODES:
        raise ValueError(f"render_mapping_link_mode must be one of {LINK_MODES}.")

    scene_root = Path(scene_dir)
    trained_ply = Path(trained_ply_path)
    eval_dir = Path(evaluations_dir) if evaluations_dir is not None else scene_root / "evaluations"
    splats_out = Path(splats_path) if splats_path is not None else scene_root / f"{method_name}_splats.npz"
    variant_root = Path(variants_dir) if variants_dir is not None else scene_root / "trained_3dgs_variants" / method_name
    render_root = Path(render_output_root) if render_output_root is not None else scene_root / "renders" / f"{method_name}_variants"
    eval_dir.mkdir(parents=True, exist_ok=True)

    convert = convert_3dgs_ply_to_splats(ply_path=trained_ply, output_splats_path=splats_out, opacity_mode=opacity_mode)
    gaussian_count = int(convert["status"]["splat_count"])

    scoring = None
    calibration = None
    if gate_path is None:
        if gpis_model_path is None:
            raise ValueError("Pass --gpis-model-path or --gate-path.")
        scoring = run_tanks_temples_gpis_field_score_diagnostics(
            scene_dir=scene_root,
            splats_path=splats_out,
            model_path=gpis_model_path,
            output_dir=eval_dir,
            method_name=method_name,
            thresholds=thresholds,
            topk_fractions=topk_fractions or (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
            score_lambdas=score_lambdas or default_score_lambdas(),
            max_pred_points=max_pred_points,
            max_gt_points=max_gt_points,
            seed=seed,
        )
        calibration = run_gpis_splat_score_calibration(
            field_scores_path=scoring["field_scores_path"],
            output_dir=eval_dir,
            method_name=method_name,
            thresholds=thresholds,
            topk_fractions=topk_fractions or (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
            feature_sets=feature_sets or ("gpis_core", "gpis_plus_current_gate", "gpis_all"),
            seed=seed,
            gate_count=gaussian_count,
            missing_gate_value=missing_gate_value,
            primary_threshold=calibration_threshold,
        )
        effective_gate_path = Path(calibration["primary_gate_path"])
    else:
        effective_gate_path = Path(gate_path)

    variants = export_3dgs_gpis_variants(
        input_ply_path=trained_ply,
        gate_path=effective_gate_path,
        output_dir=variant_root,
        method_name=method_name,
        iteration=iteration,
        gate_thresholds=gate_thresholds,
        include_baseline=include_baseline,
        write_scaled=write_scaled,
        write_filtered=write_filtered,
        opacity_mode=opacity_mode,
        opacity_scale_floor=opacity_scale_floor,
        template_model_dir=template_model_dir,
    )

    predictions_root: Path | None = None
    effective_prediction_subdir = prediction_subdir
    render_status = None
    if renderer == "gsplat":
        render_status = render_3dgs_manifest_with_gsplat(
            manifest_path=variants["manifest_path"],
            scene_dir=scene_root,
            output_root=render_root,
            method_name=f"{method_name}_gsplat",
            split=render_split,
            projection_convention=gsplat_projection_convention,
            device=gsplat_device,
            dtype=gsplat_dtype,
            opacity_mode=opacity_mode,
            color_mode=gsplat_color_mode,
            sh_degree=gsplat_sh_degree,
            strict_3dgs_fidelity=gsplat_strict_3dgs_fidelity,
            background_mode=gsplat_background_mode,
            background_color=gsplat_background_color,
            near_plane=gsplat_near_plane,
            far_plane=gsplat_far_plane,
            radius_clip=gsplat_radius_clip,
            eps2d=gsplat_eps2d,
            tile_size=gsplat_tile_size,
            packed=gsplat_packed,
            render_mode=gsplat_render_mode,
            rasterize_mode=gsplat_rasterize_mode,
            channel_chunk=gsplat_channel_chunk,
            max_frames=gsplat_max_frames,
            max_gaussians=gsplat_max_gaussians,
        )
        predictions_root = render_root
        effective_prediction_subdir = effective_prediction_subdir or f"{render_split}/ours_{iteration}/renders"
    elif renderer == "external":
        if not render_command_template:
            raise ValueError("renderer='external' requires render_command_template.")
        predictions_root = render_external_variants(variants["manifest_path"], render_root, scene_root, render_command_template, iteration)
    elif renderer == "precomputed":
        if rendered_predictions_root is None:
            raise ValueError("renderer='precomputed' requires rendered_predictions_root.")
        predictions_root = Path(rendered_predictions_root)
    elif rendered_predictions_root is not None:
        predictions_root = Path(rendered_predictions_root)

    mapped_statuses = []
    if predictions_root is not None and render_name_map_path is not None:
        predictions_root, effective_prediction_subdir, mapped_statuses = map_rendered_variants(
            manifest_path=variants["manifest_path"],
            predictions_root=predictions_root,
            scene_dir=scene_root,
            method_name=method_name,
            prediction_subdir=effective_prediction_subdir,
            render_name_map_path=render_name_map_path,
            link_mode=render_mapping_link_mode,
            require_all=require_all_images,
        )

    render_evaluation = None
    if predictions_root is not None:
        render_evaluation = evaluate_3dgs_variant_renders(
            manifest_path=variants["manifest_path"],
            scene_dir=scene_root,
            predictions_root=predictions_root,
            output_dir=eval_dir,
            method_name=method_name,
            split=render_split,
            prediction_subdir=effective_prediction_subdir,
            compute_lpips=compute_lpips,
            require_all_images=require_all_images,
            require_all_variants=require_all_variants,
            benchmark_target=benchmark_target,
        )

    status_path = eval_dir / f"{method_name}_trained_3dgs_experiment_status.json"
    report_path = eval_dir / f"{method_name}_trained_3dgs_experiment_report.md"
    status = {
        "schema_version": 1,
        "method": method_name,
        "scene_dir": str(scene_root),
        "trained_ply_path": str(trained_ply),
        "splats_path": str(splats_out),
        "gaussian_count": gaussian_count,
        "gpis_model_path": None if gpis_model_path is None else str(gpis_model_path),
        "gate_path": str(effective_gate_path),
        "calibration_threshold": calibration_threshold,
        "variants_manifest_path": str(variants["manifest_path"]),
        "renderer": renderer,
        "render_predictions_root": None if predictions_root is None else str(predictions_root),
        "prediction_subdir": effective_prediction_subdir,
        "render_evaluation_path": None if render_evaluation is None else str(render_evaluation["comparison_path"]),
        "status_path": str(status_path),
        "report_path": str(report_path),
    }
    write_json(status_path, status)
    report_path.write_text(format_report(status, variants.get("manifest"), None if render_evaluation is None else render_evaluation.get("comparison")), encoding="utf-8")
    return {"status": status, "status_path": status_path, "report_path": report_path, "convert": convert, "scoring": scoring, "calibration": calibration, "variants": variants, "render_status": render_status, "mapped_render_statuses": mapped_statuses, "render_evaluation": render_evaluation}


def render_external_variants(manifest_path: str | Path, render_root: Path, scene_dir: Path, command_template: str, iteration: int) -> Path:
    manifest = pd.read_csv(manifest_path)
    render_root.mkdir(parents=True, exist_ok=True)
    for row in manifest.to_dict(orient="records"):
        variant = str(row["variant"])
        output_dir = render_root / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        command = command_template.format(model_dir=row["model_dir"], output_dir=output_dir, scene_dir=scene_dir, variant=variant, iteration=iteration, point_cloud_path=row["point_cloud_path"])
        subprocess.run(command, shell=True, check=True)
    return render_root


def map_rendered_variants(*, manifest_path: str | Path, predictions_root: Path, scene_dir: Path, method_name: str, prediction_subdir: str, render_name_map_path: str | Path, link_mode: str, require_all: bool) -> tuple[Path, str, list[dict[str, Any]]]:
    manifest = pd.read_csv(manifest_path)
    map_path = Path(render_name_map_path)
    if not map_path.is_absolute() and not map_path.exists():
        map_path = scene_dir / map_path
    mapped_root = scene_dir / "renders" / f"{method_name}_mapped_variants"
    statuses = []
    for row in manifest.itertuples(index=False):
        raw_dir = resolve_3dgs_variant_prediction_dir(predictions_root, manifest_row=row, method_name=method_name, prediction_subdir=prediction_subdir)
        if raw_dir is None:
            raise FileNotFoundError(f"Could not find rendered prediction directory for variant {row.variant!r} under {predictions_root}.")
        result = map_3dgs_renders_to_prepared_scene(map_path=map_path, renders_dir=raw_dir, output_dir=mapped_root / str(row.variant), link_mode=link_mode, require_all=require_all, overwrite=True)
        statuses.append(result["status"])
    return mapped_root, "", statuses


def format_report(status: dict[str, Any], manifest: pd.DataFrame | None, comparison: pd.DataFrame | None) -> str:
    lines = [
        "# Trained 3DGS GPIS Experiment",
        "",
        f"- Method: `{status['method']}`",
        f"- Scene: `{status['scene_dir']}`",
        f"- Trained PLY: `{status['trained_ply_path']}`",
        f"- Gaussians: `{status['gaussian_count']}`",
        f"- Gate: `{status['gate_path']}`",
        f"- Variants: `{status['variants_manifest_path']}`",
        f"- Renderer: `{status['renderer']}`",
        f"- Render evaluation: `{status['render_evaluation_path'] or 'not run'}`",
    ]
    if manifest is not None and not manifest.empty:
        lines.extend(["", "## Variants", "", manifest[["variant", "variant_kind", "retained_count", "retention_fraction", "opacity_scaled"]].to_markdown(index=False)])
    if comparison is not None and not comparison.empty:
        cols = [c for c in ["variant", "variant_kind", "retained_count", "retention_fraction", "mean_psnr", "mean_ssim", "mean_lpips_vgg", "image_count"] if c in comparison.columns]
        lines.extend(["", "## Render metrics", "", comparison[cols].to_markdown(index=False)])
    return "\n".join(lines) + "\n"
