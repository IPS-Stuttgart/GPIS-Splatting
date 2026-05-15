from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from PIL import Image

from gpis_splatting.gsplat_fidelity_adapter import render_3dgs_ply_with_gsplat
from gpis_splatting.render_consistency import DEFAULT_AA_DOWNSAMPLE_FACTORS, evaluate_render_consistency
from gpis_splatting.serialization import read_json, write_json


def run_render_scale_diagnostics(
    *,
    scene_dir: str | Path,
    input_ply_path: str | Path,
    output_dir: str | Path,
    method_name: str = "trained_3dgs_scale_aa",
    split: str = "test",
    render_scale_factors: Sequence[float] = (0.5, 1.0, 2.0),
    include_gsplat_antialiased: bool = True,
    output_resolution: str = "target",
    aa_downsample_factors: Sequence[int] | None = DEFAULT_AA_DOWNSAMPLE_FACTORS,
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
    channel_chunk: int = 32,
    max_frames: int | None = None,
    max_gaussians: int | None = None,
    require_all: bool = True,
    max_temporal_pairs: int | None = None,
    rasterization_fn: Any | None = None,
) -> dict[str, Any]:
    """Render controlled scale/AA variants and run the render-consistency evaluator."""
    scene_root = Path(scene_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if output_resolution not in {"target", "render"}:
        raise ValueError("output_resolution must be 'target' or 'render'.")
    factors = validate_render_scale_factors(render_scale_factors)
    if 1.0 not in factors:
        factors = (1.0, *factors)

    base_dir: Path | None = None
    rows: list[dict[str, Any]] = []
    scale_dirs: dict[str, Path] = {}
    modes = ["classic", "antialiased"] if include_gsplat_antialiased else ["classic"]
    for factor in factors:
        scaled_scene = write_scaled_prepared_scene(scene_root, out_root / "scaled_scenes" / scale_label(factor), factor)
        for rasterize_mode in modes:
            label = variant_label(factor, rasterize_mode)
            pred_dir = out_root / "renders" / label
            render_result = render_3dgs_ply_with_gsplat(
                input_ply_path=input_ply_path,
                scene_dir=scaled_scene,
                output_dir=pred_dir,
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
                rasterization_fn=rasterization_fn,
            )
            if output_resolution == "target":
                resize_predictions_to_target(pred_dir, scene_root, split=split)
            rows.append(
                {
                    "method": method_name,
                    "variant": label,
                    "render_scale": factor,
                    "rasterize_mode": rasterize_mode,
                    "predictions_dir": str(pred_dir),
                    "scaled_scene_dir": str(scaled_scene),
                    "output_resolution": output_resolution,
                    "render_report_path": str(render_result["report_path"]),
                    "image_count": int(render_result["report"]["image_count"]),
                    "rendered_gaussian_count": int(render_result["report"]["rendered_gaussian_count"]),
                }
            )
            if factor == 1.0 and rasterize_mode == "classic":
                base_dir = pred_dir
            else:
                scale_dirs[label] = pred_dir
    if base_dir is None:
        raise RuntimeError("Internal error: missing scale1_classic base render.")

    manifest = pd.DataFrame(rows)
    manifest_path = out_root / f"{method_name}_scale_aa_render_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    consistency = evaluate_render_consistency(
        scene_dir=scene_root,
        predictions_dir=base_dir,
        output_dir=out_root / "evaluations",
        method_name=method_name,
        split=split,
        scale_prediction_dirs=scale_dirs,
        require_all=require_all,
        max_temporal_pairs=max_temporal_pairs,
        aa_downsample_factors=aa_downsample_factors,
    )
    status = {
        "schema_version": 1,
        "method": method_name,
        "scene_dir": str(scene_root),
        "input_ply_path": str(Path(input_ply_path)),
        "output_dir": str(out_root),
        "split": split,
        "render_scale_factors": list(factors),
        "include_gsplat_antialiased": include_gsplat_antialiased,
        "output_resolution": output_resolution,
        "manifest_path": str(manifest_path),
        "base_predictions_dir": str(base_dir),
        "scale_prediction_dirs": {key: str(value) for key, value in scale_dirs.items()},
        "consistency_summary_path": str(consistency["summary_path"]),
        "summary": consistency["summary"],
    }
    status_path = out_root / f"{method_name}_scale_aa_diagnostics_status.json"
    report_path = out_root / f"{method_name}_scale_aa_diagnostics_report.md"
    write_json(status_path, status)
    report_path.write_text(format_scale_aa_report(status, manifest), encoding="utf-8")
    return {"status_path": status_path, "report_path": report_path, "manifest_path": manifest_path, "status": status, "manifest": manifest, "consistency": consistency}


def validate_render_scale_factors(factors: Sequence[float]) -> tuple[float, ...]:
    values: list[float] = []
    for factor in factors:
        value = float(factor)
        if value <= 0.0:
            raise ValueError("Render scale factors must be positive.")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("At least one render scale factor is required.")
    return tuple(sorted(values))


def write_scaled_prepared_scene(source_scene: Path, target_scene: Path, render_scale: float) -> Path:
    target_scene.mkdir(parents=True, exist_ok=True)
    for filename in ("real_scene.json", "splits.json"):
        write_json(target_scene / filename, read_json(source_scene / filename))
    cameras = read_json(source_scene / "cameras.json")
    frames = []
    for frame in cameras["frames"]:
        scaled = dict(frame)
        intrinsics = dict(frame.get("intrinsics", {}))
        width = int(round(float(frame.get("width") or intrinsics.get("width")) * render_scale))
        height = int(round(float(frame.get("height") or intrinsics.get("height")) * render_scale))
        width = max(1, width)
        height = max(1, height)
        scaled["width"] = width
        scaled["height"] = height
        for key in ("fx", "fy", "cx", "cy"):
            if key in intrinsics:
                intrinsics[key] = float(intrinsics[key]) * render_scale
        intrinsics["width"] = width
        intrinsics["height"] = height
        scaled["intrinsics"] = intrinsics
        frames.append(scaled)
    cameras["frames"] = frames
    cameras["render_scale"] = float(render_scale)
    write_json(target_scene / "cameras.json", cameras)
    return target_scene


def resize_predictions_to_target(predictions_dir: Path, scene_dir: Path, *, split: str) -> None:
    cameras = read_json(scene_dir / "cameras.json")
    splits = read_json(scene_dir / "splits.json")
    frames = cameras["frames"]
    for frame_index in splits[split]:
        frame = frames[int(frame_index)]
        filename = str(frame.get("file_name") or Path(str(frame["image_path"])).name)
        path = predictions_dir / filename
        if not path.exists():
            continue
        intrinsics = frame.get("intrinsics", {})
        width = int(frame.get("width") or intrinsics.get("width"))
        height = int(frame.get("height") or intrinsics.get("height"))
        with Image.open(path) as image:
            if image.size == (width, height):
                continue
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image.convert("RGB").resize((width, height), resample=resampling).save(path)


def scale_label(factor: float) -> str:
    return f"scale{factor:g}".replace(".", "p")


def variant_label(factor: float, rasterize_mode: str) -> str:
    return f"{scale_label(factor)}_{rasterize_mode}"


def format_scale_aa_report(status: dict[str, Any], manifest: pd.DataFrame) -> str:
    summary = status["summary"]
    lines = [
        "# Render Scale and Anti-Aliasing Diagnostics",
        "",
        f"- Method: `{status['method']}`",
        f"- Scene: `{status['scene_dir']}`",
        f"- Input PLY: `{status['input_ply_path']}`",
        f"- Split: `{status['split']}`",
        f"- Output resolution: `{status['output_resolution']}`",
        f"- Render scale factors: `{status['render_scale_factors']}`",
        f"- Includes gsplat antialiased variants: `{status['include_gsplat_antialiased']}`",
        f"- Base predictions: `{status['base_predictions_dir']}`",
        f"- Mean scale instability score: `{summary.get('mean_scale_instability_score')}`",
        f"- Mean AA instability score: `{summary.get('mean_aa_instability_score')}`",
        f"- Manifest: `{status['manifest_path']}`",
        "",
        "## Rendered Variants",
        "",
        "| variant | render scale | rasterize mode | images | gaussians | predictions |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in manifest.itertuples(index=False):
        lines.append(f"| `{row.variant}` | {row.render_scale:g} | `{row.rasterize_mode}` | {row.image_count} | {row.rendered_gaussian_count} | `{row.predictions_dir}` |")
    return "\n".join(lines) + "\n"
