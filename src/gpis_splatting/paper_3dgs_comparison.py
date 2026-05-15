from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.external_3dgs import convert_3dgs_ply_to_splats, evaluate_3dgs_variant_renders
from gpis_splatting.real_geometry import evaluate_tanks_temples_geometry
from gpis_splatting.serialization import read_json, write_json

PAPER_METRICS = (
    "mean_psnr",
    "mean_ssim",
    "mean_lpips_vgg",
    "f_score",
    "chamfer_l1",
    "chamfer_l2",
    "gaussian_count",
    "fps",
    "peak_vram_mb",
)
HIGHER_IS_BETTER = {"mean_psnr", "mean_ssim", "f_score", "fps"}
LOWER_IS_BETTER = {"mean_lpips_vgg", "chamfer_l1", "chamfer_l2", "gaussian_count", "peak_vram_mb"}


@dataclass(frozen=True)
class Paper3DGSComparisonConfig:
    output_dir: Path
    scenes: tuple[dict[str, Any], ...]
    comparison_name: str = "paper_3dgs"
    prepared_root: Path = Path("real_scenes")
    split: str = "test"
    method_name: str = "trained_3dgs"
    prediction_subdir: str = ""
    primary_geometry_threshold: float = 0.05
    geometry_thresholds: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1)
    compute_photometry: bool = True
    compute_geometry: bool = True
    compute_lpips: bool = True
    require_all_images: bool = True
    require_all_variants: bool = True
    max_pred_points: int | None = 100_000
    max_gt_points: int | None = 100_000
    seed: int = 13
    apply_alignment: bool | None = None
    invert_alignment: bool = False
    use_crop: bool = True
    distance_chunk_size: int = 256
    fail_on_missing: bool = False
    require_lpips: bool = False
    require_performance: bool = False


def run_paper_3dgs_comparison(config: Paper3DGSComparisonConfig) -> dict[str, Any]:
    validate_config(config)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    scene_statuses: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(config.scenes):
        result = evaluate_scene(scene, config=config, scene_index=scene_index)
        rows.extend(result["rows"])
        scene_statuses.append(result["status"])

    comparison = pd.DataFrame(rows)
    checks = build_metric_checks(comparison, config=config)
    passed = bool(checks["passed"].all()) if not checks.empty else True
    if config.fail_on_missing and not passed:
        missing = ", ".join(checks.loc[~checks["passed"], "metric"].astype(str).tolist())
        raise ValueError(f"Paper-grade 3DGS comparison is missing required metrics: {missing}")

    aggregate = aggregate_comparison(comparison)
    paper_table = build_paper_table(aggregate)
    paths = write_artifacts(out_dir, config, comparison, aggregate, paper_table, checks, scene_statuses, passed)
    return {
        "comparison_path": paths["comparison"],
        "aggregate_path": paths["aggregate"],
        "paper_table_path": paths["paper_table"],
        "checks_path": paths["checks"],
        "status_path": paths["status"],
        "config_path": paths["config"],
        "report_path": paths["report"],
        "comparison": comparison,
        "aggregate": aggregate,
        "paper_table": paper_table,
        "checks": checks,
        "passed": passed,
    }


def validate_config(config: Paper3DGSComparisonConfig) -> None:
    if not config.scenes:
        raise ValueError("At least one scene must be configured.")
    if config.primary_geometry_threshold <= 0.0:
        raise ValueError("primary_geometry_threshold must be positive.")
    if not config.geometry_thresholds or any(threshold <= 0.0 for threshold in config.geometry_thresholds):
        raise ValueError("geometry_thresholds must contain positive values.")
    if config.distance_chunk_size < 1:
        raise ValueError("distance_chunk_size must be positive.")


def evaluate_scene(scene: dict[str, Any], *, config: Paper3DGSComparisonConfig, scene_index: int) -> dict[str, Any]:
    scene_name = str(scene.get("scene") or scene.get("name") or f"scene_{scene_index:03d}")
    scene_dir = resolve_scene_dir(scene, config=config, scene_name=scene_name)
    manifest_path = required_path(scene, "manifest_path")
    output_dir = Path(scene.get("output_dir") or config.output_dir / scene_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    photometry = load_or_compute_photometry(scene, config=config, scene_name=scene_name, scene_dir=scene_dir, manifest_path=manifest_path, output_dir=output_dir)
    geometry = load_or_compute_geometry(scene, config=config, scene_name=scene_name, scene_dir=scene_dir, manifest=manifest, output_dir=output_dir)
    performance = load_performance(scene)

    variants = collect_variants(manifest, photometry, geometry, performance)
    rows = []
    for variant in variants:
        manifest_row = select_variant_row(manifest, variant)
        row = {
            "scene": scene_name,
            "scene_dir": str(scene_dir),
            "variant": variant,
            "variant_kind": scalar_from_row(manifest_row, "variant_kind"),
            "point_cloud_path": scalar_from_row(manifest_row, "point_cloud_path"),
            "model_dir": scalar_from_row(manifest_row, "model_dir"),
            "gaussian_count": first_number(
                scalar_from_row(manifest_row, "retained_count"),
                lookup_number(photometry, variant, "retained_count"),
                lookup_number(performance, variant, "rendered_gaussian_count"),
            ),
            "retention_fraction": first_number(scalar_from_row(manifest_row, "retention_fraction"), lookup_number(photometry, variant, "retention_fraction")),
            "gate_threshold": first_number(scalar_from_row(manifest_row, "gate_threshold"), lookup_number(photometry, variant, "gate_threshold")),
            "mean_psnr": lookup_number(photometry, variant, "mean_psnr"),
            "mean_ssim": lookup_number(photometry, variant, "mean_ssim"),
            "mean_lpips_vgg": lookup_number(photometry, variant, "mean_lpips_vgg"),
            "image_count": lookup_number(photometry, variant, "image_count"),
            "precision": lookup_number(geometry, variant, "precision"),
            "recall": lookup_number(geometry, variant, "recall"),
            "f_score": lookup_number(geometry, variant, "f_score"),
            "chamfer_l1": lookup_number(geometry, variant, "chamfer_l1"),
            "chamfer_l2": lookup_number(geometry, variant, "chamfer_l2"),
            "accuracy_mean": lookup_number(geometry, variant, "accuracy_mean"),
            "completion_mean": lookup_number(geometry, variant, "completion_mean"),
            "geometry_threshold": lookup_number(geometry, variant, "geometry_threshold"),
            "fps": lookup_number(performance, variant, "fps"),
            "mean_frame_seconds": lookup_number(performance, variant, "mean_frame_seconds"),
            "total_render_seconds": lookup_number(performance, variant, "total_render_seconds"),
            "peak_vram_mb": first_number(lookup_number(performance, variant, "peak_vram_mb"), lookup_number(performance, variant, "cuda_peak_allocated_mb"), lookup_number(performance, variant, "cuda_peak_reserved_mb")),
            "device": lookup_value(performance, variant, "device"),
        }
        row["complete_metric_set"] = has_complete_paper_metrics(row, config=config)
        rows.append(row)

    return {
        "rows": rows,
        "status": {
            "scene": scene_name,
            "scene_dir": str(scene_dir),
            "manifest_path": str(manifest_path),
            "variant_count": len(rows),
            "photometry_rows": int(len(photometry)),
            "geometry_rows": int(len(geometry)),
            "performance_rows": int(len(performance)),
        },
    }


def resolve_scene_dir(scene: dict[str, Any], *, config: Paper3DGSComparisonConfig, scene_name: str) -> Path:
    if scene.get("scene_dir"):
        return Path(scene["scene_dir"])
    return Path(scene.get("prepared_root") or config.prepared_root) / scene_name


def required_path(scene: dict[str, Any], key: str) -> Path:
    value = scene.get(key)
    if not value:
        raise ValueError(f"Scene {scene.get('scene') or scene.get('name') or '<unnamed>'} is missing required field {key!r}.")
    return Path(value)


def load_or_compute_photometry(
    scene: dict[str, Any],
    *,
    config: Paper3DGSComparisonConfig,
    scene_name: str,
    scene_dir: Path,
    manifest_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    provided = scene.get("render_comparison_path") or scene.get("photometry_path")
    if provided:
        return normalize_photometry_table(pd.read_csv(provided))
    if not config.compute_photometry:
        return pd.DataFrame()
    predictions_root = scene.get("predictions_root")
    if not predictions_root:
        return pd.DataFrame()
    result = evaluate_3dgs_variant_renders(
        manifest_path=manifest_path,
        scene_dir=scene_dir,
        predictions_root=predictions_root,
        output_dir=output_dir / "photometry",
        method_name=str(scene.get("method_name") or config.method_name),
        split=str(scene.get("split") or config.split),
        prediction_subdir=str(scene.get("prediction_subdir") if scene.get("prediction_subdir") is not None else config.prediction_subdir),
        compute_lpips=bool(scene.get("compute_lpips", config.compute_lpips)),
        require_all_images=bool(scene.get("require_all_images", config.require_all_images)),
        require_all_variants=bool(scene.get("require_all_variants", config.require_all_variants)),
        benchmark_target=scene.get("benchmark_target"),
    )
    return normalize_photometry_table(result["comparison"])


def normalize_photometry_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    if "variant" not in table.columns and "method" in table.columns:
        table = table.rename(columns={"method": "variant"})
    return table.copy()


def load_or_compute_geometry(
    scene: dict[str, Any],
    *,
    config: Paper3DGSComparisonConfig,
    scene_name: str,
    scene_dir: Path,
    manifest: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    provided = scene.get("geometry_comparison_path") or scene.get("geometry_path")
    if provided:
        return normalize_geometry_table(pd.read_csv(provided), primary_threshold=float(scene.get("primary_geometry_threshold", config.primary_geometry_threshold)))
    if not config.compute_geometry or not scene.get("ground_truth_path"):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for manifest_row in manifest.itertuples(index=False):
        variant = str(manifest_row.variant)
        point_cloud_path = Path(str(manifest_row.point_cloud_path))
        splats_path = output_dir / "geometry_splats" / f"{scene_name}_{variant}.npz"
        convert_3dgs_ply_to_splats(ply_path=point_cloud_path, output_splats_path=splats_path, opacity_mode=str(scene.get("opacity_mode", "logit")))
        result = evaluate_tanks_temples_geometry(
            scene_dir=scene_dir,
            splats_path=splats_path,
            ground_truth_path=scene.get("ground_truth_path"),
            alignment_path=scene.get("alignment_path"),
            crop_path=scene.get("crop_path"),
            output_dir=output_dir / "geometry" / variant,
            method_name=f"{scene_name}_{variant}",
            thresholds=tuple(float(v) for v in scene.get("geometry_thresholds", config.geometry_thresholds)),
            max_pred_points=optional_int(scene.get("max_pred_points", config.max_pred_points)),
            max_gt_points=optional_int(scene.get("max_gt_points", config.max_gt_points)),
            seed=int(scene.get("seed", config.seed)),
            apply_alignment=scene.get("apply_alignment", config.apply_alignment),
            invert_alignment=bool(scene.get("invert_alignment", config.invert_alignment)),
            use_crop=bool(scene.get("use_crop", config.use_crop)),
            distance_chunk_size=int(scene.get("distance_chunk_size", config.distance_chunk_size)),
        )
        threshold_table = pd.read_csv(result["threshold_metrics_path"])
        summary_table = pd.read_csv(result["summary_path"])
        selected_threshold = select_threshold(threshold_table, float(scene.get("primary_geometry_threshold", config.primary_geometry_threshold)))
        threshold_row = selected_threshold[selected_threshold["group"].astype(str) == "all"].iloc[0] if "group" in selected_threshold and "all" in set(selected_threshold["group"].astype(str)) else selected_threshold.iloc[0]
        summary_row = summary_table[summary_table["group"].astype(str) == "all"].iloc[0] if "group" in summary_table and "all" in set(summary_table["group"].astype(str)) else summary_table.iloc[0]
        rows.append({
            "variant": variant,
            "geometry_threshold": float(threshold_row["threshold"]),
            "precision": float(threshold_row["precision"]),
            "recall": float(threshold_row["recall"]),
            "f_score": float(threshold_row["f_score"]),
            "chamfer_l1": float(summary_row["chamfer_l1"]),
            "chamfer_l2": float(summary_row["chamfer_l2"]),
            "accuracy_mean": float(summary_row["accuracy_mean"]),
            "completion_mean": float(summary_row["completion_mean"]),
        })
    return pd.DataFrame(rows)


def normalize_geometry_table(table: pd.DataFrame, *, primary_threshold: float) -> pd.DataFrame:
    if table.empty:
        return table
    data = table.copy()
    if "variant" not in data.columns:
        if "group" in data.columns:
            data = data.rename(columns={"group": "variant"})
        elif "method" in data.columns:
            data = data.rename(columns={"method": "variant"})
    if "geometry_threshold" not in data.columns and "threshold" in data.columns:
        data["geometry_threshold"] = data["threshold"]
    if "geometry_threshold" in data.columns:
        data = select_threshold(data, primary_threshold, threshold_column="geometry_threshold")
    return data


def select_threshold(table: pd.DataFrame, primary_threshold: float, *, threshold_column: str = "threshold") -> pd.DataFrame:
    if table.empty or threshold_column not in table.columns:
        return table
    distances = (table[threshold_column].astype(float) - float(primary_threshold)).abs()
    threshold = float(table.loc[distances.idxmin(), threshold_column])
    return table[table[threshold_column].astype(float) == threshold]


def load_performance(scene: dict[str, Any]) -> pd.DataFrame:
    paths = [scene.get("performance_path"), scene.get("render_manifest_path")]
    frames = []
    for path in paths:
        if path:
            p = Path(path)
            if p.exists():
                frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "variant" not in merged.columns and "method" in merged.columns:
        merged = merged.rename(columns={"method": "variant"})
    return merged


def collect_variants(*tables: pd.DataFrame) -> list[str]:
    variants: set[str] = set()
    for table in tables:
        if not table.empty and "variant" in table.columns:
            variants.update(table["variant"].dropna().astype(str))
    return sorted(variants)


def select_variant_row(table: pd.DataFrame, variant: str) -> pd.Series:
    if table.empty or "variant" not in table.columns:
        return pd.Series(dtype=object)
    rows = table[table["variant"].astype(str) == variant]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def lookup_row(table: pd.DataFrame, variant: str) -> pd.Series:
    return select_variant_row(table, variant)


def lookup_value(table: pd.DataFrame, variant: str, column: str) -> Any:
    row = lookup_row(table, variant)
    return scalar_from_row(row, column)


def lookup_number(table: pd.DataFrame, variant: str, column: str) -> float:
    return to_float(lookup_value(table, variant, column))


def scalar_from_row(row: pd.Series, column: str) -> Any:
    if row is None or row.empty or column not in row:
        return None
    value = row[column]
    if is_missing(value):
        return None
    return value.item() if hasattr(value, "item") else value


def to_float(value: Any) -> float:
    if is_missing(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def first_number(*values: Any) -> float:
    for value in values:
        number = to_float(value)
        if not is_missing(number):
            return number
    return np.nan


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def has_complete_paper_metrics(row: dict[str, Any], *, config: Paper3DGSComparisonConfig) -> bool:
    required = ["mean_psnr", "mean_ssim", "f_score", "chamfer_l1", "gaussian_count"]
    if config.require_lpips:
        required.append("mean_lpips_vgg")
    if config.require_performance:
        required.extend(["fps", "peak_vram_mb"])
    return all(not is_missing(row.get(metric)) for metric in required)


def build_metric_checks(comparison: pd.DataFrame, *, config: Paper3DGSComparisonConfig) -> pd.DataFrame:
    required = ["mean_psnr", "mean_ssim", "f_score", "chamfer_l1", "gaussian_count"]
    if config.require_lpips:
        required.append("mean_lpips_vgg")
    if config.require_performance:
        required.extend(["fps", "peak_vram_mb"])
    rows = []
    for metric in required:
        present = metric in comparison.columns and comparison[metric].notna().any()
        rows.append({"metric": metric, "required": True, "passed": bool(present), "non_missing_rows": int(comparison[metric].notna().sum()) if metric in comparison else 0})
    for metric in PAPER_METRICS:
        if metric in required:
            continue
        present = metric in comparison.columns and comparison[metric].notna().any()
        rows.append({"metric": metric, "required": False, "passed": True, "non_missing_rows": int(comparison[metric].notna().sum()) if metric in comparison else 0})
    return pd.DataFrame(rows)


def aggregate_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    rows = []
    for (variant, variant_kind), group in comparison.groupby(["variant", "variant_kind"], dropna=False):
        row: dict[str, Any] = {"variant": variant, "variant_kind": variant_kind, "scene_count": int(group["scene"].nunique())}
        for metric in PAPER_METRICS:
            if metric not in group.columns:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_count"] = 0
                continue
            series = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(series.mean()) if not series.empty else np.nan
            row[f"{metric}_std"] = float(series.std(ddof=0)) if len(series) > 1 else 0.0 if len(series) == 1 else np.nan
            row[f"{metric}_count"] = int(series.shape[0])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant_kind", "variant"], na_position="last")


def build_paper_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    rows = []
    for row in aggregate.itertuples(index=False):
        item = {"Variant": row.variant, "Kind": row.variant_kind, "Scenes": row.scene_count}
        for metric, label in [
            ("mean_psnr", "PSNR↑"),
            ("mean_ssim", "SSIM↑"),
            ("mean_lpips_vgg", "LPIPS↓"),
            ("f_score", "F-score↑"),
            ("chamfer_l1", "Chamfer-L1↓"),
            ("chamfer_l2", "Chamfer-L2↓"),
            ("gaussian_count", "Gaussians↓"),
            ("fps", "FPS↑"),
            ("peak_vram_mb", "VRAM MB↓"),
        ]:
            item[label] = mean_std_cell(getattr(row, f"{metric}_mean"), getattr(row, f"{metric}_std"), getattr(row, f"{metric}_count"))
        rows.append(item)
    return pd.DataFrame(rows)


def mean_std_cell(mean: Any, std: Any, count: Any) -> str:
    if is_missing(mean) or int(count or 0) == 0:
        return ""
    if int(count) <= 1 or is_missing(std):
        return f"{float(mean):.4g}"
    return f"{float(mean):.4g} ± {float(std):.3g}"


def write_artifacts(
    output_dir: Path,
    config: Paper3DGSComparisonConfig,
    comparison: pd.DataFrame,
    aggregate: pd.DataFrame,
    paper_table: pd.DataFrame,
    checks: pd.DataFrame,
    scene_statuses: list[dict[str, Any]],
    passed: bool,
) -> dict[str, Path]:
    name = config.comparison_name
    paths = {
        "comparison": output_dir / f"{name}_scene_comparison.csv",
        "aggregate": output_dir / f"{name}_aggregate_by_variant.csv",
        "paper_table": output_dir / f"{name}_paper_table.csv",
        "checks": output_dir / f"{name}_checks.csv",
        "status": output_dir / f"{name}_status.json",
        "config": output_dir / f"{name}_config.json",
        "report": output_dir / f"{name}_report.md",
    }
    comparison.to_csv(paths["comparison"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    paper_table.to_csv(paths["paper_table"], index=False)
    checks.to_csv(paths["checks"], index=False)
    status = {
        "schema_version": 1,
        "comparison_name": name,
        "passed": passed,
        "scene_count": len(config.scenes),
        "variant_row_count": int(len(comparison)),
        "split": config.split,
        "primary_geometry_threshold": config.primary_geometry_threshold,
        "require_lpips": config.require_lpips,
        "require_performance": config.require_performance,
        "paths": {key: str(value) for key, value in paths.items()},
        "scenes": scene_statuses,
    }
    config_payload = {
        "comparison_name": config.comparison_name,
        "output_dir": str(config.output_dir),
        "prepared_root": str(config.prepared_root),
        "split": config.split,
        "method_name": config.method_name,
        "prediction_subdir": config.prediction_subdir,
        "primary_geometry_threshold": config.primary_geometry_threshold,
        "geometry_thresholds": list(config.geometry_thresholds),
        "compute_photometry": config.compute_photometry,
        "compute_geometry": config.compute_geometry,
        "compute_lpips": config.compute_lpips,
        "require_lpips": config.require_lpips,
        "require_performance": config.require_performance,
        "scenes": list(config.scenes),
    }
    write_json(paths["status"], status)
    write_json(paths["config"], config_payload)
    paths["report"].write_text(format_report(status, comparison, aggregate, paper_table, checks), encoding="utf-8")
    return paths


def format_report(status: dict[str, Any], comparison: pd.DataFrame, aggregate: pd.DataFrame, paper_table: pd.DataFrame, checks: pd.DataFrame) -> str:
    lines = [
        "# Paper-grade trained-3DGS comparison",
        "",
        f"- Comparison: `{status['comparison_name']}`",
        f"- Passed checks: `{status['passed']}`",
        f"- Scenes: `{status['scene_count']}`",
        f"- Variant rows: `{status['variant_row_count']}`",
        f"- Split: `{status['split']}`",
        f"- Primary geometry threshold: `{status['primary_geometry_threshold']}`",
        "",
        "## Paper table",
        "",
        markdown_table(paper_table),
        "",
        "## Metric checks",
        "",
        markdown_table(checks),
        "",
        "## Aggregate by variant",
        "",
        markdown_table(aggregate),
    ]
    return "\n".join(lines) + "\n"


def markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "No rows."
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(format_cell(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("\n", " ").replace("|", "\\|")


def config_from_json(path: str | Path) -> Paper3DGSComparisonConfig:
    payload = read_json(path)
    return Paper3DGSComparisonConfig(
        output_dir=Path(payload.get("output_dir", "paper_results/trained_3dgs_comparison")),
        scenes=tuple(payload["scenes"]),
        comparison_name=str(payload.get("comparison_name", "paper_3dgs")),
        prepared_root=Path(payload.get("prepared_root", "real_scenes")),
        split=str(payload.get("split", "test")),
        method_name=str(payload.get("method_name", "trained_3dgs")),
        prediction_subdir=str(payload.get("prediction_subdir", "")),
        primary_geometry_threshold=float(payload.get("primary_geometry_threshold", 0.05)),
        geometry_thresholds=tuple(float(v) for v in payload.get("geometry_thresholds", [0.01, 0.02, 0.05, 0.1])),
        compute_photometry=bool(payload.get("compute_photometry", True)),
        compute_geometry=bool(payload.get("compute_geometry", True)),
        compute_lpips=bool(payload.get("compute_lpips", True)),
        require_all_images=bool(payload.get("require_all_images", True)),
        require_all_variants=bool(payload.get("require_all_variants", True)),
        max_pred_points=optional_int(payload.get("max_pred_points", 100000)),
        max_gt_points=optional_int(payload.get("max_gt_points", 100000)),
        seed=int(payload.get("seed", 13)),
        apply_alignment=payload.get("apply_alignment"),
        invert_alignment=bool(payload.get("invert_alignment", False)),
        use_crop=bool(payload.get("use_crop", True)),
        distance_chunk_size=int(payload.get("distance_chunk_size", 256)),
        fail_on_missing=bool(payload.get("fail_on_missing", False)),
        require_lpips=bool(payload.get("require_lpips", False)),
        require_performance=bool(payload.get("require_performance", False)),
    )
