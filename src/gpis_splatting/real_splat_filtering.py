from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gpis_splatting.real_benchmark import evaluate_real_renders
from gpis_splatting.real_geometry import evaluate_tanks_temples_geometry, format_threshold_label, resolve_scene_file
from gpis_splatting.real_pipeline import render_real_splats
from gpis_splatting.serialization import write_json
from gpis_splatting.splats import SplatCloud, load_splats, save_splats


@dataclass(frozen=True)
class SplatFilterVariant:
    name: str
    kind: str
    splats_path: Path
    gate_path: Path | None
    retained_count: int
    retention_fraction: float
    gate_threshold: float | None
    tau_scaled: bool
    gates: np.ndarray
    random_seed: int | None = None


def run_tanks_temples_calibrated_splat_filtering(
    *,
    scene_dir: str | Path,
    splats_path: str | Path | None = None,
    gate_path: str | Path | None = None,
    method_name: str = "calibrated_splat_filtering",
    output_dir: str | Path | None = None,
    gate_thresholds: tuple[float, ...] = (0.25, 0.5, 0.75),
    include_baseline: bool = True,
    write_scaled: bool = True,
    write_filtered: bool = True,
    include_random_baselines: bool = False,
    random_baseline_seeds: tuple[int, ...] = (0, 1, 2),
    tau_scale_floor: float = 0.0,
    ground_truth_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
    crop_path: str | Path | None = None,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.1),
    max_pred_points: int | None = 100_000,
    max_gt_points: int | None = 100_000,
    seed: int = 13,
    apply_alignment: bool | None = None,
    invert_alignment: bool = False,
    use_crop: bool = True,
    distance_chunk_size: int = 256,
    render_split: str = "test",
    render_max_frames: int = 0,
    evaluate_render_metrics: bool = True,
    benchmark_target: str | Path | None = None,
) -> dict[str, Any]:
    validate_filtering_config(
        gate_thresholds=gate_thresholds,
        include_baseline=include_baseline,
        write_scaled=write_scaled,
        write_filtered=write_filtered,
        include_random_baselines=include_random_baselines,
        random_baseline_seeds=random_baseline_seeds,
        tau_scale_floor=tau_scale_floor,
        render_max_frames=render_max_frames,
    )
    scene_root = Path(scene_dir).resolve()
    out_dir = Path(output_dir).resolve() if output_dir is not None else scene_root / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_splats = resolve_scene_file(scene_root, splats_path, "real_splats.npz")
    if gate_path is None:
        raise ValueError("gate_path is required for calibrated splat filtering.")
    resolved_gate = resolve_scene_file(scene_root, gate_path, "real_splat_gates.npz")
    splats = load_splats(str(resolved_splats))
    gates = load_gate_array(resolved_gate, expected_count=int(splats.centers.shape[0]))

    variants = build_filter_variants(
        splats=splats,
        gates=gates,
        splats_path=resolved_splats,
        gate_path=resolved_gate,
        out_dir=out_dir,
        method_name=method_name,
        gate_thresholds=gate_thresholds,
        include_baseline=include_baseline,
        write_scaled=write_scaled,
        write_filtered=write_filtered,
        include_random_baselines=include_random_baselines,
        random_baseline_seeds=random_baseline_seeds,
        tau_scale_floor=tau_scale_floor,
    )

    comparison_rows = []
    variant_statuses = []
    for variant in variants:
        variant_method = f"{method_name}_{variant.name}"
        geometry_result = evaluate_tanks_temples_geometry(
            scene_dir=scene_root,
            splats_path=variant.splats_path,
            ground_truth_path=ground_truth_path,
            alignment_path=alignment_path,
            crop_path=crop_path,
            output_dir=out_dir,
            method_name=variant_method,
            thresholds=thresholds,
            max_pred_points=max_pred_points,
            max_gt_points=max_gt_points,
            seed=seed,
            apply_alignment=apply_alignment,
            invert_alignment=invert_alignment,
            use_crop=use_crop,
            distance_chunk_size=distance_chunk_size,
        )
        geometry_summary = pd.read_csv(geometry_result["summary_path"])
        geometry_thresholds = pd.read_csv(geometry_result["threshold_metrics_path"])
        render_status = None
        render_summary: dict[str, Any] = {}
        if render_max_frames > 0:
            render_result = render_real_splats(
                scene_dir=scene_root,
                splats_path=variant.splats_path,
                output_dir=scene_root / "renders" / variant_method,
                method_name=variant_method,
                split=render_split,
                use_gpis_gate=False,
                max_frames=render_max_frames,
            )
            if evaluate_render_metrics:
                render_status = evaluate_real_renders(
                    scene_dir=scene_root,
                    predictions_dir=render_result["output_dir"],
                    output_dir=out_dir,
                    method_name=variant_method,
                    split=render_split,
                    benchmark_target=benchmark_target,
                    compute_lpips=False,
                    require_all=False,
                    allow_diagnostic_proxy=True,
                )
                render_summary = render_status["summary"]

        all_summary = geometry_summary[geometry_summary["group"] == "all"].iloc[0].to_dict()
        all_thresholds = geometry_thresholds[geometry_thresholds["group"] == "all"]
        for row in all_thresholds.to_dict(orient="records"):
            comparison_rows.append(
                {
                    "method": method_name,
                    "variant": variant.name,
                    "variant_kind": variant.kind,
                    "splats_path": str(variant.splats_path),
                    "gate_path": str(variant.gate_path) if variant.gate_path is not None else None,
                    "retained_count": variant.retained_count,
                    "retention_fraction": variant.retention_fraction,
                    "gate_threshold": variant.gate_threshold,
                    "tau_scaled": variant.tau_scaled,
                    "random_seed": variant.random_seed,
                    "gate_min": float(variant.gates.min()) if variant.gates.size else None,
                    "gate_max": float(variant.gates.max()) if variant.gates.size else None,
                    "gate_mean": float(variant.gates.mean()) if variant.gates.size else None,
                    "geometry_threshold": row["threshold"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f_score": row["f_score"],
                    "chamfer_l1": all_summary["chamfer_l1"],
                    "chamfer_l2": all_summary["chamfer_l2"],
                    "mean_psnr": render_summary.get("mean_psnr"),
                    "mean_ssim": render_summary.get("mean_ssim"),
                    "rendered_image_count": render_summary.get("image_count", 0),
                    "missing_render_count": render_summary.get("missing_count", 0),
                }
            )
        variant_statuses.append(
            {
                "name": variant.name,
                "kind": variant.kind,
                "splats_path": str(variant.splats_path),
                "gate_path": str(variant.gate_path) if variant.gate_path is not None else None,
                "retained_count": variant.retained_count,
                "retention_fraction": variant.retention_fraction,
                "gate_threshold": variant.gate_threshold,
                "tau_scaled": variant.tau_scaled,
                "random_seed": variant.random_seed,
                "geometry_summary_path": str(geometry_result["summary_path"]),
                "geometry_threshold_metrics_path": str(geometry_result["threshold_metrics_path"]),
                "render_status": render_status,
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = out_dir / f"{method_name}_splat_filtering_comparison.csv"
    status_path = out_dir / f"{method_name}_splat_filtering_status.json"
    report_path = out_dir / f"{method_name}_splat_filtering_report.md"
    comparison.to_csv(comparison_path, index=False)
    status = {
        "schema_version": 1,
        "method": method_name,
        "scene_dir": str(scene_root),
        "input_splats_path": str(resolved_splats),
        "input_gate_path": str(resolved_gate),
        "output_dir": str(out_dir),
        "gate_thresholds": list(gate_thresholds),
        "include_baseline": include_baseline,
        "write_scaled": write_scaled,
        "write_filtered": write_filtered,
        "include_random_baselines": include_random_baselines,
        "random_baseline_seeds": list(random_baseline_seeds),
        "tau_scale_floor": tau_scale_floor,
        "render_split": render_split,
        "render_max_frames": render_max_frames,
        "variant_count": len(variants),
        "variants": variant_statuses,
        "comparison_path": str(comparison_path),
        "report_path": str(report_path),
    }
    write_json(status_path, status)
    report_path.write_text(format_filtering_report(status, comparison), encoding="utf-8")
    return {
        "comparison_path": comparison_path,
        "status_path": status_path,
        "report_path": report_path,
        "comparison": comparison,
        "status": status,
    }


def build_filter_variants(
    *,
    splats: SplatCloud,
    gates: np.ndarray,
    splats_path: Path,
    gate_path: Path,
    out_dir: Path,
    method_name: str,
    gate_thresholds: tuple[float, ...],
    include_baseline: bool,
    write_scaled: bool,
    write_filtered: bool,
    include_random_baselines: bool,
    random_baseline_seeds: tuple[int, ...],
    tau_scale_floor: float,
) -> list[SplatFilterVariant]:
    variants = []
    splat_count = int(splats.centers.shape[0])
    all_mask = np.ones((splat_count,), dtype=bool)
    if include_baseline:
        variants.append(
            SplatFilterVariant(
                name="baseline",
                kind="baseline",
                splats_path=splats_path,
                gate_path=None,
                retained_count=splat_count,
                retention_fraction=1.0,
                gate_threshold=None,
                tau_scaled=False,
                gates=gates,
            )
        )
    if write_scaled:
        variants.append(
            save_splat_filter_variant(
                splats=splats,
                gates=gates,
                mask=all_mask,
                out_dir=out_dir,
                method_name=method_name,
                name="gate_scaled",
                kind="gate_scaled",
                source_gate_path=gate_path,
                gate_threshold=None,
                tau_scaled=True,
                tau_scale_floor=tau_scale_floor,
            )
        )
    if write_filtered:
        for threshold in sorted(set(gate_thresholds)):
            mask = gates >= threshold
            if not np.any(mask):
                continue
            label = format_threshold_label(threshold)
            variants.append(
                save_splat_filter_variant(
                    splats=splats,
                    gates=gates,
                    mask=mask,
                    out_dir=out_dir,
                    method_name=method_name,
                    name=f"gate_ge_{label}",
                    kind="gate_threshold",
                    source_gate_path=gate_path,
                    gate_threshold=threshold,
                    tau_scaled=False,
                    tau_scale_floor=tau_scale_floor,
                )
            )
            if include_random_baselines:
                variants.extend(
                    save_random_same_retention_variants(
                        splats=splats,
                        gates=gates,
                        retained_count=int(mask.sum()),
                        out_dir=out_dir,
                        method_name=method_name,
                        source_gate_path=gate_path,
                        gate_threshold=threshold,
                        threshold_label=label,
                        random_baseline_seeds=random_baseline_seeds,
                    )
                )
    return variants


def save_random_same_retention_variants(
    *,
    splats: SplatCloud,
    gates: np.ndarray,
    retained_count: int,
    out_dir: Path,
    method_name: str,
    source_gate_path: Path,
    gate_threshold: float,
    threshold_label: str,
    random_baseline_seeds: tuple[int, ...],
) -> list[SplatFilterVariant]:
    splat_count = int(gates.shape[0])
    variants = []
    for random_seed in random_baseline_seeds:
        rng = np.random.default_rng(random_seed)
        selected = rng.choice(splat_count, size=retained_count, replace=False)
        mask = np.zeros((splat_count,), dtype=bool)
        mask[selected] = True
        variants.append(
            save_splat_filter_variant(
                splats=splats,
                gates=gates,
                mask=mask,
                out_dir=out_dir,
                method_name=method_name,
                name=f"random_same_retention_{threshold_label}_seed{random_seed}",
                kind="random_same_retention",
                source_gate_path=source_gate_path,
                gate_threshold=gate_threshold,
                tau_scaled=False,
                tau_scale_floor=0.0,
                random_seed=random_seed,
            )
        )
    return variants


def save_splat_filter_variant(
    *,
    splats: SplatCloud,
    gates: np.ndarray,
    mask: np.ndarray,
    out_dir: Path,
    method_name: str,
    name: str,
    kind: str,
    source_gate_path: Path,
    gate_threshold: float | None,
    tau_scaled: bool,
    tau_scale_floor: float,
    random_seed: int | None = None,
) -> SplatFilterVariant:
    selected = np.flatnonzero(mask).astype(np.int64)
    selected_gates = gates[selected]
    mask_tensor = torch.from_numpy(mask)
    tau = splats.tau[mask_tensor].clone()
    if tau_scaled:
        tau_multiplier = torch.from_numpy(gate_multiplier(selected_gates, tau_scale_floor)).to(dtype=tau.dtype)
        tau = tau * tau_multiplier
    variant = SplatCloud(
        centers=splats.centers[mask_tensor].clone(),
        colors=splats.colors[mask_tensor].clone(),
        tau=tau,
        sigma=splats.sigma[mask_tensor].clone(),
        is_surface=splats.is_surface[mask_tensor].clone(),
    )
    splats_out = out_dir / f"{method_name}_{name}_splats.npz"
    gate_out = out_dir / f"{method_name}_{name}_gate.npz"
    save_splats(str(splats_out), variant)
    np.savez_compressed(
        gate_out,
        gate=selected_gates,
        raw_gate=selected_gates,
        splat_index=np.arange(selected.shape[0], dtype=np.int64),
        source_splat_index=selected,
        source_gate_path=np.asarray(str(source_gate_path)),
        gate_threshold=np.asarray(np.nan if gate_threshold is None else float(gate_threshold), dtype=np.float64),
        tau_scaled=np.asarray(bool(tau_scaled)),
        tau_scale_floor=np.asarray(float(tau_scale_floor), dtype=np.float64),
        random_seed=np.asarray(-1 if random_seed is None else int(random_seed), dtype=np.int64),
    )
    return SplatFilterVariant(
        name=name,
        kind=kind,
        splats_path=splats_out,
        gate_path=gate_out,
        retained_count=int(selected.shape[0]),
        retention_fraction=float(selected.shape[0] / gates.shape[0]),
        gate_threshold=gate_threshold,
        tau_scaled=tau_scaled,
        gates=selected_gates,
        random_seed=random_seed,
    )


def load_gate_array(path: str | Path, *, expected_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        key = "gate" if "gate" in data.files else "raw_gate"
        gates = np.asarray(data[key], dtype=np.float64).reshape(-1)
    if gates.shape[0] != expected_count:
        raise ValueError(f"Gate count {gates.shape[0]} does not match splat count {expected_count}.")
    return np.clip(gates, 0.0, 1.0)


def gate_multiplier(gates: np.ndarray, tau_scale_floor: float) -> np.ndarray:
    return np.clip(tau_scale_floor + (1.0 - tau_scale_floor) * gates, 0.0, 1.0)


def validate_filtering_config(
    *,
    gate_thresholds: tuple[float, ...],
    include_baseline: bool,
    write_scaled: bool,
    write_filtered: bool,
    include_random_baselines: bool,
    random_baseline_seeds: tuple[int, ...],
    tau_scale_floor: float,
    render_max_frames: int,
) -> None:
    if not include_baseline and not write_scaled and not write_filtered:
        raise ValueError("Enable at least one of baseline, scaled, or filtered variants.")
    if any(not 0.0 <= threshold <= 1.0 for threshold in gate_thresholds):
        raise ValueError("gate_thresholds must be in [0, 1].")
    if include_random_baselines and not write_filtered:
        raise ValueError("Random same-retention baselines require filtered gate-threshold variants.")
    if include_random_baselines and not random_baseline_seeds:
        raise ValueError("At least one random_baseline_seed is required when random baselines are enabled.")
    if not 0.0 <= tau_scale_floor <= 1.0:
        raise ValueError("tau_scale_floor must be in [0, 1].")
    if render_max_frames < 0:
        raise ValueError("render_max_frames must be non-negative.")


def format_filtering_report(status: dict[str, Any], comparison: pd.DataFrame) -> str:
    lines = [
        "# Calibrated Splat Filtering",
        "",
        f"- Method: `{status['method']}`",
        f"- Input splats: `{status['input_splats_path']}`",
        f"- Input gate: `{status['input_gate_path']}`",
        f"- Variants evaluated: `{status['variant_count']}`",
        f"- Random same-retention baselines: `{status.get('include_random_baselines', False)}`",
        f"- Comparison CSV: `{status['comparison_path']}`",
    ]
    if not comparison.empty:
        lines.extend(["", "## Comparison", "", format_comparison_table(comparison)])
    return "\n".join(lines) + "\n"


def format_comparison_table(comparison: pd.DataFrame) -> str:
    columns = ["variant", "variant_kind", "geometry_threshold", "retained_count", "retention_fraction", "precision", "recall", "f_score", "chamfer_l1", "mean_psnr"]
    table = comparison[columns].copy()
    lines = [
        "| variant | kind | threshold | retained | retention | precision | recall | f_score | chamfer_l1 | psnr |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table.itertuples(index=False):
        psnr = "n/a" if pd.isna(row.mean_psnr) else f"{row.mean_psnr:.6g}"
        lines.append(
            f"| `{row.variant}` | `{row.variant_kind}` | {row.geometry_threshold:.6g} | {row.retained_count} | {row.retention_fraction:.6g} | "
            f"{row.precision:.6g} | {row.recall:.6g} | {row.f_score:.6g} | {row.chamfer_l1:.6g} | {psnr} |"
        )
    return "\n".join(lines)
