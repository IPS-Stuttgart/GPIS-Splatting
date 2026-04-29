from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.real_bootstrap import SAMPLE_TYPE_IDS, bootstrap_real_gpis
from gpis_splatting.real_gate_diagnostics import run_tanks_temples_gate_diagnostics
from gpis_splatting.real_pipeline import fit_real_gpis
from gpis_splatting.real_scene import load_prepared_scene
from gpis_splatting.serialization import write_json

CONSTRUCTION_MODES = ("existing", "surface_free", "strong_free", "behind_surface", "normal_offsets")
NORMAL_POSITIVE_TYPE_ID = 3
NORMAL_NEGATIVE_TYPE_ID = 4


@dataclass(frozen=True)
class SweepArtifacts:
    mode: str
    samples_path: Path
    splats_path: Path
    bootstrap_report_path: Path | None


def run_real_gpis_gate_model_sweep(
    *,
    scene_dir: str | Path,
    sweep_name: str = "real_gate_model_sweep",
    construction_modes: tuple[str, ...] = ("surface_free", "behind_surface", "normal_offsets"),
    samples_path: str | Path | None = None,
    splats_path: str | Path | None = None,
    point_source: str = "auto",
    point_path: str | Path | None = None,
    lengthscales: tuple[float, ...] = (0.15, 0.25, 0.4),
    noise_stds: tuple[float, ...] = (0.03, 0.06),
    epsilons: tuple[float, ...] = (0.08, 0.16, 0.24),
    gate_floors: tuple[float, ...] = (0.0, 0.25),
    variance: float = 1.0,
    jitter: float = 1e-6,
    use_observation_noise: bool = True,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.1),
    topk_fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
    num_bins: int = 10,
    max_bootstrap_points: int | None = 5000,
    max_train_points: int | None = 1200,
    max_pred_points: int | None = 100_000,
    max_gt_points: int | None = 100_000,
    seed: int = 13,
    gate_batch_size: int = 4096,
    distance_chunk_size: int = 256,
    normal_offset_distance: float = 0.04,
    normal_offset_noise_std: float = 0.05,
    normal_offset_neighbors: int = 12,
    max_normal_offset_points: int | None = 3000,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    validate_sweep_config(
        construction_modes=construction_modes,
        lengthscales=lengthscales,
        noise_stds=noise_stds,
        epsilons=epsilons,
        gate_floors=gate_floors,
    )
    scene_root = Path(scene_dir).resolve()
    scene_meta, _, _ = load_prepared_scene(scene_root)
    artifact_root = (Path(output_dir) if output_dir is not None else scene_root / "model_sweeps" / sweep_name).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    evaluations_dir = artifact_root / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    artifacts_by_mode = {
        mode: prepare_construction_mode(
            scene_root=scene_root,
            artifact_root=artifact_root,
            sweep_name=sweep_name,
            mode=mode,
            samples_path=samples_path,
            splats_path=splats_path,
            point_source=point_source,
            point_path=point_path,
            max_bootstrap_points=max_bootstrap_points,
            seed=seed,
            normal_offset_distance=normal_offset_distance,
            normal_offset_noise_std=normal_offset_noise_std,
            normal_offset_neighbors=normal_offset_neighbors,
            max_normal_offset_points=max_normal_offset_points,
        )
        for mode in construction_modes
    }

    rows = []
    failures: list[dict[str, Any]] = []
    for mode in construction_modes:
        artifacts = artifacts_by_mode[mode]
        for lengthscale in lengthscales:
            for noise_std in noise_stds:
                model_tag = f"{sweep_name}_{mode}_ls{format_label(lengthscale)}_n{format_label(noise_std)}"
                model_path = artifact_root / f"{model_tag}_gpis_model.npz"
                try:
                    fit_result = fit_real_gpis(
                        scene_dir=scene_root,
                        samples_path=artifacts.samples_path,
                        output_model=model_path,
                        lengthscale=lengthscale,
                        variance=variance,
                        noise_std=noise_std,
                        jitter=jitter,
                        max_train_points=max_train_points,
                        seed=seed + 17,
                        use_observation_noise=use_observation_noise,
                    )
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        {
                            "construction_mode": mode,
                            "lengthscale": lengthscale,
                            "noise_std": noise_std,
                            "stage": "fit_real_gpis",
                            "error": str(exc),
                        }
                    )
                    continue
                for epsilon in epsilons:
                    for gate_floor in gate_floors:
                        method = f"{model_tag}_eps{format_label(epsilon)}_floor{format_label(gate_floor)}"
                        try:
                            diagnostic = run_tanks_temples_gate_diagnostics(
                                scene_dir=scene_root,
                                splats_path=artifacts.splats_path,
                                output_dir=evaluations_dir,
                                method_name=method,
                                thresholds=thresholds,
                                topk_fractions=topk_fractions,
                                num_bins=num_bins,
                                max_pred_points=max_pred_points,
                                max_gt_points=max_gt_points,
                                seed=seed,
                                model_path=model_path,
                                epsilon=epsilon,
                                gate_floor=gate_floor,
                                gate_batch_size=gate_batch_size,
                                distance_chunk_size=distance_chunk_size,
                            )
                        except Exception as exc:  # noqa: BLE001
                            failures.append(
                                {
                                    "construction_mode": mode,
                                    "lengthscale": lengthscale,
                                    "noise_std": noise_std,
                                    "epsilon": epsilon,
                                    "gate_floor": gate_floor,
                                    "stage": "diagnose_tanks_temples_gates",
                                    "error": str(exc),
                                }
                            )
                            continue
                        rows.extend(
                            summarize_diagnostic(
                                scene=scene_meta["scene"],
                                dataset=scene_meta.get("dataset"),
                                sweep_name=sweep_name,
                                construction_mode=mode,
                                samples_path=artifacts.samples_path,
                                splats_path=artifacts.splats_path,
                                model_path=model_path,
                                fit_report=fit_result["report"],
                                diagnostic=diagnostic,
                                lengthscale=lengthscale,
                                noise_std=noise_std,
                                epsilon=epsilon,
                                gate_floor=gate_floor,
                                variance=variance,
                                use_observation_noise=use_observation_noise,
                            )
                        )

    summary = pd.DataFrame(rows)
    summary_path = artifact_root / f"{sweep_name}_summary.csv"
    status_path = artifact_root / f"{sweep_name}_status.json"
    report_path = artifact_root / f"{sweep_name}_report.md"
    summary.to_csv(summary_path, index=False)
    best = best_sweep_rows(summary)
    status = {
        "schema_version": 1,
        "scene": scene_meta["scene"],
        "dataset": scene_meta.get("dataset"),
        "sweep_name": sweep_name,
        "construction_modes": list(construction_modes),
        "lengthscales": list(lengthscales),
        "noise_stds": list(noise_stds),
        "epsilons": list(epsilons),
        "gate_floors": list(gate_floors),
        "variance": variance,
        "jitter": jitter,
        "use_observation_noise": use_observation_noise,
        "thresholds": list(thresholds),
        "topk_fractions": list(topk_fractions),
        "artifact_root": str(artifact_root),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "failure_count": len(failures),
        "failures": failures,
        "best_by_gate_error_spearman": best["by_spearman"],
        "best_by_delta_f_score": best["by_delta_f_score"],
        "row_count": int(summary.shape[0]),
    }
    write_json(status_path, status)
    report_path.write_text(format_model_sweep_report(status, summary), encoding="utf-8")
    return {
        "summary_path": summary_path,
        "status_path": status_path,
        "report_path": report_path,
        "summary": summary,
        "status": status,
        "artifact_root": artifact_root,
    }


def validate_sweep_config(
    *,
    construction_modes: tuple[str, ...],
    lengthscales: tuple[float, ...],
    noise_stds: tuple[float, ...],
    epsilons: tuple[float, ...],
    gate_floors: tuple[float, ...],
) -> None:
    if not construction_modes:
        raise ValueError("At least one construction mode is required.")
    unknown = sorted(set(construction_modes) - set(CONSTRUCTION_MODES))
    if unknown:
        raise ValueError(f"Unsupported construction modes: {', '.join(unknown)}")
    if any(value <= 0.0 for value in lengthscales):
        raise ValueError("Lengthscales must be positive.")
    if any(value <= 0.0 for value in noise_stds):
        raise ValueError("Noise standard deviations must be positive.")
    if any(value <= 0.0 for value in epsilons):
        raise ValueError("Epsilons must be positive.")
    if any(value < 0.0 or value > 1.0 for value in gate_floors):
        raise ValueError("Gate floors must be in [0, 1].")


def prepare_construction_mode(
    *,
    scene_root: Path,
    artifact_root: Path,
    sweep_name: str,
    mode: str,
    samples_path: str | Path | None,
    splats_path: str | Path | None,
    point_source: str,
    point_path: str | Path | None,
    max_bootstrap_points: int | None,
    seed: int,
    normal_offset_distance: float,
    normal_offset_noise_std: float,
    normal_offset_neighbors: int,
    max_normal_offset_points: int | None,
) -> SweepArtifacts:
    if mode == "existing":
        return SweepArtifacts(
            mode=mode,
            samples_path=resolve_scene_file(scene_root, samples_path, "real_samples.npz"),
            splats_path=resolve_scene_file(scene_root, splats_path, "real_splats.npz"),
            bootstrap_report_path=None,
        )

    params = construction_mode_bootstrap_params(mode)
    output_prefix = f"{sweep_name}_{mode}"
    bootstrap = bootstrap_real_gpis(
        scene_dir=scene_root,
        point_source=point_source,
        point_path=point_path,
        output_prefix=output_prefix,
        max_points=max_bootstrap_points,
        seed=seed,
        **params,
    )
    mode_samples = Path(bootstrap["samples_path"])
    mode_splats = Path(bootstrap["splats_path"])
    if mode == "normal_offsets":
        augmented_samples = artifact_root / f"{output_prefix}_normal_offsets_samples.npz"
        add_normal_offset_samples(
            samples_path=mode_samples,
            output_path=augmented_samples,
            scene_dir=scene_root,
            offset_distance=normal_offset_distance,
            noise_std=normal_offset_noise_std,
            neighbors=normal_offset_neighbors,
            max_points=max_normal_offset_points,
            seed=seed + 101,
        )
        mode_samples = augmented_samples
    return SweepArtifacts(
        mode=mode,
        samples_path=mode_samples,
        splats_path=mode_splats,
        bootstrap_report_path=Path(bootstrap["report_path"]),
    )


def construction_mode_bootstrap_params(mode: str) -> dict[str, Any]:
    base = {
        "free_space_samples_per_point": 2,
        "free_space_min_fraction": 0.2,
        "free_space_max_fraction": 0.85,
        "add_behind_surface_samples": False,
        "behind_surface_fraction": 1.08,
        "max_sample_distance": 0.35,
        "surface_noise_std": 0.03,
        "free_space_noise_std": 0.08,
        "behind_surface_noise_std": 0.12,
        "splat_tau": 0.45,
        "splat_sigma": 0.025,
    }
    if mode == "surface_free":
        return base
    if mode == "strong_free":
        return {
            **base,
            "free_space_samples_per_point": 4,
            "free_space_max_fraction": 0.92,
            "max_sample_distance": 0.55,
            "free_space_noise_std": 0.05,
        }
    if mode == "behind_surface":
        return {**base, "add_behind_surface_samples": True}
    if mode == "normal_offsets":
        return {**base, "add_behind_surface_samples": True}
    raise ValueError(f"Unsupported construction mode {mode!r}.")


def add_normal_offset_samples(
    *,
    samples_path: str | Path,
    output_path: str | Path,
    scene_dir: str | Path,
    offset_distance: float,
    noise_std: float,
    neighbors: int,
    max_points: int | None,
    seed: int,
) -> Path:
    if offset_distance <= 0.0:
        raise ValueError("normal offset distance must be positive.")
    if noise_std <= 0.0:
        raise ValueError("normal offset noise must be positive.")
    if neighbors < 3:
        raise ValueError("normal offset neighbors must be at least 3.")

    scene_root = Path(scene_dir)
    _, frames, splits = load_prepared_scene(scene_root)
    train_frames = [frames[index] for index in splits.get("train", [])]
    camera_centers = np.asarray([np.asarray(frame["camera_to_world"], dtype=np.float64)[:3, 3] for frame in train_frames], dtype=np.float64)
    with np.load(samples_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    points = np.asarray(arrays["points"], dtype=np.float64)
    sdf = np.asarray(arrays["sdf"], dtype=np.float64).reshape(-1)
    noise = np.asarray(arrays["observation_noise_std"], dtype=np.float64).reshape(-1)
    sample_type = np.asarray(arrays["sample_type"], dtype=np.int64).reshape(-1)
    source_index = np.asarray(arrays["source_point_index"], dtype=np.int64).reshape(-1)
    camera_index = np.asarray(arrays["camera_index"], dtype=np.int64).reshape(-1)
    ray_distance = np.asarray(arrays["ray_distance"], dtype=np.float64).reshape(-1)
    surface = np.flatnonzero(sample_type == SAMPLE_TYPE_IDS["surface"])
    if surface.size == 0:
        raise ValueError("No surface samples are available for normal-offset augmentation.")
    if max_points is not None and max_points > 0 and surface.size > max_points:
        rng = np.random.default_rng(seed)
        surface = np.sort(rng.choice(surface, size=max_points, replace=False)).astype(np.int64)

    surface_points = points[surface]
    normals = estimate_pca_normals(surface_points, neighbors=neighbors)
    camera_vectors = camera_centers[np.clip(camera_index[surface], 0, camera_centers.shape[0] - 1)] - surface_points
    orientation = np.sum(normals * camera_vectors, axis=1)
    normals[orientation < 0.0] *= -1.0
    weak = np.linalg.norm(camera_vectors, axis=1) <= 1e-9
    if np.any(weak):
        normals[weak] = fallback_ray_normals(surface_points[weak], camera_centers[0])

    positive_points = surface_points + normals * offset_distance
    negative_points = surface_points - normals * offset_distance
    augmented = {
        **arrays,
        "points": np.concatenate((points, positive_points, negative_points), axis=0),
        "sdf": np.concatenate((sdf, np.full(surface.shape[0], offset_distance), np.full(surface.shape[0], -offset_distance))),
        "observation_noise_std": np.concatenate((noise, np.full(surface.shape[0], noise_std), np.full(surface.shape[0], noise_std))),
        "sample_type": np.concatenate(
            (
                sample_type,
                np.full(surface.shape[0], NORMAL_POSITIVE_TYPE_ID, dtype=np.int64),
                np.full(surface.shape[0], NORMAL_NEGATIVE_TYPE_ID, dtype=np.int64),
            )
        ),
        "source_point_index": np.concatenate((source_index, source_index[surface], source_index[surface])),
        "camera_index": np.concatenate((camera_index, camera_index[surface], camera_index[surface])),
        "ray_distance": np.concatenate((ray_distance, ray_distance[surface], ray_distance[surface])),
        "sample_type_names": np.asarray(["surface", "free_space", "behind_surface", "normal_positive", "normal_negative"]),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **augmented)
    return output


def estimate_pca_normals(points: np.ndarray, *, neighbors: int) -> np.ndarray:
    normals = np.empty_like(points)
    neighbor_count = min(max(3, neighbors), points.shape[0])
    for index, point in enumerate(points):
        squared = np.sum((points - point[None, :]) ** 2, axis=1)
        nearest = np.argpartition(squared, kth=neighbor_count - 1)[:neighbor_count]
        centered = points[nearest] - points[nearest].mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(neighbor_count - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        normal = eigvecs[:, int(np.argmin(eigvals))]
        norm = float(np.linalg.norm(normal))
        normals[index] = normal / norm if norm > 1e-12 else np.asarray([0.0, 0.0, 1.0])
    return normals


def fallback_ray_normals(points: np.ndarray, camera_center: np.ndarray) -> np.ndarray:
    vectors = camera_center[None, :] - points
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def summarize_diagnostic(
    *,
    scene: str,
    dataset: str | None,
    sweep_name: str,
    construction_mode: str,
    samples_path: Path,
    splats_path: Path,
    model_path: Path,
    fit_report: dict[str, Any],
    diagnostic: dict[str, Any],
    lengthscale: float,
    noise_std: float,
    epsilon: float,
    gate_floor: float,
    variance: float,
    use_observation_noise: bool,
) -> list[dict[str, Any]]:
    status = diagnostic["status"]
    ranked = diagnostic["ranked_quality"]
    correlations = status["correlations"]
    rows = []
    for best in status["best_topk_by_f_score"]:
        threshold = float(best["geometry_threshold"])
        full = ranked[(ranked["geometry_threshold"] == threshold) & (ranked["topk_fraction"] == 1.0)]
        full_row = full.iloc[0] if not full.empty else None
        rows.append(
            {
                "scene": scene,
                "dataset": dataset,
                "sweep_name": sweep_name,
                "construction_mode": construction_mode,
                "lengthscale": lengthscale,
                "variance": variance,
                "noise_std": noise_std,
                "epsilon": epsilon,
                "gate_floor": gate_floor,
                "use_observation_noise": use_observation_noise,
                "geometry_threshold": threshold,
                "spearman_gate_vs_negative_distance": correlations["spearman_gate_vs_negative_distance"],
                "pearson_gate_vs_negative_distance": correlations["pearson_gate_vs_negative_distance"],
                "best_topk_fraction": float(best["topk_fraction"]),
                "best_retention_fraction": float(best["retention_fraction"]),
                "best_selected_pred_point_count": int(best["selected_pred_point_count"]),
                "best_precision": float(best["precision"]),
                "best_recall": float(best["recall"]),
                "best_f_score": float(best["f_score"]),
                "full_precision": float(full_row["precision"]) if full_row is not None else np.nan,
                "full_recall": float(full_row["recall"]) if full_row is not None else np.nan,
                "full_f_score": float(full_row["f_score"]) if full_row is not None else np.nan,
                "delta_best_f_score_vs_full": float(best["f_score"] - full_row["f_score"]) if full_row is not None else np.nan,
                "train_sample_count": int(fit_report["train_sample_count"]),
                "available_sample_count": int(fit_report["available_sample_count"]),
                "samples_path": str(samples_path),
                "splats_path": str(splats_path),
                "model_path": str(model_path),
                "diagnostic_status_path": str(diagnostic["status_path"]),
                "diagnostic_report_path": str(diagnostic["report_path"]),
            }
        )
    return rows


def best_sweep_rows(summary: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if summary.empty:
        return {"by_spearman": [], "by_delta_f_score": []}
    by_spearman = []
    by_delta = []
    for threshold, group in summary.groupby("geometry_threshold"):
        spearman_row = group.sort_values(["spearman_gate_vs_negative_distance", "best_f_score"], ascending=[False, False]).iloc[0]
        delta_row = group.sort_values(["delta_best_f_score_vs_full", "best_f_score"], ascending=[False, False]).iloc[0]
        by_spearman.append(compact_best_row(spearman_row, threshold=threshold))
        by_delta.append(compact_best_row(delta_row, threshold=threshold))
    return {"by_spearman": by_spearman, "by_delta_f_score": by_delta}


def compact_best_row(row: pd.Series, *, threshold: float) -> dict[str, Any]:
    return {
        "geometry_threshold": float(threshold),
        "construction_mode": row["construction_mode"],
        "lengthscale": float(row["lengthscale"]),
        "noise_std": float(row["noise_std"]),
        "epsilon": float(row["epsilon"]),
        "gate_floor": float(row["gate_floor"]),
        "spearman_gate_vs_negative_distance": maybe_float(row["spearman_gate_vs_negative_distance"]),
        "best_topk_fraction": float(row["best_topk_fraction"]),
        "best_f_score": float(row["best_f_score"]),
        "delta_best_f_score_vs_full": float(row["delta_best_f_score_vs_full"]),
    }


def format_model_sweep_report(status: dict[str, Any], summary: pd.DataFrame) -> str:
    lines = [
        "# Real GPIS Gate Model Sweep",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Sweep: `{status['sweep_name']}`",
        f"- Rows: `{status['row_count']}`",
        f"- Failures: `{status['failure_count']}`",
        f"- Summary CSV: `{status['summary_path']}`",
        "",
        "## Best By Spearman(gate, -distance)",
        "",
    ]
    for row in status["best_by_gate_error_spearman"]:
        lines.append(format_best_bullet(row))
    lines.extend(["", "## Best By F-Score Gain Over Full Retention", ""])
    for row in status["best_by_delta_f_score"]:
        lines.append(format_best_bullet(row))
    if not summary.empty:
        lines.extend(["", "## Summary Table", "", format_summary_table(summary)])
    if status["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in status["failures"]:
            lines.append(f"- `{failure.get('stage')}` `{failure.get('construction_mode')}`: {failure.get('error')}")
    return "\n".join(lines) + "\n"


def format_best_bullet(row: dict[str, Any]) -> str:
    return (
        f"- threshold `{row['geometry_threshold']:.6g}`: `{row['construction_mode']}`, "
        f"lengthscale `{row['lengthscale']:.6g}`, noise `{row['noise_std']:.6g}`, epsilon `{row['epsilon']:.6g}`, "
        f"floor `{row['gate_floor']:.6g}`, Spearman `{format_optional(row['spearman_gate_vs_negative_distance'])}`, "
        f"best F-score `{row['best_f_score']:.6g}`, delta `{row['delta_best_f_score_vs_full']:.6g}`"
    )


def format_summary_table(summary: pd.DataFrame) -> str:
    columns = [
        "construction_mode",
        "lengthscale",
        "noise_std",
        "epsilon",
        "gate_floor",
        "geometry_threshold",
        "spearman_gate_vs_negative_distance",
        "best_topk_fraction",
        "best_f_score",
        "delta_best_f_score_vs_full",
    ]
    lines = [
        "| mode | lengthscale | noise | epsilon | floor | threshold | spearman | top_k | best_f | delta_f |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary[columns].sort_values(["geometry_threshold", "spearman_gate_vs_negative_distance"], ascending=[True, False]).itertuples(index=False):
        lines.append(
            f"| `{row.construction_mode}` | {row.lengthscale:.6g} | {row.noise_std:.6g} | {row.epsilon:.6g} | {row.gate_floor:.6g} | "
            f"{row.geometry_threshold:.6g} | {format_optional(row.spearman_gate_vs_negative_distance)} | {row.best_topk_fraction:.6g} | "
            f"{row.best_f_score:.6g} | {row.delta_best_f_score_vs_full:.6g} |"
        )
    return "\n".join(lines)


def resolve_scene_file(scene_root: Path, path: str | Path | None, default_name: str) -> Path:
    resolved = Path(default_name) if path is None else Path(path)
    if not resolved.is_absolute():
        resolved = scene_root / resolved
    return resolved


def format_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def format_optional(value: Any) -> str:
    parsed = maybe_float(value)
    return "n/a" if parsed is None else f"{parsed:.6g}"
