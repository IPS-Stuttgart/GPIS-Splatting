from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.external_3dgs import load_3dgs_ply, load_gate_array, opacity_to_alpha, vertex_centers, vertex_colors, vertex_sigma
from gpis_splatting.real_bootstrap import load_ply_point_cloud
from gpis_splatting.real_geometry import crop_mask, deterministic_subsample, evaluate_geometry_group, format_threshold_label, load_alignment_matrix, transform_points
from gpis_splatting.serialization import read_json, write_json

SURFACE_EXTRACTION_MODES = ("centers", "surfels", "opacity_field")


@dataclass(frozen=True)
class SurfaceMesh:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray | None = None

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


def extract_gpis_gated_gaussian_surfaces(
    *,
    input_ply_path: str | Path,
    output_dir: str | Path,
    gate_path: str | Path | None = None,
    method_name: str = "gpis_gated_3dgs_surface",
    gate_thresholds: tuple[float, ...] = (0.5,),
    extraction_modes: tuple[str, ...] = SURFACE_EXTRACTION_MODES,
    include_baseline: bool = True,
    opacity_mode: str = "logit",
    surfel_scale: float = 1.0,
    min_surfel_radius: float = 1e-5,
    max_surfel_radius: float | None = None,
    opacity_field_resolution: int = 48,
    opacity_field_threshold: float = 0.15,
    opacity_field_sigma_scale: float = 1.0,
    opacity_field_margin_sigma: float = 3.0,
    max_field_gaussians: int | None = 20_000,
    field_query_chunk_size: int = 4096,
    field_gaussian_chunk_size: int = 1024,
    seed: int = 13,
) -> dict[str, Any]:
    validate_extraction_config(
        method_name=method_name,
        gate_thresholds=gate_thresholds,
        extraction_modes=extraction_modes,
        include_baseline=include_baseline,
        surfel_scale=surfel_scale,
        min_surfel_radius=min_surfel_radius,
        max_surfel_radius=max_surfel_radius,
        opacity_field_resolution=opacity_field_resolution,
        opacity_field_threshold=opacity_field_threshold,
        opacity_field_sigma_scale=opacity_field_sigma_scale,
        opacity_field_margin_sigma=opacity_field_margin_sigma,
        max_field_gaussians=max_field_gaussians,
        field_query_chunk_size=field_query_chunk_size,
        field_gaussian_chunk_size=field_gaussian_chunk_size,
    )
    if opacity_mode not in {"logit", "linear"}:
        raise ValueError("opacity_mode must be 'logit' or 'linear'.")

    ply = load_3dgs_ply(input_ply_path)
    centers = vertex_centers(ply.vertices)
    colors = vertex_colors(ply.vertices)
    alpha = vertex_alpha_safe(ply.vertices, opacity_mode=opacity_mode)
    sigma = vertex_sigma(ply.vertices)
    gates = load_gate_array(gate_path, expected_count=ply.vertex_count) if gate_path is not None else None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    variants = gaussian_selection_variants(ply.vertex_count, gates=gates, gate_thresholds=gate_thresholds, include_baseline=include_baseline)
    for variant in variants:
        mask = np.asarray(variant["mask"], dtype=bool)
        if not np.any(mask):
            continue
        selected_gates = gates[mask] if gates is not None else None
        for mode in extraction_modes:
            if mode == "centers":
                mesh = SurfaceMesh(centers[mask], np.zeros((0, 3), dtype=np.int64), colors[mask])
                representation = "point_cloud"
            elif mode == "surfels":
                mesh = build_oriented_surfel_mesh(ply.vertices, mask=mask, colors=colors, surfel_scale=surfel_scale, min_radius=min_surfel_radius, max_radius=max_surfel_radius)
                representation = "surfel_mesh"
            elif mode == "opacity_field":
                mesh = build_opacity_field_mesh(
                    centers=centers,
                    sigma=sigma,
                    alpha=alpha,
                    colors=colors,
                    mask=mask,
                    resolution=opacity_field_resolution,
                    opacity_threshold=opacity_field_threshold,
                    sigma_scale=opacity_field_sigma_scale,
                    margin_sigma=opacity_field_margin_sigma,
                    max_gaussians=max_field_gaussians,
                    query_chunk_size=field_query_chunk_size,
                    gaussian_chunk_size=field_gaussian_chunk_size,
                    seed=seed,
                )
                representation = "opacity_field_mesh"
            else:
                raise ValueError(f"Unsupported extraction mode {mode!r}.")
            path = out_dir / f"{method_name}_{variant['name']}_{mode}.ply"
            write_surface_ply(path, mesh)
            rows.append(
                {
                    "method": method_name,
                    "variant": variant["name"],
                    "variant_kind": variant["kind"],
                    "extraction_method": mode,
                    "representation": representation,
                    "geometry_path": str(path),
                    "retained_count": int(mask.sum()),
                    "retention_fraction": float(mask.mean()),
                    "gate_threshold": variant.get("gate_threshold"),
                    "gate_min": optional_stat(selected_gates, "min"),
                    "gate_max": optional_stat(selected_gates, "max"),
                    "gate_mean": optional_stat(selected_gates, "mean"),
                    "surface_vertex_count": mesh.vertex_count,
                    "surface_face_count": mesh.face_count,
                }
            )
    if not rows:
        raise ValueError("No surface variants were written. Check gate thresholds and extraction modes.")

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / f"{method_name}_surface_manifest.csv"
    status_path = out_dir / f"{method_name}_surface_status.json"
    report_path = out_dir / f"{method_name}_surface_report.md"
    manifest.to_csv(manifest_path, index=False)
    status = {
        "schema_version": 1,
        "method": method_name,
        "input_ply_path": str(Path(input_ply_path)),
        "gate_path": str(Path(gate_path)) if gate_path is not None else None,
        "output_dir": str(out_dir),
        "input_gaussian_count": ply.vertex_count,
        "gate_available": gates is not None,
        "gate_thresholds": list(gate_thresholds),
        "extraction_modes": list(extraction_modes),
        "surfel_scale": surfel_scale,
        "opacity_field_resolution": opacity_field_resolution,
        "opacity_field_threshold": opacity_field_threshold,
        "surface_count": int(len(rows)),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
    }
    write_json(status_path, status)
    report_path.write_text(format_surface_extraction_report(status, manifest), encoding="utf-8")
    return {"manifest_path": manifest_path, "status_path": status_path, "report_path": report_path, "manifest": manifest, "status": status}


def evaluate_gaussian_surface_geometry(
    *,
    manifest_path: str | Path,
    ground_truth_path: str | Path,
    output_dir: str | Path,
    method_name: str = "gpis_gated_3dgs_surface",
    thresholds: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1),
    max_pred_points: int | None = 100_000,
    max_gt_points: int | None = 100_000,
    seed: int = 13,
    alignment_path: str | Path | None = None,
    invert_alignment: bool = False,
    crop_path: str | Path | None = None,
    use_crop: bool = True,
    distance_chunk_size: int = 256,
) -> dict[str, Any]:
    if not thresholds or any(threshold <= 0.0 for threshold in thresholds):
        raise ValueError("Distance thresholds must be positive and non-empty.")
    if distance_chunk_size < 1:
        raise ValueError("distance_chunk_size must be positive.")
    manifest_file = Path(manifest_path)
    manifest = pd.read_csv(manifest_file)
    validate_surface_manifest(manifest)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alignment_matrix = None
    if alignment_path is not None:
        alignment_matrix = load_alignment_matrix(alignment_path)
        if invert_alignment:
            alignment_matrix = np.linalg.inv(alignment_matrix)
    crop = read_json(crop_path) if crop_path is not None and use_crop else None

    gt_points = load_ply_point_cloud(ground_truth_path).points.astype(np.float64)
    gt_count_input = int(gt_points.shape[0])
    if crop is not None:
        gt_points = gt_points[crop_mask(gt_points, crop)]
    gt_points, gt_indices = deterministic_subsample(gt_points, max_points=max_gt_points, seed=seed + 1)
    if gt_points.size == 0:
        raise ValueError("No ground-truth points remain after cropping/subsampling.")

    summary_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(manifest.itertuples(index=False)):
        geometry_path = Path(str(row.geometry_path))
        if not geometry_path.is_absolute():
            geometry_path = manifest_file.parent / geometry_path
        surface = read_surface_ply(geometry_path)
        pred_points = surface_points_for_evaluation(surface, max_points=max_pred_points, seed=seed + row_index)
        pred_count_input = int(pred_points.shape[0])
        if alignment_matrix is not None:
            pred_points = transform_points(pred_points, alignment_matrix)
        if crop is not None:
            pred_points = pred_points[crop_mask(pred_points, crop)]
        if pred_points.size == 0:
            skipped_rows.append({"geometry_path": str(geometry_path), "reason": "no predicted points after crop/subsample"})
            continue
        pred_points, pred_indices = deterministic_subsample(pred_points, max_points=max_pred_points, seed=seed + row_index)
        summary, metrics = evaluate_geometry_group(pred_points, gt_points, thresholds=thresholds, distance_chunk_size=distance_chunk_size)
        base = {
            "method": method_name,
            "variant": str(row.variant),
            "variant_kind": str(row.variant_kind),
            "extraction_method": str(row.extraction_method),
            "representation": str(row.representation),
            "geometry_path": str(geometry_path),
            "retained_count": int(row.retained_count),
            "retention_fraction": float(row.retention_fraction),
            "gate_threshold": None if pd.isna(row.gate_threshold) else float(row.gate_threshold),
            "surface_vertex_count": int(row.surface_vertex_count),
            "surface_face_count": int(row.surface_face_count),
            "pred_point_count_input": pred_count_input,
            "pred_point_count_evaluated": int(pred_points.shape[0]),
            "gt_point_count_evaluated": int(gt_points.shape[0]),
            "pred_sample_indices_count": int(pred_indices.shape[0]),
        }
        summary_rows.append({**base, **summary})
        threshold_rows.extend({**base, **metric} for metric in metrics)
    if not summary_rows:
        raise ValueError("No surface geometry rows were evaluated.")

    summary_df = pd.DataFrame(summary_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    summary_path = out_dir / f"{method_name}_surface_geometry_summary.csv"
    threshold_path = out_dir / f"{method_name}_surface_geometry_thresholds.csv"
    status_path = out_dir / f"{method_name}_surface_geometry_status.json"
    report_path = out_dir / f"{method_name}_surface_geometry_report.md"
    summary_df.to_csv(summary_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    status = {
        "schema_version": 1,
        "method": method_name,
        "manifest_path": str(manifest_file),
        "ground_truth_path": str(Path(ground_truth_path)),
        "output_dir": str(out_dir),
        "thresholds": list(thresholds),
        "max_pred_points": max_pred_points,
        "max_gt_points": max_gt_points,
        "gt_count_input": gt_count_input,
        "gt_count_evaluated": int(gt_points.shape[0]),
        "gt_sample_indices_count": int(gt_indices.shape[0]),
        "alignment_path": str(Path(alignment_path)) if alignment_path is not None else None,
        "invert_alignment": invert_alignment,
        "crop_path": str(Path(crop_path)) if crop_path is not None else None,
        "crop_enabled": crop is not None,
        "evaluated_surface_count": int(len(summary_rows)),
        "skipped_rows": skipped_rows,
        "summary_path": str(summary_path),
        "threshold_metrics_path": str(threshold_path),
        "report_path": str(report_path),
    }
    write_json(status_path, status)
    report_path.write_text(format_surface_geometry_report(status, summary_df, threshold_df), encoding="utf-8")
    return {"summary_path": summary_path, "threshold_metrics_path": threshold_path, "status_path": status_path, "report_path": report_path, "summary": summary_rows, "status": status}


def validate_extraction_config(**kwargs: Any) -> None:
    method_name = kwargs["method_name"]
    gate_thresholds = kwargs["gate_thresholds"]
    extraction_modes = kwargs["extraction_modes"]
    if not method_name:
        raise ValueError("method_name must be non-empty.")
    if not extraction_modes:
        raise ValueError("At least one extraction mode is required.")
    unknown = sorted(set(extraction_modes) - set(SURFACE_EXTRACTION_MODES))
    if unknown:
        raise ValueError(f"Unsupported extraction modes: {', '.join(unknown)}.")
    if any(not 0.0 <= threshold <= 1.0 for threshold in gate_thresholds):
        raise ValueError("gate_thresholds must be in [0, 1].")
    if not kwargs["include_baseline"] and not gate_thresholds:
        raise ValueError("Enable a baseline or at least one gate threshold.")
    for key in ("surfel_scale", "min_surfel_radius", "opacity_field_sigma_scale"):
        if kwargs[key] <= 0.0:
            raise ValueError(f"{key} must be positive.")
    max_radius = kwargs["max_surfel_radius"]
    if max_radius is not None and max_radius < kwargs["min_surfel_radius"]:
        raise ValueError("max_surfel_radius must be >= min_surfel_radius.")
    if kwargs["opacity_field_resolution"] < 2:
        raise ValueError("opacity_field_resolution must be at least 2.")
    if not 0.0 < kwargs["opacity_field_threshold"] < 1.0:
        raise ValueError("opacity_field_threshold must be in (0, 1).")
    if kwargs["opacity_field_margin_sigma"] < 0.0:
        raise ValueError("opacity_field_margin_sigma must be non-negative.")
    if kwargs["max_field_gaussians"] is not None and kwargs["max_field_gaussians"] <= 0:
        raise ValueError("max_field_gaussians must be positive when set.")
    if kwargs["field_query_chunk_size"] <= 0 or kwargs["field_gaussian_chunk_size"] <= 0:
        raise ValueError("field chunk sizes must be positive.")


def validate_surface_manifest(manifest: pd.DataFrame) -> None:
    required = {"variant", "variant_kind", "extraction_method", "representation", "geometry_path", "retained_count", "retention_fraction", "gate_threshold", "surface_vertex_count", "surface_face_count"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Surface manifest is missing columns: {', '.join(missing)}.")
    if manifest.empty:
        raise ValueError("Surface manifest is empty.")


def gaussian_selection_variants(vertex_count: int, *, gates: np.ndarray | None, gate_thresholds: tuple[float, ...], include_baseline: bool) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    if include_baseline:
        variants.append({"name": "baseline", "kind": "baseline", "mask": np.ones((vertex_count,), dtype=bool), "gate_threshold": None})
    if gates is None:
        if gate_thresholds and not include_baseline:
            raise ValueError("Gate thresholds require --gate-path when baseline extraction is disabled.")
        return variants
    for threshold in sorted(set(gate_thresholds)):
        variants.append({"name": f"gate_ge_{format_threshold_label(threshold)}", "kind": "gate_threshold", "mask": gates >= threshold, "gate_threshold": float(threshold)})
    return variants


def vertex_alpha_safe(vertices: np.ndarray, *, opacity_mode: str) -> np.ndarray:
    if "opacity" not in (vertices.dtype.names or ()):
        return np.ones((vertices.shape[0],), dtype=np.float64)
    return opacity_to_alpha(vertices["opacity"].astype(np.float64), opacity_mode=opacity_mode)


def build_oriented_surfel_mesh(vertices: np.ndarray, *, mask: np.ndarray, colors: np.ndarray, surfel_scale: float, min_radius: float, max_radius: float | None) -> SurfaceMesh:
    centers = vertex_centers(vertices)[mask]
    selected_colors = colors[mask]
    scales, axes = gaussian_scales_and_axes(vertices)
    selected_scales = scales[mask]
    selected_axes = axes[mask]
    mesh_vertices = np.empty((centers.shape[0] * 4, 3), dtype=np.float64)
    mesh_colors = np.empty((centers.shape[0] * 4, 3), dtype=np.float64)
    faces = np.empty((centers.shape[0] * 2, 3), dtype=np.int64)
    for index, center in enumerate(centers):
        scale = selected_scales[index]
        order = np.argsort(scale)
        tangent_a = selected_axes[index, :, order[1]]
        tangent_b = selected_axes[index, :, order[2]]
        radius_a = clip_radius(float(scale[order[1]]) * surfel_scale, min_radius=min_radius, max_radius=max_radius)
        radius_b = clip_radius(float(scale[order[2]]) * surfel_scale, min_radius=min_radius, max_radius=max_radius)
        base = index * 4
        mesh_vertices[base + 0] = center - radius_a * tangent_a - radius_b * tangent_b
        mesh_vertices[base + 1] = center + radius_a * tangent_a - radius_b * tangent_b
        mesh_vertices[base + 2] = center + radius_a * tangent_a + radius_b * tangent_b
        mesh_vertices[base + 3] = center - radius_a * tangent_a + radius_b * tangent_b
        mesh_colors[base : base + 4] = selected_colors[index]
        face_base = index * 2
        faces[face_base + 0] = [base, base + 1, base + 2]
        faces[face_base + 1] = [base, base + 2, base + 3]
    return SurfaceMesh(mesh_vertices, faces, mesh_colors)


def build_opacity_field_mesh(*, centers: np.ndarray, sigma: np.ndarray, alpha: np.ndarray, colors: np.ndarray, mask: np.ndarray, resolution: int, opacity_threshold: float, sigma_scale: float, margin_sigma: float, max_gaussians: int | None, query_chunk_size: int, gaussian_chunk_size: int, seed: int) -> SurfaceMesh:
    selected_centers = centers[mask]
    selected_sigma = np.maximum(sigma[mask].reshape(-1) * sigma_scale, 1e-9)
    selected_alpha = np.clip(alpha[mask].reshape(-1), 0.0, 1.0)
    selected_colors = colors[mask]
    if selected_centers.shape[0] == 0:
        return SurfaceMesh(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.float64))
    if max_gaussians is not None and selected_centers.shape[0] > max_gaussians:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(selected_centers.shape[0], size=max_gaussians, replace=False))
        selected_centers = selected_centers[keep]
        selected_sigma = selected_sigma[keep]
        selected_alpha = selected_alpha[keep]
        selected_colors = selected_colors[keep]
    lo, hi = opacity_field_bounds(selected_centers, selected_sigma, margin_sigma=margin_sigma)
    occupied = opacity_grid(selected_centers, selected_sigma, selected_alpha, lo=lo, hi=hi, resolution=resolution, opacity_threshold=opacity_threshold, query_chunk_size=query_chunk_size, gaussian_chunk_size=gaussian_chunk_size)
    mean_color = np.clip(selected_colors.mean(axis=0), 0.0, 1.0) if selected_colors.size else np.full((3,), 0.7, dtype=np.float64)
    return occupied_voxel_boundary_mesh(occupied, lo=lo, hi=hi, color=mean_color)


def gaussian_scales_and_axes(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = set(vertices.dtype.names or ())
    if {"scale_0", "scale_1", "scale_2"}.issubset(names):
        scales = np.exp(np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1).astype(np.float64))
    else:
        scales = np.repeat(vertex_sigma(vertices).reshape(-1, 1), 3, axis=1)
    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names):
        quat = np.stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=1).astype(np.float64)
        axes = quaternions_to_rotation_matrices(quat)
    else:
        axes = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], scales.shape[0], axis=0)
    return np.maximum(scales, 1e-12), axes


def quaternions_to_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    invalid = norms[:, 0] <= 1e-12
    norms[invalid] = 1.0
    q = q / norms
    q[invalid] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrices = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def clip_radius(radius: float, *, min_radius: float, max_radius: float | None) -> float:
    clipped = max(float(radius), min_radius)
    return min(clipped, max_radius) if max_radius is not None else clipped


def opacity_field_bounds(centers: np.ndarray, sigma: np.ndarray, *, margin_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    margin = float(np.max(sigma) * margin_sigma) if sigma.size else 0.0
    lo = centers.min(axis=0) - margin
    hi = centers.max(axis=0) + margin
    degenerate = (hi - lo) <= 1e-9
    lo[degenerate] -= 0.5
    hi[degenerate] += 0.5
    return lo.astype(np.float64), hi.astype(np.float64)


def opacity_grid(centers: np.ndarray, sigma: np.ndarray, alpha: np.ndarray, *, lo: np.ndarray, hi: np.ndarray, resolution: int, opacity_threshold: float, query_chunk_size: int, gaussian_chunk_size: int) -> np.ndarray:
    axes = [np.linspace(lo[dim], hi[dim], resolution, endpoint=False, dtype=np.float64) + (hi[dim] - lo[dim]) / (2.0 * resolution) for dim in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    occupied_flat = np.zeros((grid.shape[0],), dtype=bool)
    for start in range(0, grid.shape[0], query_chunk_size):
        query = grid[start : start + query_chunk_size]
        density = np.zeros((query.shape[0],), dtype=np.float64)
        for gaussian_start in range(0, centers.shape[0], gaussian_chunk_size):
            c = centers[gaussian_start : gaussian_start + gaussian_chunk_size]
            s = sigma[gaussian_start : gaussian_start + gaussian_chunk_size]
            a = alpha[gaussian_start : gaussian_start + gaussian_chunk_size]
            diff = query[:, None, :] - c[None, :, :]
            scaled_squared = np.sum(diff * diff, axis=2) / np.maximum(s[None, :] ** 2, 1e-18)
            density += np.sum(a[None, :] * np.exp(-0.5 * np.minimum(scaled_squared, 80.0)), axis=1)
        occupied_flat[start : start + query.shape[0]] = (1.0 - np.exp(-density)) >= opacity_threshold
    return occupied_flat.reshape((resolution, resolution, resolution))


def occupied_voxel_boundary_mesh(occupied: np.ndarray, *, lo: np.ndarray, hi: np.ndarray, color: np.ndarray) -> SurfaceMesh:
    resolution = int(occupied.shape[0])
    vertices: list[list[float]] = []
    colors: list[list[float]] = []
    faces: list[list[int]] = []
    vertex_index: dict[tuple[int, int, int], int] = {}
    steps = (hi - lo) / float(resolution)
    face_specs = [
        ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
        ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
        ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
        ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
        ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
        ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    ]

    def add_vertex(node: tuple[int, int, int]) -> int:
        if node in vertex_index:
            return vertex_index[node]
        vertex_index[node] = len(vertices)
        vertices.append((lo + steps * np.asarray(node, dtype=np.float64)).tolist())
        colors.append(color.tolist())
        return vertex_index[node]

    for i, j, k in np.argwhere(occupied):
        for direction, corners in face_specs:
            ni, nj, nk = int(i + direction[0]), int(j + direction[1]), int(k + direction[2])
            if 0 <= ni < resolution and 0 <= nj < resolution and 0 <= nk < resolution and bool(occupied[ni, nj, nk]):
                continue
            nodes = [add_vertex((int(i + dx), int(j + dy), int(k + dz))) for dx, dy, dz in corners]
            faces.append([nodes[0], nodes[1], nodes[2]])
            faces.append([nodes[0], nodes[2], nodes[3]])
    return SurfaceMesh(np.asarray(vertices, dtype=np.float64).reshape(-1, 3), np.asarray(faces, dtype=np.int64).reshape(-1, 3), np.asarray(colors, dtype=np.float64).reshape(-1, 3))


def write_surface_ply(path: str | Path, mesh: SurfaceMesh) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_colors = mesh.colors is not None and mesh.colors.shape[0] == mesh.vertices.shape[0]
    header = ["ply", "format ascii 1.0", f"element vertex {mesh.vertex_count}", "property float x", "property float y", "property float z"]
    if has_colors:
        header.extend(["property uchar red", "property uchar green", "property uchar blue"])
    header.extend([f"element face {mesh.face_count}", "property list uchar int vertex_indices", "end_header"])
    lines = [*header]
    rgb = np.clip(np.rint((mesh.colors if has_colors else np.full((mesh.vertex_count, 3), 0.7)) * 255.0), 0, 255).astype(np.uint8)
    for index, vertex in enumerate(mesh.vertices):
        if has_colors:
            lines.append(f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} {int(rgb[index, 0])} {int(rgb[index, 1])} {int(rgb[index, 2])}")
        else:
            lines.append(f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}")
    for face in mesh.faces:
        lines.append(f"3 {int(face[0])} {int(face[1])} {int(face[2])}")
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def read_surface_ply(path: str | Path) -> SurfaceMesh:
    ply_path = Path(path)
    lines = ply_path.read_text(encoding="ascii").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{ply_path} is not a PLY file.")
    vertex_count: int | None = None
    face_count = 0
    vertex_properties: list[str] = []
    in_vertex = False
    header_end: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "end_header":
            header_end = index
            break
        parts = stripped.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            vertex_count = int(parts[2])
            in_vertex = True
        elif len(parts) >= 3 and parts[0] == "element" and parts[1] == "face":
            face_count = int(parts[2])
            in_vertex = False
        elif in_vertex and len(parts) == 3 and parts[0] == "property":
            vertex_properties.append(parts[2])
    if header_end is None or vertex_count is None:
        raise ValueError(f"{ply_path} is missing a complete PLY header.")
    prop_index = {name: index for index, name in enumerate(vertex_properties)}
    for required in ("x", "y", "z"):
        if required not in prop_index:
            raise ValueError(f"{ply_path} is missing vertex property {required!r}.")
    vertex_lines = lines[header_end + 1 : header_end + 1 + vertex_count]
    split_vertices = [line.split() for line in vertex_lines]
    vertices = np.asarray([[float(parts[prop_index["x"]]), float(parts[prop_index["y"]]), float(parts[prop_index["z"]])] for parts in split_vertices], dtype=np.float64)
    colors = None
    if {"red", "green", "blue"}.issubset(prop_index):
        colors = np.asarray([[float(parts[prop_index["red"]]), float(parts[prop_index["green"]]), float(parts[prop_index["blue"]])] for parts in split_vertices], dtype=np.float64)
        colors = np.clip(colors / 255.0, 0.0, 1.0)
    faces: list[list[int]] = []
    face_start = header_end + 1 + vertex_count
    for line in lines[face_start : face_start + face_count]:
        parts = line.split()
        if not parts:
            continue
        count = int(parts[0])
        indices = [int(value) for value in parts[1 : 1 + count]]
        for offset in range(1, max(count - 1, 1)):
            if count >= 3:
                faces.append([indices[0], indices[offset], indices[offset + 1]])
    return SurfaceMesh(vertices, np.asarray(faces, dtype=np.int64).reshape(-1, 3), colors)


def surface_points_for_evaluation(surface: SurfaceMesh, *, max_points: int | None, seed: int) -> np.ndarray:
    if surface.vertices.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if surface.faces.shape[0] == 0:
        points, _ = deterministic_subsample(surface.vertices, max_points=max_points, seed=seed)
        return points
    sample_count = int(max_points) if max_points is not None and max_points > 0 else max(surface.faces.shape[0] * 2, surface.vertices.shape[0])
    return sample_mesh_surface(surface.vertices, surface.faces, sample_count=sample_count, seed=seed)


def sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, *, sample_count: int, seed: int) -> np.ndarray:
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    valid = areas > 1e-18
    if not np.any(valid):
        points, _ = deterministic_subsample(vertices, max_points=sample_count, seed=seed)
        return points
    valid_faces = faces[valid]
    probabilities = areas[valid] / areas[valid].sum()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid_faces.shape[0], size=sample_count, replace=True, p=probabilities)
    selected = vertices[valid_faces[chosen]]
    u = rng.random(sample_count)
    v = rng.random(sample_count)
    sqrt_u = np.sqrt(u)
    return (1.0 - sqrt_u)[:, None] * selected[:, 0] + (sqrt_u * (1.0 - v))[:, None] * selected[:, 1] + (sqrt_u * v)[:, None] * selected[:, 2]


def optional_stat(values: np.ndarray | None, name: str) -> float | None:
    if values is None or values.size == 0:
        return None
    if name == "min":
        return float(values.min())
    if name == "max":
        return float(values.max())
    if name == "mean":
        return float(values.mean())
    raise ValueError(f"Unsupported optional statistic {name!r}.")


def format_surface_extraction_report(status: dict[str, Any], manifest: pd.DataFrame) -> str:
    lines = [
        "# GPIS-Gated Gaussian Surface Extraction",
        "",
        f"- Method: `{status['method']}`",
        f"- Input PLY: `{status['input_ply_path']}`",
        f"- Gate NPZ: `{status.get('gate_path') or 'n/a'}`",
        f"- Input Gaussians: `{status['input_gaussian_count']}`",
        f"- Extracted surfaces: `{status['surface_count']}`",
        f"- Modes: `{', '.join(status['extraction_modes'])}`",
        "",
        "The `surfels` representation uses each Gaussian's shortest scale axis as a normal proxy, matching the 2DGS-style idea of thin oriented surface elements.",
        "The `opacity_field` representation evaluates a Gaussian alpha field and writes a voxel-boundary isosurface as a GOF-style opacity-surface proxy.",
    ]
    if not manifest.empty:
        lines.extend(["", "## Surface Variants", "", format_surface_manifest_table(manifest)])
    return "\n".join(lines) + "\n"


def format_surface_manifest_table(manifest: pd.DataFrame) -> str:
    lines = ["| variant | extraction | representation | retained | vertices | faces | gate_threshold | path |", "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in manifest.itertuples(index=False):
        threshold = "n/a" if pd.isna(row.gate_threshold) else f"{float(row.gate_threshold):.6g}"
        lines.append(f"| `{row.variant}` | `{row.extraction_method}` | `{row.representation}` | {row.retained_count} | {row.surface_vertex_count} | {row.surface_face_count} | {threshold} | `{row.geometry_path}` |")
    return "\n".join(lines)


def format_surface_geometry_report(status: dict[str, Any], summary: pd.DataFrame, thresholds: pd.DataFrame) -> str:
    lines = [
        "# GPIS-Gated Gaussian Surface Geometry Comparison",
        "",
        f"- Method: `{status['method']}`",
        f"- Manifest: `{status['manifest_path']}`",
        f"- Ground truth: `{status['ground_truth_path']}`",
        f"- Surfaces evaluated: `{status['evaluated_surface_count']}`",
        f"- Ground-truth points evaluated: `{status['gt_count_evaluated']}`",
        f"- Alignment: `{status.get('alignment_path') or 'n/a'}`",
        f"- Crop enabled: `{status['crop_enabled']}`",
        f"- Summary CSV: `{status['summary_path']}`",
        f"- Threshold CSV: `{status['threshold_metrics_path']}`",
    ]
    if status.get("skipped_rows"):
        lines.extend(["", f"Skipped rows: `{len(status['skipped_rows'])}`"])
    if not summary.empty:
        lines.extend(["", "## Summary", "", format_surface_summary_table(summary)])
    if not thresholds.empty:
        lines.extend(["", "## Threshold Metrics", "", format_surface_threshold_table(thresholds)])
    return "\n".join(lines) + "\n"


def format_surface_summary_table(summary: pd.DataFrame) -> str:
    lines = ["| variant | extraction | representation | pred_points | chamfer_l1 | accuracy_mean | completion_mean |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summary[["variant", "extraction_method", "representation", "pred_point_count_evaluated", "chamfer_l1", "accuracy_mean", "completion_mean"]].itertuples(index=False):
        lines.append(f"| `{row.variant}` | `{row.extraction_method}` | `{row.representation}` | {row.pred_point_count_evaluated} | {row.chamfer_l1:.6g} | {row.accuracy_mean:.6g} | {row.completion_mean:.6g} |")
    return "\n".join(lines)


def format_surface_threshold_table(thresholds: pd.DataFrame) -> str:
    lines = ["| variant | extraction | threshold | precision | recall | f_score |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in thresholds[["variant", "extraction_method", "threshold", "precision", "recall", "f_score"]].itertuples(index=False):
        lines.append(f"| `{row.variant}` | `{row.extraction_method}` | {row.threshold:.6g} | {row.precision:.6g} | {row.recall:.6g} | {row.f_score:.6g} |")
    return "\n".join(lines)
