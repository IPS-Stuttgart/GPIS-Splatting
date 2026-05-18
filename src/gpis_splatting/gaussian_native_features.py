from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.external_3dgs import load_3dgs_ply, opacity_to_alpha, vertex_colors, vertex_sigma
from gpis_splatting.serialization import write_json

NATIVE_3DGS_FEATURE_COLUMNS = (
    "gaussian_alpha",
    "gaussian_opacity_logit",
    "gaussian_scale_mean",
    "gaussian_scale_min",
    "gaussian_scale_max",
    "gaussian_scale_std",
    "gaussian_scale_anisotropy",
    "gaussian_log_scale_mean",
    "gaussian_log_scale_range",
    "gaussian_log_volume",
    "gaussian_sh_dc_norm",
    "gaussian_sh_rest_energy",
    "gaussian_color_norm",
    "gaussian_rotation_norm_deviation",
)
NATIVE_3DGS_SCORE_COLUMNS = (
    "score_gaussian_alpha",
    "score_negative_gaussian_scale_mean",
    "score_negative_gaussian_scale_max",
    "score_negative_gaussian_anisotropy",
    "score_negative_gaussian_log_volume",
    "score_negative_gaussian_sh_rest_energy",
)
NATIVE_3DGS_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "gpis_3dgs_native": (
        "gaussian_alpha",
        "gaussian_scale_mean",
        "gaussian_scale_max",
        "gaussian_scale_anisotropy",
        "gaussian_log_volume",
        "gaussian_sh_dc_norm",
        "gaussian_sh_rest_energy",
    ),
    "gpis_3dgs_fused": (
        "abs_mu",
        "sigma",
        "grad_norm",
        "abs_signed_distance",
        "distance_std",
        "score_current_gate",
        "score_raw_surface_band",
        "gaussian_alpha",
        "gaussian_scale_mean",
        "gaussian_scale_max",
        "gaussian_scale_anisotropy",
        "gaussian_log_volume",
        "gaussian_sh_dc_norm",
        "gaussian_sh_rest_energy",
    ),
    "gpis_3dgs_scores": (
        "score_current_gate",
        "score_raw_surface_band",
        "score_variance_penalized_band",
        "score_variance_penalized_exp",
        "score_gaussian_alpha",
        "score_negative_gaussian_scale_mean",
        "score_negative_gaussian_scale_max",
        "score_negative_gaussian_anisotropy",
        "score_negative_gaussian_log_volume",
        "score_negative_gaussian_sh_rest_energy",
    ),
}


def register_3dgs_native_calibration_feature_sets() -> None:
    """Expose 3DGS-native feature sets to the existing calibration registry."""
    from gpis_splatting import real_score_calibration

    real_score_calibration.DEFAULT_FEATURE_SETS.update(NATIVE_3DGS_FEATURE_SETS)


def default_trained_3dgs_feature_sets() -> tuple[str, ...]:
    register_3dgs_native_calibration_feature_sets()
    return ("gpis_field", "gpis_with_gate", "gpis_scores", "gpis_3dgs_native", "gpis_3dgs_fused", "gpis_3dgs_scores")


def build_3dgs_native_feature_table(*, ply_path: str | Path, opacity_mode: str = "logit") -> pd.DataFrame:
    """Return per-Gaussian native 3DGS attributes that can help confidence calibration."""
    ply = load_3dgs_ply(ply_path)
    vertices = ply.vertices
    alpha = vertex_alpha_for_features(vertices, opacity_mode=opacity_mode)
    logit_opacity = opacity_logits_for_features(vertices, alpha=alpha, opacity_mode=opacity_mode)
    scale_stats = gaussian_scale_statistics(vertices)
    sh_dc_norm, sh_rest_energy = spherical_harmonic_energy_features(vertices)
    colors = vertex_colors(vertices)
    rotation_norm_deviation = gaussian_rotation_norm_deviation(vertices)

    table = pd.DataFrame(
        {
            "splat_index": np.arange(vertices.shape[0], dtype=np.int64),
            "gaussian_alpha": alpha,
            "gaussian_opacity_logit": logit_opacity,
            "gaussian_scale_mean": scale_stats["mean"],
            "gaussian_scale_min": scale_stats["min"],
            "gaussian_scale_max": scale_stats["max"],
            "gaussian_scale_std": scale_stats["std"],
            "gaussian_scale_anisotropy": scale_stats["anisotropy"],
            "gaussian_log_scale_mean": scale_stats["log_mean"],
            "gaussian_log_scale_range": scale_stats["log_range"],
            "gaussian_log_volume": scale_stats["log_volume"],
            "gaussian_sh_dc_norm": sh_dc_norm,
            "gaussian_sh_rest_energy": sh_rest_energy,
            "gaussian_color_norm": np.linalg.norm(colors, axis=1),
            "gaussian_rotation_norm_deviation": rotation_norm_deviation,
        }
    )
    add_native_quality_scores(table)
    return table


def append_3dgs_native_features_to_field_scores(
    *,
    field_scores_path: str | Path,
    ply_path: str | Path,
    output_path: str | Path | None = None,
    opacity_mode: str = "logit",
    overwrite: bool = True,
) -> dict[str, Any]:
    """Append 3DGS-native Gaussian features to an existing GPIS field-score CSV."""
    field_path = Path(field_scores_path)
    table = pd.read_csv(field_path)
    if "splat_index" not in table.columns:
        table["splat_index"] = np.arange(len(table), dtype=np.int64)
    native = build_3dgs_native_feature_table(ply_path=ply_path, opacity_mode=opacity_mode)
    native_by_index = native.set_index("splat_index", verify_integrity=True)
    splat_index = table["splat_index"].to_numpy(dtype=np.int64)
    missing = sorted(set(int(index) for index in splat_index) - set(int(index) for index in native_by_index.index.to_numpy(dtype=np.int64)))
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"Cannot append 3DGS-native features because splat_index values are outside the trained PLY: {preview}{suffix}")
    aligned = native_by_index.loc[splat_index].reset_index(drop=True)
    added_columns: list[str] = []
    skipped_columns: list[str] = []
    for column in [*NATIVE_3DGS_FEATURE_COLUMNS, *NATIVE_3DGS_SCORE_COLUMNS]:
        if column not in aligned.columns:
            continue
        if column in table.columns and not overwrite:
            skipped_columns.append(column)
            continue
        table[column] = aligned[column].to_numpy(dtype=np.float64)
        added_columns.append(column)

    out_path = Path(output_path) if output_path is not None else field_path.with_name(f"{field_path.stem}_with_3dgs_native.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    status_path = out_path.with_name(f"{out_path.stem}_native_feature_status.json")
    status = {
        "schema_version": 1,
        "field_scores_path": str(field_path),
        "output_field_scores_path": str(out_path),
        "trained_ply_path": str(Path(ply_path)),
        "opacity_mode": opacity_mode,
        "row_count": int(len(table)),
        "native_gaussian_count": int(len(native)),
        "added_columns": added_columns,
        "skipped_columns": skipped_columns,
        "registered_feature_sets": list(NATIVE_3DGS_FEATURE_SETS),
    }
    write_json(status_path, status)
    return {"field_scores_path": out_path, "status_path": status_path, "table": table, "native_features": native, "status": status}


def vertex_alpha_for_features(vertices: np.ndarray, *, opacity_mode: str) -> np.ndarray:
    if "opacity" not in (vertices.dtype.names or ()):  # trained 3DGS PLYs normally have this, but keep fallback robust.
        return np.ones((vertices.shape[0],), dtype=np.float64)
    return opacity_to_alpha(vertices["opacity"].astype(np.float64), opacity_mode=opacity_mode)


def opacity_logits_for_features(vertices: np.ndarray, *, alpha: np.ndarray, opacity_mode: str) -> np.ndarray:
    if "opacity" in (vertices.dtype.names or ()) and opacity_mode == "logit":
        return vertices["opacity"].astype(np.float64)
    clipped = np.clip(alpha, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def gaussian_scale_statistics(vertices: np.ndarray) -> dict[str, np.ndarray]:
    names = set(vertices.dtype.names or ())
    scale_names = [name for name in ("scale_0", "scale_1", "scale_2") if name in names]
    if scale_names:
        raw_scales = np.stack([vertices[name] for name in scale_names], axis=1).astype(np.float64)
        scales = np.exp(np.clip(raw_scales, -60.0, 60.0))
    else:
        sigma = vertex_sigma(vertices).astype(np.float64)
        raw_scales = np.log(np.clip(sigma, 1e-12, None))[:, None]
        scales = sigma[:, None]
    scale_min = scales.min(axis=1)
    scale_max = scales.max(axis=1)
    log_min = raw_scales.min(axis=1)
    log_max = raw_scales.max(axis=1)
    return {
        "mean": scales.mean(axis=1),
        "min": scale_min,
        "max": scale_max,
        "std": scales.std(axis=1),
        "anisotropy": scale_max / np.maximum(scale_min, 1e-12),
        "log_mean": raw_scales.mean(axis=1),
        "log_range": log_max - log_min,
        "log_volume": raw_scales.sum(axis=1),
    }


def spherical_harmonic_energy_features(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = vertices.dtype.names or ()
    dc_names = [name for name in ("f_dc_0", "f_dc_1", "f_dc_2") if name in names]
    if dc_names:
        sh_dc = np.stack([vertices[name] for name in dc_names], axis=1).astype(np.float64)
        dc_norm = np.linalg.norm(sh_dc, axis=1)
    else:
        dc_norm = np.linalg.norm(vertex_colors(vertices), axis=1)
    rest_names = sorted(name for name in names if name.startswith("f_rest_"))
    if rest_names:
        rest = np.stack([vertices[name] for name in rest_names], axis=1).astype(np.float64)
        rest_energy = np.linalg.norm(rest, axis=1)
    else:
        rest_energy = np.zeros((vertices.shape[0],), dtype=np.float64)
    return dc_norm, rest_energy


def gaussian_rotation_norm_deviation(vertices: np.ndarray) -> np.ndarray:
    names = set(vertices.dtype.names or ())
    rot_names = [name for name in ("rot_0", "rot_1", "rot_2", "rot_3") if name in names]
    if len(rot_names) != 4:
        return np.zeros((vertices.shape[0],), dtype=np.float64)
    rotations = np.stack([vertices[name] for name in rot_names], axis=1).astype(np.float64)
    return np.abs(np.linalg.norm(rotations, axis=1) - 1.0)


def add_native_quality_scores(table: pd.DataFrame) -> None:
    table["score_gaussian_alpha"] = np.clip(table["gaussian_alpha"].to_numpy(dtype=np.float64), 0.0, 1.0)
    table["score_negative_gaussian_scale_mean"] = -table["gaussian_scale_mean"].to_numpy(dtype=np.float64)
    table["score_negative_gaussian_scale_max"] = -table["gaussian_scale_max"].to_numpy(dtype=np.float64)
    table["score_negative_gaussian_anisotropy"] = -table["gaussian_scale_anisotropy"].to_numpy(dtype=np.float64)
    table["score_negative_gaussian_log_volume"] = -table["gaussian_log_volume"].to_numpy(dtype=np.float64)
    table["score_negative_gaussian_sh_rest_energy"] = -table["gaussian_sh_rest_energy"].to_numpy(dtype=np.float64)
