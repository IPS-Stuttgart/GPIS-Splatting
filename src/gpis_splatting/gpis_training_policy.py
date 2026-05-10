from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.external_3dgs import load_3dgs_ply, opacity_to_alpha
from gpis_splatting.serialization import write_json


def export_training_policy(
    *,
    input_ply_path: str | Path,
    gate_path: str | Path,
    output_dir: str | Path,
    field_scores_path: str | Path | None = None,
    candidate_metadata_path: str | Path | None = None,
    method_name: str = "gpis_training_policy",
    init_threshold: float = 0.75,
    densify_threshold: float = 0.55,
    prune_threshold: float = 0.25,
    uncertainty_quantile: float = 0.8,
    opacity_strength: float = 1.0,
    opacity_floor: float = 0.05,
    max_init_points: int | None = None,
    opacity_mode: str = "logit",
) -> dict[str, Any]:
    ply = load_3dgs_ply(input_ply_path)
    gate = load_gate(gate_path)
    field = None if field_scores_path is None else pd.read_csv(field_scores_path)
    meta = None if candidate_metadata_path is None else pd.read_csv(candidate_metadata_path)
    if field is not None and len(field) != gate.size:
        raise ValueError("field_scores_path row count must match gate count")
    if meta is not None and len(meta) != gate.size:
        raise ValueError("candidate_metadata_path row count must match gate count")
    unc = signal(field, ("distance_std", "sigma", "variance", "uncertainty"), np.ones_like(gate))
    ev = signal(field, ("score_raw_surface_band", "score_variance_penalized_band", "score_exp_neg_abs_distance", "evidence"), gate)
    pgate, punc, pev = per_gaussian(ply.vertex_count, gate, unc, ev, meta)
    plaus = np.clip(pgate * (0.5 + 0.5 * pev), 0.0, 1.0)
    dens_w = np.clip(pgate * (0.25 + 0.75 * punc) * (0.5 + 0.5 * pev), 0.0, 1.0)
    dens_m = (pgate >= densify_threshold) & (punc >= q(punc, uncertainty_quantile))
    prune_w = np.clip((1.0 - pgate) * (1.0 - plaus), 0.0, 1.0)
    prune_m = pgate <= prune_threshold
    alpha = alpha_values(ply.vertices, opacity_mode)
    target = np.clip(alpha * (opacity_floor + (1.0 - opacity_floor) * plaus), 1e-6, 1.0 - 1e-6)
    op_w = np.clip(opacity_strength * (1.0 - plaus), 0.0, None)
    init = init_points(ply.vertices, gate, unc, ev, meta, init_threshold, max_init_points)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prior_path = out / f"{method_name}.npz"
    np.savez_compressed(
        prior_path,
        gate=pgate.astype(np.float32),
        densify_weight=dens_w.astype(np.float32),
        densify_candidate_mask=dens_m.astype(bool),
        prune_weight=prune_w.astype(np.float32),
        prune_candidate_mask=prune_m.astype(bool),
        opacity_target_alpha=target.astype(np.float32),
        opacity_regularization_weight=op_w.astype(np.float32),
        initialization_points=init["points"].astype(np.float32),
        initialization_confidence=init["confidence"].astype(np.float32),
        initialization_weight=init["weight"].astype(np.float32),
        initialization_source_splat_index=init["source"].astype(np.int64),
        initialization_candidate_index=init["candidate"].astype(np.int64),
    )
    pd.DataFrame({"x": init["points"][:, 0] if init["points"].size else [], "y": init["points"][:, 1] if init["points"].size else [], "z": init["points"][:, 2] if init["points"].size else [], "confidence": init["confidence"], "weight": init["weight"], "source_splat_index": init["source"], "candidate_index": init["candidate"]}).to_csv(out / f"{method_name}_initialization.csv", index=False)
    status = {"training_prior_path": str(prior_path), "gaussian_count": int(ply.vertex_count), "gate_count": int(gate.size), "initialization_candidate_count": int(init["points"].shape[0]), "densify_candidate_count": int(dens_m.sum()), "prune_candidate_count": int(prune_m.sum())}
    status_path = out / f"{method_name}_status.json"
    write_json(status_path, status)
    return {"prior_path": prior_path, "status_path": status_path, "status": status}


def load_gate(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as d:
        key = next((k for k in ("gate", "confidence", "calibrated_confidence", "raw_gate") if k in d.files), d.files[0])
        return np.clip(np.nan_to_num(np.asarray(d[key], dtype=np.float64).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def signal(table: pd.DataFrame | None, names: tuple[str, ...], default: np.ndarray) -> np.ndarray:
    if table is None:
        return norm(default)
    for name in names:
        if name in table.columns:
            return norm(table[name].to_numpy(dtype=np.float64))
    return norm(default)


def per_gaussian(n: int, gate: np.ndarray, unc: np.ndarray, ev: np.ndarray, meta: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if gate.size == n:
        return gate, unc, ev
    if meta is None or "source_splat_index" not in meta.columns:
        raise ValueError("candidate-level gates require candidate metadata with source_splat_index")
    src = meta["source_splat_index"].to_numpy(dtype=np.int64)
    return agg(n, src, gate), agg(n, src, unc), agg(n, src, ev)


def agg(n: int, src: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    for i, v in zip(src, values):
        if 0 <= int(i) < n:
            out[int(i)] = max(out[int(i)], float(v))
    return out


def init_points(vertices: np.ndarray, gate: np.ndarray, unc: np.ndarray, ev: np.ndarray, meta: pd.DataFrame | None, threshold: float, max_count: int | None) -> dict[str, np.ndarray]:
    if meta is None:
        pts = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float64)
        src = np.arange(pts.shape[0], dtype=np.int64)
        cand = np.arange(pts.shape[0], dtype=np.int64)
    else:
        cols = point_cols(meta)
        pts = meta[list(cols)].to_numpy(dtype=np.float64)
        src = meta["source_splat_index"].to_numpy(dtype=np.int64) if "source_splat_index" in meta.columns else np.full(pts.shape[0], -1, dtype=np.int64)
        cand = meta["candidate_index"].to_numpy(dtype=np.int64) if "candidate_index" in meta.columns else np.arange(pts.shape[0], dtype=np.int64)
    w = np.clip(gate * (0.5 + 0.5 * ev) * (0.25 + 0.75 * unc), 0.0, 1.0)
    keep = np.flatnonzero(gate >= threshold)
    if keep.size == 0 and w.size:
        keep = np.asarray([int(np.argmax(w))])
    keep = keep[np.argsort(-w[keep], kind="mergesort")]
    if max_count is not None:
        keep = keep[:max(0, int(max_count))]
    return {"points": pts[keep], "confidence": gate[keep], "weight": w[keep], "source": src[keep], "candidate": cand[keep]}


def point_cols(table: pd.DataFrame) -> tuple[str, str, str]:
    for cols in (("x", "y", "z"), ("query_x", "query_y", "query_z"), ("eval_x", "eval_y", "eval_z")):
        if set(cols).issubset(table.columns):
            return cols
    raise ValueError("candidate metadata needs x/y/z columns")


def alpha_values(vertices: np.ndarray, mode: str) -> np.ndarray:
    if "opacity" not in (vertices.dtype.names or ()): return np.ones(vertices.shape[0], dtype=np.float64)
    return opacity_to_alpha(vertices["opacity"].astype(np.float64), opacity_mode=mode)


def norm(x: np.ndarray) -> np.ndarray:
    y = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    if y.size == 0 or y.max() <= y.min() + 1e-12:
        return np.zeros_like(y)
    return np.clip((y - y.min()) / (y.max() - y.min()), 0.0, 1.0)


def q(x: np.ndarray, p: float) -> float:
    return float(np.quantile(x, np.clip(p, 0.0, 1.0))) if x.size else 0.0
