from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.external_3dgs import load_3dgs_ply, opacity_to_alpha, write_3dgs_ply
from gpis_splatting.serialization import write_json

UNCERTAINTY_COLUMNS = ("gpis_uncertainty", "uncertainty", "distance_std", "sigma", "variance", "field_variance", "posterior_variance")
EVIDENCE_COLUMNS = ("gpis_evidence", "evidence", "score_gpis_surface_likelihood", "score_raw_surface_band", "score_variance_penalized_band")


@dataclass(frozen=True)
class GpisTrainingPriorConfig:
    initialization_confidence_threshold: float = 0.75
    densify_confidence_threshold: float = 0.55
    prune_confidence_threshold: float = 0.25
    high_uncertainty_quantile: float = 0.8
    opacity_regularization_strength: float = 1.0
    opacity_scale_floor: float = 0.05
    clone_top_count: int | None = None
    opacity_mode: str = "logit"


def export_gpis_training_prior(
    *,
    input_ply_path: str | Path,
    gate_path: str | Path,
    output_dir: str | Path,
    field_scores_path: str | Path | None = None,
    method_name: str = "gpis_confidence_training_prior",
    config: GpisTrainingPriorConfig | None = None,
) -> dict[str, Any]:
    cfg = config or GpisTrainingPriorConfig()
    ply = load_3dgs_ply(input_ply_path)
    gate = load_gate_array(gate_path, expected_count=ply.vertex_count)
    field_scores = load_field_scores(field_scores_path, expected_count=ply.vertex_count)
    uncertainty = resolve_signal(field_scores, UNCERTAINTY_COLUMNS, default=np.ones_like(gate), transform="normalize")
    evidence = resolve_signal(field_scores, EVIDENCE_COLUMNS, default=gate, transform="normalize")
    current_alpha = vertex_alpha(ply.vertices, opacity_mode=cfg.opacity_mode)

    plausibility = np.clip(gate * (0.5 + 0.5 * evidence), 0.0, 1.0)
    high_uncertainty = uncertainty >= quantile_safe(uncertainty, cfg.high_uncertainty_quantile)
    densify_weight = np.clip(gate * (0.25 + 0.75 * uncertainty) * (0.5 + 0.5 * evidence), 0.0, 1.0)
    densify_candidate_mask = (gate >= cfg.densify_confidence_threshold) & high_uncertainty
    prune_weight = np.clip((1.0 - gate) * (1.0 - plausibility), 0.0, 1.0)
    prune_candidate_mask = gate <= cfg.prune_confidence_threshold
    opacity_confidence_scale = np.clip(cfg.opacity_scale_floor + (1.0 - cfg.opacity_scale_floor) * plausibility, cfg.opacity_scale_floor, 1.0)
    opacity_target_alpha = np.clip(current_alpha * opacity_confidence_scale, 1e-6, 1.0 - 1e-6)
    opacity_regularization_weight = np.clip(cfg.opacity_regularization_strength * (1.0 - plausibility), 0.0, None)
    init_mask = gate >= cfg.initialization_confidence_threshold

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prior_path = out_dir / f"{method_name}_training_prior.npz"
    np.savez_compressed(
        prior_path,
        gate=gate.astype(np.float32),
        uncertainty=uncertainty.astype(np.float32),
        evidence=evidence.astype(np.float32),
        geometry_plausibility=plausibility.astype(np.float32),
        densify_weight=densify_weight.astype(np.float32),
        densify_candidate_mask=densify_candidate_mask.astype(bool),
        prune_weight=prune_weight.astype(np.float32),
        prune_candidate_mask=prune_candidate_mask.astype(bool),
        opacity_regularization_weight=opacity_regularization_weight.astype(np.float32),
        opacity_target_alpha=opacity_target_alpha.astype(np.float32),
        opacity_current_alpha=current_alpha.astype(np.float32),
        opacity_confidence_scale=opacity_confidence_scale.astype(np.float32),
        initialization_candidate_mask=init_mask.astype(bool),
    )

    seed_ply_path = out_dir / f"{method_name}_initialization_seed.ply"
    seed_vertices = build_initialization_vertices(ply.vertices, gate=gate, densify_weight=densify_weight, init_mask=init_mask, clone_top_count=cfg.clone_top_count)
    write_3dgs_ply(seed_ply_path, ply, vertices=seed_vertices)

    hooks_path = out_dir / f"{method_name}_trainer_hooks.md"
    hooks_path.write_text(format_trainer_hooks(method_name, prior_path, seed_ply_path, cfg), encoding="utf-8")
    status = {
        "schema_version": 1,
        "method": method_name,
        "input_ply_path": str(Path(input_ply_path)),
        "gate_path": str(Path(gate_path)),
        "field_scores_path": None if field_scores_path is None else str(Path(field_scores_path)),
        "output_dir": str(out_dir),
        "training_prior_path": str(prior_path),
        "initialization_seed_ply_path": str(seed_ply_path),
        "trainer_hooks_path": str(hooks_path),
        "gaussian_count": int(ply.vertex_count),
        "seed_count": int(seed_vertices.shape[0]),
        "initialization_candidate_count": int(init_mask.sum()),
        "densify_candidate_count": int(densify_candidate_mask.sum()),
        "prune_candidate_count": int(prune_candidate_mask.sum()),
        "config": asdict(cfg),
    }
    status_path = out_dir / f"{method_name}_training_prior_status.json"
    write_json(status_path, status)
    return {"prior_path": prior_path, "initialization_seed_ply_path": seed_ply_path, "trainer_hooks_path": hooks_path, "status_path": status_path, "status": status}


def load_gate_array(path: str | Path, *, expected_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        for key in ("gate", "confidence", "calibrated_confidence", "raw_gate"):
            if key in data.files:
                gate = np.asarray(data[key], dtype=np.float64).reshape(-1)
                break
        else:
            gate = np.asarray(data[data.files[0]], dtype=np.float64).reshape(-1)
    if gate.shape[0] != expected_count:
        raise ValueError(f"Gate count {gate.shape[0]} does not match Gaussian count {expected_count}.")
    return np.clip(np.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def load_field_scores(path: str | Path | None, *, expected_count: int) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(index=np.arange(expected_count))
    table = pd.read_csv(path)
    if len(table) != expected_count:
        raise ValueError(f"Field-score row count {len(table)} does not match Gaussian count {expected_count}.")
    return table


def resolve_signal(table: pd.DataFrame, columns: tuple[str, ...], *, default: np.ndarray, transform: str) -> np.ndarray:
    for column in columns:
        if column in table.columns:
            values = np.asarray(table[column], dtype=np.float64).reshape(-1)
            break
    else:
        values = np.asarray(default, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(values, nan=np.nanmedian(values) if values.size else 0.0, posinf=np.nanmax(values) if values.size else 1.0, neginf=np.nanmin(values) if values.size else 0.0)
    if transform == "normalize":
        values = minmax01(values)
    return np.clip(values, 0.0, 1.0)


def minmax01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.min(values)) if values.size else 0.0
    hi = float(np.max(values)) if values.size else 1.0
    if hi <= lo + 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def quantile_safe(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, np.clip(q, 0.0, 1.0))) if values.size else 0.0


def vertex_alpha(vertices: np.ndarray, *, opacity_mode: str) -> np.ndarray:
    if "opacity" not in (vertices.dtype.names or ()): 
        return np.ones((vertices.shape[0],), dtype=np.float64)
    return opacity_to_alpha(vertices["opacity"].astype(np.float64), opacity_mode=opacity_mode)


def build_initialization_vertices(vertices: np.ndarray, *, gate: np.ndarray, densify_weight: np.ndarray, init_mask: np.ndarray, clone_top_count: int | None) -> np.ndarray:
    selected = vertices[init_mask].copy()
    if clone_top_count is None or clone_top_count <= 0:
        return selected
    count = min(int(clone_top_count), int(vertices.shape[0]))
    order = np.argsort(-(gate * densify_weight))[:count]
    clones = vertices[order].copy()
    if clones.size and "opacity" in (clones.dtype.names or ()):
        clones["opacity"] = np.asarray(clones["opacity"], dtype=clones["opacity"].dtype)
    if selected.size == 0:
        return clones
    return np.concatenate([selected, clones], axis=0)


def format_trainer_hooks(method_name: str, prior_path: Path, seed_ply_path: Path, cfg: GpisTrainingPriorConfig) -> str:
    return "\n".join(
        [
            f"# {method_name} training prior",
            "",
            "This artifact turns calibrated GPIS confidence into training-time 3DGS priors, not only final filtering.",
            "",
            "## Initialization",
            f"- Seed or warm-start from `{seed_ply_path}`.",
            f"- Initialization threshold: `{cfg.initialization_confidence_threshold}`.",
            "",
            "## Densification / pruning",
            f"- Load `{prior_path}` and use `densify_weight` / `densify_candidate_mask` to promote high-confidence, high-uncertainty regions.",
            "- Use `prune_weight` / `prune_candidate_mask` to suppress low-confidence floaters.",
            "",
            "## Opacity regularization",
            "- Add a soft penalty instead of hard-deleting low-confidence Gaussians too early:",
            "",
            "```text",
            "loss += lambda_gpis_opacity * mean(opacity_regularization_weight * (alpha - opacity_target_alpha)^2)",
            "```",
            "",
            "The `.npz` also contains `geometry_plausibility`, `evidence`, `uncertainty`, and the original `gate` for custom trainer policies.",
            "",
        ]
    )
