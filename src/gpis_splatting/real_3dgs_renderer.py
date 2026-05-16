from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from gpis_splatting.external_3dgs import load_3dgs_ply, scale_opacity, vertex_centers, write_3dgs_ply
from gpis_splatting.gpis import load_model
from gpis_splatting.gsplat_fidelity_adapter import render_3dgs_ply_with_gsplat
from gpis_splatting.real_pipeline import load_external_gate
from gpis_splatting.real_scene import load_prepared_scene
from gpis_splatting.serialization import write_json
from gpis_splatting.splats import SplatCloud, gpis_gate_for_splats

REAL_RENDER_BACKENDS = ("proxy", "gsplat")


def render_real_3dgs_splats(
    *,
    scene_dir: str | Path,
    input_ply_path: str | Path,
    model_path: str | Path | None = None,
    gate_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    method_name: str | None = None,
    split: str = "test",
    use_gpis_gate: bool = True,
    epsilon: float = 0.09,
    gate_floor: float = 0.0,
    projection_convention: str = "auto",
    near_plane: float = 1e-2,
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gate_batch_size: int = 4096,
    max_frames: int | None = None,
    opacity_mode: str = "logit",
    gsplat_device: str = "auto",
    gsplat_dtype: str = "float32",
    gsplat_color_mode: str = "auto",
    gsplat_sh_degree: int | str | None = "auto",
    gsplat_strict_3dgs_fidelity: bool = True,
    gsplat_far_plane: float = 1.0e10,
    gsplat_radius_clip: float = 0.0,
    gsplat_eps2d: float = 0.3,
    gsplat_tile_size: int = 16,
    gsplat_packed: bool = True,
    gsplat_render_mode: str = "RGB",
    gsplat_rasterize_mode: str = "classic",
    gsplat_channel_chunk: int = 32,
    gsplat_max_gaussians: int | None = None,
    gsplat_rasterization_fn: Any | None = None,
) -> dict[str, Any]:
    """Render trained 3DGS Gaussians with gsplat and optional GPIS opacity gating.

    This is the photometric real-scene renderer. Unlike the legacy CPU proxy path, it renders from the
    original trained 3DGS PLY and preserves anisotropic scales, rotations, opacity, and SH color
    coefficients whenever they are present in the PLY.
    """
    if not 0.0 <= gate_floor <= 1.0:
        raise ValueError("gate_floor must be in [0, 1].")
    if opacity_mode not in {"logit", "linear"}:
        raise ValueError("opacity_mode must be 'logit' or 'linear'.")

    scene_root = Path(scene_dir)
    scene_meta, _, _ = load_prepared_scene(scene_root)
    resolved_ply = _resolve_scene_relative_path(scene_root, input_ply_path)
    resolved_method = method_name or (
        "real_external_gate_gsplat" if gate_path is not None else ("real_gpis_gate_gsplat" if use_gpis_gate else "real_3dgs_gsplat")
    )
    resolved_output = Path(output_dir) if output_dir is not None else scene_root / "renders" / resolved_method
    resolved_output.mkdir(parents=True, exist_ok=True)

    ply = load_3dgs_ply(resolved_ply)
    centers = vertex_centers(ply.vertices)
    splat_count = int(centers.shape[0])
    gate = torch.ones((splat_count,), dtype=torch.float64)
    resolved_model: Path | None = None
    resolved_gate_input: Path | None = None
    output_gate_path: Path | None = None
    gate_summary: dict[str, Any] = {
        "enabled": False,
        "source": "none",
        "epsilon": epsilon,
        "gate_floor": gate_floor,
        "min": 1.0,
        "max": 1.0,
        "mean": 1.0,
    }

    if gate_path is not None:
        resolved_gate_input = _resolve_scene_file(scene_root, gate_path, "real_splat_gates.npz")
        raw_gate_np = load_external_gate(resolved_gate_input, expected_count=splat_count)
        raw_gate = torch.from_numpy(raw_gate_np).to(dtype=torch.float64)
        gate = torch.clamp(gate_floor + (1.0 - gate_floor) * raw_gate, min=0.0, max=1.0)
        gate_np = gate.detach().cpu().numpy()
        output_gate_path = resolved_output / "real_splat_gates.npz"
        np.savez_compressed(
            output_gate_path,
            gate=gate_np,
            raw_gate=raw_gate_np,
            gate_floor=np.array(gate_floor),
            input_ply_path=np.array(str(resolved_ply)),
            source_gate_path=np.array(str(resolved_gate_input)),
        )
        gate_summary = _summarize_gate(source="external", raw_gate_np=raw_gate_np, gate_np=gate_np, epsilon=None, gate_floor=gate_floor)
        gate_summary["gate_path"] = str(resolved_gate_input)
    elif use_gpis_gate:
        resolved_model = _resolve_scene_file(scene_root, model_path, "real_gpis_model.npz")
        model, _ = load_model(str(resolved_model))
        centers_t = torch.from_numpy(centers).to(dtype=torch.float64)
        gate_splats = SplatCloud(
            centers=centers_t,
            colors=torch.zeros((splat_count, 3), dtype=torch.float64),
            tau=torch.ones((splat_count,), dtype=torch.float64),
            sigma=torch.ones((splat_count,), dtype=torch.float64),
            is_surface=torch.ones((splat_count,), dtype=torch.bool),
        )
        raw_gate = gpis_gate_for_splats(gate_splats, model, epsilon, batch_size=gate_batch_size)
        gate = torch.clamp(gate_floor + (1.0 - gate_floor) * raw_gate, min=0.0, max=1.0)
        raw_gate_np = raw_gate.detach().cpu().numpy()
        gate_np = gate.detach().cpu().numpy()
        output_gate_path = resolved_output / "real_splat_gates.npz"
        np.savez_compressed(
            output_gate_path,
            gate=gate_np,
            raw_gate=raw_gate_np,
            epsilon=np.array(epsilon),
            gate_floor=np.array(gate_floor),
            input_ply_path=np.array(str(resolved_ply)),
            model_path=np.array(str(resolved_model)),
        )
        gate_summary = _summarize_gate(source="gpis", raw_gate_np=raw_gate_np, gate_np=gate_np, epsilon=epsilon, gate_floor=gate_floor)
    else:
        gate_np = gate.detach().cpu().numpy()

    render_ply_path = resolved_ply
    gated_ply_path: Path | None = None
    if bool(gate_summary["enabled"]):
        gated_ply_path = resolved_output / "gated_point_cloud.ply"
        gated_vertices = scale_opacity(ply.vertices, gate_np, opacity_mode=opacity_mode, opacity_scale_floor=0.0)
        write_3dgs_ply(gated_ply_path, ply, vertices=gated_vertices)
        render_ply_path = gated_ply_path

    gsplat_result = render_3dgs_ply_with_gsplat(
        input_ply_path=render_ply_path,
        scene_dir=scene_root,
        output_dir=resolved_output,
        split=split,
        projection_convention=projection_convention,
        device=gsplat_device,
        dtype=gsplat_dtype,
        opacity_mode=opacity_mode,
        color_mode=gsplat_color_mode,
        sh_degree=gsplat_sh_degree,
        strict_3dgs_fidelity=gsplat_strict_3dgs_fidelity,
        background_mode="rgb",
        background_color=background_color,
        near_plane=near_plane,
        far_plane=gsplat_far_plane,
        radius_clip=gsplat_radius_clip,
        eps2d=gsplat_eps2d,
        tile_size=gsplat_tile_size,
        packed=gsplat_packed,
        render_mode=gsplat_render_mode,
        rasterize_mode=gsplat_rasterize_mode,
        channel_chunk=gsplat_channel_chunk,
        max_frames=max_frames,
        max_gaussians=gsplat_max_gaussians,
        rasterization_fn=gsplat_rasterization_fn,
    )
    gsplat_report = gsplat_result["report"]
    report_path = resolved_output / "real_render_report.json"
    report = {
        "schema_version": 2,
        "scene": scene_meta["scene"],
        "method": resolved_method,
        "renderer_backend": "gsplat",
        "backend_report_path": str(gsplat_result["report_path"]),
        "split": split,
        "scene_dir": str(scene_root),
        "input_ply_path": str(resolved_ply),
        "render_ply_path": str(render_ply_path),
        "gated_ply_path": str(gated_ply_path) if gated_ply_path is not None else None,
        "model_path": str(resolved_model) if resolved_model is not None else None,
        "input_gate_path": str(resolved_gate_input) if resolved_gate_input is not None else None,
        "output_dir": str(resolved_output),
        "gate_path": str(output_gate_path) if output_gate_path is not None else None,
        "use_gpis_gate": use_gpis_gate,
        "gate_summary": gate_summary,
        "gate_floor": gate_floor,
        "splat_count": splat_count,
        "source_gaussian_count": int(gsplat_report["source_gaussian_count"]),
        "rendered_gaussian_count": int(gsplat_report["rendered_gaussian_count"]),
        "image_count": int(gsplat_report["image_count"]),
        "projection_convention": gsplat_report["projection_convention"],
        "opacity_mode": opacity_mode,
        "color": gsplat_report.get("color"),
        "near_plane": near_plane,
        "background_color": list(background_color),
        "outputs": gsplat_report["outputs"],
    }
    write_json(report_path, report)
    return {"output_dir": resolved_output, "report_path": report_path, "gate_path": output_gate_path, "report": report}


def _summarize_gate(
    *,
    source: str,
    raw_gate_np: np.ndarray,
    gate_np: np.ndarray,
    epsilon: float | None,
    gate_floor: float,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "source": source,
        "epsilon": epsilon,
        "gate_floor": gate_floor,
        "raw_min": float(raw_gate_np.min()) if raw_gate_np.size else 0.0,
        "raw_max": float(raw_gate_np.max()) if raw_gate_np.size else 0.0,
        "raw_mean": float(raw_gate_np.mean()) if raw_gate_np.size else 0.0,
        "min": float(gate_np.min()) if gate_np.size else 0.0,
        "max": float(gate_np.max()) if gate_np.size else 0.0,
        "mean": float(gate_np.mean()) if gate_np.size else 0.0,
    }


def _resolve_scene_file(scene_root: Path, path: str | Path | None, default_name: str) -> Path:
    if path is None:
        return scene_root / default_name
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = scene_root / resolved
    return resolved


def _resolve_scene_relative_path(scene_root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = scene_root / resolved
    return resolved
