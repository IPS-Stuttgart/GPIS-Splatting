from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gpis_splatting.real_bootstrap import binary_ply_vertex_dtype, parse_ply_header, split_ply_header
from gpis_splatting.real_geometry import format_threshold_label
from gpis_splatting.real_splat_filtering import gate_multiplier
from gpis_splatting.serialization import write_json
from gpis_splatting.splats import SplatCloud, save_splats

SH_C0 = 0.28209479177387814
PLY_SCALAR_TYPES = {
    "char",
    "int8",
    "uchar",
    "uint8",
    "short",
    "int16",
    "ushort",
    "uint16",
    "int",
    "int32",
    "uint",
    "uint32",
    "float",
    "float32",
    "double",
    "float64",
}


@dataclass(frozen=True)
class GaussianPly:
    path: Path
    ply_format: str
    properties: tuple[tuple[str, str], ...]
    vertices: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])


def load_3dgs_ply(path: str | Path) -> GaussianPly:
    ply_path = Path(path)
    data = ply_path.read_bytes()
    header_text, body = split_ply_header(data, ply_path)
    header = parse_ply_header(header_text, ply_path)
    ensure_supported_3dgs_header(header_text, header, ply_path)
    properties = tuple(header["properties"])
    if header["format"] == "ascii":
        vertices = read_ascii_vertices(body, properties=properties, vertex_count=int(header["vertex_count"]), path=ply_path)
    elif header["format"] in {"binary_little_endian", "binary_big_endian"}:
        endian = "<" if header["format"] == "binary_little_endian" else ">"
        dtype = binary_ply_vertex_dtype(list(properties), endian=endian, path=ply_path)
        expected_bytes = int(header["vertex_count"]) * dtype.itemsize
        if len(body) < expected_bytes:
            raise ValueError(f"{ply_path} has fewer binary vertex bytes than declared.")
        vertices = np.frombuffer(body[:expected_bytes], dtype=dtype, count=int(header["vertex_count"])).copy()
    else:
        raise ValueError(f"{ply_path} uses unsupported PLY format {header['format']!r}.")
    validate_gaussian_vertices(vertices, path=ply_path)
    return GaussianPly(path=ply_path, ply_format=str(header["format"]), properties=properties, vertices=vertices)


def ensure_supported_3dgs_header(header_text: str, header: dict[str, Any], path: Path) -> None:
    for line in header_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("element ") and not stripped.startswith("element vertex"):
            raise ValueError(f"{path} contains non-vertex PLY elements, which are not preserved by this exporter.")
    for name, property_type in header["properties"]:
        if property_type not in PLY_SCALAR_TYPES:
            raise ValueError(f"{path} has unsupported scalar property {name!r} with type {property_type!r}.")


def read_ascii_vertices(body: bytes, *, properties: tuple[tuple[str, str], ...], vertex_count: int, path: Path) -> np.ndarray:
    lines = body.decode("ascii").splitlines()
    if len(lines) < vertex_count:
        raise ValueError(f"{path} has fewer vertex rows than declared.")
    dtype = binary_ply_vertex_dtype(list(properties), endian="<", path=path)
    rows = []
    for line in lines[:vertex_count]:
        parts = line.split()
        if len(parts) < len(properties):
            raise ValueError(f"{path} has a short vertex row: {line!r}")
        rows.append(tuple(cast_ascii_value(value, dtype.fields[name][0]) for value, (name, _type_name) in zip(parts, properties, strict=False)))
    return np.asarray(rows, dtype=dtype)


def cast_ascii_value(value: str, dtype: np.dtype) -> int | float:
    if dtype.kind in {"i", "u"}:
        return int(value)
    return float(value)


def validate_gaussian_vertices(vertices: np.ndarray, *, path: Path) -> None:
    names = set(vertices.dtype.names or ())
    missing = sorted({"x", "y", "z"} - names)
    if missing:
        raise ValueError(f"{path} is missing required 3DGS center properties: {', '.join(missing)}.")


def write_3dgs_ply(path: str | Path, ply: GaussianPly, *, vertices: np.ndarray | None = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ply.vertices if vertices is None else vertices
    header = "\n".join(
        [
            "ply",
            f"format {ply.ply_format} 1.0",
            f"element vertex {int(rows.shape[0])}",
            *[f"property {property_type} {name}" for name, property_type in ply.properties],
            "end_header",
            "",
        ]
    ).encode("ascii")
    if ply.ply_format == "ascii":
        output_path.write_text(header.decode("ascii") + format_ascii_vertices(rows, ply.properties), encoding="ascii")
    elif ply.ply_format in {"binary_little_endian", "binary_big_endian"}:
        output_path.write_bytes(header + rows.tobytes())
    else:
        raise ValueError(f"Unsupported output PLY format {ply.ply_format!r}.")


def format_ascii_vertices(vertices: np.ndarray, properties: tuple[tuple[str, str], ...]) -> str:
    lines = []
    for row in vertices:
        values = []
        for name, _property_type in properties:
            value = row[name].item()
            if isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            else:
                values.append(f"{float(value):.9g}")
        lines.append(" ".join(values))
    return "\n".join(lines) + ("\n" if lines else "")


def convert_3dgs_ply_to_splats(
    *,
    ply_path: str | Path,
    output_splats_path: str | Path,
    opacity_mode: str = "logit",
) -> dict[str, Any]:
    ply = load_3dgs_ply(ply_path)
    centers = vertex_centers(ply.vertices)
    colors = vertex_colors(ply.vertices)
    tau = vertex_alpha(ply.vertices, opacity_mode=opacity_mode)
    sigma = vertex_sigma(ply.vertices)
    splats = SplatCloud(
        centers=torch.from_numpy(centers).to(dtype=torch.float64),
        colors=torch.from_numpy(colors).to(dtype=torch.float64),
        tau=torch.from_numpy(tau).to(dtype=torch.float64),
        sigma=torch.from_numpy(sigma).to(dtype=torch.float64),
        is_surface=torch.ones((centers.shape[0],), dtype=torch.bool),
    )
    output_path = Path(output_splats_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_splats(str(output_path), splats)
    status = {
        "schema_version": 1,
        "input_ply_path": str(Path(ply_path)),
        "output_splats_path": str(output_path),
        "splat_count": int(centers.shape[0]),
        "opacity_mode": opacity_mode,
        "has_opacity": "opacity" in (ply.vertices.dtype.names or ()),
        "has_scale": any(name.startswith("scale_") for name in (ply.vertices.dtype.names or ())),
    }
    status_path = output_path.with_suffix(".json")
    write_json(status_path, status)
    return {"splats_path": output_path, "status_path": status_path, "status": status}


def export_3dgs_gpis_variants(
    *,
    input_ply_path: str | Path,
    gate_path: str | Path,
    output_dir: str | Path,
    method_name: str = "gpis_confidence_3dgs",
    iteration: int = 30000,
    gate_thresholds: tuple[float, ...] = (0.25, 0.5, 0.75),
    include_baseline: bool = True,
    write_scaled: bool = True,
    write_filtered: bool = True,
    opacity_mode: str = "logit",
    opacity_scale_floor: float = 0.0,
) -> dict[str, Any]:
    validate_export_config(
        iteration=iteration,
        gate_thresholds=gate_thresholds,
        include_baseline=include_baseline,
        write_scaled=write_scaled,
        write_filtered=write_filtered,
        opacity_mode=opacity_mode,
        opacity_scale_floor=opacity_scale_floor,
    )
    ply = load_3dgs_ply(input_ply_path)
    gates = load_gate_array(gate_path, expected_count=ply.vertex_count)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    if include_baseline:
        rows.append(write_variant(ply=ply, out_dir=out_dir, method_name=method_name, name="baseline", kind="baseline", iteration=iteration, gates=gates))
    if write_scaled:
        rows.append(
            write_variant(
                ply=ply,
                out_dir=out_dir,
                method_name=method_name,
                name="gate_scaled",
                kind="gate_scaled",
                iteration=iteration,
                gates=gates,
                opacity_mode=opacity_mode,
                opacity_scale_floor=opacity_scale_floor,
                opacity_scaled=True,
            )
        )
    if write_filtered:
        for threshold in sorted(set(gate_thresholds)):
            mask = gates >= threshold
            if not np.any(mask):
                continue
            label = format_threshold_label(threshold)
            rows.append(
                write_variant(
                    ply=ply,
                    out_dir=out_dir,
                    method_name=method_name,
                    name=f"gate_ge_{label}",
                    kind="gate_threshold",
                    iteration=iteration,
                    gates=gates,
                    mask=mask,
                    gate_threshold=threshold,
                )
            )

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / f"{method_name}_3dgs_variant_manifest.csv"
    status_path = out_dir / f"{method_name}_3dgs_variant_status.json"
    report_path = out_dir / f"{method_name}_3dgs_variant_report.md"
    manifest.to_csv(manifest_path, index=False)
    status = {
        "schema_version": 1,
        "method": method_name,
        "input_ply_path": str(Path(input_ply_path)),
        "gate_path": str(Path(gate_path)),
        "output_dir": str(out_dir),
        "iteration": iteration,
        "opacity_mode": opacity_mode,
        "opacity_scale_floor": opacity_scale_floor,
        "input_gaussian_count": ply.vertex_count,
        "gate_min": float(gates.min()) if gates.size else None,
        "gate_max": float(gates.max()) if gates.size else None,
        "gate_mean": float(gates.mean()) if gates.size else None,
        "variant_count": int(len(rows)),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
    }
    write_json(status_path, status)
    report_path.write_text(format_3dgs_variant_report(status, manifest), encoding="utf-8")
    return {"manifest_path": manifest_path, "status_path": status_path, "report_path": report_path, "manifest": manifest, "status": status}


def write_variant(
    *,
    ply: GaussianPly,
    out_dir: Path,
    method_name: str,
    name: str,
    kind: str,
    iteration: int,
    gates: np.ndarray,
    mask: np.ndarray | None = None,
    gate_threshold: float | None = None,
    opacity_mode: str = "logit",
    opacity_scale_floor: float = 0.0,
    opacity_scaled: bool = False,
) -> dict[str, Any]:
    selected = np.ones((ply.vertex_count,), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    vertices = ply.vertices[selected].copy()
    selected_gates = gates[selected]
    if opacity_scaled:
        vertices = scale_opacity(vertices, selected_gates, opacity_mode=opacity_mode, opacity_scale_floor=opacity_scale_floor)
    variant_dir = out_dir / f"{method_name}_{name}"
    ply_path = variant_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    write_3dgs_ply(ply_path, ply, vertices=vertices)
    return {
        "variant": name,
        "variant_kind": kind,
        "model_dir": str(variant_dir),
        "point_cloud_path": str(ply_path),
        "retained_count": int(vertices.shape[0]),
        "retention_fraction": float(vertices.shape[0] / ply.vertex_count),
        "gate_threshold": gate_threshold,
        "opacity_scaled": bool(opacity_scaled),
        "gate_min": float(selected_gates.min()) if selected_gates.size else None,
        "gate_max": float(selected_gates.max()) if selected_gates.size else None,
        "gate_mean": float(selected_gates.mean()) if selected_gates.size else None,
    }


def scale_opacity(vertices: np.ndarray, gates: np.ndarray, *, opacity_mode: str, opacity_scale_floor: float) -> np.ndarray:
    if "opacity" not in (vertices.dtype.names or ()):
        raise ValueError("Cannot write gate_scaled 3DGS variant because the PLY has no opacity property.")
    output = vertices.copy()
    alpha = opacity_to_alpha(output["opacity"].astype(np.float64), opacity_mode=opacity_mode)
    scaled_alpha = np.clip(alpha * gate_multiplier(gates, opacity_scale_floor), 1e-6, 1.0 - 1e-6)
    output["opacity"] = alpha_to_opacity(scaled_alpha, opacity_mode=opacity_mode).astype(output["opacity"].dtype)
    return output


def vertex_centers(vertices: np.ndarray) -> np.ndarray:
    return np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float64)


def vertex_colors(vertices: np.ndarray) -> np.ndarray:
    names = set(vertices.dtype.names or ())
    if {"red", "green", "blue"}.issubset(names):
        colors = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.float64) / 255.0
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        colors = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1).astype(np.float64)
        colors = colors * SH_C0 + 0.5
    else:
        colors = np.full((vertices.shape[0], 3), 0.7, dtype=np.float64)
    return np.clip(colors, 0.0, 1.0)


def vertex_alpha(vertices: np.ndarray, *, opacity_mode: str) -> np.ndarray:
    if "opacity" not in (vertices.dtype.names or ()):
        return np.ones((vertices.shape[0],), dtype=np.float64)
    return opacity_to_alpha(vertices["opacity"].astype(np.float64), opacity_mode=opacity_mode)


def vertex_sigma(vertices: np.ndarray) -> np.ndarray:
    names = set(vertices.dtype.names or ())
    scale_names = [name for name in ("scale_0", "scale_1", "scale_2") if name in names]
    if not scale_names:
        return np.full((vertices.shape[0],), 0.025, dtype=np.float64)
    scales = np.stack([vertices[name] for name in scale_names], axis=1).astype(np.float64)
    return np.mean(np.exp(scales), axis=1)


def opacity_to_alpha(values: np.ndarray, *, opacity_mode: str) -> np.ndarray:
    if opacity_mode == "logit":
        return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))
    if opacity_mode == "linear":
        return np.clip(values, 0.0, 1.0)
    raise ValueError("opacity_mode must be 'logit' or 'linear'.")


def alpha_to_opacity(alpha: np.ndarray, *, opacity_mode: str) -> np.ndarray:
    clipped = np.clip(alpha, 1e-6, 1.0 - 1e-6)
    if opacity_mode == "logit":
        return np.log(clipped / (1.0 - clipped))
    if opacity_mode == "linear":
        return clipped
    raise ValueError("opacity_mode must be 'logit' or 'linear'.")


def load_gate_array(path: str | Path, *, expected_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        key = "gate" if "gate" in data.files else "raw_gate"
        gates = np.asarray(data[key], dtype=np.float64).reshape(-1)
    if gates.shape[0] != expected_count:
        raise ValueError(f"Gate count {gates.shape[0]} does not match 3DGS Gaussian count {expected_count}.")
    return np.clip(gates, 0.0, 1.0)


def validate_export_config(
    *,
    iteration: int,
    gate_thresholds: tuple[float, ...],
    include_baseline: bool,
    write_scaled: bool,
    write_filtered: bool,
    opacity_mode: str,
    opacity_scale_floor: float,
) -> None:
    if iteration < 0:
        raise ValueError("iteration must be non-negative.")
    if not include_baseline and not write_scaled and not write_filtered:
        raise ValueError("Enable at least one output variant.")
    if any(not 0.0 <= threshold <= 1.0 for threshold in gate_thresholds):
        raise ValueError("gate_thresholds must be in [0, 1].")
    if opacity_mode not in {"logit", "linear"}:
        raise ValueError("opacity_mode must be 'logit' or 'linear'.")
    if not 0.0 <= opacity_scale_floor <= 1.0:
        raise ValueError("opacity_scale_floor must be in [0, 1].")


def format_3dgs_variant_report(status: dict[str, Any], manifest: pd.DataFrame) -> str:
    lines = [
        "# GPIS-Gated 3DGS Variants",
        "",
        f"- Method: `{status['method']}`",
        f"- Input PLY: `{status['input_ply_path']}`",
        f"- Gate NPZ: `{status['gate_path']}`",
        f"- Input Gaussians: `{status['input_gaussian_count']}`",
        f"- Variants: `{status['variant_count']}`",
        "",
        "Render each `model_dir` with the standard 3DGS renderer and evaluate predictions with `evaluate_real_renders` to obtain PSNR/SSIM/LPIPS.",
    ]
    if not manifest.empty:
        lines.extend(["", "## Variants", "", format_manifest_table(manifest)])
    return "\n".join(lines) + "\n"


def format_manifest_table(manifest: pd.DataFrame) -> str:
    lines = [
        "| variant | kind | retained | retention | opacity_scaled | gate_threshold | model_dir |",
        "| --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for row in manifest.itertuples(index=False):
        threshold = "n/a" if pd.isna(row.gate_threshold) else f"{row.gate_threshold:.6g}"
        lines.append(
            f"| `{row.variant}` | `{row.variant_kind}` | {row.retained_count} | {row.retention_fraction:.6g} | "
            f"`{row.opacity_scaled}` | {threshold} | `{row.model_dir}` |"
        )
    return "\n".join(lines)
