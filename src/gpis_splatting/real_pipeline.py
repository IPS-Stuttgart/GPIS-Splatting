from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from gpis_splatting.gpis import fit_dense_gpis, load_model, save_model
from gpis_splatting.real_scene import load_prepared_scene
from gpis_splatting.renderer import save_image
from gpis_splatting.serialization import write_json
from gpis_splatting.splats import SplatCloud, gpis_gate_for_splats, load_splats

PROJECTION_CONVENTIONS = ("auto", "opencv", "opengl")


def fit_real_gpis(
    *,
    scene_dir: str | Path,
    samples_path: str | Path | None = None,
    output_model: str | Path | None = None,
    lengthscale: float = 0.25,
    variance: float = 1.0,
    noise_std: float = 0.05,
    jitter: float = 1e-6,
    max_train_points: int | None = 1200,
    seed: int = 11,
    use_observation_noise: bool = True,
) -> dict[str, Any]:
    scene_root = Path(scene_dir)
    scene_meta, _, _ = load_prepared_scene(scene_root)
    resolved_samples = _resolve_scene_file(scene_root, samples_path, "real_samples.npz")
    resolved_model = _resolve_scene_file(scene_root, output_model, "real_gpis_model.npz")

    with np.load(resolved_samples, allow_pickle=False) as samples:
        points = np.asarray(samples["points"], dtype=np.float64)
        sdf = np.asarray(samples["sdf"], dtype=np.float64).reshape(-1)
        observation_noise_std = np.asarray(samples["observation_noise_std"], dtype=np.float64).reshape(-1) if "observation_noise_std" in samples.files else None
        sample_type = np.asarray(samples["sample_type"], dtype=np.int64).reshape(-1) if "sample_type" in samples.files else None

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("real_samples.npz must contain points with shape (N, 3).")
    if sdf.shape[0] != points.shape[0]:
        raise ValueError("real_samples.npz must contain one sdf value per point.")
    if observation_noise_std is not None and observation_noise_std.shape[0] != points.shape[0]:
        raise ValueError("observation_noise_std must contain one value per point.")

    indices = select_training_indices(sample_type=sample_type, sample_count=points.shape[0], max_train_points=max_train_points, seed=seed)
    selected_points = points[indices]
    selected_sdf = sdf[indices]
    selected_noise = observation_noise_std[indices] if observation_noise_std is not None and use_observation_noise else None

    model = fit_dense_gpis(
        torch.from_numpy(selected_points),
        torch.from_numpy(selected_sdf),
        lengthscale=lengthscale,
        variance=variance,
        noise_std=noise_std,
        observation_noise_std=torch.from_numpy(selected_noise) if selected_noise is not None else None,
        jitter=jitter,
    )
    save_model(
        str(resolved_model),
        model,
        metadata={
            "scene": scene_meta["scene"],
            "source": "real_samples",
            "samples_path": str(resolved_samples),
            "available_sample_count": int(points.shape[0]),
            "train_sample_count": int(selected_points.shape[0]),
            "lengthscale": float(lengthscale),
            "variance": float(variance),
            "noise_std": float(noise_std),
            "uses_observation_noise": bool(selected_noise is not None),
        },
    )

    report_path = resolved_model.with_name(f"{resolved_model.stem}_fit_report.json")
    report = {
        "schema_version": 1,
        "scene": scene_meta["scene"],
        "samples_path": str(resolved_samples),
        "model_path": str(resolved_model),
        "available_sample_count": int(points.shape[0]),
        "train_sample_count": int(selected_points.shape[0]),
        "max_train_points": max_train_points,
        "seed": seed,
        "lengthscale": lengthscale,
        "variance": variance,
        "noise_std": noise_std,
        "jitter": jitter,
        "uses_observation_noise": bool(selected_noise is not None),
        "sample_type_counts": sample_type_counts(sample_type[indices] if sample_type is not None else None),
        "sdf_min": float(selected_sdf.min()),
        "sdf_max": float(selected_sdf.max()),
        "sdf_mean": float(selected_sdf.mean()),
    }
    write_json(report_path, report)
    return {
        "model_path": resolved_model,
        "report_path": report_path,
        "report": report,
    }


def render_real_splats(
    *,
    scene_dir: str | Path,
    splats_path: str | Path | None = None,
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    method_name: str | None = None,
    split: str = "test",
    use_gpis_gate: bool = True,
    epsilon: float = 0.09,
    projection_convention: str = "auto",
    near_plane: float = 1e-4,
    kernel_radius: float = 3.0,
    min_sigma_px: float = 0.6,
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gate_batch_size: int = 4096,
    max_frames: int | None = None,
) -> dict[str, Any]:
    if projection_convention not in PROJECTION_CONVENTIONS:
        raise ValueError(f"Unsupported projection convention {projection_convention!r}. Expected one of {', '.join(PROJECTION_CONVENTIONS)}.")

    scene_root = Path(scene_dir)
    scene_meta, frames, splits = load_prepared_scene(scene_root)
    resolved_splats = _resolve_scene_file(scene_root, splats_path, "real_splats.npz")
    resolved_method = method_name or ("real_gpis_gate" if use_gpis_gate else "real_splats_plain")
    resolved_output = Path(output_dir) if output_dir is not None else scene_root / "renders" / resolved_method
    resolved_output.mkdir(parents=True, exist_ok=True)

    splats = load_splats(str(resolved_splats))
    gate = torch.ones_like(splats.tau)
    resolved_model: Path | None = None
    gate_path: Path | None = None
    gate_summary = {
        "enabled": False,
        "epsilon": epsilon,
        "min": 1.0,
        "max": 1.0,
        "mean": 1.0,
    }
    if use_gpis_gate:
        resolved_model = _resolve_scene_file(scene_root, model_path, "real_gpis_model.npz")
        model, _ = load_model(str(resolved_model))
        gate = gpis_gate_for_splats(splats, model, epsilon, batch_size=gate_batch_size)
        gate_np = gate.detach().cpu().numpy()
        gate_path = resolved_output / "real_splat_gates.npz"
        np.savez_compressed(gate_path, gate=gate_np, epsilon=np.array(epsilon), splats_path=np.array(str(resolved_splats)), model_path=np.array(str(resolved_model)))
        gate_summary = {
            "enabled": True,
            "epsilon": epsilon,
            "min": float(gate_np.min()) if gate_np.size else 0.0,
            "max": float(gate_np.max()) if gate_np.size else 0.0,
            "mean": float(gate_np.mean()) if gate_np.size else 0.0,
        }

    frame_indices = resolve_frame_indices(splits, frame_count=len(frames), split=split)
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    if not frame_indices:
        raise ValueError(f"Split {split!r} did not resolve to any frames.")

    outputs = []
    for frame_index in frame_indices:
        frame = frames[int(frame_index)]
        convention = resolve_projection_convention(scene_meta, projection_convention)
        image, stats = render_real_splat_image(
            splats,
            frame,
            gate=gate,
            projection_convention=convention,
            near_plane=near_plane,
            kernel_radius=kernel_radius,
            min_sigma_px=min_sigma_px,
            background_color=background_color,
        )
        image_path = resolved_output / frame["file_name"]
        save_image(image_path, image)
        outputs.append(
            {
                "frame_index": int(frame_index),
                "image_path": frame["image_path"],
                "prediction_path": str(image_path),
                "projection_convention": convention,
                **stats,
            }
        )

    report_path = resolved_output / "real_render_report.json"
    report = {
        "schema_version": 1,
        "scene": scene_meta["scene"],
        "method": resolved_method,
        "split": split,
        "scene_dir": str(scene_root),
        "splats_path": str(resolved_splats),
        "model_path": str(resolved_model) if resolved_model is not None else None,
        "output_dir": str(resolved_output),
        "gate_path": str(gate_path) if gate_path is not None else None,
        "use_gpis_gate": use_gpis_gate,
        "gate_summary": gate_summary,
        "splat_count": int(splats.centers.shape[0]),
        "image_count": len(outputs),
        "near_plane": near_plane,
        "kernel_radius": kernel_radius,
        "min_sigma_px": min_sigma_px,
        "background_color": list(background_color),
        "outputs": outputs,
    }
    write_json(report_path, report)
    return {
        "output_dir": resolved_output,
        "report_path": report_path,
        "gate_path": gate_path,
        "report": report,
    }


def render_real_splat_image(
    splats: SplatCloud,
    frame: dict[str, Any],
    *,
    gate: torch.Tensor,
    projection_convention: str,
    near_plane: float,
    kernel_radius: float,
    min_sigma_px: float,
    background_color: tuple[float, float, float],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if projection_convention not in {"opencv", "opengl"}:
        raise ValueError("projection_convention must be resolved to 'opencv' or 'opengl'.")

    intrinsics = frame["intrinsics"]
    width = int(frame.get("width") or intrinsics.get("width"))
    height = int(frame.get("height") or intrinsics.get("height"))
    fx = _required_intrinsic(intrinsics, "fx")
    fy = _required_intrinsic(intrinsics, "fy")

    projected = project_splats_to_frame(
        splats,
        frame,
        projection_convention=projection_convention,
        near_plane=near_plane,
    )
    centers_px = projected["centers_px"]
    depth = projected["depth"]
    valid_projection = projected["valid"]

    colors = splats.colors.detach().cpu().numpy()
    tau = splats.tau.detach().cpu().numpy()
    sigma = splats.sigma.detach().cpu().numpy()
    gate_np = gate.detach().cpu().numpy()

    focal = 0.5 * (fx + fy)
    sigma_px = np.maximum(sigma * focal / np.clip(depth, near_plane, None), min_sigma_px)
    order = np.argsort(depth)
    accum = np.zeros((height, width, 3), dtype=np.float64)
    transmittance = np.ones((height, width), dtype=np.float64)
    drawn_splats = 0

    for splat_index in order:
        if not valid_projection[splat_index]:
            continue
        sx = float(centers_px[splat_index, 0])
        sy = float(centers_px[splat_index, 1])
        spx = float(sigma_px[splat_index])
        radius = max(1, int(np.ceil(kernel_radius * spx)))
        x0 = max(0, int(np.floor(sx - radius)))
        x1 = min(width, int(np.ceil(sx + radius + 1)))
        y0 = max(0, int(np.floor(sy - radius)))
        y1 = min(height, int(np.ceil(sy + radius + 1)))
        if x0 >= x1 or y0 >= y1:
            continue

        xs = np.arange(x0, x1, dtype=np.float64)
        ys = np.arange(y0, y1, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        weight = np.exp(-0.5 * (((xx - sx) / spx) ** 2 + ((yy - sy) / spx) ** 2))
        optical = max(float(tau[splat_index] * gate_np[splat_index]), 0.0) * weight
        alpha = 1.0 - np.exp(-optical)
        patch_trans = transmittance[y0:y1, x0:x1].copy()
        accum[y0:y1, x0:x1, :] += patch_trans[..., None] * alpha[..., None] * colors[splat_index]
        transmittance[y0:y1, x0:x1] = patch_trans * np.exp(-optical)
        drawn_splats += 1

    background = np.asarray(background_color, dtype=np.float64).reshape(1, 1, 3)
    image = np.clip(accum + transmittance[..., None] * background, 0.0, 1.0)
    valid_depth = depth[valid_projection]
    stats = {
        "width": width,
        "height": height,
        "projected_splat_count": int(valid_projection.sum()),
        "drawn_splat_count": int(drawn_splats),
        "min_depth": float(valid_depth.min()) if valid_depth.size else None,
        "max_depth": float(valid_depth.max()) if valid_depth.size else None,
    }
    return torch.from_numpy(image), stats


def project_splats_to_frame(
    splats: SplatCloud,
    frame: dict[str, Any],
    *,
    projection_convention: str,
    near_plane: float,
) -> dict[str, np.ndarray]:
    centers = splats.centers.detach().cpu().numpy()
    world_to_camera = np.asarray(frame["world_to_camera"], dtype=np.float64)
    if world_to_camera.shape != (4, 4):
        raise ValueError("Prepared frames must contain a 4x4 world_to_camera matrix.")
    homogeneous = np.concatenate((centers, np.ones((centers.shape[0], 1), dtype=np.float64)), axis=1)
    camera_xyz = homogeneous @ world_to_camera.T
    camera_xyz = camera_xyz[:, :3]

    intrinsics = frame["intrinsics"]
    fx = _required_intrinsic(intrinsics, "fx")
    fy = _required_intrinsic(intrinsics, "fy")
    cx = _required_intrinsic(intrinsics, "cx")
    cy = _required_intrinsic(intrinsics, "cy")
    with np.errstate(divide="ignore", invalid="ignore"):
        if projection_convention == "opencv":
            depth = camera_xyz[:, 2]
            u = fx * camera_xyz[:, 0] / depth + cx
            v = fy * camera_xyz[:, 1] / depth + cy
        elif projection_convention == "opengl":
            depth = -camera_xyz[:, 2]
            u = fx * camera_xyz[:, 0] / depth + cx
            v = cy - fy * camera_xyz[:, 1] / depth
        else:
            raise ValueError("projection_convention must be 'opencv' or 'opengl'.")

    centers_px = np.stack((u, v), axis=1)
    valid = (depth > near_plane) & np.isfinite(depth) & np.isfinite(centers_px).all(axis=1)
    return {
        "centers_px": centers_px,
        "depth": depth,
        "valid": valid,
        "camera_xyz": camera_xyz,
    }


def resolve_frame_indices(splits: dict[str, Any], *, frame_count: int, split: str) -> list[int]:
    if split == "all":
        return list(range(frame_count))
    if split not in splits:
        raise ValueError(f"Split {split!r} does not exist. Available splits: {', '.join(sorted(k for k in splits if k != 'schema_version'))}.")
    indices = [int(index) for index in splits[split]]
    return [index for index in indices if 0 <= index < frame_count]


def resolve_projection_convention(scene_meta: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    if scene_meta.get("source_format") == "transforms":
        return "opengl"
    return "opencv"


def select_training_indices(
    *,
    sample_type: np.ndarray | None,
    sample_count: int,
    max_train_points: int | None,
    seed: int,
) -> np.ndarray:
    if max_train_points is None or max_train_points <= 0 or sample_count <= max_train_points:
        return np.arange(sample_count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if sample_type is None:
        return np.sort(rng.choice(sample_count, size=max_train_points, replace=False)).astype(np.int64)

    selected: list[np.ndarray] = []
    for label in np.unique(sample_type):
        group = np.flatnonzero(sample_type == label)
        quota = max(1, int(round(max_train_points * group.shape[0] / sample_count)))
        quota = min(quota, group.shape[0])
        selected.append(rng.choice(group, size=quota, replace=False))
    merged = np.unique(np.concatenate(selected)).astype(np.int64)
    if merged.shape[0] > max_train_points:
        merged = np.sort(rng.choice(merged, size=max_train_points, replace=False)).astype(np.int64)
    elif merged.shape[0] < max_train_points:
        missing = max_train_points - merged.shape[0]
        remaining = np.setdiff1d(np.arange(sample_count, dtype=np.int64), merged, assume_unique=False)
        extra = rng.choice(remaining, size=min(missing, remaining.shape[0]), replace=False)
        merged = np.unique(np.concatenate((merged, extra))).astype(np.int64)
    return np.sort(merged)


def sample_type_counts(sample_type: np.ndarray | None) -> dict[str, int]:
    if sample_type is None:
        return {}
    labels, counts = np.unique(sample_type, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts, strict=True)}


def parse_rgb_triplet(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected background color as r,g,b.")
    rgb = tuple(float(part) for part in parts)
    if any(channel < 0.0 or channel > 1.0 for channel in rgb):
        raise ValueError("Background color channels must be in [0, 1].")
    return rgb


def _resolve_scene_file(scene_root: Path, path: str | Path | None, default_name: str) -> Path:
    if path is None:
        return scene_root / default_name
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = scene_root / resolved
    return resolved


def _required_intrinsic(intrinsics: dict[str, Any], key: str) -> float:
    value = intrinsics.get(key)
    if value is None:
        raise ValueError(f"Prepared camera is missing required pinhole intrinsic {key!r}.")
    return float(value)
