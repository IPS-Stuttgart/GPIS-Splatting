from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.zero_level_initialization import ZeroLevelInitializationConfig, initialize_zero_level_splats_from_files


def optional_positive_float(value: str) -> float | None:
    parsed = float(value)
    if parsed <= 0.0:
        return None
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize GPIS zero-level splats with anisotropic covariance aligned to posterior mean gradients.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--model-path", default=None, help="Defaults to <scene-dir>/real_gpis_model.npz when --scene or --scene-dir is passed.")
    parser.add_argument("--seed-splats-path", default=None, help="Optional reference splats for surface-biased sampling and nearest-neighbor color transfer.")
    parser.add_argument("--output-splats", default=None, help="Defaults to <scene-dir>/gpis_zero_level_splats.npz when --scene or --scene-dir is passed.")
    parser.add_argument("--output-gate", default=None, help="Defaults to <output-splats-stem>_confidence_gate.npz.")
    parser.add_argument("--output-report", default=None, help="Defaults to <output-splats-stem>_report.json.")
    parser.add_argument("--output-ply", default=None, help="Optional standard 3DGS binary PLY with scale_* and rot_* fields.")
    parser.add_argument("--num-candidates", type=int, default=40000)
    parser.add_argument("--target-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--projection-iterations", type=int, default=4)
    parser.add_argument("--max-projection-step", type=float, default=0.08)
    parser.add_argument("--surface-band", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--max-distance-std", type=optional_positive_float, default=None, help="Use <=0 to disable this filter.")
    parser.add_argument("--min-gradient-norm", type=float, default=1e-4)
    parser.add_argument("--nms-radius", type=float, default=0.015)
    parser.add_argument("--bounds-margin-fraction", type=float, default=0.05)
    parser.add_argument("--surface-seed-fraction", type=float, default=0.75)
    parser.add_argument("--seed-jitter-scale", type=optional_positive_float, default=None, help="Use <=0 to select the default jitter scale.")
    parser.add_argument("--tangent-scale", type=float, default=0.025)
    parser.add_argument("--normal-scale", type=float, default=0.006)
    parser.add_argument("--normal-uncertainty-scale", type=float, default=0.0)
    parser.add_argument("--min-scale", type=float, default=1e-4)
    parser.add_argument("--max-scale", type=optional_positive_float, default=None, help="Use <=0 to disable scale clipping.")
    parser.add_argument("--tau", type=float, default=0.45)
    parser.add_argument("--color-source-max-points", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    scene_dir = resolve_scene_dir(scene=args.scene, prepared_root=args.prepared_root, scene_dir=args.scene_dir)
    model_path = resolve_scene_file(scene_dir, args.model_path, "real_gpis_model.npz")
    output_splats = resolve_scene_file(scene_dir, args.output_splats, "gpis_zero_level_splats.npz")
    seed_splats_path = resolve_optional_scene_file(scene_dir, args.seed_splats_path, "real_splats.npz")
    output_gate = resolve_optional_scene_file(scene_dir, args.output_gate, None)
    output_report = resolve_optional_scene_file(scene_dir, args.output_report, None)
    output_ply = resolve_optional_scene_file(scene_dir, args.output_ply, None)
    config = ZeroLevelInitializationConfig(
        num_candidates=args.num_candidates,
        target_count=args.target_count,
        seed=args.seed,
        projection_iterations=args.projection_iterations,
        max_projection_step=args.max_projection_step,
        surface_band=args.surface_band,
        min_confidence=args.min_confidence,
        max_distance_std=args.max_distance_std,
        min_gradient_norm=args.min_gradient_norm,
        nms_radius=args.nms_radius,
        bounds_margin_fraction=args.bounds_margin_fraction,
        surface_seed_fraction=args.surface_seed_fraction,
        seed_jitter_scale=args.seed_jitter_scale,
        tangent_scale=args.tangent_scale,
        normal_scale=args.normal_scale,
        normal_uncertainty_scale=args.normal_uncertainty_scale,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        tau=args.tau,
        color_source_max_points=args.color_source_max_points,
        batch_size=args.batch_size,
    )
    result = initialize_zero_level_splats_from_files(
        model_path=model_path,
        output_splats_path=output_splats,
        seed_splats_path=seed_splats_path,
        output_gate_path=output_gate,
        output_report_path=output_report,
        output_ply_path=output_ply,
        config=config,
    )
    print(f"Wrote {result['splats_path']}")
    print(f"Wrote {result['gate_path']}")
    print(f"Wrote {result['report_path']}")
    if result["ply_path"] is not None:
        print(f"Wrote {result['ply_path']}")
    print(f"selected_splats: {result['report']['selected_splat_count']}")
    print(f"accepted_before_nms: {result['report']['accepted_before_nms_count']}")


def resolve_scene_dir(*, scene: str | None, prepared_root: str, scene_dir: str | None) -> Path | None:
    if scene_dir is not None:
        return Path(scene_dir)
    if scene is not None:
        return Path(prepared_root) / scene
    return None


def resolve_scene_file(scene_dir: Path | None, value: str | None, default_name: str) -> Path:
    if value is not None:
        path = Path(value)
        return path if path.is_absolute() or scene_dir is None else scene_dir / path
    if scene_dir is None:
        raise ValueError(f"Pass an explicit path or provide --scene/--scene-dir so {default_name!r} can be resolved.")
    return scene_dir / default_name


def resolve_optional_scene_file(scene_dir: Path | None, value: str | None, default_name: str | None) -> Path | None:
    if value is not None:
        path = Path(value)
        return path if path.is_absolute() or scene_dir is None else scene_dir / path
    if default_name is not None and scene_dir is not None:
        candidate = scene_dir / default_name
        return candidate if candidate.exists() else None
    return None


if __name__ == "__main__":
    main()
