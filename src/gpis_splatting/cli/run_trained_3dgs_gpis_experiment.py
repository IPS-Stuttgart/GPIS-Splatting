from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.cli.evaluate_real_renders import str_to_bool
from gpis_splatting.trained_3dgs_evaluation import TRAINED_3DGS_RENDERERS, coerce_optional_positive_int, run_trained_3dgs_gpis_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trained-3DGS GPIS scoring, variant export, optional rendering, and photometric evaluation experiment.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--trained-ply-path", required=True)
    parser.add_argument("--gpis-model-path", default=None, help="Required unless --gate-path is supplied.")
    parser.add_argument("--gate-path", default=None, help="Precomputed gate NPZ aligned to the trained PLY.")
    parser.add_argument("--method-name", default="trained_3dgs")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--calibration-threshold", type=float, default=0.05)
    parser.add_argument("--gate-thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--max-pred-points", type=int, default=0, help="Use 0 to score all trained Gaussians.")
    parser.add_argument("--max-gt-points", type=int, default=150000, help="Use 0 for all GT points.")
    parser.add_argument("--missing-gate-value", type=float, default=1.0)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--opacity-mode", choices=["logit", "linear"], default="logit")
    parser.add_argument("--opacity-scale-floor", type=float, default=0.0)
    parser.add_argument("--opacity-scale-floors", type=float, nargs="*", default=[], help="Additional conservative opacity floors to export as gate_floor_<value> variants.")
    parser.add_argument("--renderer", choices=TRAINED_3DGS_RENDERERS, default="none")
    parser.add_argument("--render-command-template", default=None)
    parser.add_argument("--rendered-predictions-root", default=None)
    parser.add_argument("--prediction-subdir", default="")
    parser.add_argument("--render-name-map-path", default=None)
    parser.add_argument("--render-mapping-link-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--render-split", default="test")
    parser.add_argument("--compute-lpips", type=str_to_bool, default=False)
    parser.add_argument("--require-all-images", type=str_to_bool, default=True)
    parser.add_argument("--require-all-variants", type=str_to_bool, default=True)
    parser.add_argument("--benchmark-target", default=None)
    parser.add_argument("--gsplat-device", default="auto")
    parser.add_argument("--gsplat-max-frames", type=int, default=None)
    parser.add_argument("--gsplat-max-gaussians", type=int, default=None)
    return parser


def resolve_scene_dir(args: argparse.Namespace) -> Path:
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    return Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_trained_3dgs_gpis_experiment(
        scene_dir=resolve_scene_dir(args),
        trained_ply_path=args.trained_ply_path,
        method_name=args.method_name,
        gpis_model_path=args.gpis_model_path,
        gate_path=args.gate_path,
        thresholds=tuple(args.thresholds),
        calibration_threshold=args.calibration_threshold,
        gate_thresholds=tuple(args.gate_thresholds),
        max_pred_points=coerce_optional_positive_int(args.max_pred_points),
        max_gt_points=coerce_optional_positive_int(args.max_gt_points),
        missing_gate_value=args.missing_gate_value,
        iteration=args.iteration,
        opacity_mode=args.opacity_mode,
        opacity_scale_floor=args.opacity_scale_floor,
        opacity_scale_floors=tuple(args.opacity_scale_floors),
        renderer=args.renderer,
        render_command_template=args.render_command_template,
        rendered_predictions_root=args.rendered_predictions_root,
        prediction_subdir=args.prediction_subdir,
        render_name_map_path=args.render_name_map_path,
        render_mapping_link_mode=args.render_mapping_link_mode,
        render_split=args.render_split,
        compute_lpips=args.compute_lpips,
        require_all_images=args.require_all_images,
        require_all_variants=args.require_all_variants,
        benchmark_target=args.benchmark_target,
        gsplat_device=args.gsplat_device,
        gsplat_max_frames=args.gsplat_max_frames,
        gsplat_max_gaussians=args.gsplat_max_gaussians,
    )
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"Wrote {result['variants']['manifest_path']}")
    if result["render_evaluation"] is not None:
        print(f"Wrote {result['render_evaluation']['comparison_path']}")


if __name__ == "__main__":
    main()
