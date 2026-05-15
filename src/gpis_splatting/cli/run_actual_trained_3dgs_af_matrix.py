from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.actual_trained_3dgs_af_matrix import ActualTrained3DGSAFConfig, ActualTrained3DGSInput, run_actual_trained_3dgs_af_matrix
from gpis_splatting.cli.evaluate_real_renders import str_to_bool
from gpis_splatting.trained_3dgs_evaluation import TRAINED_3DGS_RENDERERS, coerce_optional_positive_int


def optional_path(value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the actual A-F matrix from plain and GPIS-regularized trained 3DGS models.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/evaluations/<matrix-name>.")
    parser.add_argument("--matrix-name", default="trained_3dgs_af_matrix")
    parser.add_argument("--gpis-model-path", required=True, help="GPIS model path, absolute or relative to the prepared scene.")
    parser.add_argument("--baseline-ply", required=True, help="Plain trained 3DGS point_cloud.ply.")
    parser.add_argument("--regularized-ply", required=True, help="GPIS-regularized trained 3DGS point_cloud.ply.")
    parser.add_argument("--baseline-method-name", default="plain_3dgs")
    parser.add_argument("--regularized-method-name", default="gpis_regularized_3dgs")
    parser.add_argument("--baseline-rendered-predictions-root", default=None)
    parser.add_argument("--baseline-raw-rendered-predictions-root", default=None)
    parser.add_argument("--regularized-rendered-predictions-root", default=None)
    parser.add_argument("--regularized-raw-rendered-predictions-root", default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--primary-geometry-threshold", type=float, default=0.05)
    parser.add_argument("--calibration-threshold", type=float, default=0.05)
    parser.add_argument("--gate-thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--max-pred-points", type=int, default=0, help="Use 0 for all trained Gaussians.")
    parser.add_argument("--max-gt-points", type=int, default=150000, help="Use 0 for all GT points.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--missing-gate-value", type=float, default=1.0)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--opacity-mode", choices=["logit", "linear"], default="logit")
    parser.add_argument("--opacity-scale-floor", type=float, default=0.0)
    parser.add_argument("--renderer", choices=TRAINED_3DGS_RENDERERS, default="none")
    parser.add_argument("--render-command-template", default=None)
    parser.add_argument("--prediction-subdir", default="")
    parser.add_argument("--render-split", default="test")
    parser.add_argument("--compute-lpips", type=str_to_bool, default=False)
    parser.add_argument("--require-all-images", type=str_to_bool, default=True)
    parser.add_argument("--require-all-variants", type=str_to_bool, default=True)
    parser.add_argument("--require-render-metrics", type=str_to_bool, default=True)
    parser.add_argument("--require-full-matrix", type=str_to_bool, default=True)
    parser.add_argument("--benchmark-target", default=None)
    parser.add_argument("--gsplat-device", default="auto")
    parser.add_argument("--gsplat-max-frames", type=int, default=None)
    parser.add_argument("--gsplat-max-gaussians", type=int, default=None)
    return parser


def resolve_scene_dir(args: argparse.Namespace) -> Path:
    if args.scene_dir is not None:
        return Path(args.scene_dir)
    if args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    return Path(args.prepared_root) / args.scene


def resolve_output_dir(args: argparse.Namespace, scene_dir: Path) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    return scene_dir / "evaluations" / args.matrix_name


def resolve_scene_relative_path(scene_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    scene_relative = scene_dir / path
    if scene_relative.exists():
        return scene_relative
    return path


def build_config(args: argparse.Namespace) -> ActualTrained3DGSAFConfig:
    scene_dir = resolve_scene_dir(args)
    return ActualTrained3DGSAFConfig(
        scene_dir=scene_dir,
        gpis_model_path=resolve_scene_relative_path(scene_dir, args.gpis_model_path),
        baseline=ActualTrained3DGSInput(
            name=args.baseline_method_name,
            trained_ply_path=Path(args.baseline_ply),
            calibrated_rendered_predictions_root=optional_path(args.baseline_rendered_predictions_root),
            raw_rendered_predictions_root=optional_path(args.baseline_raw_rendered_predictions_root),
        ),
        regularized=ActualTrained3DGSInput(
            name=args.regularized_method_name,
            trained_ply_path=Path(args.regularized_ply),
            calibrated_rendered_predictions_root=optional_path(args.regularized_rendered_predictions_root),
            raw_rendered_predictions_root=optional_path(args.regularized_raw_rendered_predictions_root),
        ),
        output_dir=resolve_output_dir(args, scene_dir),
        matrix_name=args.matrix_name,
        thresholds=tuple(args.thresholds),
        primary_geometry_threshold=args.primary_geometry_threshold,
        calibration_threshold=args.calibration_threshold,
        gate_thresholds=tuple(args.gate_thresholds),
        max_pred_points=coerce_optional_positive_int(args.max_pred_points),
        max_gt_points=coerce_optional_positive_int(args.max_gt_points),
        seed=args.seed,
        missing_gate_value=args.missing_gate_value,
        iteration=args.iteration,
        opacity_mode=args.opacity_mode,
        opacity_scale_floor=args.opacity_scale_floor,
        renderer=args.renderer,
        render_command_template=args.render_command_template,
        prediction_subdir=args.prediction_subdir,
        render_split=args.render_split,
        compute_lpips=args.compute_lpips,
        require_all_images=args.require_all_images,
        require_all_variants=args.require_all_variants,
        require_render_metrics=args.require_render_metrics,
        require_full_matrix=args.require_full_matrix,
        benchmark_target=optional_path(args.benchmark_target),
        gsplat_device=args.gsplat_device,
        gsplat_max_frames=args.gsplat_max_frames,
        gsplat_max_gaussians=args.gsplat_max_gaussians,
    )


def main(argv: list[str] | None = None) -> None:
    result = run_actual_trained_3dgs_af_matrix(build_config(build_parser().parse_args(argv)))
    print(f"Wrote {result['results_path']}")
    print(f"Wrote {result['checks_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"passed: {result['passed']}")


if __name__ == "__main__":
    main()
