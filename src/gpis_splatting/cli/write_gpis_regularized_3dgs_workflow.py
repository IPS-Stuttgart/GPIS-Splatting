from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.cli.evaluate_real_renders import str_to_bool
from gpis_splatting.gpis_training_result_workflow import GPISRegularized3DGSWorkflowConfig, write_gpis_regularized_3dgs_workflow


def none_or_int(value: str) -> int | None:
    lowered = value.strip().lower()
    return None if lowered in {"none", "null", "all", "0", "-1"} else int(value)


def optional_path(value: str | None) -> Path | None:
    return None if value is None or value == "" else Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a reproducible baseline-vs-GPIS-regularized 3DGS workflow bundle for A-F cases E/F.")
    parser.add_argument("--scene", required=True, help="Prepared scene name under --prepared-root.")
    parser.add_argument("--gpis-model-path", required=True, help="GPIS model path used by the Graphdeco training patch and A-F evaluation.")
    parser.add_argument("--output-dir", required=True, help="Directory for the generated patch, guide, command script, and status/report files.")
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--trainer-dir", default="_external/gaussian-splatting")
    parser.add_argument("--colmap-scene-dir", default=None)
    parser.add_argument("--baseline-model-dir", default=None)
    parser.add_argument("--regularized-model-dir", default=None)
    parser.add_argument("--export-split", choices=("train", "all"), default="train")
    parser.add_argument("--max-points", type=none_or_int, default=100000, help="Sparse points exported to COLMAP; use 0/all/none for all points.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--matrix-name", default="trained_3dgs_af_matrix")
    parser.add_argument("--gpis-epsilon", type=float, default=0.08)
    parser.add_argument("--gpis-surface-weight", type=float, default=0.01)
    parser.add_argument("--gpis-opacity-weight", type=float, default=0.001)
    parser.add_argument("--gpis-normal-weight", type=float, default=0.001)
    parser.add_argument("--gpis-surface-confidence-floor", type=float, default=0.05)
    parser.add_argument("--gpis-start-iteration", type=int, default=500)
    parser.add_argument("--gpis-stop-iteration", type=int, default=-1)
    parser.add_argument("--gpis-ramp-iterations", type=int, default=1000)
    parser.add_argument("--gpis-interval", type=int, default=1)
    parser.add_argument("--gpis-max-gaussians", type=int, default=65536)
    parser.add_argument("--gpis-batch-size", type=int, default=8192)
    parser.add_argument("--gpis-prune-start-iteration", type=int, default=3000)
    parser.add_argument("--gpis-prune-interval", type=int, default=0)
    parser.add_argument("--gpis-prune-confidence-threshold", type=float, default=0.05)
    parser.add_argument("--gpis-prune-opacity-threshold", type=float, default=0.01)
    parser.add_argument("--gpis-max-prune-fraction", type=float, default=0.02)
    parser.add_argument("--gpis-densification-boost-start-iteration", type=int, default=3000)
    parser.add_argument("--gpis-densification-boost-interval", type=int, default=0)
    parser.add_argument("--gpis-densification-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--gpis-densification-min-distance-std", type=float, default=-1.0)
    parser.add_argument("--gpis-densification-gradient-boost", type=float, default=0.0)
    parser.add_argument("--renderer", default="none")
    parser.add_argument("--render-command-template", default=None)
    parser.add_argument("--prediction-subdir", default="")
    parser.add_argument("--render-split", default="test")
    parser.add_argument("--compute-lpips", type=str_to_bool, default=False)
    parser.add_argument("--require-render-metrics", type=str_to_bool, default=False)
    parser.add_argument("--require-full-matrix", type=str_to_bool, default=True)
    return parser


def build_config(args: argparse.Namespace) -> GPISRegularized3DGSWorkflowConfig:
    return GPISRegularized3DGSWorkflowConfig(
        prepared_scene=args.scene,
        gpis_model_path=Path(args.gpis_model_path),
        output_dir=Path(args.output_dir),
        prepared_root=Path(args.prepared_root),
        trainer_dir=Path(args.trainer_dir),
        colmap_scene_dir=optional_path(args.colmap_scene_dir),
        baseline_model_dir=optional_path(args.baseline_model_dir),
        regularized_model_dir=optional_path(args.regularized_model_dir),
        export_split=args.export_split,
        max_points=args.max_points,
        seed=args.seed,
        iterations=args.iterations,
        matrix_name=args.matrix_name,
        gpis_epsilon=args.gpis_epsilon,
        gpis_surface_weight=args.gpis_surface_weight,
        gpis_opacity_weight=args.gpis_opacity_weight,
        gpis_normal_weight=args.gpis_normal_weight,
        gpis_surface_confidence_floor=args.gpis_surface_confidence_floor,
        gpis_start_iteration=args.gpis_start_iteration,
        gpis_stop_iteration=args.gpis_stop_iteration,
        gpis_ramp_iterations=args.gpis_ramp_iterations,
        gpis_interval=args.gpis_interval,
        gpis_max_gaussians=args.gpis_max_gaussians,
        gpis_batch_size=args.gpis_batch_size,
        gpis_prune_start_iteration=args.gpis_prune_start_iteration,
        gpis_prune_interval=args.gpis_prune_interval,
        gpis_prune_confidence_threshold=args.gpis_prune_confidence_threshold,
        gpis_prune_opacity_threshold=args.gpis_prune_opacity_threshold,
        gpis_max_prune_fraction=args.gpis_max_prune_fraction,
        gpis_densification_boost_start_iteration=args.gpis_densification_boost_start_iteration,
        gpis_densification_boost_interval=args.gpis_densification_boost_interval,
        gpis_densification_confidence_threshold=args.gpis_densification_confidence_threshold,
        gpis_densification_min_distance_std=args.gpis_densification_min_distance_std,
        gpis_densification_gradient_boost=args.gpis_densification_gradient_boost,
        renderer=args.renderer,
        render_command_template=args.render_command_template,
        prediction_subdir=args.prediction_subdir,
        render_split=args.render_split,
        compute_lpips=args.compute_lpips,
        require_render_metrics=args.require_render_metrics,
        require_full_matrix=args.require_full_matrix,
    )


def main(argv: list[str] | None = None) -> None:
    result = write_gpis_regularized_3dgs_workflow(build_config(build_parser().parse_args(argv)))
    print(f"Wrote {result['patch_path']}")
    print(f"Wrote {result['guide_path']}")
    print(f"Wrote {result['script_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")


if __name__ == "__main__":
    main()
