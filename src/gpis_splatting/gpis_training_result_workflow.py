from __future__ import annotations

import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gpis_splatting.graphdeco_patch import GraphdecoGpisPatchConfig, write_graphdeco_patch_bundle
from gpis_splatting.serialization import write_json


@dataclass(frozen=True)
class GPISRegularized3DGSWorkflowConfig:
    """Configuration for a reproducible training-time GPIS 3DGS result workflow.

    The workflow makes experiment-matrix cases E and F explicit: train a plain 3DGS
    baseline, train a second 3DGS model with the GPIS regularizer inside the optimizer,
    and evaluate both with the actual trained-3DGS A-F matrix runner.
    """

    prepared_scene: str
    gpis_model_path: Path
    output_dir: Path
    prepared_root: Path = Path("real_scenes")
    trainer_dir: Path = Path("_external/gaussian-splatting")
    colmap_scene_dir: Path | None = None
    baseline_model_dir: Path | None = None
    regularized_model_dir: Path | None = None
    export_split: str = "train"
    max_points: int | None = 100_000
    seed: int = 13
    iterations: int = 30_000
    matrix_name: str = "trained_3dgs_af_matrix"
    gpis_epsilon: float = 0.08
    gpis_surface_weight: float = 0.01
    gpis_opacity_weight: float = 0.001
    gpis_normal_weight: float = 0.001
    gpis_surface_confidence_floor: float = 0.05
    gpis_start_iteration: int = 500
    gpis_stop_iteration: int = -1
    gpis_ramp_iterations: int = 1000
    gpis_interval: int = 1
    gpis_max_gaussians: int = 65_536
    gpis_batch_size: int = 8192
    gpis_prune_start_iteration: int = 3000
    gpis_prune_interval: int = 0
    gpis_prune_confidence_threshold: float = 0.05
    gpis_prune_opacity_threshold: float = 0.01
    gpis_max_prune_fraction: float = 0.02
    gpis_densification_boost_start_iteration: int = 3000
    gpis_densification_boost_interval: int = 0
    gpis_densification_confidence_threshold: float = 0.35
    gpis_densification_min_distance_std: float = -1.0
    gpis_densification_gradient_boost: float = 0.0
    renderer: str = "none"
    render_command_template: str | None = None
    prediction_subdir: str = ""
    render_split: str = "test"
    compute_lpips: bool = False
    require_render_metrics: bool = False
    require_full_matrix: bool = True


@dataclass(frozen=True)
class GPISRegularized3DGSWorkflowPaths:
    output_dir: Path
    patch_path: Path
    guide_path: Path
    script_path: Path
    report_path: Path
    status_path: Path
    colmap_scene_dir: Path
    baseline_model_dir: Path
    regularized_model_dir: Path
    baseline_ply_path: Path
    regularized_ply_path: Path
    matrix_output_dir: Path


@dataclass(frozen=True)
class GPISRegularized3DGSCommands:
    export_scene: str
    apply_graphdeco_patch: str
    train_baseline: str
    train_gpis_regularized: str
    evaluate_af_matrix: str


def resolve_workflow_paths(config: GPISRegularized3DGSWorkflowConfig) -> GPISRegularized3DGSWorkflowPaths:
    output_dir = Path(config.output_dir)
    colmap_scene_dir = Path(config.colmap_scene_dir) if config.colmap_scene_dir is not None else output_dir / "colmap_scene"
    baseline_model_dir = Path(config.baseline_model_dir) if config.baseline_model_dir is not None else output_dir / "baseline_3dgs"
    regularized_model_dir = Path(config.regularized_model_dir) if config.regularized_model_dir is not None else output_dir / "gpis_regularized_3dgs"
    baseline_ply_path = baseline_model_dir / "point_cloud" / f"iteration_{int(config.iterations)}" / "point_cloud.ply"
    regularized_ply_path = regularized_model_dir / "point_cloud" / f"iteration_{int(config.iterations)}" / "point_cloud.ply"
    return GPISRegularized3DGSWorkflowPaths(
        output_dir=output_dir,
        patch_path=output_dir / "graphdeco_gpis_regularizer.patch",
        guide_path=output_dir / "graphdeco_gpis_patch.md",
        script_path=output_dir / "train_gpis_regularized_3dgs_workflow.sh",
        report_path=output_dir / "gpis_regularized_3dgs_workflow.md",
        status_path=output_dir / "gpis_regularized_3dgs_workflow.json",
        colmap_scene_dir=colmap_scene_dir,
        baseline_model_dir=baseline_model_dir,
        regularized_model_dir=regularized_model_dir,
        baseline_ply_path=baseline_ply_path,
        regularized_ply_path=regularized_ply_path,
        matrix_output_dir=output_dir / config.matrix_name,
    )


def build_training_time_result_commands(config: GPISRegularized3DGSWorkflowConfig) -> GPISRegularized3DGSCommands:
    """Build shell commands for baseline, GPIS-regularized training and A-F evaluation."""
    validate_workflow_config(config)
    paths = resolve_workflow_paths(config)
    trainer_train_py = Path(config.trainer_dir) / "train.py"
    return GPISRegularized3DGSCommands(
        export_scene=format_export_scene_command(config, paths),
        apply_graphdeco_patch=format_apply_patch_command(config, paths),
        train_baseline=format_graphdeco_train_command(
            train_py=trainer_train_py,
            colmap_scene_dir=paths.colmap_scene_dir,
            model_dir=paths.baseline_model_dir,
            iterations=config.iterations,
            gpis_flags=(),
        ),
        train_gpis_regularized=format_graphdeco_train_command(
            train_py=trainer_train_py,
            colmap_scene_dir=paths.colmap_scene_dir,
            model_dir=paths.regularized_model_dir,
            iterations=config.iterations,
            gpis_flags=gpis_regularizer_flags(config),
        ),
        evaluate_af_matrix=format_af_matrix_command(config, paths),
    )


def write_gpis_regularized_3dgs_workflow(config: GPISRegularized3DGSWorkflowConfig) -> dict[str, Any]:
    """Write a self-contained workflow bundle for experiment-matrix cases E and F."""
    validate_workflow_config(config)
    paths = resolve_workflow_paths(config)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    patch_config = GraphdecoGpisPatchConfig(
        default_gpis_epsilon=config.gpis_epsilon,
        default_surface_weight=config.gpis_surface_weight,
        default_opacity_weight=config.gpis_opacity_weight,
        default_normal_weight=config.gpis_normal_weight,
        default_start_iteration=config.gpis_start_iteration,
        default_ramp_iterations=config.gpis_ramp_iterations,
        default_max_gaussians=config.gpis_max_gaussians,
        default_batch_size=config.gpis_batch_size,
    )
    write_graphdeco_patch_bundle(paths.patch_path, paths.guide_path, config=patch_config)
    commands = build_training_time_result_commands(config)
    paths.script_path.write_text(format_workflow_script(commands), encoding="utf-8")
    paths.script_path.chmod(0o755)
    status = workflow_status(config, paths, commands)
    write_json(paths.status_path, status)
    paths.report_path.write_text(format_workflow_report(status), encoding="utf-8")
    return {
        "paths": paths,
        "commands": commands,
        "status": status,
        "patch_path": paths.patch_path,
        "guide_path": paths.guide_path,
        "script_path": paths.script_path,
        "status_path": paths.status_path,
        "report_path": paths.report_path,
    }


def format_export_scene_command(config: GPISRegularized3DGSWorkflowConfig, paths: GPISRegularized3DGSWorkflowPaths) -> str:
    return join_shell_command(
        [
            "export_prepared_scene_to_colmap_3dgs",
            "--scene",
            config.prepared_scene,
            "--prepared-root",
            str(config.prepared_root),
            "--output-dir",
            str(paths.colmap_scene_dir),
            "--split",
            config.export_split,
            "--max-points",
            str(0 if config.max_points is None else int(config.max_points)),
            "--seed",
            str(int(config.seed)),
        ]
    )


def format_apply_patch_command(config: GPISRegularized3DGSWorkflowConfig, paths: GPISRegularized3DGSWorkflowPaths) -> str:
    return join_shell_command(
        [
            "git",
            "-C",
            str(config.trainer_dir),
            "apply",
            "--reject",
            "--whitespace=fix",
            str(paths.patch_path),
        ]
    )


def format_graphdeco_train_command(
    *,
    train_py: Path,
    colmap_scene_dir: Path,
    model_dir: Path,
    iterations: int,
    gpis_flags: tuple[str, ...],
) -> str:
    return join_shell_command(
        [
            "python",
            str(train_py),
            "-s",
            str(colmap_scene_dir),
            "-m",
            str(model_dir),
            "--iterations",
            str(int(iterations)),
            "--save_iterations",
            str(int(iterations)),
            "--test_iterations",
            "-1",
            "--checkpoint_iterations",
            "-1",
            *gpis_flags,
        ]
    )


def gpis_regularizer_flags(config: GPISRegularized3DGSWorkflowConfig) -> tuple[str, ...]:
    return (
        "--gpis_model",
        str(config.gpis_model_path),
        "--gpis_epsilon",
        str(config.gpis_epsilon),
        "--gpis_surface_weight",
        str(config.gpis_surface_weight),
        "--gpis_opacity_weight",
        str(config.gpis_opacity_weight),
        "--gpis_normal_weight",
        str(config.gpis_normal_weight),
        "--gpis_surface_confidence_floor",
        str(config.gpis_surface_confidence_floor),
        "--gpis_start_iteration",
        str(int(config.gpis_start_iteration)),
        "--gpis_stop_iteration",
        str(int(config.gpis_stop_iteration)),
        "--gpis_ramp_iterations",
        str(int(config.gpis_ramp_iterations)),
        "--gpis_interval",
        str(int(config.gpis_interval)),
        "--gpis_max_gaussians",
        str(int(config.gpis_max_gaussians)),
        "--gpis_batch_size",
        str(int(config.gpis_batch_size)),
        "--gpis_prune_start_iteration",
        str(int(config.gpis_prune_start_iteration)),
        "--gpis_prune_interval",
        str(int(config.gpis_prune_interval)),
        "--gpis_prune_confidence_threshold",
        str(config.gpis_prune_confidence_threshold),
        "--gpis_prune_opacity_threshold",
        str(config.gpis_prune_opacity_threshold),
        "--gpis_max_prune_fraction",
        str(config.gpis_max_prune_fraction),
        "--gpis_densification_boost_start_iteration",
        str(int(config.gpis_densification_boost_start_iteration)),
        "--gpis_densification_boost_interval",
        str(int(config.gpis_densification_boost_interval)),
        "--gpis_densification_confidence_threshold",
        str(config.gpis_densification_confidence_threshold),
        "--gpis_densification_min_distance_std",
        str(config.gpis_densification_min_distance_std),
        "--gpis_densification_gradient_boost",
        str(config.gpis_densification_gradient_boost),
    )


def format_af_matrix_command(config: GPISRegularized3DGSWorkflowConfig, paths: GPISRegularized3DGSWorkflowPaths) -> str:
    args = [
        "run_actual_trained_3dgs_af_matrix",
        "--scene",
        config.prepared_scene,
        "--prepared-root",
        str(config.prepared_root),
        "--output-dir",
        str(paths.matrix_output_dir),
        "--matrix-name",
        config.matrix_name,
        "--gpis-model-path",
        str(config.gpis_model_path),
        "--baseline-ply",
        str(paths.baseline_ply_path),
        "--regularized-ply",
        str(paths.regularized_ply_path),
        "--baseline-method-name",
        "plain_3dgs",
        "--regularized-method-name",
        "gpis_training_regularized_3dgs",
        "--max-pred-points",
        "0",
        "--renderer",
        config.renderer,
        "--prediction-subdir",
        config.prediction_subdir,
        "--render-split",
        config.render_split,
        "--compute-lpips",
        bool_arg(config.compute_lpips),
        "--require-render-metrics",
        bool_arg(config.require_render_metrics),
        "--require-full-matrix",
        bool_arg(config.require_full_matrix),
    ]
    if config.render_command_template:
        args.extend(["--render-command-template", config.render_command_template])
    return join_shell_command(args)


def format_workflow_script(commands: GPISRegularized3DGSCommands) -> str:
    sections = (
        ("Export prepared scene", commands.export_scene),
        ("Apply Graphdeco GPIS patch", commands.apply_graphdeco_patch),
        ("Train plain 3DGS baseline", commands.train_baseline),
        ("Train GPIS-regularized 3DGS", commands.train_gpis_regularized),
        ("Evaluate actual A-F matrix", commands.evaluate_af_matrix),
    )
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for title, command in sections:
        lines.extend([f"echo {shlex.quote('==> ' + title)}", command, ""])
    return "\n".join(lines)


def workflow_status(
    config: GPISRegularized3DGSWorkflowConfig,
    paths: GPISRegularized3DGSWorkflowPaths,
    commands: GPISRegularized3DGSCommands,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": "gpis_regularized_3dgs_training_result",
        "purpose": "train GPIS-regularized 3DGS and evaluate experiment-matrix cases E/F",
        "config": stringify_paths(asdict(config)),
        "paths": stringify_paths(asdict(paths)),
        "commands": asdict(commands),
        "primary_cases": ["E", "F"],
        "scoring_policy": "run_actual_trained_3dgs_af_matrix is generated with --max-pred-points 0 so all Gaussians are scored.",
    }


def format_workflow_report(status: dict[str, Any]) -> str:
    paths = status["paths"]
    commands = status["commands"]
    return "\n".join(
        [
            "# GPIS-Regularized 3DGS Training Result Workflow",
            "",
            "This bundle promotes training-time GPIS regularization to the result-producing path for experiment-matrix cases E and F.",
            "",
            "## Artifacts",
            "",
            f"- Graphdeco patch: `{paths['patch_path']}`",
            f"- Patch guide: `{paths['guide_path']}`",
            f"- Executable workflow script: `{paths['script_path']}`",
            f"- A-F matrix output directory: `{paths['matrix_output_dir']}`",
            "",
            "## Commands",
            "",
            "```bash",
            commands["export_scene"],
            commands["apply_graphdeco_patch"],
            commands["train_baseline"],
            commands["train_gpis_regularized"],
            commands["evaluate_af_matrix"],
            "```",
            "",
            "The generated A-F matrix command uses `--max-pred-points 0` so post-training GPIS scoring covers every trained Gaussian instead of trusting unscored splats.",
            "",
        ]
    )


def validate_workflow_config(config: GPISRegularized3DGSWorkflowConfig) -> None:
    if not config.prepared_scene:
        raise ValueError("prepared_scene must be non-empty.")
    if config.iterations < 1:
        raise ValueError("iterations must be positive.")
    if config.max_points is not None and config.max_points < 1:
        raise ValueError("max_points must be positive, or None to export all points.")
    if config.gpis_interval < 1:
        raise ValueError("gpis_interval must be positive.")
    if config.gpis_ramp_iterations < 0:
        raise ValueError("gpis_ramp_iterations must be non-negative.")
    if config.gpis_max_gaussians < 1:
        raise ValueError("gpis_max_gaussians must be positive.")
    if config.gpis_batch_size < 1:
        raise ValueError("gpis_batch_size must be positive.")
    if config.gpis_prune_interval < 0 or config.gpis_densification_boost_interval < 0:
        raise ValueError("GPIS density-control intervals must be non-negative.")
    if not 0.0 <= config.gpis_max_prune_fraction <= 1.0:
        raise ValueError("gpis_max_prune_fraction must be in [0, 1].")


def join_shell_command(args: list[str] | tuple[str, ...]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: stringify_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [stringify_paths(item) for item in value]
    return value


__all__ = [
    "GPISRegularized3DGSCommands",
    "GPISRegularized3DGSWorkflowConfig",
    "GPISRegularized3DGSWorkflowPaths",
    "build_training_time_result_commands",
    "format_af_matrix_command",
    "format_graphdeco_train_command",
    "gpis_regularizer_flags",
    "resolve_workflow_paths",
    "write_gpis_regularized_3dgs_workflow",
]
