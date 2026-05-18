from __future__ import annotations

from pathlib import Path

from gpis_splatting.gpis_training_result_workflow import GPISRegularized3DGSWorkflowConfig, build_training_time_result_commands, write_gpis_regularized_3dgs_workflow


def test_training_time_workflow_commands_promote_regularized_cases(tmp_path: Path) -> None:
    config = GPISRegularized3DGSWorkflowConfig(
        prepared_scene="barn",
        gpis_model_path=Path("real_scenes/barn/gpis_model.npz"),
        output_dir=tmp_path / "workflow",
        trainer_dir=Path("external/gaussian-splatting"),
        iterations=1234,
        gpis_surface_weight=0.02,
    )

    commands = build_training_time_result_commands(config)

    assert "export_prepared_scene_to_colmap_3dgs" in commands.export_scene
    assert "git -C external/gaussian-splatting apply" in commands.apply_graphdeco_patch
    assert "--gpis_model" not in commands.train_baseline
    assert "--gpis_model real_scenes/barn/gpis_model.npz" in commands.train_gpis_regularized
    assert "--gpis_surface_weight 0.02" in commands.train_gpis_regularized
    assert "gpis_regularized_3dgs" in commands.train_gpis_regularized
    assert "run_actual_trained_3dgs_af_matrix" in commands.evaluate_af_matrix
    assert "--regularized-ply" in commands.evaluate_af_matrix
    assert "--max-pred-points 0" in commands.evaluate_af_matrix


def test_training_time_workflow_bundle_writes_patch_script_and_status(tmp_path: Path) -> None:
    config = GPISRegularized3DGSWorkflowConfig(
        prepared_scene="ignatius",
        gpis_model_path=Path("real_scenes/ignatius/gpis_model.npz"),
        output_dir=tmp_path / "regularized_workflow",
        iterations=30000,
    )

    result = write_gpis_regularized_3dgs_workflow(config)

    assert result["patch_path"].exists()
    assert result["guide_path"].exists()
    assert result["script_path"].exists()
    assert result["status_path"].exists()
    assert result["report_path"].exists()
    script = result["script_path"].read_text(encoding="utf-8")
    assert "Train GPIS-regularized 3DGS" in script
    assert "--gpis_model" in script
    assert "run_actual_trained_3dgs_af_matrix" in script
    report = result["report_path"].read_text(encoding="utf-8")
    assert "experiment-matrix cases E and F" in report
    assert "--max-pred-points 0" in report
