from __future__ import annotations

from pathlib import Path

from gpis_splatting.graphdeco_patch import GraphdecoGpisPatchConfig, graphdeco_integration_guide, graphdeco_train_py_patch, write_graphdeco_patch_bundle


def test_graphdeco_patch_contains_regularizer_flags_and_hooks() -> None:
    patch = graphdeco_train_py_patch(GraphdecoGpisPatchConfig(default_surface_weight=0.02, default_start_iteration=123))

    assert "GPIS3DGSTrainingRegularizer.from_model_path" in patch
    assert "GPIS3DGSOptimizationLoop" in patch
    assert "--gpis_model" in patch
    assert "--gpis_surface_weight" in patch
    assert "default=0.02" in patch
    assert "--gpis_start_iteration" in patch
    assert "default=123" in patch
    assert "gpis_loop.augment_loss" in patch
    assert "gpis_loop.after_backward" in patch
    assert "gpis_loop.after_optimizer_step" in patch
    assert "gpis_densification_gradient_boost" in patch


def test_graphdeco_patch_places_density_hooks_at_safe_anchors() -> None:
    patch = graphdeco_train_py_patch()

    assert "gpis_train_step = None" in patch
    assert "gpis_loop.augment_loss" in patch
    assert "gpis_train_step.total_loss.backward()" in patch
    assert "gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)" in patch
    assert "gpis_loop.after_backward(gaussians, gpis_train_step)" in patch
    assert "gaussians.optimizer.step()" in patch
    assert "gpis_loop.after_optimizer_step(gaussians, gpis_train_step)" in patch
    assert patch.index("gpis_train_step.total_loss.backward()") < patch.index("gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)")
    assert patch.index("gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)") < patch.index("gpis_loop.after_backward(gaussians, gpis_train_step)")
    assert patch.index("gaussians.optimizer.step()") < patch.index("gpis_loop.after_optimizer_step(gaussians, gpis_train_step)")


def test_graphdeco_guide_and_bundle_are_written(tmp_path: Path) -> None:
    patch_path = tmp_path / "patches" / "graphdeco_gpis_regularizer.patch"
    guide_path = tmp_path / "docs" / "graphdeco_gpis_patch.md"

    result = write_graphdeco_patch_bundle(patch_path, guide_path)

    assert result["patch_path"] == patch_path
    assert result["guide_path"] == guide_path
    assert patch_path.exists()
    assert guide_path.exists()
    assert "--gpis_model" in patch_path.read_text(encoding="utf-8")
    assert "experiment-matrix case E" in guide_path.read_text(encoding="utf-8")
    assert "regularized_3dgs_render_comparison" in graphdeco_integration_guide()
    assert "Hook-order checks" in graphdeco_integration_guide()
