from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GraphdecoGpisPatchConfig:
    """Options for writing a Graphdeco 3DGS GPIS integration patch bundle."""

    default_gpis_epsilon: float = 0.08
    default_surface_weight: float = 0.01
    default_opacity_weight: float = 0.001
    default_normal_weight: float = 0.001
    default_start_iteration: int = 500
    default_ramp_iterations: int = 1000
    default_max_gaussians: int = 65536
    default_batch_size: int = 8192


def graphdeco_train_py_patch(config: GraphdecoGpisPatchConfig | None = None) -> str:
    """Return a patch fragment for adding GPIS regularization to Graphdeco train.py.

    The upstream Graphdeco trainer changes over time, so this intentionally emits a
    reviewable patch fragment with stable insertion anchors instead of mutating an
    external checkout in-place. The fragment covers imports, CLI flags, regularizer
    construction, loss augmentation, logging, and optional density-control hooks.

    Hook ordering matters: the GPIS loss is added before backward, densification-stat
    boosting is applied only after Graphdeco has updated ``xyz_gradient_accum`` via
    ``add_densification_stats``, and GPIS pruning runs after the optimizer step so it
    does not invalidate gradients or optimizer-state reads from the current iteration.
    """
    cfg = GraphdecoGpisPatchConfig() if config is None else config
    return f"""diff --git a/train.py b/train.py
--- a/train.py
+++ b/train.py
@@
 import torch
+from gpis_splatting.gpis_regularization import GPISRegularizationConfig
+from gpis_splatting.gpis_3dgs_regularization import (
+    GPIS3DGSDensityControlConfig,
+    GPIS3DGSRegularizerConfig,
+    GPIS3DGSTrainingRegularizer,
+)
+from gpis_splatting.gpis_3dgs_optimization import GPIS3DGSOptimizationLoop, GPIS3DGSOptimizationLoopConfig
@@
 def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
@@
     gaussians = GaussianModel(dataset.sh_degree)
+    gpis_regularizer = None
+    gpis_loop = None
+    if getattr(opt, "gpis_model", ""):
+        gpis_regularizer = GPIS3DGSTrainingRegularizer.from_model_path(
+            opt.gpis_model,
+            loss_config=GPISRegularizationConfig(
+                epsilon=opt.gpis_epsilon,
+                surface_weight=opt.gpis_surface_weight,
+                opacity_weight=opt.gpis_opacity_weight,
+                normal_weight=opt.gpis_normal_weight,
+                surface_confidence_floor=opt.gpis_surface_confidence_floor,
+            ),
+            schedule_config=GPIS3DGSRegularizerConfig(
+                start_iteration=opt.gpis_start_iteration,
+                stop_iteration=None if opt.gpis_stop_iteration < 0 else opt.gpis_stop_iteration,
+                ramp_iterations=opt.gpis_ramp_iterations,
+                interval=opt.gpis_interval,
+                max_regularized_gaussians=opt.gpis_max_gaussians,
+                batch_size=opt.gpis_batch_size,
+            ),
+            density_config=GPIS3DGSDensityControlConfig(
+                prune_start_iteration=opt.gpis_prune_start_iteration,
+                prune_interval=opt.gpis_prune_interval,
+                prune_confidence_threshold=opt.gpis_prune_confidence_threshold,
+                prune_opacity_threshold=opt.gpis_prune_opacity_threshold,
+                max_prune_fraction=opt.gpis_max_prune_fraction,
+                densification_boost_start_iteration=opt.gpis_densification_boost_start_iteration,
+                densification_boost_interval=opt.gpis_densification_boost_interval,
+                densification_confidence_threshold=opt.gpis_densification_confidence_threshold,
+                densification_min_distance_std=None if opt.gpis_densification_min_distance_std < 0 else opt.gpis_densification_min_distance_std,
+                densification_gradient_boost=opt.gpis_densification_gradient_boost,
+            ),
+        )
+        gpis_loop = GPIS3DGSOptimizationLoop(
+            gpis_regularizer,
+            GPIS3DGSOptimizationLoopConfig(step_optimizer=False, apply_densification_boost=True, apply_pruning=True, prune_after_optimizer_step=True),
+        )
@@
-        loss.backward()
+        gpis_train_step = None
+        gpis_step = None
+        if gpis_loop is None:
+            loss.backward()
+        else:
+            gpis_train_step = gpis_loop.augment_loss(base_loss=loss, gaussians=gaussians, iteration=iteration)
+            gpis_train_step.total_loss.backward()
+            gpis_step = gpis_train_step.gpis_step
+            loss = gpis_train_step.total_loss
@@
         with torch.no_grad():
@@
+            if tb_writer is not None and gpis_step is not None:
+                for name, value in gpis_step.log_dict().items():
+                    tb_writer.add_scalar(name, float(value.detach().cpu()), iteration)
+
             # Densification
             if iteration < opt.densify_until_iter:
@@
                 gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                 gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
+                if gpis_loop is not None and gpis_train_step is not None:
+                    gpis_loop.after_backward(gaussians, gpis_train_step)
@@
                 else:
                     gaussians.optimizer.step()
                     gaussians.optimizer.zero_grad(set_to_none = True)
+                if gpis_loop is not None and gpis_train_step is not None:
+                    gpis_loop.after_optimizer_step(gaussians, gpis_train_step)
@@
 if __name__ == "__main__":
@@
     parser.add_argument('--debug_from', type=int, default=-1)
+    parser.add_argument('--gpis_model', type=str, default='', help='Optional GPIS model .npz used for training-time 3DGS regularization.')
+    parser.add_argument('--gpis_epsilon', type=float, default={cfg.default_gpis_epsilon})
+    parser.add_argument('--gpis_surface_weight', type=float, default={cfg.default_surface_weight})
+    parser.add_argument('--gpis_opacity_weight', type=float, default={cfg.default_opacity_weight})
+    parser.add_argument('--gpis_normal_weight', type=float, default={cfg.default_normal_weight})
+    parser.add_argument('--gpis_surface_confidence_floor', type=float, default=0.05)
+    parser.add_argument('--gpis_start_iteration', type=int, default={cfg.default_start_iteration})
+    parser.add_argument('--gpis_stop_iteration', type=int, default=-1)
+    parser.add_argument('--gpis_ramp_iterations', type=int, default={cfg.default_ramp_iterations})
+    parser.add_argument('--gpis_interval', type=int, default=1)
+    parser.add_argument('--gpis_max_gaussians', type=int, default={cfg.default_max_gaussians})
+    parser.add_argument('--gpis_batch_size', type=int, default={cfg.default_batch_size})
+    parser.add_argument('--gpis_prune_start_iteration', type=int, default=3000)
+    parser.add_argument('--gpis_prune_interval', type=int, default=0)
+    parser.add_argument('--gpis_prune_confidence_threshold', type=float, default=0.05)
+    parser.add_argument('--gpis_prune_opacity_threshold', type=float, default=0.01)
+    parser.add_argument('--gpis_max_prune_fraction', type=float, default=0.02)
+    parser.add_argument('--gpis_densification_boost_start_iteration', type=int, default=3000)
+    parser.add_argument('--gpis_densification_boost_interval', type=int, default=0)
+    parser.add_argument('--gpis_densification_confidence_threshold', type=float, default=0.35)
+    parser.add_argument('--gpis_densification_min_distance_std', type=float, default=-1.0)
+    parser.add_argument('--gpis_densification_gradient_boost', type=float, default=0.0)
"""


def graphdeco_integration_guide(config: GraphdecoGpisPatchConfig | None = None) -> str:
    cfg = GraphdecoGpisPatchConfig() if config is None else config
    return f"""# Graphdeco 3DGS GPIS training integration

This bundle turns the repository's training-time GPIS regularizer into an external Graphdeco `train.py` integration.

## Apply

Generate the patch fragment:

```bash
write_graphdeco_gpis_patch --output patches/graphdeco_gpis_regularizer.patch --guide docs/graphdeco_gpis_patch.md
```

Apply it to a local `graphdeco-inria/gaussian-splatting` checkout manually or with `git apply` after resolving any upstream-context drift. The patch is intentionally reviewable because upstream Graphdeco `train.py` evolves.

## Hook-order checks

Keep the hook order as generated:

1. add the GPIS loss before `backward()`;
2. call `gpis_loop.after_backward(...)` after `gaussians.add_densification_stats(...)` and before Graphdeco densification/pruning;
3. call `gpis_loop.after_optimizer_step(...)` after `gaussians.optimizer.step()`.

This avoids boosting stale densification statistics and avoids pruning Gaussians before the optimizer has consumed the current gradients.

## Conservative first run

```bash
python train.py \\
  -s <colmap_scene> \\
  -m outputs/<scene>_E_gpis_regularized \\
  --gpis_model real_scenes/<scene>/real20k_sigma_0p04_gpis_model.npz \\
  --gpis_epsilon {cfg.default_gpis_epsilon} \\
  --gpis_surface_weight {cfg.default_surface_weight} \\
  --gpis_opacity_weight {cfg.default_opacity_weight} \\
  --gpis_normal_weight {cfg.default_normal_weight} \\
  --gpis_start_iteration {cfg.default_start_iteration} \\
  --gpis_ramp_iterations {cfg.default_ramp_iterations} \\
  --gpis_max_gaussians {cfg.default_max_gaussians} \\
  --gpis_prune_interval 0 \\
  --gpis_densification_gradient_boost 0.0
```

This produces experiment-matrix case E. After validating that PSNR/SSIM do not regress strongly, enable conservative density control:

```bash
--gpis_prune_interval 1000 \\
--gpis_densification_boost_interval 100 \\
--gpis_densification_gradient_boost 0.25
```

## Matrix handoff

Evaluate the regularized checkpoint exactly like the plain trained-3DGS baseline, then pass the render-comparison and optional geometry-summary CSVs as `regularized_3dgs_render_comparison` and `regularized_geometry_summary` artifacts in the A-F experiment matrix.
"""


def write_graphdeco_patch_bundle(output_path: str | Path, guide_path: str | Path | None = None, *, config: GraphdecoGpisPatchConfig | None = None) -> dict[str, Path]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(graphdeco_train_py_patch(config), encoding="utf-8")
    result = {"patch_path": output}
    if guide_path is not None:
        guide = Path(guide_path)
        guide.parent.mkdir(parents=True, exist_ok=True)
        guide.write_text(graphdeco_integration_guide(config), encoding="utf-8")
        result["guide_path"] = guide
    return result
