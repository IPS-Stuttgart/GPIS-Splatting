# Runtime GPIS training prior for 3DGS

This adapter consumes `.npz` artifacts from `export_gpis_training_prior` or `export_gpis_training_policy` and uses them during 3D Gaussian Splatting training. It complements the live GPIS regularizer: the live regularizer queries a GPIS model at current centers, while this adapter uses precomputed confidence, opacity targets, pruning weights, and densification weights.

## Export

```bash
export_gpis_training_prior \
  --input-ply outputs/<scene>/point_cloud/iteration_30000/point_cloud.ply \
  --gate-path outputs/<scene>/calibrated_gpis_gate.npz \
  --field-scores-path outputs/<scene>/gpis_field_scores.csv \
  --output-dir outputs/<scene>/gpis_training_prior
```

## Trainer integration

```python
from gpis_splatting.gpis_3dgs_optimization import GPIS3DGSOptimizationLoop, GPIS3DGSOptimizationLoopConfig
from gpis_splatting.gpis_3dgs_training_prior import GPIS3DGSTrainingPriorConfig, GPIS3DGSTrainingPriorRegularizer

prior_regularizer = GPIS3DGSTrainingPriorRegularizer.from_prior_path(
    "outputs/<scene>/gpis_training_prior/gpis_confidence_training_prior_training_prior.npz",
    config=GPIS3DGSTrainingPriorConfig(
        start_iteration=500,
        ramp_iterations=1000,
        opacity_weight=1e-3,
        prune_start_iteration=3000,
        prune_interval=1000,
        prune_weight_threshold=0.5,
        max_prune_fraction=0.02,
        densification_boost_start_iteration=3000,
        densification_boost_interval=100,
        densification_gradient_boost=0.25,
    ),
    dtype=gaussians.get_xyz.dtype,
    device=gaussians.get_xyz.device,
)

loop = GPIS3DGSOptimizationLoop(
    prior_regularizer,
    GPIS3DGSOptimizationLoopConfig(step_optimizer=False, apply_densification_boost=True, apply_pruning=True),
)

step = loop.augment_loss(base_loss=loss, gaussians=gaussians, iteration=iteration)
step.total_loss.backward()
loop.after_backward(gaussians, step)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
loop.after_optimizer_step(gaussians, step)
```

## Gaussian-count changes

3DGS changes the number of Gaussians during densification and pruning. The default `count_mismatch="pad"` pads newly created Gaussians with neutral prior values. If the external trainer prunes Gaussians independently, call:

```python
prior_regularizer.apply_prune_mask(external_prune_mask)
```

immediately after the trainer mutates Gaussian tensors.

## Conservative first run

```text
opacity_weight=1e-3
start_iteration=500
ramp_iterations=1000
prune_interval=0
densification_gradient_boost=0.0
```

After photometric quality is stable, enable density control with conservative pruning and densification-boost intervals. This makes GPIS a training-time prior rather than only a post-hoc gate while preserving the external trainer's ownership of rendering, visibility, optimizer-state surgery, densification, and pruning semantics.
