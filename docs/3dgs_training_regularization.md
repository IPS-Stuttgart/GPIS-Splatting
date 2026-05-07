# Training-time GPIS regularization for 3DGS

This adapter moves GPIS from a post-hoc gate into the 3D Gaussian Splatting optimization loop. It is designed to be imported by an external 3DGS trainer, especially the reference `graphdeco-inria/gaussian-splatting` training loop, without vendoring a CUDA renderer into this repository.

## What the adapter does

`gpis_splatting.gpis_3dgs_regularization.GPIS3DGSTrainingRegularizer` reads a fixed GPIS model and computes differentiable losses on the current Gaussian centers:

- surface loss: pulls centers toward the GPIS zero level set using the signed-distance proxy `mean / ||grad mean||`;
- opacity loss: suppresses opacity for low-confidence off-surface Gaussians;
- normal loss: aligns the shortest Gaussian axis with the GPIS field normal when scaling and rotation are available;
- density-control hooks: optional low-confidence pruning and optional densification-gradient boosting.

The adapter understands the common 3DGS `GaussianModel` attributes/properties:

```text
get_xyz
get_opacity
get_scaling
get_rotation
xyz_gradient_accum
prune_points(mask)
```

It can also be called with explicit tensors for custom trainers.

## Minimal integration into a 3DGS `train.py`

Add imports:

```python
from gpis_splatting.gpis_regularization import GPISRegularizationConfig
from gpis_splatting.gpis_3dgs_regularization import (
    GPIS3DGSDensityControlConfig,
    GPIS3DGSRegularizerConfig,
    GPIS3DGSTrainingRegularizer,
)
```

Instantiate after the 3DGS `gaussians` object exists:

```python
gpis_regularizer = GPIS3DGSTrainingRegularizer.from_model_path(
    args.gpis_model,
    loss_config=GPISRegularizationConfig(
        epsilon=args.gpis_epsilon,
        surface_weight=args.gpis_surface_weight,
        opacity_weight=args.gpis_opacity_weight,
        normal_weight=args.gpis_normal_weight,
        surface_confidence_floor=args.gpis_surface_confidence_floor,
    ),
    schedule_config=GPIS3DGSRegularizerConfig(
        start_iteration=args.gpis_start_iteration,
        stop_iteration=args.gpis_stop_iteration,
        ramp_iterations=args.gpis_ramp_iterations,
        interval=args.gpis_interval,
        max_regularized_gaussians=args.gpis_max_gaussians,
        batch_size=args.gpis_batch_size,
    ),
    density_config=GPIS3DGSDensityControlConfig(
        prune_start_iteration=args.gpis_prune_start_iteration,
        prune_interval=args.gpis_prune_interval,
        prune_confidence_threshold=args.gpis_prune_confidence_threshold,
        prune_opacity_threshold=args.gpis_prune_opacity_threshold,
        max_prune_fraction=args.gpis_max_prune_fraction,
        densification_boost_interval=args.gpis_densification_boost_interval,
        densification_confidence_threshold=args.gpis_densification_confidence_threshold,
        densification_min_distance_std=args.gpis_densification_min_distance_std,
        densification_gradient_boost=args.gpis_densification_gradient_boost,
    ),
)
```

Inside the training iteration, after the photometric loss has been computed and before `loss.backward()`:

```python
gpis_step = gpis_regularizer.compute(gaussians, iteration=iteration)
if gpis_step is not None:
    loss = loss + gpis_step.loss
```

For TensorBoard or WandB logging:

```python
if gpis_step is not None:
    for name, value in gpis_step.log_dict().items():
        tb_writer.add_scalar(name, float(value.detach().cpu()), iteration)
```

Before the usual 3DGS densification call, optionally boost the accumulated densification gradients for GPIS-uncertain Gaussians:

```python
if gpis_step is not None:
    gpis_regularizer.maybe_boost_densification_stats(gaussians, gpis_step, iteration=iteration)
```

At the same point where the trainer already performs opacity/size pruning, optionally apply the GPIS low-confidence prune mask:

```python
if gpis_step is not None:
    gpis_regularizer.maybe_prune(gaussians, gpis_step, iteration=iteration)
```

`maybe_prune` mutates the Gaussian model, so call it after the optimizer step and in the same section where the 3DGS trainer already mutates its Gaussian tensors.

## Conservative default knobs

Recommended first run:

```text
gpis_surface_weight=0.01
gpis_opacity_weight=0.001
gpis_normal_weight=0.001
gpis_start_iteration=500
gpis_ramp_iterations=1000
gpis_interval=1
gpis_max_gaussians=65536
gpis_prune_interval=0
gpis_densification_gradient_boost=0.0
```

Then enable density control after verifying that PSNR/SSIM do not regress during warmup:

```text
gpis_prune_start_iteration=3000
gpis_prune_interval=1000
gpis_prune_confidence_threshold=0.05
gpis_prune_opacity_threshold=0.01
gpis_max_prune_fraction=0.02
gpis_densification_boost_interval=100
gpis_densification_confidence_threshold=0.35
gpis_densification_min_distance_std=<scene-scale dependent>
gpis_densification_gradient_boost=0.25
```

The pruning hook defaults to requiring both low GPIS confidence and low opacity. That makes it a cleanup mechanism after the opacity loss has already pushed floaters down, rather than an aggressive geometry-only deleter.
