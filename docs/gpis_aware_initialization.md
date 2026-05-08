# GPIS-Aware Gaussian Initialization

This workflow converts a fitted real-scene GPIS posterior into an initial anisotropic Gaussian cloud.

The initializer starts from sparse seed splats, creates jittered proposals, projects each proposal onto the GPIS zero level-set, ranks candidates by posterior surface confidence and uncertainty, and writes both internal splats and a 3DGS-compatible PLY.

```bash
bootstrap_real_gpis \
  --scene-dir real_scenes/ignatius \
  --point-source ply \
  --point-path reconstruction/Ignatius.ply \
  --output-prefix real

fit_real_gpis \
  --scene-dir real_scenes/ignatius \
  --samples-path real_samples.npz \
  --output-model real_gpis_model.npz

initialize_gpis_splats \
  --scene-dir real_scenes/ignatius \
  --model-path real_gpis_model.npz \
  --splats-path real_splats.npz \
  --output-prefix gpis_init \
  --target-count 50000 \
  --proposals-per-seed 4 \
  --epsilon 0.08 \
  --min-view-count 1 \
  --min-separation 0.01
```

Outputs:

- `gpis_init_gaussians.npz`: centers, RGB colors, opacity, anisotropic scales, rotations, normals, confidence, and selected source indices.
- `gpis_init_splats.npz`: an isotropic internal `SplatCloud` view for the CPU renderer.
- `gpis_init_3dgs.ply`: 3DGS-style Gaussian PLY with `f_dc_*`, zero `f_rest_*`, logit opacity, log-scales, and scalar-first quaternions.
- `gpis_init_field_scores.csv`: all projected candidates with GPIS posterior features, surface-band score, view count, selected flag, and rank.
- `gpis_init_status.json` and `gpis_init_report.md`: reproducibility metadata.

A calibrated confidence bundle can be used for opacity and ranking instead of the analytic variance-penalized GPIS band score:

```bash
initialize_gpis_splats \
  --scene-dir real_scenes/ignatius \
  --confidence-model-path evaluations/ignatius_calibrated_confidence_model.json \
  --confidence-threshold 0.05
```

Scale convention: `scale_0` is aligned with the GPIS normal and should usually be smaller than `scale_1` and `scale_2`, which span the tangent plane. The PLY writer stores scales in log-space because trained 3DGS PLYs use log scale parameters.
