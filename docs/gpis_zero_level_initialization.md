# GPIS Zero-Level Splat Initialization

`initialize_gpis_zero_level_splats` builds a new splat cloud directly from a fitted GPIS field. It samples candidate points, projects them to the posterior mean zero level, filters them by GPIS posterior confidence, and writes anisotropic Gaussian fields aligned to the GPIS normal.

```powershell
initialize_gpis_zero_level_splats `
  --scene ignatius_tnt64 `
  --model-path real_gpis_model.npz `
  --seed-splats-path real_splats.npz `
  --output-splats gpis_zero_level_splats.npz `
  --output-ply gpis_zero_level_point_cloud.ply `
  --num-candidates 80000 `
  --target-count 20000 `
  --surface-band 0.03 `
  --tangent-scale 0.025 `
  --normal-scale 0.006
```

The output `.npz` remains compatible with existing internal splat workflows. In addition to `centers`, `colors`, `tau`, `sigma`, and `is_surface`, it stores:

- `normals`: normalized GPIS posterior mean gradients at initialized centers.
- `scales`: local Gaussian axes `[tangent_1, tangent_2, normal]` in world units.
- `rotations`: normalized quaternions `[w, x, y, z]` mapping local Gaussian axes to world axes.
- `covariances`: full world-space covariance matrices `R diag(scales^2) R^T`.
- `confidence`: GPIS surface-band probability used as the initialized splat confidence gate.

When `--output-ply` is passed, the command also writes a binary little-endian 3DGS-style PLY with `scale_0..2` and `rot_0..3` fields. Scales are stored in the logarithmic convention used by standard 3DGS point-cloud PLYs, and colors are written as SH DC coefficients.

The initializer is intentionally conservative by default: most candidates are sampled near reference seed splats, then projected to the implicit surface. Increase `--surface-seed-fraction` for sparse reconstructions where COLMAP points are reliable, or decrease it to let the GPIS field hallucinate more zero-level coverage inside the inferred bounds. Use `--max-distance-std` to reject high-uncertainty zero-level candidates when the GPIS posterior is underconstrained.
