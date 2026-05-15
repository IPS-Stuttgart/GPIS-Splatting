# Geometry-Aware Splat Finalization

`initialize_gpis_splats` writes a rich `*_gaussians.npz` artifact containing centers, colors, opacity, anisotropic scales, rotations, normals, confidence, and field scores. The companion `finalize_geometry_aware_splats` command converts that rich artifact into an internal `SplatCloud` that preserves the geometry fields needed by later GPIS/3DGS analysis.

```bash
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

finalize_geometry_aware_splats \
  --gaussians-path real_scenes/ignatius/gpis_init_gaussians.npz \
  --output-splats real_scenes/ignatius/gpis_init_splats.npz
```

The finalized internal splat file keeps:

- `normals`: normalized GPIS posterior-gradient normals.
- `scales`: anisotropic Gaussian axes with the GPIS-normal axis first.
- `rotations`: scalar-first quaternions.
- `covariances`: world-space covariance matrices `R diag(scales^2) R^T`.
- `confidence`: GPIS or calibrated confidence used for ranking and opacity.

For legacy CPU rendering, the scalar `sigma` field is set to the tangent-plane footprint, i.e. the mean of `scale_1` and `scale_2`, rather than averaging in the deliberately thin normal axis. The `tau` field is Beer-Lambert optical thickness derived from alpha/opacity via `tau = -log(1 - alpha)`.

A sidecar `<output-splats-stem>_geometry_status.json` records the conventions and preserved fields.
