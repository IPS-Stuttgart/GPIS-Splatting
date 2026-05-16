# Faithful real-scene 3DGS rendering

`render_real_splats` now has two explicit backends:

- `proxy`: the legacy CPU diagnostic renderer for `real_splats.npz` files. It is useful for sparse debugging, but it collapses splats to isotropic screen-space kernels and should not be used for photometric claims about trained 3DGS models.
- `gsplat`: a faithful trained-3DGS path that renders the original `point_cloud.ply` with anisotropic scales, rotations, opacity, and spherical-harmonic color coefficients when they are present.

Example:

```bash
render_real_splats \
  --scene-dir real_scenes/garden \
  --renderer-backend gsplat \
  --input-ply outputs/trained_model/point_cloud/iteration_30000/point_cloud.ply \
  --gate-path real_scenes/garden/calibrated_primary_confidence_gate.npz \
  --method-name gpis_confidence_3dgs_gsplat \
  --split test \
  --gsplat-color-mode auto \
  --gsplat-sh-degree auto \
  --gsplat-strict-3dgs-fidelity true \
  --gsplat-device cuda:0
```

When a GPIS or external gate is active, the backend writes a `gated_point_cloud.ply` in the render directory. Only opacity is changed, by scaling alpha values; the remaining 3DGS PLY fields are preserved.

The output directory contains `real_render_report.json` and the underlying `gsplat_render_report.json`. Use the `gsplat` backend for PSNR/SSIM/LPIPS evaluation of trained 3DGS models.
