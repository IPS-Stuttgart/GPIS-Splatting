# Scale-robust 3DGS rendering experiments

This workflow evaluates whether GPIS-gated or GPIS-regularized trained 3DGS variants remain stable under changes in test-time image scale and rasterizer anti-aliasing.

It is intentionally an experiment harness rather than a full Mip-Splatting reimplementation. The command keeps the trained Gaussian field fixed, renders the same camera views at several image scales, compares `classic` and `antialiased` gsplat rasterization, resizes the target image to the rendered size, and reports PSNR/SSIM/LPIPS deltas.

## Single trained PLY

```bash
run_scale_robust_rendering_experiment \
  --input-ply outputs/model/point_cloud/iteration_30000/point_cloud.ply \
  --scene-dir real_scenes/garden \
  --output-dir real_scenes/garden/evaluations/scale_robust_baseline \
  --method-name baseline_scale_robust \
  --split test \
  --scales 0.5,1.0,2.0 \
  --rasterize-modes classic,antialiased \
  --device cuda:0
```

## GPIS-gated variants

```bash
run_scale_robust_rendering_experiment \
  --manifest-path real_scenes/garden/3dgs_variants/gpis_confidence_3dgs_3dgs_variant_manifest.csv \
  --scene-dir real_scenes/garden \
  --output-dir real_scenes/garden/evaluations/gpis_scale_robust \
  --method-name gpis_confidence_3dgs_scale_robust \
  --split test \
  --scales 0.5,1.0,2.0 \
  --rasterize-modes classic,antialiased \
  --color-mode auto \
  --sh-degree auto \
  --strict-3dgs-fidelity true \
  --device cuda:0
```

## Outputs

The command writes:

```text
<output-dir>/<method>_scale_robust_render_manifest.csv
<output-dir>/<method>_scale_robust_image_metrics.csv
<output-dir>/<method>_scale_robust_summary.csv
<output-dir>/<method>_scale_robust_status.json
<output-dir>/<method>_scale_robust_report.md
<output-dir>/renders/<rasterize_mode>/<scale>/<variant>/<split>/ours_<iteration>/renders/*.png
```

The summary compares each variant/rasterizer/scale cell and reports deltas versus the matching `classic` rasterizer cell. A useful paper-grade matrix is:

```text
variants:        baseline, gate_scaled, calibrated, thresholded, GPIS-regularized
scales:          0.5, 1.0, 2.0
rasterize modes: classic, antialiased
metrics:         PSNR, SSIM, LPIPS, Gaussian count, retention fraction
```
