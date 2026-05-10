# gsplat 3DGS renderer adapter

This adapter renders trained 3D Gaussian Splatting `point_cloud.ply` files directly from prepared-scene cameras with the optional `gsplat` backend. It complements the existing external-3DGS workflow: `export_3dgs_gpis_variants` writes baseline, opacity-scaled, and thresholded PLY variants; `render_3dgs_with_gsplat` renders those variants without invoking the original Graphdeco renderer.

The fidelity path uses Graphdeco-style anisotropic scales, rotations, opacity logits, and spherical-harmonic color coefficients when they are present in the PLY.

## Install

The base package remains CPU/install friendly. Install `gsplat` only when rendering trained 3DGS models:

```bash
pip install -e ".[gsplat]"
```

`gsplat` usually requires a compatible PyTorch/CUDA environment. On a CPU-only machine the adapter can still be imported, but rendering will fail unless a compatible backend is installed or a test rasterizer is injected.

## Render one trained 3DGS PLY

```bash
render_3dgs_with_gsplat \
  --input-ply outputs/trained_model/point_cloud/iteration_30000/point_cloud.ply \
  --scene-dir real_scenes/garden \
  --output-dir real_scenes/garden/renders/gsplat_3dgs \
  --split test \
  --projection-convention auto \
  --color-mode auto \
  --sh-degree auto \
  --strict-3dgs-fidelity true \
  --device cuda:0
```

The command writes RGB images plus `gsplat_render_report.json` in the output directory. The report records whether the effective color path was `sh` or `rgb` and which SH degree was used.

## Color modes

- `--color-mode auto` uses SH rendering when `f_dc_*` and `f_rest_*` coefficients are present; otherwise it falls back to RGB/DC colors.
- `--color-mode sh` requires SH-compatible PLY fields when `--strict-3dgs-fidelity true`.
- `--color-mode rgb` forces the fallback path using RGB properties or DC-color conversion.
- `--sh-degree auto` uses the highest complete SH degree stored in the PLY. A numeric degree such as `1`, `2`, or `3` can be passed to restrict the active basis.

For Graphdeco PLYs, `f_dc_0..2` are loaded as the degree-0 coefficients. The `f_rest_*` block is reshaped from Graphdeco layout into the `[N, K, 3]` coefficient layout expected by `gsplat.rasterization`, and the selected `sh_degree` is passed to the rasterizer.

## Render GPIS-gated 3DGS variants

```bash
export_3dgs_gpis_variants \
  --input-ply outputs/trained_model/point_cloud/iteration_30000/point_cloud.ply \
  --gate-path real_scenes/garden/calibrated_primary_confidence_gate.npz \
  --output-dir real_scenes/garden/3dgs_variants \
  --method-name gpis_confidence_3dgs \
  --iteration 30000

render_3dgs_with_gsplat \
  --manifest-path real_scenes/garden/3dgs_variants/gpis_confidence_3dgs_3dgs_variant_manifest.csv \
  --scene-dir real_scenes/garden \
  --output-dir real_scenes/garden/renders/gsplat_3dgs_variants \
  --method-name gpis_confidence_3dgs_gsplat \
  --split test \
  --color-mode auto \
  --sh-degree auto \
  --device cuda:0
```

Manifest rendering writes the prediction layout expected by `evaluate_3dgs_variant_renders`:

```text
<output-dir>/<variant>/<split>/ours_<iteration>/renders/<frame>.png
```

Evaluate the rendered variants with:

```bash
evaluate_3dgs_variant_renders \
  --manifest-path real_scenes/garden/3dgs_variants/gpis_confidence_3dgs_3dgs_variant_manifest.csv \
  --scene-dir real_scenes/garden \
  --predictions-root real_scenes/garden/renders/gsplat_3dgs_variants \
  --prediction-subdir test/ours_30000/renders \
  --method-name gpis_confidence_3dgs_gsplat
```

## Baseline photometry validation

Before using GPIS-gated variants for photometric claims, validate that the `gsplat` baseline agrees with a trusted reference renderer when reference renders are available:

```bash
validate_3dgs_baseline_photometry \
  --input-ply outputs/trained_model/point_cloud/iteration_30000/point_cloud.ply \
  --scene-dir real_scenes/garden \
  --reference-renders-dir outputs/reference_3dgs/test/ours_30000/renders \
  --color-mode auto \
  --sh-degree auto \
  --strict-3dgs-fidelity true \
  --device cuda:0
```

Without `--reference-renders-dir`, the command still reports GT PSNR/SSIM/LPIPS for the rendered images, but it cannot pass the reference-render agreement gate.

## Camera conventions

Prepared COLMAP scenes use OpenCV camera coordinates and are passed directly to `gsplat`. Prepared NeRF-style `transforms.json` scenes use OpenGL camera coordinates; the adapter converts them to OpenCV coordinates with `diag(1, -1, -1, 1)` before rasterization. `--projection-convention auto` follows the same scene metadata convention as the existing CPU real-splat renderer.

## PLY interpretation

The adapter reads standard trained-3DGS PLY properties:

- `x`, `y`, `z` -> Gaussian means.
- `scale_0`, `scale_1`, `scale_2` -> exponentiated anisotropic scales.
- `rot_0`, `rot_1`, `rot_2`, `rot_3` -> normalized `wxyz` quaternions.
- `opacity` -> alpha via either logit or linear decoding.
- `f_dc_*` and `f_rest_*` -> full SH color rendering when enabled.
- RGB properties or `f_dc_*` only -> RGB/DC fallback when SH coefficients are unavailable or disabled.
