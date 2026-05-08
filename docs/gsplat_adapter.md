# gsplat 3DGS renderer adapter

This adapter renders trained 3D Gaussian Splatting `point_cloud.ply` files directly from prepared-scene cameras with the optional `gsplat` backend. It complements the existing external-3DGS workflow: `export_3dgs_gpis_variants` still writes baseline, opacity-scaled, and thresholded PLY variants; `render_3dgs_with_gsplat` can then render those variants without invoking the original Graphdeco renderer.

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
  --device cuda:0
```

The command writes RGB images plus `gsplat_render_report.json` in the output directory.

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

## Camera conventions

Prepared COLMAP scenes use OpenCV camera coordinates and are passed directly to `gsplat`. Prepared NeRF-style `transforms.json` scenes use OpenGL camera coordinates; the adapter converts them to OpenCV coordinates with `diag(1, -1, -1, 1)` before rasterization. `--projection-convention auto` follows the same scene metadata convention as the existing CPU real-splat renderer.

## PLY interpretation

The adapter reads standard trained-3DGS PLY properties:

- `x`, `y`, `z` -> Gaussian means.
- `scale_0`, `scale_1`, `scale_2` -> exponentiated anisotropic scales.
- `rot_0`, `rot_1`, `rot_2`, `rot_3` -> normalized `wxyz` quaternions.
- `opacity` -> alpha via either logit or linear decoding.
- `f_dc_*` -> RGB fallback matching the existing external-3DGS conversion path.
