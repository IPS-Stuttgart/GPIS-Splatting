# Render scale and anti-aliasing diagnostics

`run_render_scale_diagnostics` connects controlled renderer-side scale tests with the existing render-consistency evaluator. It renders a trained 3DGS PLY at multiple camera resolutions, optionally includes the native `gsplat` antialiased rasterization mode, and compares all variants against the `scale1_classic` base render.

The command writes run-local renders and CSV/JSON/Markdown diagnostics under the selected output directory. It does not add trained models or rendered images to the repository.

## Example

```bash
run_render_scale_diagnostics \
  --scene ignatius_sparse \
  --prepared-root real_scenes \
  --input-ply outputs/ignatius/point_cloud/iteration_30000/point_cloud.ply \
  --method-name trained_3dgs_scale_aa \
  --split test \
  --render-scale-factor 0.5 \
  --render-scale-factor 1.0 \
  --render-scale-factor 2.0 \
  --include-gsplat-antialiased true \
  --output-resolution target \
  --aa-downsample-factor 2 \
  --aa-downsample-factor 4
```

## Outputs

For method `trained_3dgs_scale_aa`, the default output directory is:

```text
<scene-dir>/scale_aa_diagnostics/trained_3dgs_scale_aa/
```

Important artifacts are:

- `trained_3dgs_scale_aa_scale_aa_render_manifest.csv`
- `trained_3dgs_scale_aa_scale_aa_diagnostics_status.json`
- `trained_3dgs_scale_aa_scale_aa_diagnostics_report.md`
- `evaluations/trained_3dgs_scale_aa_test_scale_consistency.csv`
- `evaluations/trained_3dgs_scale_aa_test_antialiasing_consistency.csv`
- `evaluations/trained_3dgs_scale_aa_test_render_consistency_report.md`

Use `--output-resolution target` for direct comparisons at the original camera resolution. In that mode, `render-scale > 1` becomes a supersampling test and `render-scale < 1` becomes a low-resolution robustness test.
