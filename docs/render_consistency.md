# Render consistency evaluation

`evaluate_render_consistency` adds diagnostics that complement PSNR/SSIM render evaluation and the existing real-render audit.

It evaluates three failure modes that are easy to miss with per-frame image metrics:

1. **Pose-aware adjacent-view consistency**: consecutive frames in a prepared-scene split are compared in prediction space and target space. The report highlights excess prediction deltas above the target-image delta, which is a simple proxy for popping, unstable depth ordering, and view-dependent splat artifacts. When camera poses are available, the rows also report camera-center translation, relative rotation, and view-motion-normalized instability. Optional view-motion filters let you restrict the report to small-baseline pairs.
2. **Scale / external anti-aliasing consistency**: optional render directories from different resolution or anti-aliasing settings are compared against a base render directory after resizing to a common resolution. This exposes renders that look good at one resolution but change materially when projected or filtered differently.
3. **Built-in anti-aliasing round-trip consistency**: each base prediction is low-pass filtered by an anti-aliased downsample and then upsampled back to the original resolution. The residual against the original image estimates high-frequency/aliasing sensitivity even when no extra render directory exists. The same round trip is computed on target images, and excess prediction residuals are reported separately from natural target-image high frequencies.

The command writes CSV, JSON, and Markdown outputs under `<scene-dir>/evaluations` by default.

```bash
evaluate_render_consistency \
  --scene ignatius_sparse \
  --prepared-root real_scenes \
  --predictions-dir real_scenes/ignatius_sparse/renders/real_gpis_gate \
  --method-name real_gpis_gate \
  --split test \
  --scale-predictions-dir lowres=real_scenes/ignatius_sparse/renders/real_gpis_gate_lowres \
  --scale-predictions-dir highaa=real_scenes/ignatius_sparse/renders/real_gpis_gate_highaa \
  --aa-downsample-factor 2 \
  --aa-downsample-factor 4 \
  --max-view-translation 0.05 \
  --max-view-rotation-deg 5
```

Use `--disable-aa-roundtrip true` to skip the built-in anti-aliasing round-trip diagnostics. Without explicit `--aa-downsample-factor` values, the command evaluates factor `2`.

## Outputs

For method `real_gpis_gate` and split `test`, the tool writes:

- `real_gpis_gate_test_temporal_consistency.csv`
- `real_gpis_gate_test_scale_consistency.csv`
- `real_gpis_gate_test_antialiasing_consistency.csv`
- `real_gpis_gate_test_render_consistency_summary.csv`
- `real_gpis_gate_test_render_consistency_status.json`
- `real_gpis_gate_test_render_consistency_report.md`

## Key metrics

Temporal/view rows include:

- `prediction_delta_mad`: mean absolute RGB change between adjacent predictions.
- `target_delta_mad`: mean absolute RGB change between adjacent ground-truth frames.
- `delta_mad_excess`: `prediction_delta_mad - target_delta_mad`.
- `edge_delta_mad_excess`: excess adjacent-frame change in simple image-gradient magnitude.
- `camera_translation_delta`: Euclidean camera-center distance between adjacent frames when `camera_to_world` or `world_to_camera` poses are available.
- `camera_rotation_delta_deg`: relative camera rotation angle in degrees.
- `view_motion_score`: `camera_translation_delta + camera_rotation_delta_rad`.
- `view_instability_score`: `temporal_instability_score / view_motion_score`.
- `temporal_instability_score`: positive excess RGB change plus positive excess edge change.

Scale rows include:

- `scale_psnr` / `scale_ssim`: similarity between the base prediction and the scale/AA variant.
- `scale_mad`: mean absolute RGB difference between base and variant.
- `scale_edge_mad`: edge-map difference between base and variant.
- `scale_instability_score`: `scale_mad + scale_edge_mad`.

Anti-aliasing rows include:

- `aa_downsample_factor`: low-pass downsample factor used for the round trip.
- `aa_psnr` / `aa_ssim`: similarity between the base prediction and its anti-aliased round-trip reconstruction.
- `aa_mad`: mean absolute RGB difference between base prediction and low-pass round-trip reconstruction.
- `aa_edge_mad`: edge-map difference between base prediction and low-pass round-trip reconstruction.
- `target_aa_mad` / `target_aa_edge_mad`: the same low-pass residuals on the target frame.
- `aa_mad_excess` / `aa_edge_mad_excess`: prediction residuals minus target residuals.
- `aa_instability_score`: positive excess RGB residual plus positive excess edge residual.

These metrics are diagnostic rather than leaderboard metrics. They are intended to find popping, aliasing, and resolution-sensitive splats before deeper renderer-level analysis.
