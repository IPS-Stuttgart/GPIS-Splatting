# Render consistency evaluation

`evaluate_render_consistency` adds diagnostics that complement PSNR/SSIM render evaluation and the existing real-render audit.

It evaluates two failure modes that are easy to miss with per-frame image metrics:

1. **Adjacent-view consistency**: consecutive frames in a prepared-scene split are compared in prediction space and target space. The report highlights excess prediction deltas above the target-image delta, which is a simple proxy for popping, unstable depth ordering, and view-dependent splat artifacts.
2. **Scale / anti-aliasing consistency**: optional render directories from different resolution or anti-aliasing settings are compared against a base render directory after resizing to a common resolution. This exposes renders that look good at one resolution but change materially when projected or filtered differently.

The command writes CSV, JSON, and Markdown outputs under `<scene-dir>/evaluations` by default.

```bash
evaluate_render_consistency \
  --scene ignatius_sparse \
  --prepared-root real_scenes \
  --predictions-dir real_scenes/ignatius_sparse/renders/real_gpis_gate \
  --method-name real_gpis_gate \
  --split test \
  --scale-predictions-dir lowres=real_scenes/ignatius_sparse/renders/real_gpis_gate_lowres \
  --scale-predictions-dir highaa=real_scenes/ignatius_sparse/renders/real_gpis_gate_highaa
```

## Outputs

For method `real_gpis_gate` and split `test`, the tool writes:

- `real_gpis_gate_test_temporal_consistency.csv`
- `real_gpis_gate_test_scale_consistency.csv`
- `real_gpis_gate_test_render_consistency_summary.csv`
- `real_gpis_gate_test_render_consistency_status.json`
- `real_gpis_gate_test_render_consistency_report.md`

## Key metrics

Temporal rows include:

- `prediction_delta_mad`: mean absolute RGB change between adjacent predictions.
- `target_delta_mad`: mean absolute RGB change between adjacent ground-truth frames.
- `delta_mad_excess`: `prediction_delta_mad - target_delta_mad`.
- `edge_delta_mad_excess`: excess adjacent-frame change in simple image-gradient magnitude.
- `temporal_instability_score`: positive excess RGB change plus positive excess edge change.

Scale rows include:

- `scale_psnr` / `scale_ssim`: similarity between the base prediction and the scale/AA variant.
- `scale_mad`: mean absolute RGB difference between base and variant.
- `scale_edge_mad`: edge-map difference between base and variant.
- `scale_instability_score`: `scale_mad + scale_edge_mad`.

These metrics are diagnostic rather than leaderboard metrics. They are intended to find popping, aliasing, and resolution-sensitive splats before deeper renderer-level analysis.
