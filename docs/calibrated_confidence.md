# Calibrated GPIS Confidence

The first Ignatius preliminary runs showed a useful distinction:

- the analytic GPIS zero-band gate is a weak splat-quality ranking signal on sparse real splats;
- calibrated GPIS posterior-field features are much stronger on mixed source and hard-negative candidates.

Accordingly, the real-data confidence interface should treat the analytic `p0,epsilon` gate as a diagnostic baseline and use calibrated GPIS posterior features as the primary splat-confidence signal.

## Workflow

The main CLI is:

```bash
run_tanks_temples_calibrated_confidence \
  --scene ignatius_tnt64 \
  --splats-path preliminary_splats.npz \
  --model-path preliminary_gpis_model.npz \
  --method-name calibrated_confidence \
  --calibration-threshold 0.05 \
  --thresholds 0.02 0.05 0.1 \
  --gate-thresholds 0.1 0.25 0.5 0.75 \
  --render-max-frames 4
```

It performs three steps:

1. Generate hard-negative splat candidates and score them with GPIS posterior field diagnostics.
2. Calibrate those GPIS-derived features into probabilities of geometric correctness.
3. Use the calibrated probability as a gate-compatible confidence signal for tau-scaled and filtered splat variants.

## Expected artifacts

The workflow writes artifacts under `real_scenes/<scene>/evaluations/`, including:

- `*_hard_negative_workflow_report.md`
- `*_hard_negative_calibrated_calibration_summary.csv`
- `*_hard_negative_calibrated_gate_<threshold>.npz`
- `*_filtering_splat_filtering_comparison.csv`
- `*_calibrated_confidence_report.md`
- `*_calibrated_confidence_status.json`

## Interpretation

The key question is no longer whether the raw analytic zero-band probability alone ranks splat quality well. It often does not. The relevant question is whether GPIS posterior information, after calibration, improves geometric filtering or optical-thickness scaling without unacceptable render degradation.

For proposal evidence, report the raw gate as a baseline and the calibrated GPIS-field confidence as the primary result.
