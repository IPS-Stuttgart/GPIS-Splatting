# A-F Experiment Matrix

`python -m gpis_splatting.cli.run_experiment_matrix` writes one reproducible comparison table for the GPIS/3DGS cases that should be tracked across scenes and commits.

| Case | Method | Purpose |
| --- | --- | --- |
| A | Plain trained 3DGS | Baseline trained 3DGS without GPIS gating, calibration, or GPIS training regularization. |
| B | Raw GPIS gate post-hoc | Tests whether the analytic GPIS zero-band/surface gate is useful without learned calibration. |
| C | Calibrated GPIS confidence post-hoc | Tests calibrated GPIS posterior-field confidence as the primary post-hoc gate. |
| D | Calibrated pruning/refinement | Tests thresholded or tau-scaled confidence variants, including retention/quality trade-offs. |
| E | GPIS training-time regularizer | Tests losses from `GPIS3DGSTrainingRegularizer` inside an external 3DGS trainer. |
| F | Regularizer plus calibrated confidence | Tests the full combination of GPIS-guided training and calibrated post-hoc confidence/filtering. |

The command can be run early as a planning step. Missing cases are kept as explicit placeholders instead of silently disappearing from the report:

```powershell
python -m gpis_splatting.cli.run_experiment_matrix `
  --scene ignatius_tnt64 `
  --matrix-name ignatius_af_matrix
```

This writes the manifest, summary, checks, status JSON, config JSON, and Markdown report under `real_scenes/<scene>/evaluations/<matrix-name>/`.

## Aggregating existing artifacts

Pass any subset of the existing workflow outputs. The matrix aggregates available metrics, picks the primary geometry threshold, chooses the best matching gate-threshold row where appropriate, and computes deltas against case A.

```powershell
python -m gpis_splatting.cli.run_experiment_matrix `
  --scene ignatius_tnt64 `
  --matrix-name ignatius_af_matrix `
  --primary-geometry-threshold 0.05 `
  --trained-3dgs-render-comparison real_scenes\ignatius_tnt64\evaluations\trained_3dgs_3dgs_render_comparison.csv `
  --raw-gate-sweep real_scenes\ignatius_tnt64\evaluations\real20k_sigma_0p04_gate_sweep.csv `
  --calibrated-filtering-comparison real_scenes\ignatius_tnt64\evaluations\ignatius_confidence_filter_v1_splat_filtering_comparison.csv
```

Use `--fail-on-missing` when the report should fail unless all six cases have at least one matched artifact row.

## Artifact roles

The supported artifact roles are:

| Role | Expected source |
| --- | --- |
| `trained_3dgs_render_comparison` | CSV from `evaluate_3dgs_variant_renders` for the plain trained-3DGS baseline. |
| `trained_3dgs_geometry_summary` | Optional geometry summary CSV for the trained-3DGS baseline. |
| `raw_gate_sweep` | CSV from `run_tanks_temples_gate_sweep`. |
| `calibrated_gate_sweep` | Optional gate-sweep CSV using a calibrated confidence gate. |
| `calibrated_confidence_metrics` | Optional calibrator validation metrics CSV. |
| `calibrated_filtering_comparison` | CSV from `run_tanks_temples_calibrated_splat_filtering`. |
| `regularized_3dgs_render_comparison` | Render comparison CSV for 3DGS trained with the GPIS regularizer. |
| `regularized_geometry_summary` | Optional geometry summary CSV for the regularized 3DGS run. |
| `regularized_calibrated_render_comparison` | Render comparison CSV for regularized 3DGS plus calibrated confidence. |
| `regularized_calibrated_filtering_comparison` | Filtering comparison CSV for regularized 3DGS plus calibrated confidence. |

You can also pass artifact paths generically:

```powershell
python -m gpis_splatting.cli.run_experiment_matrix `
  --artifact raw_gate_sweep=real_scenes\ignatius_tnt64\evaluations\raw_gate_sweep.csv `
  --artifact regularized_3dgs_render_comparison=C:\runs\gpis_regularized\comparison.csv
```

## Output files

For a matrix named `ignatius_af_matrix`, the command writes:

- `ignatius_af_matrix_manifest.csv`: planned A-F rows, hypotheses, expected artifacts, and command hints.
- `ignatius_af_matrix_summary.csv`: normalized metrics by case, source artifacts, and deltas against the baseline case.
- `ignatius_af_matrix_checks.csv`: coverage and sanity checks.
- `ignatius_af_matrix_status.json`: machine-readable status for CI and self-hosted workflows.
- `ignatius_af_matrix_config.json`: exact input paths and threshold settings.
- `ignatius_af_matrix_report.md`: compact human-readable report.

The normalized summary includes geometry metrics (`precision`, `recall`, `f_score`, `chamfer_l1`, `chamfer_l2`), render metrics (`mean_psnr`, `mean_ssim`, `mean_lpips_vgg`), retention metrics, gate thresholds, and improvement-style deltas against case A.
