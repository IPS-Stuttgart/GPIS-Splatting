# Calibrated Confidence API

This API makes calibrated GPIS confidence a reusable artifact rather than a one-off CSV. It separates four concerns:

1. deterministic feature extraction from GPIS posterior and splat diagnostics,
2. leakage-aware train/validation splitting,
3. calibrator serialization,
4. reliability diagnostics.

## CLI

```bash
fit_calibrated_confidence \
  --field-scores-path real_scenes/ignatius_tnt64/evaluations/method_gpis_field_scores.csv \
  --metadata-path real_scenes/ignatius_tnt64/evaluations/method_hard_negative_candidates.csv \
  --group-columns source_splat_index \
  --thresholds 0.02 0.05 0.1 \
  --method-name method_confidence_api
```

The command writes, next to the field-score CSV unless `--output-dir` is passed:

- `*_calibrated_confidence_model.json`, a reusable serialized calibrator bundle;
- `*_confidence_features.csv`, the exact feature table used for fitting;
- `*_confidence_split.csv`, train/validation assignment and split group keys;
- `*_confidence_summary.csv`, validation metrics for candidate calibrators;
- `*_calibrated_confidence.csv`, calibrated per-splat probabilities;
- `*_reliability_<threshold>.csv` and `*_reliability_<threshold>.png`;
- `*_confidence_api_status.json` and `*_confidence_api_report.md`.

## Leakage-free splits

For hard-negative candidates, pass `--metadata-path` and group by the source identity:

```bash
fit_calibrated_confidence \
  --field-scores-path evaluations/hard_negative_gpis_field_scores.csv \
  --metadata-path evaluations/hard_negative_candidates.csv \
  --group-columns source_splat_index
```

Rows that share `source_splat_index` stay on the same side of the split. Random candidates with `source_splat_index == -1` are treated as row-level groups so they do not collapse into one large validation group. If no grouping column is available, the API falls back to a deterministic row-level split.

Spatial grouping is also available:

```bash
fit_calibrated_confidence \
  --field-scores-path evaluations/method_gpis_field_scores.csv \
  --spatial-cell-size 0.05 \
  --coordinate-columns query_x query_y query_z
```

## Feature extraction

The default extractor adds derived GPIS features such as `abs_mu`, `abs_signed_distance`, `distance_snr`, `abs_mu_over_sigma`, and `score_gpis_surface_likelihood`. It excludes obvious leakage columns such as `nearest_gt_distance`, `within_*`, `label*`, `gt_*`, and identity/group columns.

Use explicit features only when needed:

```bash
fit_calibrated_confidence \
  --field-scores-path evaluations/method_gpis_field_scores.csv \
  --feature-columns abs_signed_distance distance_std score_current_gate
```

Explicit label-like features are rejected unless `--allow-label-like-features` is set, which should only be used for deliberate leakage baselines.

## Python API

```python
from gpis_splatting.calibrated_confidence_api import (
    ConfidenceFitConfig,
    ConfidenceSplitConfig,
    run_calibrated_confidence_fit,
)

result = run_calibrated_confidence_fit(
    field_scores_path="evaluations/method_gpis_field_scores.csv",
    method_name="method_confidence_api",
    config=ConfidenceFitConfig(
        split_config=ConfidenceSplitConfig(group_columns=("source_splat_index",)),
    ),
)
```

The serialized JSON model can be applied later with `apply_calibrated_confidence_model` or the existing `apply_calibrated_confidence` CLI.
