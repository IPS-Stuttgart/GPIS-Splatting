# Large-scale GPIS scoring

This path scores large 3DGS Gaussian-center arrays with an inducing-point GPIS posterior and bounded accelerator memory.

## What changed

- `score_large_scale_gpis()` streams query centers in chunks and copies outputs to CPU by default.
- Inducing-point GPIS inference can run on CUDA/float32 while preserving the existing CPU/float64 fitting path.
- The inducing gradient is computed with matrix products instead of allocating a `batch x num_inducing x 3` tensor.
- The `score_large_scale_gpis` CLI reads centers from an NPZ file and writes `gate`, `mean`, `variance`, `gradient`, `distance`, and `distance_std` arrays.
- Trained-3DGS result export now requires full gate coverage by default. Use `--max-pred-points 0` in `run_trained_3dgs_gpis_experiment`, or pass a full-length `--gate-path` produced by this large-scale scorer.

## Score one million Gaussian centers

```bash
score_large_scale_gpis \
  --backend-model experiments/real_gpis_backend.npz \
  --points-npz experiments/trained_3dgs_splats.npz \
  --points-key centers \
  --output experiments/gpis_large_scale_scores.npz \
  --stats-json experiments/gpis_large_scale_scores.json \
  --epsilon 0.08 \
  --prediction-device cuda \
  --prediction-dtype float32 \
  --output-device cpu \
  --memory-budget-mib 1024
```

Use `--gate-only` when only the GPIS gate is needed for filtering/pruning and full diagnostics are not required.

## Use full large-scale gates in the trained-3DGS experiment

```bash
run_trained_3dgs_gpis_experiment \
  --scene-dir real_scenes/Truck \
  --trained-ply-path outputs/truck/point_cloud/iteration_30000/point_cloud.ply \
  --gate-path experiments/gpis_large_scale_scores.npz \
  --max-pred-points 0
```

The gate file must contain exactly one `gate` value per Gaussian in the trained PLY. Calibration gates with fallback-filled missing entries are rejected unless `--allow-partial-gpis-scores true` is explicitly supplied for diagnostics-only runs.

## CPU smoke run

```bash
score_large_scale_gpis \
  --backend-model experiments/real_gpis_backend.npz \
  --points-npz experiments/trained_3dgs_splats.npz \
  --points-key centers \
  --output experiments/gpis_scores_gate_only.npz \
  --prediction-device cpu \
  --prediction-dtype float32 \
  --gate-only
```
