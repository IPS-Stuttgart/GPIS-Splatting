# GPIS backend benchmark

`benchmark_gpis_backends` compares the dense, local-exact, and inducing-point GPIS backends on the same deterministic pseudo-SDF samples. It is intended to answer two practical questions before running large 3DGS experiments:

- how expensive is each backend for a requested number of observations and query points?
- how far do approximate backends deviate from dense exact predictions when a dense reference is still feasible?

## Quick smoke run

```powershell
benchmark_gpis_backends `
  --output-dir experiments/gpis_backend_benchmark_smoke `
  --benchmark-name smoke_backend_benchmark `
  --n-train 512 `
  --n-query 256 `
  --num-neighbors 64 `
  --num-inducing 128
```

Expected artifacts:

```text
experiments/gpis_backend_benchmark_smoke/smoke_backend_benchmark.csv
experiments/gpis_backend_benchmark_smoke/smoke_backend_benchmark_config.json
experiments/gpis_backend_benchmark_smoke/smoke_backend_benchmark_status.json
experiments/gpis_backend_benchmark_smoke/smoke_backend_benchmark_report.md
```

## Larger approximate-backend run

Dense exact GPIS is skipped by default when `n_train` is above `--skip-dense-over-points`. Dense-reference error columns are only populated when `n_train <= --max-dense-reference-points`.

```powershell
benchmark_gpis_backends `
  --output-dir experiments/gpis_backend_benchmark_large `
  --benchmark-name large_backend_benchmark `
  --backend local_exact `
  --backend inducing_points `
  --shape torus `
  --n-train 20000 `
  --n-query 5000 `
  --num-neighbors 96 `
  --num-inducing 1024 `
  --fit-batch-size 4096 `
  --batch-size 4096
```

## Reported columns

The CSV records fit time, prediction time, queries per second, tensor-backed model storage, effective training/inducing/neighbor counts, and RMSE/max-absolute deviations for mean, variance, gradient, and GPIS surface-band gate versus the dense reference when available.

Use this benchmark to choose a backend before running GPIS scoring on trained 3DGS Gaussian centers. For paper-style comparisons, include the benchmark CSV next to the scene's A-F experiment matrix so speed/accuracy trade-offs are documented rather than implicit.
