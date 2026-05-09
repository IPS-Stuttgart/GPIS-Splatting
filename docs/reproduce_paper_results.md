# Reproduce Evaluation Results

This guide gives deterministic commands for producing the repository's synthetic, real-scene, calibrated-confidence, geometry, and trained-3DGS artifacts. Paths follow the PowerShell style used in the README; replace backticks with backslashes on POSIX shells.

## 1. Environment

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m ruff check .
python -m pytest -q
```

Record the commit before running experiments:

```powershell
git rev-parse HEAD
```

## 2. Synthetic CI smoke result

Use this for pull requests and for checking that report generation works:

```powershell
run_evaluation `
  --preset-config configs/evaluation/synthetic_ci.json `
  --experiment-name evaluation_synthetic_ci

write_reproducibility_report `
  --experiment-root experiments/evaluation_synthetic_ci `
  --config configs/evaluation/synthetic_ci.json `
  --run-command "run_evaluation --preset-config configs/evaluation/synthetic_ci.json --experiment-name evaluation_synthetic_ci"
```

Expected key artifacts:

```text
experiments/evaluation_synthetic_ci/evaluation_config.json
experiments/evaluation_synthetic_ci/evaluation_checks.csv
experiments/evaluation_synthetic_ci/evaluation_status.json
experiments/evaluation_synthetic_ci/evaluation_report.md
experiments/evaluation_synthetic_ci/reproducibility_report.md
```

## 3. Synthetic quick benchmark

Use this as the default synthetic comparison across representative shapes:

```powershell
run_evaluation `
  --preset-config configs/evaluation/synthetic_quick.json `
  --experiment-name evaluation_synthetic_quick `
  --benchmark-target benchmarks/mipnerf360_sparse_12view.json

write_reproducibility_report `
  --experiment-root experiments/evaluation_synthetic_quick `
  --config configs/evaluation/synthetic_quick.json `
  --run-command "run_evaluation --preset-config configs/evaluation/synthetic_quick.json --experiment-name evaluation_synthetic_quick --benchmark-target benchmarks/mipnerf360_sparse_12view.json"
```

Compare `evaluation_checks.csv`, `summary/ablation_summary.csv`, and `summary/ablation_winners.csv` across commits.

## 4. Real-scene laptop smoke run

The reduced Nerfstudio poster workflow is small enough for local smoke tests:

```powershell
run_real_evaluation `
  --scene poster8_smoke `
  --max-download-images 24 `
  --max-points 800 `
  --max-train-points 600 `
  --max-frames 4 `
  --epsilons 0.08 0.16 0.24 `
  --gate-floors 0.0 0.25 `
  --splat-sigmas 0.015 0.025 0.04

write_reproducibility_report `
  --experiment-root real_scenes/poster8_smoke/evaluations `
  --config configs/real/poster8_smoke.json `
  --run-command "run_real_evaluation --scene poster8_smoke --max-download-images 24 --max-points 800 --max-train-points 600 --max-frames 4 --epsilons 0.08 0.16 0.24 --gate-floors 0.0 0.25 --splat-sigmas 0.015 0.025 0.04"
```

Use this run to verify camera normalization, GPIS bootstrap, dense GPIS fitting, gated rendering, and report artifacts.

## 5. Tanks-and-Temples geometry and confidence

Prepare the scene:

```powershell
download_tanks_temples_scene `
  --scene Ignatius `
  --output-root real_scenes/_downloads `
  --max-images 64

prepare_tanks_temples_scene `
  --input-dir real_scenes/_downloads/tanks_temples/Ignatius `
  --prepared-scene ignatius_tnt64 `
  --train-view-count 12
```

Run geometry and gate-quality diagnostics:

```powershell
run_tanks_temples_gate_sweep `
  --scene ignatius_tnt64 `
  --splats-path real20k_sigma_0p04_splats.npz `
  --model-path real20k_sigma_0p04_gpis_model.npz `
  --thresholds 0.02 0.05 0.1 `
  --gate-thresholds 0.02 0.05 0.1 0.2 0.35 0.5

diagnose_tanks_temples_gpis_field_scores `
  --scene ignatius_tnt64 `
  --splats-path real20k_sigma_0p04_splats.npz `
  --model-path real20k_sigma_0p04_gpis_model.npz `
  --thresholds 0.02 0.05 0.1 `
  --score-lambdas 0.25 0.5 1.0

run_tanks_temples_hard_negative_calibration `
  --scene ignatius_tnt64 `
  --splats-path real20k_sigma_0p04_splats.npz `
  --model-path real20k_sigma_0p04_gpis_model.npz `
  --method-name ignatius_hard_negative_v1 `
  --thresholds 0.02 0.05 0.1
```

Write a reproducibility report over the scene evaluation directory:

```powershell
write_reproducibility_report `
  --experiment-root real_scenes/ignatius_tnt64/evaluations `
  --config configs/tanks_temples/ignatius_geometry.json
```

The key comparison artifacts are the gate sweep CSV, GPIS field-score CSV, calibrated-confidence model JSON, primary confidence CSV, gate-compatible NPZ files, and geometry metrics for filtered/tau-scaled variants.

## 6. Trained 3DGS photometric comparison

Export a prepared scene to the standard 3DGS COLMAP layout, train an external 3DGS model, then score and export GPIS variants:

```powershell
export_prepared_scene_to_colmap_3dgs `
  --scene ignatius_tnt64 `
  --output-dir C:\runs\3dgs\ignatius_gpis_scene `
  --split train `
  --max-points 100000

convert_3dgs_ply_to_splats `
  --input-ply C:\runs\3dgs\ignatius\point_cloud\iteration_30000\point_cloud.ply `
  --output-splats real_scenes\ignatius_tnt64\trained_3dgs_splats.npz

diagnose_tanks_temples_gpis_field_scores `
  --scene ignatius_tnt64 `
  --splats-path trained_3dgs_splats.npz `
  --model-path real20k_sigma_0p04_gpis_model.npz `
  --method-name trained_3dgs `
  --max-pred-points 0

calibrate_gpis_splat_scores `
  --field-scores-path real_scenes\ignatius_tnt64\evaluations\trained_3dgs_gpis_field_scores.csv `
  --gate-count 123456

export_3dgs_gpis_variants `
  --input-ply C:\runs\3dgs\ignatius\point_cloud\iteration_30000\point_cloud.ply `
  --gate-path real_scenes\ignatius_tnt64\evaluations\trained_3dgs_calibrated_gate_0p05.npz `
  --output-dir C:\runs\3dgs\ignatius_gpis_variants `
  --iteration 30000
```

Render the exported model directories with the external 3DGS renderer, then evaluate:

```powershell
evaluate_3dgs_variant_renders `
  --manifest-path real_scenes\ignatius_tnt64\trained_3dgs_variants\trained_3dgs\trained_3dgs_3dgs_variant_manifest.csv `
  --scene ignatius_tnt64 `
  --predictions-root real_scenes\ignatius_tnt64\renders\trained_3dgs_variants `
  --prediction-subdir test\ours_30000\renders `
  --method-name trained_3dgs `
  --split test
```

## 7. Result-bundle checklist

For every reported experiment, include:

```text
*_status.json
*_report.md
*_metrics.csv or *_comparison.csv
config JSON used for the run
reproducibility_report.md
commit SHA
```

For trained 3DGS runs, also include the variant manifest and the exact external renderer command template used to produce predictions.
