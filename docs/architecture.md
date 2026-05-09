# Architecture

This document summarizes the GPIS-Splatting execution paths, artifact contracts, and extension points used by the synthetic, real-scene, Tanks-and-Temples, calibrated-confidence, and trained-3DGS workflows.

## System overview

```text
synthetic scene / prepared real scene / trained 3DGS PLY
        |
        v
surface, free-space, and optional behind-surface pseudo-SDF observations
        |
        v
GPIS backend -> posterior mean, variance, gradient, distance proxy, confidence features
        |
        +--> initialized or imported splats
        |        |
        |        +--> renderer / external 3DGS interop
        |        +--> geometry metrics and gate sweeps
        |        +--> calibrated confidence and hard-negative filtering
        |
        +--> reproducibility reports and command/config manifests
```

The repository keeps the GPIS, splat conversion, confidence calibration, diagnostics, and reproducibility layers explicit. CUDA training/rendering for full 3DGS remains an external interop path rather than a vendored renderer.

## Main package areas

```text
src/gpis_splatting/
  GPIS fitting, posterior queries, splat representation, rendering, diagnostics,
  confidence calibration, initialization, 3DGS interop, and reports.

src/gpis_splatting/cli/
  Console-entry wrappers for synthetic, real-scene, Tanks-and-Temples, confidence,
  trained-3DGS, and reproducibility workflows.

docs/
  Human-readable architecture and reproduction guidance.

configs/
  Declarative evaluation presets and command plans.

tests/
  Small deterministic tests for CLI/report/config behavior and core data contracts.
```

## Artifact contracts

### GPIS models

GPIS model files are immutable model snapshots. Changing backend, lengthscale, variance, noise, train-point subsampling, or pseudo-SDF construction should produce a new model artifact and a status JSON recording the settings.

### Splat files

Internal splat files are used for CPU rendering, GPIS scoring, geometry checks, calibrated filtering, and 3DGS PLY export. Filtering steps should preserve source indices or write an explicit manifest so confidence arrays can be mapped back to the source Gaussian set.

### Gate files

Gate files contain one confidence value per source splat or Gaussian. When a workflow filters candidates before scoring, unscored candidates should be filled with the configured missing value and that choice should be recorded in the status JSON.

### Reports

Every metric-producing workflow should write three artifact classes:

1. A machine-readable status JSON with pass/fail state, thresholds, selected settings, and paths.
2. One or more CSV files with per-frame, per-splat, per-threshold, per-variant, or per-check metrics.
3. A Markdown report that summarizes the command, artifacts, and major checks.

## Reproducibility configs

Evaluation preset JSON files under `configs/evaluation/` can be passed directly to:

```powershell
run_evaluation --preset-config configs/evaluation/synthetic_ci.json
```

Other config directories contain command plans for real-scene, Tanks-and-Temples, and trained-3DGS workflows. After a run, call:

```powershell
write_reproducibility_report --experiment-root <run-dir> --config <config.json>
```

This binds the config hash, command, commit, status JSON, CSV summaries, and artifact manifest into a single Markdown report.

## Extension points

### GPIS backends

Backends should expose fit/predict behavior and return posterior mean, variance, and gradients at query points. Dense exact GPIS should remain the small-scene reference; approximate backends should be regression-tested against it on deterministic scenes.

### Calibrated confidence

New confidence models should preserve leakage-safe train/validation splits, serialize their feature set, and report AUROC/AUPRC, Brier, ECE, top-k retention, and reliability-style artifacts. Raw GPIS zero-band probability should remain a baseline rather than the only score.

### 3DGS training-time integration

The GPIS regularizer should remain renderer-agnostic. External trainers should query it at Gaussian centers and record the loss schedule, weights, pruning settings, and densification settings in the run config.

## Testing policy

Small deterministic tests should cover artifact schemas, preset loading, reproducibility report generation, confidence serialization, backend equivalence, render diagnostics, and PLY/gate export. Large real-data or trained-3DGS workflows should be represented by smoke configs and self-hosted workflow runs rather than mandatory unit tests.
