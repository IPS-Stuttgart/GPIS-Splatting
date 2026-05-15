# Tanks and Temples geometry and calibration workflows

This page collects the Tanks and Temples workflows for geometry-oriented GPIS experiments, gate diagnostics, calibrated confidence, hard negatives, and splat filtering.

## Download and prepare Ignatius

Download and prepare the Tanks and Temples `Ignatius` training scene:

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

The downloader records official Tanks and Temples provenance and license URLs. The preparer reads the Redwood `.log` camera trajectory and uses the dataset's recommended pinhole initialization (`fx=fy=0.7*W`, `cx=W/2`, `cy=H/2`).

It stores paths to the COLMAP reconstruction, alignment, crop, and ground-truth geometry when present, and writes the normalized `real_scene.json`, `cameras.json`, and `splits.json`.

## Geometry evaluation

Evaluate initialized splat geometry against the official Tanks and Temples ground truth:

```powershell
evaluate_tanks_temples_geometry `
  --scene ignatius_tnt64 `
  --splats-path real20k_sigma_0p04_splats.npz `
  --thresholds 0.01 0.02 0.05 0.1
```

The evaluator applies the stored alignment and crop metadata by default, uses deterministic point subsampling for large PLYs, and writes Chamfer, accuracy/completion, precision, recall, and F-score tables under `real_scenes/<scene>/evaluations/`. Passing `--gate-path` or `--model-path` also adds high-gate/low-gate geometry slices.

## GPIS gate threshold sweep

Run a focused gate-threshold sweep to test whether GPIS confidence selects geometrically better splats:

```powershell
run_tanks_temples_gate_sweep `
  --scene ignatius_tnt64 `
  --splats-path real20k_sigma_0p04_splats.npz `
  --model-path real20k_sigma_0p04_gpis_model.npz `
  --thresholds 0.02 0.05 0.1 `
  --gate-thresholds 0.02 0.05 0.1 0.2 0.35 0.5
```

This reuses one geometry evaluation, then writes a compact `*_gate_sweep.csv` and report with retention, precision, recall, F-score, Chamfer, and deltas versus the ungated splat set for every `gate >= threshold` subset.

## Gate-ranking diagnostics

Diagnose whether GPIS gates are useful as a ranking and calibration signal:

```powershell
diagnose_tanks_temples_gates `
  --scene ignatius_tnt64 `
  --splats-path real20k_sigma_0p04_splats.npz `
  --model-path real20k_sigma_0p04_gpis_model.npz `
  --thresholds 0.02 0.05 0.1 `
  --topk-fractions 0.01 0.02 0.05 0.1 0.2 0.35 0.5 0.75 1.0
```

This writes per-splat nearest-ground-truth distances, Spearman/Pearson gate-error correlations, gate-sorted retention curves, and calibration-style gate bins under `real_scenes/<scene>/evaluations/`.

## GPIS field-score diagnostics

Diagnose whether the GPIS field is useful through another score even when the current gate is weak:

```powershell
diagnose_tanks_temples_gpis_field_scores `
  --scene ignatius_tnt64 `
  --splats-path real20k_sigma_0p04_splats.npz `
  --model-path real20k_sigma_0p04_gpis_model.npz `
  --thresholds 0.02 0.05 0.1 `
  --score-lambdas 0.25 0.5 1.0
```

This evaluates GPIS posterior mean, variance, gradient norm, signed-distance proxy, distance uncertainty, the current gate, and alternative distance/uncertainty scores at every splat center, then ranks those scores against nearest-ground-truth error.

## Calibrate GPIS-derived splat confidence

```powershell
calibrate_gpis_splat_scores `
  --scene ignatius_tnt64 `
  --field-scores-path real20k_sigma_0p04_gpis_field_scores.csv `
  --thresholds 0.02 0.05 0.1
```

This consumes the per-splat field-score CSV, builds labels such as `nearest_gt_distance <= threshold`, compares current gate scores, isotonic calibration, and logistic calibration over GPIS posterior features, then writes validation metrics, top-k retention curves, calibrated splat confidences, gate-compatible `*_gate_<threshold>.npz` files, and a report under `real_scenes/<scene>/evaluations/`.

Those gate files can be passed to `evaluate_tanks_temples_geometry --gate-path` for geometry slices or to `render_real_splats --gate-path` to apply calibrated confidence as an optical-thickness gate. When the hard-negative workflow filters splats during crop or subsampling, unscored candidates are exported with zero confidence:

```powershell
render_real_splats `
  --scene ignatius_tnt64 `
  --splats-path evaluations/ignatius_hard_negative_v1_hard_negative_splats.npz `
  --gate-path evaluations/ignatius_hard_negative_v1_hard_negative_calibrated_gate_0p05.npz `
  --method-name calibrated_confidence_gate `
  --use-gpis-gate false
```

## Calibrated splat filtering

Turn calibrated confidence into compacted or tau-scaled splat variants and evaluate each variant:

```powershell
run_tanks_temples_calibrated_splat_filtering `
  --scene ignatius_tnt64 `
  --splats-path evaluations/ignatius_hard_negative_v1_hard_negative_splats.npz `
  --gate-path evaluations/ignatius_hard_negative_v1_hard_negative_calibrated_gate_0p05.npz `
  --method-name ignatius_confidence_filter_v1 `
  --gate-thresholds 0.25 0.5 0.75 `
  --thresholds 0.02 0.05 0.1 `
  --render-max-frames 2
```

This writes filtered splat files for every `gate >= threshold` variant, a `gate_scaled` variant with optical thickness multiplied by calibrated confidence, per-variant geometry metrics, optional render metrics, and one compact comparison report.

## Hard-negative calibration

Run a harder mixed-candidate calibration workflow with generated off-surface splat candidates:

```powershell
run_tanks_temples_hard_negative_calibration `
  --scene ignatius_tnt64 `
  --splats-path real20k_sigma_0p04_splats.npz `
  --model-path real20k_sigma_0p04_gpis_model.npz `
  --method-name ignatius_hard_negative_v1 `
  --thresholds 0.02 0.05 0.1
```

This creates source, jittered, camera-ray, behind-surface, and crop-random candidate splats, scores the mixed set with GPIS field diagnostics, and calibrates splat confidence on nearest-ground-truth labels. The workflow is intended to test whether GPIS-derived confidence rejects floating or off-surface artifacts, rather than mostly ranking already-good source splats.

It also exports the threshold-specific calibrated gates needed by downstream geometry and rendering checks.

## Real GPIS gate model sweep

Sweep GPIS pseudo-SDF construction and model hyperparameters against gate-quality diagnostics:

```powershell
run_real_gpis_gate_model_sweep `
  --scene ignatius_tnt64 `
  --sweep-name ignatius_gate_model_v1 `
  --construction-modes surface_free strong_free behind_surface normal_offsets `
  --lengthscales 0.15 0.25 0.4 `
  --noise-stds 0.03 0.06 `
  --epsilons 0.08 0.16 0.24 `
  --gate-floors 0.0 0.25 `
  --max-bootstrap-points 5000 `
  --max-train-points 1200
```

This regenerates bootstrap samples for each construction mode, fits a GPIS model for every hyperparameter combination, runs the gate-quality diagnostic for every gate setting, and writes a summary CSV/status/report under `real_scenes/<scene>/model_sweeps/<sweep-name>/`.

Use `--construction-modes existing --samples-path ... --splats-path ...` for a fast sweep over already-created samples and splats.
