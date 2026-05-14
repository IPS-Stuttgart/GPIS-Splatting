# Development and implemented scope

## Development setup

Install the local package and development tools:

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Run the baseline checks before opening a PR:

```powershell
python -m ruff check .
python -m pytest -q
python -m build
```

## Implemented scope

- Synthetic SDF scenes: `sphere`, `torus`, `two_objects`, `non_star_convex`
- Dense RBF GPIS posterior mean and variance
- Analytic posterior mean gradients for distance proxy and GPIS gates
- Orthographic CPU splat renderer with Beer-Lambert optical-thickness compositing
- GPIS gate applied as `tau_tilde_i = p_0,epsilon(x_i) * tau_i`
- Optional bidirectional feedback: high-confidence gated splats become heteroscedastic GPIS zero-level pseudo observations
- Feedback selectors for gate-only, uncertainty-weighted, and diversity-suppressed pseudo-observation promotion
- Ablation runner for comparing feedback iteration counts and selector modes across synthetic shapes
- Ablation summarizer for PSNR/RMSE/IoU deltas, selector winners, and comparison plots
- Evaluation workflow for deterministic preset runs, pass/fail checks, report artifacts, and a Mip-NeRF 360 Sparse 12-view target manifest
- Real-scene preparation and render evaluation harness for NeRF `transforms.json` and COLMAP text camera exports
- Download adapter for the public Nerfstudio `poster` scene at reduced image scale
- Real-scene GPIS bootstrap from COLMAP `points3D.txt` or ASCII PLY point clouds into pseudo-SDF observations and initial splats
- Dense real-scene GPIS fitting plus camera-aware real-splat rendering with optional GPIS optical-thickness gates
- Reproducible real-data evaluation workflow with plain/gated comparisons, epsilon, splat scale, and gate-floor sweeps
- Real-render diagnostics with target/plain/gated panels, projected splats, depth views, gate overlays, histograms, and per-frame visibility/metric CSVs
- Tanks and Temples `Ignatius` downloader and `.log` pose adapter for geometry-oriented real-data experiments
- Tanks and Temples geometry evaluator with alignment/crop handling, Chamfer, precision, recall, F-score, and gate-stratified slices
- Gate-threshold geometry sweeps for checking whether GPIS confidence is useful for selecting splats
- Gate quality diagnostics for checking whether GPIS confidence ranks and calibrates splat geometry error
- GPIS field score diagnostics for testing whether posterior mean, uncertainty, distance, or combined scores rank splat geometry error better than the current gate
- GPIS-derived splat confidence calibration with downstream gate NPZ exports, current-score baselines, isotonic calibration, and logistic feature models
- Hard-negative real-splat workflow that generates off-surface candidates, scores them with GPIS, and calibrates confidence on mixed source/negative sets
- Calibrated splat filtering workflow that writes compacted/tau-scaled splat variants and compares geometry plus render metrics
- Real GPIS gate model sweeps over pseudo-SDF construction modes, GPIS hyperparameters, epsilon, and gate floors
- Render audit workflow for checking path mistakes, exact pixel matches, image coverage, and pixel-difference panels
- Metrics: RMSE, IoU, NLL, Brier score, ECE, and PSNR for rendered images
- Unit and regression tests
- Source code is kept in `src/gpis_splatting/`, with tests in `tests/`.
