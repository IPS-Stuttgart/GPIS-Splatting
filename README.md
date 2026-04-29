# GPIS Splatting Bootstrap

This is a small synthetic prototype for the GPIS and uncertainty-aware splat-rendering plan.
It is intentionally dense and CPU-friendly: the goal is to get first visible and measurable results,
not to implement scalable SKI, CUDA kernels, dynamic scenes, or full 3DGS.

## Quick Start

From this directory:

```powershell
python -m pip install -e .
python -m gpis_splatting.cli.generate_scene --shape sphere --scene sphere_demo --num-points 180
python -m gpis_splatting.cli.fit_gpis --scene sphere_demo --grid-size 28
python -m gpis_splatting.cli.render_splats --scene sphere_demo --view all
python -m gpis_splatting.cli.evaluate --scene sphere_demo
```

Outputs are written to `experiments/<scene>/`:

- `config.json`
- `samples.npz`
- `gpis_model.npz`
- `posterior_grid.npz`
- `splats.npz`
- `render_reference_<view>.png`
- `render_plain_<view>.png`
- `render_gpis_<view>.png`
- `render_feedback_<view>.png` when `render_splats --feedback-iterations` is used
- `feedback_gpis_model.npz`, `feedback_trace.csv`, and `feedback_splat_gates.npz` when feedback is used
- `gpis_surface.png`
- `uncertainty_slice.png`
- `metrics.csv`

If installed with `pip install -e .`, the console scripts are also available:

```powershell
generate_scene --shape torus --scene torus_demo
fit_gpis --scene torus_demo
render_splats --scene torus_demo --view all
evaluate --scene torus_demo
```

To run the first bidirectional GPIS-splat feedback loop, enable one or more feedback iterations:

```powershell
render_splats --scene torus_demo --view all --feedback-iterations 2 --feedback-selector uncertainty
evaluate --scene torus_demo
```

To compare the one-way gate against multiple feedback depths across synthetic shapes:

```powershell
run_ablation --shapes sphere torus --feedback-iterations 0 1 2 --feedback-selectors gate uncertainty uncertainty_diverse
```

This writes `experiments/feedback_ablation/ablation_metrics.csv` with one row per shape, feedback setting, and selector mode.

Summarize the ablation into plots and winner tables:

```powershell
summarize_ablation --ablation-root experiments/feedback_ablation
```

This writes `ablation_summary.csv`, `ablation_winners.csv`, `ablation_summary.md`, and comparison plots under `experiments/feedback_ablation/summary/`.

Run a reproducible evaluation workflow with preset thresholds and report artifacts:

```powershell
run_evaluation --preset synthetic_quick --benchmark-target benchmarks/mipnerf360_sparse_12view.json
```

For pull requests and fast smoke checks, use the smaller preset:

```powershell
run_evaluation --preset synthetic_ci --experiment-name ci_evaluation
```

This chains ablation, summary generation, and evaluation checks. It writes `evaluation_config.json`, `evaluation_checks.csv`, `evaluation_status.json`,
and `evaluation_report.md` under `experiments/<experiment-name>/`. The GitHub Actions evaluation workflow runs `synthetic_ci` and uploads the report plus summary artifacts.

## Real-Data Benchmark Harness

Prepare a real image/camera scene in the repository's normalized format:

```powershell
prepare_real_scene `
  --input-dir C:\datasets\mipnerf360\bicycle `
  --scene bicycle_sparse12 `
  --dataset mipnerf360_sparse `
  --train-view-count 12
```

The adapter currently supports NeRF-style `transforms.json` and COLMAP text exports containing `cameras.txt` and `images.txt`.
It writes `real_scenes/<scene>/real_scene.json`, `cameras.json`, `splits.json`, copied images, and `validation.json`.

Validate a prepared scene:

```powershell
validate_real_scene --scene bicycle_sparse12
```

Evaluate held-out render images from a method:

```powershell
evaluate_real_renders `
  --scene bicycle_sparse12 `
  --predictions-dir C:\runs\gpis_splatting\bicycle\renders `
  --method-name gpis_splatting `
  --benchmark-target benchmarks/mipnerf360_sparse_12view.json
```

This writes per-image PSNR/SSIM metrics, a summary CSV, and a Markdown report under `real_scenes/<scene>/evaluations/`.
LPIPS can be enabled with `--compute-lpips true` when the optional `lpips` package is installed.

Bootstrap first GPIS observations and initial splats from sparse real geometry:

```powershell
bootstrap_real_gpis `
  --scene bicycle_sparse12 `
  --point-source auto `
  --max-points 5000
```

The bootstrapper reads COLMAP `points3D.txt` or an ASCII/binary `.ply` point cloud. It writes `real_samples.npz` with surface, free-space,
and optional behind-surface pseudo-SDF observations, `real_splats.npz` with initial colored splats, plus `real_gpis_config.json`
and `real_bootstrap_report.json`.

Download a small public real scene for laptop smoke runs:

```powershell
download_real_scene `
  --dataset nerfstudio_poster `
  --image-scale 8 `
  --max-images 24
```

This writes a local Nerfstudio `poster` subset under `real_scenes/_downloads/`, including scaled camera intrinsics for `images_8`.

Fit the dense GPIS model and render the initialized splats through the prepared real cameras:

```powershell
fit_real_gpis `
  --scene bicycle_sparse12 `
  --max-train-points 1200

render_real_splats `
  --scene bicycle_sparse12 `
  --split test `
  --use-gpis-gate true
```

This writes `real_gpis_model.npz`, a fit report, held-out render images under `real_scenes/<scene>/renders/real_gpis_gate/`,
`real_splat_gates.npz`, and `real_render_report.json`. The render directory can be passed directly to `evaluate_real_renders`.

Run a reproducible real-data plain-vs-gated smoke workflow, including a small parameter sweep:

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
```

This writes `real_evaluation_comparison.csv`, `real_evaluation_status.json`, and `real_evaluation_report.md` under
`real_scenes/<scene>/evaluations/`.

Diagnose why a real-data run is failing before starting another sweep:

```powershell
diagnose_real_render `
  --scene poster8_smoke `
  --split test `
  --max-frames 8 `
  --epsilon 0.24 `
  --gate-floor 0.0
```

This writes target/plain/gated panels, projected-splat overlays, depth visualizations, gate-colored overlays, gate histograms,
`real_render_diagnostics.csv`, and `real_render_diagnostics.md` under `real_scenes/<scene>/diagnostics/real_render/`.
Existing render directories can be passed with `--plain-renders-dir` and `--gated-renders-dir` to diagnose already-generated outputs.

Download and prepare the Tanks and Temples `Ignatius` training scene for geometry-oriented GPIS experiments:

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

The downloader records official Tanks and Temples provenance and license URLs. The preparer reads the Redwood `.log` camera trajectory,
uses the dataset's recommended pinhole initialization (`fx=fy=0.7*W`, `cx=W/2`, `cy=H/2`), stores paths to the COLMAP reconstruction,
alignment, crop, and ground-truth geometry when present, and writes the normalized `real_scene.json`, `cameras.json`, and `splits.json`.

Evaluate initialized splat geometry against the official Tanks and Temples ground truth:

```powershell
evaluate_tanks_temples_geometry `
  --scene ignatius_tnt64 `
  --splats-path real20k_sigma_0p04_splats.npz `
  --thresholds 0.01 0.02 0.05 0.1
```

The evaluator applies the stored alignment and crop metadata by default, uses deterministic point subsampling for large PLYs, and writes
Chamfer, accuracy/completion, precision, recall, and F-score tables under `real_scenes/<scene>/evaluations/`. Passing `--gate-path` or
`--model-path` also adds high-gate/low-gate geometry slices.

## Development

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

## Implemented Scope

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
- Metrics: RMSE, IoU, NLL, Brier score, ECE, and PSNR for rendered images
- Unit and regression tests
- Source code is kept in `src/gpis_splatting/`, with tests in `tests/`.
