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

Audit a render evaluation when PSNR/SSIM look suspicious:

```powershell
audit_real_renders `
  --scene bicycle_sparse12 `
  --predictions-dir C:\runs\gpis_splatting\bicycle\renders `
  --method-name gpis_splatting
```

This checks that target and prediction paths are not identical, records per-image MSE and pixel-difference statistics, includes render
report coverage fields such as drawn splat count when available, and writes target/prediction/difference panels.

Diagnose whether bad render metrics are caused by camera/projection alignment before tuning GPIS gates:

```powershell
diagnose_real_alignment `
  --scene bicycle_sparse12 `
  --render-dir real_scenes\bicycle_sparse12\renders\real_gpis_gate `
  --split test `
  --max-frames 16
```

This joins PSNR/SSIM with projection diagnostics such as valid-depth fraction, behind-camera count, in-frame splat fraction, approximate
projected coverage, depth histograms, target/projected-splat overlays, target/prediction/difference panels, and a ranked failure-mode CSV.

Sweep renderer appearance parameters before comparing GPIS gates:

```powershell
run_real_render_parameter_sweep `
  --scene bicycle_sparse12 `
  --split test `
  --max-frames 16 `
  --sigma-scales 0.5 1.0 1.5 `
  --tau-scales 0.5 1.0 2.0 `
  --min-sigma-pxs 0.6 1.0 `
  --kernel-radii 2.0 3.0 `
  --background-colors 0,0,0 1,1,1
```

This writes a per-variant `render_parameter_sweep.csv`, ranked CSV, `best_render_parameters.json`, a copied `best_render/` directory,
and per-variant render metrics, audits, and optional alignment summaries under `real_scenes/<scene>/evaluations/<method-name>/`.

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

Run a focused gate-threshold sweep to test whether GPIS confidence selects geometrically better splats:

```powershell
run_tanks_temples_gate_sweep `
  --scene ignatius_tnt64 `
  --splats-path real20k_sigma_0p04_splats.npz `
  --model-path real20k_sigma_0p04_gpis_model.npz `
  --thresholds 0.02 0.05 0.1 `
  --gate-thresholds 0.02 0.05 0.1 0.2 0.35 0.5
```

This reuses one geometry evaluation, then writes a compact `*_gate_sweep.csv` and report with retention, precision, recall, F-score,
Chamfer, and deltas versus the ungated splat set for every `gate >= threshold` subset.

Diagnose whether GPIS gates are useful as a ranking and calibration signal:

```powershell
diagnose_tanks_temples_gates `
  --scene ignatius_tnt64 `
  --splats-path real20k_sigma_0p04_splats.npz `
  --model-path real20k_sigma_0p04_gpis_model.npz `
  --thresholds 0.02 0.05 0.1 `
  --topk-fractions 0.01 0.02 0.05 0.1 0.2 0.35 0.5 0.75 1.0
```

This writes per-splat nearest-ground-truth distances, Spearman/Pearson gate-error correlations, gate-sorted retention curves,
and calibration-style gate bins under `real_scenes/<scene>/evaluations/`.

Diagnose whether the GPIS field is useful through another score even when the current gate is weak:

```powershell
diagnose_tanks_temples_gpis_field_scores `
  --scene ignatius_tnt64 `
  --splats-path real20k_sigma_0p04_splats.npz `
  --model-path real20k_sigma_0p04_gpis_model.npz `
  --thresholds 0.02 0.05 0.1 `
  --score-lambdas 0.25 0.5 1.0
```

This evaluates GPIS posterior mean, variance, gradient norm, signed-distance proxy, distance uncertainty, the current gate,
and alternative distance/uncertainty scores at every splat center, then ranks those scores against nearest-ground-truth error.

Calibrate those GPIS-derived features into splat-confidence predictions:

```powershell
calibrate_gpis_splat_scores `
  --scene ignatius_tnt64 `
  --field-scores-path real20k_sigma_0p04_gpis_field_scores.csv `
  --thresholds 0.02 0.05 0.1
```

This consumes the per-splat field-score CSV, builds labels such as `nearest_gt_distance <= threshold`, compares current gate scores,
isotonic calibration, and logistic calibration over GPIS posterior features, then writes validation metrics, top-k retention curves,
calibrated splat confidences, gate-compatible `*_gate_<threshold>.npz` files, and a report under `real_scenes/<scene>/evaluations/`.
Those gate files can be passed to `evaluate_tanks_temples_geometry --gate-path` for geometry slices or to `render_real_splats --gate-path`
to apply calibrated confidence as an optical-thickness gate. When the hard-negative workflow filters splats during crop or subsampling,
unscored candidates are exported with zero confidence:

```powershell
render_real_splats `
  --scene ignatius_tnt64 `
  --splats-path evaluations/ignatius_hard_negative_v1_hard_negative_splats.npz `
  --gate-path evaluations/ignatius_hard_negative_v1_hard_negative_calibrated_gate_0p05.npz `
  --method-name calibrated_confidence_gate `
  --use-gpis-gate false
```

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

This writes filtered splat files for every `gate >= threshold` variant, a `gate_scaled` variant with optical thickness multiplied by
calibrated confidence, per-variant geometry metrics, optional render metrics, and one compact comparison report.

Run a harder mixed-candidate calibration workflow with generated off-surface splat candidates:

```powershell
run_tanks_temples_hard_negative_calibration `
  --scene ignatius_tnt64 `
  --splats-path real20k_sigma_0p04_splats.npz `
  --model-path real20k_sigma_0p04_gpis_model.npz `
  --method-name ignatius_hard_negative_v1 `
  --thresholds 0.02 0.05 0.1
```

This creates source, jittered, camera-ray, behind-surface, and crop-random candidate splats, scores the mixed set with GPIS field
diagnostics, and calibrates splat confidence on nearest-ground-truth labels. The workflow is intended to test whether GPIS-derived
confidence rejects floating or off-surface artifacts, rather than mostly ranking already-good source splats. It also exports the
threshold-specific calibrated gates needed by downstream geometry and rendering checks.

## Trained 3DGS Integration

To make the GPIS confidence signal comparable with standard 3DGS render metrics, convert a trained 3DGS Gaussian PLY into the internal
splat format, score/calibrate its centers with the existing GPIS tools, then export renderable 3DGS PLY variants:

```powershell
convert_3dgs_ply_to_splats `
  --input-ply C:\runs\3dgs\barn\point_cloud\iteration_30000\point_cloud.ply `
  --output-splats real_scenes\barn_selfhosted_m25000_t2500_s12\trained_3dgs_splats.npz

$gaussianCount = 123456  # replace with the `splats:` count printed by convert_3dgs_ply_to_splats

diagnose_tanks_temples_gpis_field_scores `
  --scene barn_selfhosted_m25000_t2500_s12 `
  --splats-path trained_3dgs_splats.npz `
  --model-path selfhosted_calibrated_confidence_gpis_model.npz `
  --method-name trained_3dgs `
  --max-pred-points 0

calibrate_gpis_splat_scores `
  --field-scores-path real_scenes\barn_selfhosted_m25000_t2500_s12\evaluations\trained_3dgs_gpis_field_scores.csv `
  --gate-count $gaussianCount

export_3dgs_gpis_variants `
  --input-ply C:\runs\3dgs\barn\point_cloud\iteration_30000\point_cloud.ply `
  --gate-path real_scenes\barn_selfhosted_m25000_t2500_s12\evaluations\trained_3dgs_calibrated_gate_0p05.npz `
  --output-dir C:\runs\3dgs\barn_gpis_variants `
  --iteration 30000
```

Each exported variant is written as `model_dir/point_cloud/iteration_<n>/point_cloud.ply`, preserving the trained Gaussian properties.
Render those model directories with the standard 3DGS renderer, then pass the predictions to `evaluate_real_renders` for PSNR/SSIM and
optional LPIPS.

For a self-hosted, artifact-producing run, use the **Trained 3DGS Photometric Evaluation** workflow. It assumes the prepared real-scene
directory, GPIS model, and trained 3DGS `point_cloud.ply` already exist on the self-hosted runner. The workflow preserves untracked
runner files during checkout, converts the trained PLY, scores/calibrates all Gaussians, exports renderable variants, and can either:

- run a renderer command template once per variant; or
- evaluate an existing root of rendered variant directories.

The renderer template may use `{model_dir}`, `{output_dir}`, `{scene_dir}`, `{variant}`, `{iteration}`, and `{point_cloud_path}`
placeholders. After rendering, the workflow writes one comparison table with PSNR/SSIM/optional LPIPS:

```powershell
evaluate_3dgs_variant_renders `
  --manifest-path real_scenes\barn_selfhosted_m25000_t2500_s12\trained_3dgs_variants\trained_3dgs\trained_3dgs_3dgs_variant_manifest.csv `
  --scene barn_selfhosted_m25000_t2500_s12 `
  --predictions-root real_scenes\barn_selfhosted_m25000_t2500_s12\renders\trained_3dgs_variants `
  --prediction-subdir test\ours_30000\renders `
  --method-name trained_3dgs `
  --split test
```

Sweep GPIS pseudo-SDF construction and model hyperparameters against those gate-quality diagnostics:

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

This regenerates bootstrap samples for each construction mode, fits a GPIS model for every hyperparameter combination, runs the
gate-quality diagnostic for every gate setting, and writes a summary CSV/status/report under `real_scenes/<scene>/model_sweeps/<sweep-name>/`.
Use `--construction-modes existing --samples-path ... --splats-path ...` for a fast sweep over already-created samples and splats.

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
