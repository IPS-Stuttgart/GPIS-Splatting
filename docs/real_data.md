# Real-data scene preparation, rendering, and diagnostics

This page covers real-scene preparation, real-splat bootstrapping, render evaluation, parameter sweeps, and diagnostics.

## Prepare and validate a real scene

Prepare a real image/camera scene in the normalized repository format:

```powershell
prepare_real_scene `
  --input-dir C:\datasets\mipnerf360\bicycle `
  --scene bicycle_sparse12 `
  --dataset mipnerf360_sparse `
  --train-view-count 12
```

The adapter currently supports NeRF-style `transforms.json` and COLMAP text exports containing `cameras.txt` and `images.txt`. It writes `real_scenes/<scene>/real_scene.json`, `cameras.json`, `splits.json`, copied images, and `validation.json`.

Validate a prepared scene:

```powershell
validate_real_scene --scene bicycle_sparse12
```

## Evaluate held-out renders

```powershell
evaluate_real_renders `
  --scene bicycle_sparse12 `
  --predictions-dir C:\runs\gpis_splatting\bicycle\renders `
  --method-name gpis_splatting `
  --benchmark-target benchmarks/mipnerf360_sparse_12view.json
```

This writes per-image PSNR/SSIM metrics, a summary CSV, and a Markdown report under `real_scenes/<scene>/evaluations/`. LPIPS can be enabled with `--compute-lpips true` when the optional `lpips` package is installed.

Audit suspicious render metrics:

```powershell
audit_real_renders `
  --scene bicycle_sparse12 `
  --predictions-dir C:\runs\gpis_splatting\bicycle\renders `
  --method-name gpis_splatting
```

The audit checks that target and prediction paths are not identical, records per-image MSE and pixel-difference statistics, includes render coverage fields such as drawn splat count when available, and writes target/prediction/difference panels.

## Alignment diagnostics

Diagnose camera/projection alignment before tuning GPIS gates:

```powershell
diagnose_real_alignment `
  --scene bicycle_sparse12 `
  --render-dir real_scenes\bicycle_sparse12\renders\real_gpis_gate `
  --split test `
  --max-frames 16
```

This joins PSNR/SSIM with projection diagnostics such as valid-depth fraction, behind-camera count, in-frame splat fraction, approximate projected coverage, depth histograms, target/projected-splat overlays, target/prediction/difference panels, and a ranked failure-mode CSV.

## Renderer parameter sweep

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

This writes a per-variant `render_parameter_sweep.csv`, ranked CSV, `best_render_parameters.json`, a copied `best_render/` directory, and per-variant render metrics, audits, and optional alignment summaries under `real_scenes/<scene>/evaluations/<method-name>/`.

## Bootstrap real GPIS observations and splats

```powershell
bootstrap_real_gpis `
  --scene bicycle_sparse12 `
  --point-source auto `
  --max-points 5000
```

The bootstrapper reads COLMAP `points3D.txt` or an ASCII/binary `.ply` point cloud. It writes `real_samples.npz` with surface, free-space, and optional behind-surface pseudo-SDF observations, `real_splats.npz` with initial colored splats, plus `real_gpis_config.json` and `real_bootstrap_report.json`.

## Public smoke scene

Download a small public real scene for laptop smoke runs:

```powershell
download_real_scene `
  --dataset nerfstudio_poster `
  --image-scale 8 `
  --max-images 24
```

This writes a local Nerfstudio `poster` subset under `real_scenes/_downloads/`, including scaled camera intrinsics for `images_8`.

## Fit and render real GPIS splats

```powershell
fit_real_gpis `
  --scene bicycle_sparse12 `
  --max-train-points 1200

render_real_splats `
  --scene bicycle_sparse12 `
  --split test `
  --use-gpis-gate true
```

This writes `real_gpis_model.npz`, a fit report, held-out render images under `real_scenes/<scene>/renders/real_gpis_gate/`, `real_splat_gates.npz`, and `real_render_report.json`. The render directory can be passed directly to `evaluate_real_renders`.

## Reproducible real-data smoke workflow

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

This writes `real_evaluation_comparison.csv`, `real_evaluation_status.json`, and `real_evaluation_report.md` under `real_scenes/<scene>/evaluations/`.

## Real-render failure diagnostics

```powershell
diagnose_real_render `
  --scene poster8_smoke `
  --split test `
  --max-frames 8 `
  --epsilon 0.24 `
  --gate-floor 0.0
```

This writes target/plain/gated panels, projected-splat overlays, depth visualizations, gate-colored overlays, gate histograms, `real_render_diagnostics.csv`, and `real_render_diagnostics.md` under `real_scenes/<scene>/diagnostics/real_render/`. Existing render directories can be passed with `--plain-renders-dir` and `--gated-renders-dir` to diagnose already-generated outputs.
