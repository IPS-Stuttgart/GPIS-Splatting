# Synthetic quickstart and evaluation

This page contains the compact synthetic workflow that used to live in the README.

## Basic synthetic run

From the repository root:

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
- `render_feedback_<view>.png` when feedback is enabled
- `feedback_gpis_model.npz`, `feedback_trace.csv`, and `feedback_splat_gates.npz` when feedback is enabled
- `gpis_surface.png`
- `uncertainty_slice.png`
- `metrics.csv`

When installed with `pip install -e .`, console scripts are also available:

```powershell
generate_scene --shape torus --scene torus_demo
fit_gpis --scene torus_demo
render_splats --scene torus_demo --view all
evaluate --scene torus_demo
```

## Bidirectional feedback

Run the first bidirectional GPIS-splat feedback loop by enabling one or more feedback iterations:

```powershell
render_splats --scene torus_demo --view all --feedback-iterations 2 --feedback-selector uncertainty
evaluate --scene torus_demo
```

Compare the one-way gate against multiple feedback depths across synthetic shapes:

```powershell
run_ablation --shapes sphere torus --feedback-iterations 0 1 2 --feedback-selectors gate uncertainty uncertainty_diverse
```

This writes `experiments/feedback_ablation/ablation_metrics.csv` with one row per shape, feedback setting, and selector mode.

Summarize the ablation into plots and winner tables:

```powershell
summarize_ablation --ablation-root experiments/feedback_ablation
```

This writes `ablation_summary.csv`, `ablation_winners.csv`, `ablation_summary.md`, and comparison plots under `experiments/feedback_ablation/summary/`.

## Reproducible synthetic evaluation

Run a reproducible evaluation workflow with preset thresholds and report artifacts:

```powershell
run_evaluation --preset synthetic_quick --benchmark-target benchmarks/mipnerf360_sparse_12view.json
```

For pull requests and fast smoke checks, use the smaller preset:

```powershell
run_evaluation --preset synthetic_ci --experiment-name ci_evaluation
```

The evaluation command chains ablation, summary generation, and evaluation checks. It writes `evaluation_config.json`, `evaluation_checks.csv`, `evaluation_status.json`, and `evaluation_report.md` under `experiments/<experiment-name>/`. The GitHub Actions evaluation workflow runs `synthetic_ci` and uploads the report plus summary artifacts.
