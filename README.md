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
- Metrics: RMSE, IoU, NLL, Brier score, ECE, and PSNR for rendered images
- Unit and regression tests
- Source code is kept in `src/gpis_splatting/`, with tests in `tests/`.
