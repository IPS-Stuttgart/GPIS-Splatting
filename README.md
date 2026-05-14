# GPIS Splatting Bootstrap

CPU-friendly prototype for GPIS-guided uncertainty-aware splat rendering. The repository contains synthetic SDF experiments, real-scene preparation/evaluation tools, Tanks and Temples geometry workflows, calibrated GPIS confidence diagnostics, and trained-3DGS integration utilities.

The README is intentionally kept compact so GitHub, code-search, and connector tools can load it reliably. Detailed workflows live in `docs/`.

## Quick start

```powershell
python -m pip install -e .
python -m gpis_splatting.cli.generate_scene --shape sphere --scene sphere_demo --num-points 180
python -m gpis_splatting.cli.fit_gpis --scene sphere_demo --grid-size 28
python -m gpis_splatting.cli.render_splats --scene sphere_demo --view all
python -m gpis_splatting.cli.evaluate --scene sphere_demo
```

Outputs are written to `experiments/<scene>/`.

## Documentation map

- [Synthetic quickstart and evaluation](docs/quickstart.md)
- [Real-data scene preparation, rendering, and diagnostics](docs/real_data.md)
- [Tanks and Temples geometry/calibration workflows](docs/tanks_temples.md)
- [Trained 3DGS integration](docs/trained_3dgs_integration.md)
- [Development and implemented scope](docs/development.md)
- [Experiment matrix reporting](docs/experiment_matrix.md)
- [Calibrated confidence API](docs/calibrated_confidence_api.md)
- [Paper result reproduction](docs/reproduce_paper_results.md)

## Common commands

Install the package locally:

```powershell
python -m pip install -e .
```

Run the fast synthetic evaluation preset:

```powershell
run_evaluation --preset synthetic_ci --experiment-name ci_evaluation
```

Prepare and validate a real scene:

```powershell
prepare_real_scene --input-dir C:\datasets\mipnerf360\bicycle --scene bicycle_sparse12 --dataset mipnerf360_sparse --train-view-count 12
validate_real_scene --scene bicycle_sparse12
```

Run a small real-data smoke workflow:

```powershell
run_real_evaluation --scene poster8_smoke --max-download-images 24 --max-points 800 --max-train-points 600 --max-frames 4
```

Download and prepare a Tanks and Temples scene:

```powershell
download_tanks_temples_scene --scene Ignatius --output-root real_scenes/_downloads --max-images 64
prepare_tanks_temples_scene --input-dir real_scenes/_downloads/tanks_temples/Ignatius --prepared-scene ignatius_tnt64 --train-view-count 12
```

## Development checks

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m ruff check .
python -m pytest -q
python -m build
```
