# Reproducibility Configs

This directory contains declarative presets and command plans used to reproduce GPIS-Splatting experiments.

- `evaluation/`: JSON presets accepted by `run_evaluation --preset-config ...`.
- `real/`: command plans for laptop-scale real-scene smoke workflows.
- `tanks_temples/`: command plans for geometry, gate, field-score, and hard-negative calibration experiments.
- `trained_3dgs/`: command plans for external 3DGS training/rendering interop.

Evaluation presets must contain `description`, `ablation`, and `targets`. Command-plan configs must contain `description` and `commands`. Both schemas may include additional metadata; `write_reproducibility_report` records the config hash and command count without requiring every external path to exist.
