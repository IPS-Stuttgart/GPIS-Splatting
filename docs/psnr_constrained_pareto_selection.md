# PSNR-constrained 3DGS variant selection

`select_3dgs_pareto_variant` selects a GPIS-gated trained-3DGS variant after rendered variants have been evaluated. The selector is intended for paper tables where aggressive GPIS pruning or opacity scaling should only be reported when it stays within a prescribed photometric degradation budget relative to the unmodified trained-3DGS baseline.

The default rule is:

1. read the `*_3dgs_render_comparison.csv` produced by `evaluate_3dgs_variant_renders`,
2. use the `baseline` variant as the PSNR reference,
3. keep only variants with `baseline_psnr - mean_psnr <= 0.2 dB`,
4. compute the non-dominated frontier over retained count, PSNR, SSIM, and LPIPS when available,
5. select the smallest retained-count variant on that frontier.

This makes GPIS cleanup/compression claims less sensitive to a cherry-picked gate threshold: the selected variant must remain photometrically close to the original trained 3DGS result.

## Standalone use

```bash
select_3dgs_pareto_variant \
  --comparison-path outputs/evaluations/paper_gate_3dgs_render_comparison.csv \
  --psnr-drop-tolerance 0.2 \
  --objective min_retained
```

The command writes:

- `*_3dgs_pareto_selection.csv` with the original comparison rows plus selection columns,
- `*_3dgs_pareto_selection_status.json` with the selected variant and constraint metadata,
- `*_3dgs_pareto_selection_report.md` with a compact table for inspection.

## Integrated render evaluation

`evaluate_3dgs_variant_renders` now writes the same selection artifacts by default after the render comparison CSV is created. Disable this with:

```bash
evaluate_3dgs_variant_renders ... --write-pareto-selection false
```

Useful knobs are:

```bash
--pareto-baseline-variant baseline
--pareto-psnr-drop-tolerance 0.2
--pareto-objective min_retained
```

The objective accepts `min_retained`, `max_psnr`, `max_ssim`, and `min_lpips`.
