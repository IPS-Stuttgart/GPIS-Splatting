# Paper-grade trained-3DGS comparison

Use `run_trained_3dgs_gpis_experiment` to consolidate trained-3DGS and GPIS-variant evidence across scenes into a paper-facing table.

The comparison table covers:

- PSNR, SSIM, and optional LPIPS from rendered predictions
- F-score, precision, recall, Chamfer-L1, and Chamfer-L2 from the Tanks-and-Temples geometry evaluator
- Gaussian count and retention from the 3DGS variant manifest
- FPS and VRAM from a gsplat render manifest or an explicit performance CSV

## Inputs

Each scene entry in the JSON config should provide at least:

```json
{
  "scene": "ignatius_tnt64",
  "scene_dir": "real_scenes/ignatius_tnt64",
  "manifest_path": "paper_results/ignatius_tnt64/variants/trained_3dgs_3dgs_variant_manifest.csv",
  "predictions_root": "paper_results/ignatius_tnt64/renders/gsplat_3dgs_variants",
  "prediction_subdir": "test/ours_30000/renders",
  "ground_truth_path": "gt/ignatius.ply",
  "alignment_path": "gt/ignatius_alignment.txt",
  "crop_path": "gt/ignatius_crop.json",
  "render_manifest_path": "paper_results/ignatius_tnt64/renders/gsplat_3dgs_variants/trained_3dgs_gsplat_render_manifest.csv"
}
```

Instead of recomputing metrics, you may pass precomputed artifacts:

- `render_comparison_path` or `photometry_path`
- `geometry_comparison_path` or `geometry_path`
- `performance_path`

The performance table must contain a `variant` column and may contain `fps`, `peak_vram_mb`, `mean_frame_seconds`, `total_render_seconds`, `device`, and `rendered_gaussian_count`.

## Run

```bash
run_trained_3dgs_gpis_experiment \
  --scene-config configs/trained_3dgs/paper_grade_comparison.json \
  --fail-on-missing true \
  --require-lpips true \
  --require-performance true
```

## Outputs

For `comparison_name = paper_3dgs`, the command writes:

```text
paper_3dgs_scene_comparison.csv
paper_3dgs_aggregate_by_variant.csv
paper_3dgs_paper_table.csv
paper_3dgs_checks.csv
paper_3dgs_status.json
paper_3dgs_config.json
paper_3dgs_report.md
```

`paper_3dgs_paper_table.csv` is the paper-facing compact table with mean ± std columns for PSNR, SSIM, LPIPS, F-score, Chamfer, Gaussian count, FPS, and VRAM.
