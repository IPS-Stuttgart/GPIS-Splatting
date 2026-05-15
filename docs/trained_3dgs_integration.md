# Trained 3DGS integration

This page covers the workflow that makes the GPIS confidence signal comparable with standard 3DGS render metrics.

## Export a prepared scene for 3DGS training

First export a prepared real scene into the standard COLMAP text layout used by the reference 3DGS trainer:

```powershell
export_prepared_scene_to_colmap_3dgs `
  --scene ignatius_selfhosted_m25000_t2500_s12 `
  --output-dir C:\runs\3dgs\ignatius_gpis_scene `
  --split train `
  --max-points 100000
```

The exporter writes `images/` plus `sparse/0/cameras.txt`, `sparse/0/images.txt`, and `sparse/0/points3D.txt`. It copies the selected prepared-scene images and writes COLMAP `PINHOLE` cameras from the normalized camera metadata.

Sparse points are initialized from the scene's Tanks and Temples reconstruction, COLMAP `points3D.txt`, or an explicit `.ply`/internal splats `.npz` passed with `--points-path`.

## Train the standard 3DGS implementation

For a self-hosted training run, use the **Train Standard 3DGS Baseline** workflow. It exports the prepared scene to the COLMAP/3DGS layout, runs either the reference Graphdeco trainer or a supplied command template, validates the resulting `point_cloud/iteration_<n>/point_cloud.ply`, and uploads the trained model artifact.

The downstream **Trained 3DGS Photometric Evaluation** workflow can restore that artifact through `trained_model_artifact` using `run_id|artifact_name|repository|relative_ply_path`, or it can continue to consume an explicit runner-local `trained_ply_path`.

For non-dry training runs, the workflow checks `nvidia-smi` and `nvcc` before downloading large prepared-scene artifacts. Use `runs_on_json` to pin the job to a known GPU runner label, for example `["self-hosted","Linux","nvidia-smi"]`.

## Convert, score, calibrate, and export variants

After training, convert the trained 3DGS Gaussian PLY into the internal splat format, score/calibrate its centers with the existing GPIS tools, and export renderable 3DGS PLY variants:

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

Each exported variant is written as `model_dir/point_cloud/iteration_<n>/point_cloud.ply`, preserving the trained Gaussian properties. Render those model directories with the standard 3DGS renderer, then pass the predictions to `evaluate_real_renders` for PSNR/SSIM and optional LPIPS.

## Photometric evaluation workflow

For a self-hosted, artifact-producing run, use the **Trained 3DGS Photometric Evaluation** workflow. It assumes the prepared real-scene directory, GPIS model, and trained 3DGS `point_cloud.ply` already exist on the self-hosted runner.

The workflow preserves untracked runner files during checkout, converts the trained PLY, scores/calibrates all Gaussians, exports renderable variants, and can either:

- run a renderer command template once per variant; or
- evaluate an existing root of rendered variant directories.

The renderer template may use `{model_dir}`, `{output_dir}`, `{scene_dir}`, `{variant}`, `{iteration}`, and `{point_cloud_path}` placeholders. After rendering, the workflow writes one comparison table with PSNR/SSIM/optional LPIPS:

```powershell
evaluate_3dgs_variant_renders `
  --manifest-path real_scenes\barn_selfhosted_m25000_t2500_s12\trained_3dgs_variants\trained_3dgs\trained_3dgs_3dgs_variant_manifest.csv `
  --scene barn_selfhosted_m25000_t2500_s12 `
  --predictions-root real_scenes\barn_selfhosted_m25000_t2500_s12\renders\trained_3dgs_variants `
  --prediction-subdir test\ours_30000\renders `
  --method-name trained_3dgs `
  --split test
```
