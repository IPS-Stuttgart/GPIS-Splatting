# GPIS-gated Gaussian surface extraction

This workflow adds a geometry-facing path for trained 3DGS outputs. It extracts surface proxies that are useful for 2DGS/GOF-style comparisons while keeping the existing GPIS gate and Tanks-and-Temples metric conventions.

## Surface proxies

`extract_3dgs_gpis_surfaces` consumes a trained `point_cloud.ply` and, optionally, a GPIS gate `.npz` aligned to the Gaussian order.

It can write three representations for each variant:

- `centers`: the retained Gaussian centers as a point cloud.
- `surfels`: a 2DGS-style oriented quad for each Gaussian. The shortest Gaussian scale axis is treated as a normal proxy and the two larger axes define the tangent footprint.
- `opacity_field`: a GOF-style opacity-field proxy. The command evaluates an alpha-weighted Gaussian field on a bounded grid and writes the occupied voxel boundary as an isosurface mesh. This is dependency-light and does not require marching cubes or Open3D.

```bash
extract_3dgs_gpis_surfaces \
  --input-ply outputs/trained_3dgs/point_cloud/iteration_30000/point_cloud.ply \
  --gate-path real_scenes/ignatius/trained_3dgs_gpis_gates.npz \
  --output-dir outputs/ignatius/surfaces \
  --method-name ignatius_gpis_surface \
  --gate-thresholds 0.25 0.50 0.75 \
  --extraction-modes centers surfels opacity_field \
  --opacity-field-resolution 64
```

The command writes:

```text
<method>_surface_manifest.csv
<method>_surface_status.json
<method>_surface_report.md
<method>_<variant>_centers.ply
<method>_<variant>_surfels.ply
<method>_<variant>_opacity_field.ply
```

The manifest records the variant, extraction method, retained Gaussian count, retention fraction, gate threshold, and mesh vertex/face counts.

## Geometry comparison

`evaluate_3dgs_gpis_surfaces` evaluates all rows in the surface manifest against ground-truth geometry using the same nearest-neighbor geometry metrics as the existing Tanks-and-Temples evaluator: accuracy, completion, Chamfer L1/L2, precision, recall, and F-score at configured thresholds.

```bash
evaluate_3dgs_gpis_surfaces \
  --manifest-path outputs/ignatius/surfaces/ignatius_gpis_surface_surface_manifest.csv \
  --ground-truth-path data/tanks_temples/Ignatius/Ignatius_COLMAP.ply \
  --alignment-path real_scenes/ignatius/tanks_temples_alignment.txt \
  --crop-path real_scenes/ignatius/tanks_temples_crop.json \
  --output-dir outputs/ignatius/surface_geometry \
  --method-name ignatius_gpis_surface \
  --thresholds 0.01 0.02 0.05 0.10
```

The evaluator samples mesh surfaces by triangle area before computing metrics. Point-cloud rows use their vertices directly, with deterministic subsampling when `--max-pred-points` is set.

## Recommended paper table

For each scene, report at least:

| variant | extraction | retained | Chamfer L1 | F@0.01 | F@0.02 | F@0.05 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | centers | all | | | | |
| gate_ge_0p5 | centers | filtered | | | | |
| gate_ge_0p5 | surfels | filtered | | | | |
| gate_ge_0p5 | opacity_field | filtered | | | | |

This separates three claims:

1. whether GPIS gating improves the original 3DGS center geometry;
2. whether anisotropic Gaussian orientation produces a better 2DGS-style surface proxy;
3. whether an opacity-field extraction produces a better GOF-style surface proxy.

## Notes and limitations

The `opacity_field` extractor is a conservative, dependency-free voxel boundary extractor rather than a full GOF implementation. It is intended for controlled comparisons and ablations. For final visualization-quality meshes, export the same field to a marching-cubes implementation or Open3D post-processing step.
