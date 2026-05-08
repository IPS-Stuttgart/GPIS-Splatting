# Depth and normal confidence supervision

This workflow augments real-scene GPIS training samples with weak geometric observations from depth, depth-confidence, normal, and normal-confidence maps.

The added command is:

```bash
augment_depth_normal_supervision \
  --scene ignatius_sparse12 \
  --depth-dir depth_maps \
  --depth-confidence-dir depth_confidence \
  --normal-dir normal_maps \
  --normal-confidence-dir normal_confidence \
  --output-samples-path real_depth_normal_samples.npz
```

The resulting file can be used by the existing real GPIS fitter:

```bash
fit_real_gpis \
  --scene ignatius_sparse12 \
  --samples-path real_depth_normal_samples.npz \
  --use-observation-noise true
```

## Observation model

Depth maps add pseudo-SDF samples:

- `depth_surface`: zero-level observations at unprojected depth pixels.
- `depth_free_space`: positive SDF observations between the camera and the depth point.

Normal maps optionally add signed offset observations:

- `depth_normal_positive`: positive SDF sample at `x + d n`.
- `depth_normal_negative`: negative SDF sample at `x - d n`.

Normals are oriented toward the observing camera. This matches the existing convention where samples between the camera and the surface are positive free-space samples, while samples behind the observed surface are negative.

## Confidence handling

Confidence maps are converted to per-observation noise via a linear mapping:

```text
noise = max_noise - confidence * (max_noise - min_noise)
```

Thus high-confidence depth or normal estimates receive low `observation_noise_std`, while uncertain estimates remain present but are down-weighted by `fit_real_gpis --use-observation-noise true`.

Depth confidence controls surface/free-space samples. Normal-offset samples use the minimum of depth confidence and normal confidence.

## Map naming

For each prepared frame, the command searches the map directory by frame file name, stem, frame index, and image id, with common suffixes such as `.npy`, `.npz`, `.png`, `.tif`, and `.jpg`.

Recommended layout:

```text
real_scenes/ignatius_sparse12/
  depth_maps/
    000000.npy
    000001.npy
  depth_confidence/
    000000.npy
    000001.npy
  normal_maps/
    000000.npy
    000001.npy
  normal_confidence/
    000000.npy
    000001.npy
```

Depth maps are assumed to be metric depths unless `--depth-scale` is provided. Normal maps can be camera-space or world-space via `--normal-space camera|world`.

## Practical notes

Use `--max-pixels-per-frame` and `--pixel-stride` to keep the dense depth supervision tractable for exact GPIS fitting. For monocular depth from a model such as Depth Anything or ZoeDepth, start with conservative noise ranges and let the confidence maps down-weight unreliable areas.
