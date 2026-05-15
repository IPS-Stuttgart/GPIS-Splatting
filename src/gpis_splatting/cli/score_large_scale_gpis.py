from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from gpis_splatting.gpis_backends import load_gpis_backend
from gpis_splatting.large_scale_gpis import (
    LargeScaleGPISScoreConfig,
    score_large_scale_gpis,
    write_large_scale_scores_npz,
    write_large_scale_stats_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream GPIS scores for large 3DGS center arrays with optional CUDA inducing-point inference.")
    parser.add_argument("--backend-model", required=True, help="Path to a saved GPIS backend/model npz file.")
    parser.add_argument("--points-npz", required=True, help="Path to an npz file containing query centers.")
    parser.add_argument("--points-key", default="centers", help="Array key in --points-npz. Defaults to centers.")
    parser.add_argument("--output", required=True, help="Output npz file for gate/prediction scores.")
    parser.add_argument("--stats-json", default=None, help="Optional JSON file for runtime and score summary statistics.")
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=None, help="Explicit query batch size. If omitted, an inducing backend batch size is estimated from --memory-budget-mib.")
    parser.add_argument("--memory-budget-mib", type=int, default=512, help="Approximate scratch-memory budget for inducing query batches.")
    parser.add_argument("--prediction-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prediction-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--gate-only", action="store_true", help="Only store the GPIS gate, not mean/variance/gradient/distance diagnostics.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    backend, metadata = load_gpis_backend(args.backend_model)
    points_npz = np.load(args.points_npz, allow_pickle=False)
    if args.points_key not in points_npz.files:
        raise KeyError(f"{args.points_npz} does not contain key {args.points_key!r}. Available keys: {', '.join(points_npz.files)}")
    points = torch.from_numpy(points_npz[args.points_key])
    config = LargeScaleGPISScoreConfig(
        epsilon=args.epsilon,
        batch_size=args.batch_size,
        memory_budget_mib=args.memory_budget_mib,
        prediction_device=args.prediction_device,
        prediction_dtype=args.prediction_dtype,
        output_device=args.output_device,
        include_prediction=not args.gate_only,
    )
    result = score_large_scale_gpis(backend, points, config=config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_large_scale_scores_npz(output_path, result)
    stats_path = Path(args.stats_json) if args.stats_json else output_path.with_suffix(".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    write_large_scale_stats_json(stats_path, result, config=config)
    print(f"Wrote {output_path}")
    print(f"Wrote {stats_path}")
    print(f"backend_metadata: {metadata}")
    print(f"points_per_sec: {result.stats['points_per_sec']}")


if __name__ == "__main__":
    main()
