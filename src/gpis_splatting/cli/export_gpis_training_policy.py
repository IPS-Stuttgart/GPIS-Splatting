from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.gpis_training_policy import export_training_policy


def none_or_int(value: str) -> int | None:
    lowered = value.strip().lower()
    return None if lowered in {"none", "null", "-1"} else int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export calibrated GPIS confidence as a runtime 3DGS training policy.")
    parser.add_argument("--input-ply", required=True, help="Trained or initialized 3DGS point_cloud.ply used as the Gaussian template and per-Gaussian target set.")
    parser.add_argument("--gate-path", required=True, help="NPZ containing calibrated confidence values. Accepted keys: gate, confidence, calibrated_confidence, raw_gate.")
    parser.add_argument("--field-scores-path", default=None, help="Optional CSV with GPIS uncertainty/evidence columns, e.g. distance_std and score_raw_surface_band.")
    parser.add_argument("--candidate-metadata-path", default=None, help="Optional hard-negative/candidate CSV with x/y/z and source_splat_index. Enables candidate-coordinate initialization.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="gpis_training_policy")
    parser.add_argument("--init-threshold", type=float, default=0.75)
    parser.add_argument("--densify-threshold", type=float, default=0.55)
    parser.add_argument("--prune-threshold", type=float, default=0.25)
    parser.add_argument("--uncertainty-quantile", type=float, default=0.8)
    parser.add_argument("--opacity-strength", type=float, default=1.0)
    parser.add_argument("--opacity-floor", type=float, default=0.05)
    parser.add_argument("--max-init-points", type=none_or_int, default=None)
    parser.add_argument("--opacity-mode", choices=("logit", "linear"), default="logit")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = export_training_policy(
        input_ply_path=Path(args.input_ply),
        gate_path=Path(args.gate_path),
        output_dir=Path(args.output_dir),
        field_scores_path=None if args.field_scores_path is None else Path(args.field_scores_path),
        candidate_metadata_path=None if args.candidate_metadata_path is None else Path(args.candidate_metadata_path),
        method_name=args.method_name,
        init_threshold=args.init_threshold,
        densify_threshold=args.densify_threshold,
        prune_threshold=args.prune_threshold,
        uncertainty_quantile=args.uncertainty_quantile,
        opacity_strength=args.opacity_strength,
        opacity_floor=args.opacity_floor,
        max_init_points=args.max_init_points,
        opacity_mode=args.opacity_mode,
    )
    print(f"Wrote {result['prior_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"initialization candidates: {result['status']['initialization_candidate_count']}")
    print(f"densify candidates: {result['status']['densify_candidate_count']}")
    print(f"prune candidates: {result['status']['prune_candidate_count']}")


if __name__ == "__main__":
    main()
