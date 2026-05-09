from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.gpis_training_prior import GpisTrainingPriorConfig, export_gpis_training_prior


def none_or_int(value: str) -> int | None:
    return None if value.lower() in {"none", "null", "-1"} else int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export calibrated GPIS confidence as a training-time 3DGS prior.")
    parser.add_argument("--input-ply", required=True, help="Trained or initialized 3DGS point_cloud.ply.")
    parser.add_argument("--gate-path", required=True, help="NPZ with calibrated confidence gate/confidence values.")
    parser.add_argument("--field-scores-path", default=None, help="Optional GPIS field-score CSV for uncertainty/evidence signals.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="gpis_confidence_training_prior")
    parser.add_argument("--initialization-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--densify-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--prune-confidence-threshold", type=float, default=0.25)
    parser.add_argument("--high-uncertainty-quantile", type=float, default=0.8)
    parser.add_argument("--opacity-regularization-strength", type=float, default=1.0)
    parser.add_argument("--opacity-scale-floor", type=float, default=0.05)
    parser.add_argument("--clone-top-count", type=none_or_int, default=None)
    parser.add_argument("--opacity-mode", choices=("logit", "linear"), default="logit")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = GpisTrainingPriorConfig(
        initialization_confidence_threshold=args.initialization_confidence_threshold,
        densify_confidence_threshold=args.densify_confidence_threshold,
        prune_confidence_threshold=args.prune_confidence_threshold,
        high_uncertainty_quantile=args.high_uncertainty_quantile,
        opacity_regularization_strength=args.opacity_regularization_strength,
        opacity_scale_floor=args.opacity_scale_floor,
        clone_top_count=args.clone_top_count,
        opacity_mode=args.opacity_mode,
    )
    result = export_gpis_training_prior(
        input_ply_path=Path(args.input_ply),
        gate_path=Path(args.gate_path),
        field_scores_path=None if args.field_scores_path is None else Path(args.field_scores_path),
        output_dir=Path(args.output_dir),
        method_name=args.method_name,
        config=cfg,
    )
    print(f"Wrote {result['prior_path']}")
    print(f"Wrote {result['initialization_seed_ply_path']}")
    print(f"Wrote {result['trainer_hooks_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"initialization candidates: {result['status']['initialization_candidate_count']}")
    print(f"densify candidates: {result['status']['densify_candidate_count']}")
    print(f"prune candidates: {result['status']['prune_candidate_count']}")


if __name__ == "__main__":
    main()
