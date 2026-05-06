from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.real_score_calibration import default_feature_sets, default_topk_fractions, run_gpis_splat_score_calibration


def resolve_field_scores_path(args: argparse.Namespace) -> Path:
    if args.scene_dir is None and args.scene is None:
        if args.field_scores_path is None:
            raise ValueError("Pass --field-scores-path, or pass --scene/--scene-dir with --method-name.")
        return Path(args.field_scores_path)
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    if args.field_scores_path is None:
        if args.method_name is None:
            raise ValueError("Pass --field-scores-path or --method-name so the field-score CSV can be resolved.")
        return scene_dir / "evaluations" / f"{args.method_name}_gpis_field_scores.csv"
    requested = Path(args.field_scores_path)
    if requested.is_absolute() or requested.exists():
        return requested
    scene_relative = scene_dir / requested
    if scene_relative.exists():
        return scene_relative
    return scene_dir / "evaluations" / requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate GPIS-derived splat confidence from field-score diagnostics and nearest-GT labels.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--field-scores-path", default=None, help="CSV written by diagnose_tanks_temples_gpis_field_scores.")
    parser.add_argument("--output-dir", default=None, help="Defaults to the field-score CSV directory.")
    parser.add_argument("--method-name", default=None, help="Output prefix. Also resolves <method>_gpis_field_scores.csv when no explicit field-score path is passed.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--topk-fractions", type=float, nargs="+", default=list(default_topk_fractions()))
    parser.add_argument("--feature-sets", nargs="+", default=list(default_feature_sets()))
    parser.add_argument("--baseline-scores", nargs="+", default=None, help="score_* columns to compare through train-set min-max scaling.")
    parser.add_argument("--isotonic-scores", nargs="+", default=None, help="score_* columns to calibrate with isotonic regression.")
    parser.add_argument("--validation-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--logistic-iterations", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--num-bins", type=int, default=10, help="ECE bin count. Must be positive.")
    parser.add_argument("--gate-count", type=int, default=None, help="Optional full splat count for gate-compatible NPZ exports.")
    parser.add_argument(
        "--missing-gate-value",
        type=float,
        default=0.0,
        help="Gate value for unscored splat indices when --gate-count is used. Use 1.0 to preserve unscored trained 3DGS Gaussians during photometric evaluation.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_gpis_splat_score_calibration(
        field_scores_path=resolve_field_scores_path(args),
        output_dir=args.output_dir,
        method_name=args.method_name,
        thresholds=tuple(args.thresholds),
        topk_fractions=tuple(args.topk_fractions),
        feature_sets=tuple(args.feature_sets),
        baseline_scores=tuple(args.baseline_scores) if args.baseline_scores is not None else None,
        isotonic_scores=tuple(args.isotonic_scores) if args.isotonic_scores is not None else None,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        logistic_iterations=args.logistic_iterations,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        num_bins=args.num_bins,
        gate_count=args.gate_count,
        missing_gate_value=args.missing_gate_value,
    )
    print(f"Wrote {result['summary_path']}")
    print(f"Wrote {result['ranked_path']}")
    print(f"Wrote {result['predictions_path']}")
    print(f"Wrote {result['confidence_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    for row in result["status"]["best_by_threshold"]:
        print(
            f"best_calibrator@{row['geometry_threshold']:.6g}: "
            f"{row['method_name']} brier={row['brier']:.6g} auc={row['auc']} "
            f"top={row['best_topk_fraction']:.6g} f={row['best_f_score']:.6g}"
        )


if __name__ == "__main__":
    main()
