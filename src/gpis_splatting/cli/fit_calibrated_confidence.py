from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.calibrated_confidence_api import (
    ConfidenceFeatureConfig,
    ConfidenceFitConfig,
    ConfidenceSplitConfig,
    run_calibrated_confidence_fit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a reusable calibrated GPIS confidence model with leakage-aware validation.")
    parser.add_argument("--field-scores-path", required=True, help="CSV containing GPIS field-score diagnostics and nearest-GT labels.")
    parser.add_argument("--metadata-path", default=None, help="Optional candidate metadata CSV, e.g. hard-negative source_splat_index groups.")
    parser.add_argument("--output-dir", default=None, help="Defaults to the field-score CSV directory.")
    parser.add_argument("--method-name", default=None, help="Artifact prefix. Defaults to the field-score CSV stem.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--label-column", default="nearest_gt_distance")
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=None,
        help="Explicit calibrated-confidence features. Leakage columns are rejected unless --allow-label-like-features is set.",
    )
    parser.add_argument("--extra-feature-columns", nargs="+", default=[])
    parser.add_argument("--exclude-columns", nargs="+", default=[])
    parser.add_argument("--no-score-columns", action="store_true", help="Do not auto-select score_* columns as features.")
    parser.add_argument("--include-coordinate-columns", action="store_true", help="Allow query/eval coordinate columns as model features.")
    parser.add_argument("--no-derived-features", action="store_true", help="Disable deterministic GPIS-derived features.")
    parser.add_argument("--allow-label-like-features", action="store_true", help="Allow leakage-prone columns. Intended only for deliberate baselines.")
    parser.add_argument("--group-columns", nargs="+", default=[], help="Columns whose rows must stay in the same split, e.g. source_splat_index.")
    parser.add_argument("--no-auto-group-columns", action="store_true")
    parser.add_argument("--spatial-cell-size", type=float, default=None, help="Optional spatial grouping cell size.")
    parser.add_argument("--coordinate-columns", nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--validation-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--baseline-score-columns", nargs="+", default=None)
    parser.add_argument("--isotonic-score-columns", nargs="+", default=None)
    parser.add_argument("--logistic-iterations", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--selection-metric", default="brier", choices=["brier", "nll", "ece", "auc", "average_precision"])
    parser.add_argument("--no-reliability-plots", action="store_true")
    parser.add_argument("--gate-count", type=int, default=None)
    parser.add_argument("--missing-gate-value", type=float, default=0.0)
    return parser


def build_config(args: argparse.Namespace) -> ConfidenceFitConfig:
    return ConfidenceFitConfig(
        thresholds=tuple(args.thresholds),
        label_column=args.label_column,
        feature_config=ConfidenceFeatureConfig(
            feature_columns=None if args.feature_columns is None else tuple(args.feature_columns),
            extra_feature_columns=tuple(args.extra_feature_columns),
            exclude_columns=tuple(args.exclude_columns),
            include_score_columns=not args.no_score_columns,
            include_coordinate_columns=args.include_coordinate_columns,
            include_derived_features=not args.no_derived_features,
            allow_label_like_features=args.allow_label_like_features,
        ),
        split_config=ConfidenceSplitConfig(
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            group_columns=tuple(args.group_columns),
            auto_group_columns=not args.no_auto_group_columns,
            spatial_cell_size=args.spatial_cell_size,
            coordinate_columns=None if args.coordinate_columns is None else tuple(args.coordinate_columns),
        ),
        baseline_score_columns=tuple(args.baseline_score_columns) if args.baseline_score_columns is not None else ConfidenceFitConfig().baseline_score_columns,
        isotonic_score_columns=tuple(args.isotonic_score_columns) if args.isotonic_score_columns is not None else ConfidenceFitConfig().isotonic_score_columns,
        logistic_iterations=args.logistic_iterations,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        num_bins=args.num_bins,
        selection_metric=args.selection_metric,
        write_reliability_plots=not args.no_reliability_plots,
        gate_count=args.gate_count,
        missing_gate_value=args.missing_gate_value,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_calibrated_confidence_fit(
        field_scores_path=Path(args.field_scores_path),
        output_dir=None if args.output_dir is None else Path(args.output_dir),
        method_name=args.method_name,
        metadata_path=None if args.metadata_path is None else Path(args.metadata_path),
        config=build_config(args),
    )
    for key, path in result.artifacts.items():
        if isinstance(path, dict):
            for sub_key, sub_path in path.items():
                print(f"Wrote {key}[{sub_key}]: {sub_path}")
        else:
            print(f"Wrote {key}: {path}")


if __name__ == "__main__":
    main()
