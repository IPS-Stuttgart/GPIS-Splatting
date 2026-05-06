from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.confidence import apply_calibrated_confidence_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a serialized GPIS calibrated-confidence model bundle to a field-score CSV.")
    parser.add_argument("--model-bundle-path", required=True, help="JSON model bundle written by calibrate_gpis_splat_scores.")
    parser.add_argument("--field-scores-path", required=True, help="CSV with GPIS field-score columns for the splats to score.")
    parser.add_argument("--output-path", default=None, help="Output calibrated confidence CSV. Defaults next to the field-score CSV.")
    parser.add_argument("--threshold", type=float, default=None, help="Apply only one geometry threshold from the bundle.")
    parser.add_argument("--gate-output-dir", default=None, help="Optional directory for gate-compatible NPZ export.")
    parser.add_argument("--gate-count", type=int, default=None, help="Optional full splat count for gate-compatible NPZ export.")
    parser.add_argument("--missing-gate-value", type=float, default=0.0, help="Gate value for unscored splats when --gate-count is provided.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = apply_calibrated_confidence_bundle(
        model_bundle_path=Path(args.model_bundle_path),
        field_scores_path=Path(args.field_scores_path),
        output_path=None if args.output_path is None else Path(args.output_path),
        threshold=args.threshold,
        gate_output_dir=None if args.gate_output_dir is None else Path(args.gate_output_dir),
        gate_count=args.gate_count,
        missing_gate_value=args.missing_gate_value,
    )
    print(f"Wrote {result['predictions_path']}")
    for label, path in result["gate_paths"].items():
        print(f"Wrote gate@{label}: {path}")


if __name__ == "__main__":
    main()
