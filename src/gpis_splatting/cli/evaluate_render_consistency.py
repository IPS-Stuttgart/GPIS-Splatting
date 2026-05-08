from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.cli.evaluate_real_renders import str_to_bool
from gpis_splatting.render_consistency import DEFAULT_AA_DOWNSAMPLE_FACTORS, evaluate_render_consistency


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate adjacent-view, anti-aliasing, and optional multi-resolution consistency for real-scene render directories.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--predictions-dir", required=True, help="Base render directory to evaluate.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/evaluations.")
    parser.add_argument("--method-name", default="method")
    parser.add_argument("--split", default="test")
    parser.add_argument("--require-all", type=str_to_bool, default=True)
    parser.add_argument("--max-temporal-pairs", type=int, default=None, help="Limit evaluated adjacent-view pairs. Useful for smoke tests.")
    parser.add_argument("--max-view-translation", type=float, default=None, help="Only evaluate adjacent pairs whose camera-center distance is at most this value.")
    parser.add_argument("--max-view-rotation-deg", type=float, default=None, help="Only evaluate adjacent pairs whose relative camera rotation is at most this many degrees.")
    parser.add_argument(
        "--aa-downsample-factor",
        action="append",
        type=int,
        default=None,
        metavar="N",
        help=f"Run an anti-aliased downsample/upsample round-trip at factor N. May be passed multiple times; default: {list(DEFAULT_AA_DOWNSAMPLE_FACTORS)}.",
    )
    parser.add_argument("--disable-aa-roundtrip", type=str_to_bool, default=False, help="Disable built-in anti-aliasing round-trip diagnostics.")
    parser.add_argument(
        "--scale-predictions-dir",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Optional render directory from another resolution/AA setting. May be passed multiple times; compared against --predictions-dir.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    output_dir = Path(args.output_dir) if args.output_dir is not None else scene_dir / "evaluations"
    scale_dirs = parse_labeled_paths(args.scale_predictions_dir)
    aa_factors = () if args.disable_aa_roundtrip else tuple(args.aa_downsample_factor or DEFAULT_AA_DOWNSAMPLE_FACTORS)
    status = evaluate_render_consistency(
        scene_dir=scene_dir,
        predictions_dir=args.predictions_dir,
        output_dir=output_dir,
        method_name=args.method_name,
        split=args.split,
        scale_prediction_dirs=scale_dirs,
        require_all=args.require_all,
        max_temporal_pairs=args.max_temporal_pairs,
        max_view_translation=args.max_view_translation,
        max_view_rotation_deg=args.max_view_rotation_deg,
        aa_downsample_factors=aa_factors,
    )
    summary = status["summary"]
    print(f"Wrote {status['temporal_path']}")
    print(f"Wrote {status['scale_path']}")
    print(f"Wrote {status['aa_path']}")
    print(f"Wrote {status['summary_path']}")
    print(f"Wrote {status['report_path']}")
    print(f"temporal_pairs: {summary['temporal_pair_count']}")
    print(f"temporal_skipped_view_filter_count: {summary['temporal_skipped_view_filter_count']}")
    print(f"scale_comparisons: {summary['scale_image_count']}")
    print(f"aa_comparisons: {summary['aa_image_count']}")
    print(f"mean_temporal_instability_score: {summary['mean_temporal_instability_score']}")
    print(f"mean_view_instability_score: {summary['mean_view_instability_score']}")
    print(f"mean_scale_instability_score: {summary['mean_scale_instability_score']}")
    print(f"mean_aa_instability_score: {summary['mean_aa_instability_score']}")


def parse_labeled_paths(items: list[str]) -> dict[str, Path]:
    labeled: dict[str, Path] = {}
    for item in items:
        label, separator, value = item.partition("=")
        if not separator:
            path = Path(label)
            label = path.name or "variant"
        else:
            path = Path(value)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid empty scale label in {item!r}.")
        if label in labeled:
            raise ValueError(f"Duplicate scale label {label!r}.")
        labeled[label] = path
    return labeled


if __name__ == "__main__":
    main()
