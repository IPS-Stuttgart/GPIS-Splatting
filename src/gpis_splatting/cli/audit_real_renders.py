from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.cli.evaluate_real_renders import str_to_bool
from gpis_splatting.real_render_audit import audit_real_renders


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit real render image comparisons for path mistakes, exact matches, and pixel-difference stats.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/evaluations.")
    parser.add_argument("--method-name", default="method")
    parser.add_argument("--split", default="test")
    parser.add_argument("--require-all", type=str_to_bool, default=True)
    parser.add_argument("--max-panels", type=int, default=16, help="Maximum target/prediction/difference panels to write. Use 0 to skip panels.")
    parser.add_argument("--fail-on-suspicious", type=str_to_bool, default=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scene_dir is None and args.scene is None:
        raise ValueError("Pass either --scene-dir or --scene.")
    scene_dir = Path(args.scene_dir) if args.scene_dir is not None else Path(args.prepared_root) / args.scene
    output_dir = Path(args.output_dir) if args.output_dir is not None else scene_dir / "evaluations"
    status = audit_real_renders(
        scene_dir=scene_dir,
        predictions_dir=args.predictions_dir,
        output_dir=output_dir,
        method_name=args.method_name,
        split=args.split,
        require_all=args.require_all,
        max_panels=args.max_panels,
        fail_on_suspicious=args.fail_on_suspicious,
    )
    summary = status["summary"]
    print(f"Wrote {status['metrics_path']}")
    print(f"Wrote {status['summary_path']}")
    print(f"Wrote {status['report_path']}")
    print(f"warnings: {summary['warning_count']}")
    print(f"suspicious_infinite_psnr: {summary['suspicious_infinite_psnr_count']}")


if __name__ == "__main__":
    main()
