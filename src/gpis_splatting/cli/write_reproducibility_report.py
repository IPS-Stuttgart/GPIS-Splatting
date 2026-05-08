from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.reproducibility import write_reproducibility_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a Markdown reproducibility report for an experiment directory.")
    parser.add_argument("--experiment-root", required=True, help="Directory containing experiment artifacts.")
    parser.add_argument("--output", default=None, help="Report path. Defaults to <experiment-root>/reproducibility_report.md.")
    parser.add_argument("--config", default=None, help="Optional JSON preset/config used to produce the run.")
    parser.add_argument("--command", action="append", default=[], help="Command line to record. May be passed multiple times.")
    parser.add_argument("--commit", default=None, help="Git commit to record. Defaults to auto-detection when possible.")
    parser.add_argument("--include", action="append", default=None, help="Artifact glob to include. May be passed multiple times.")
    parser.add_argument("--max-files", type=int, default=200, help="Maximum number of artifact manifest entries.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report_path = write_reproducibility_report(
        experiment_root=args.experiment_root,
        output_path=args.output,
        config_path=args.config,
        command_lines=args.command,
        commit=args.commit,
        include_patterns=args.include,
        max_files=args.max_files,
    )
    print(f"Wrote {Path(report_path)}")


if __name__ == "__main__":
    main()
