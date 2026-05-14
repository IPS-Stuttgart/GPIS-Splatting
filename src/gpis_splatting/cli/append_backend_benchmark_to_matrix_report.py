from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = (
    "backend",
    "status",
    "fit_time_sec",
    "predict_time_sec",
    "queries_per_sec",
    "model_storage_mib",
    "mean_rmse_vs_dense",
    "variance_rmse_vs_dense",
    "gradient_rmse_vs_dense",
    "gate_rmse_vs_dense",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append a GPIS backend benchmark table to an existing GPIS/3DGS experiment-matrix report.")
    parser.add_argument("--matrix-report", required=True, help="Existing matrix Markdown report to update in place.")
    parser.add_argument("--backend-benchmark", required=True, help="CSV written by benchmark_gpis_backends.")
    parser.add_argument("--output-report", default=None, help="Optional output path. Defaults to updating --matrix-report in place.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    matrix_report = Path(args.matrix_report)
    backend_benchmark = Path(args.backend_benchmark)
    output_report = Path(args.output_report) if args.output_report else matrix_report
    append_backend_benchmark_to_report(matrix_report=matrix_report, backend_benchmark=backend_benchmark, output_report=output_report)
    print(f"Wrote {output_report}")


def append_backend_benchmark_to_report(*, matrix_report: Path, backend_benchmark: Path, output_report: Path) -> None:
    if not matrix_report.exists():
        raise FileNotFoundError(f"Missing matrix report: {matrix_report}")
    if not backend_benchmark.exists():
        raise FileNotFoundError(f"Missing backend benchmark CSV: {backend_benchmark}")
    report_text = matrix_report.read_text(encoding="utf-8").rstrip()
    benchmark = pd.read_csv(backend_benchmark)
    columns = [column for column in DEFAULT_COLUMNS if column in benchmark.columns]
    if not columns:
        raise ValueError("Backend benchmark CSV does not contain any supported report columns.")
    section = format_backend_benchmark_section(benchmark[columns], source_path=backend_benchmark)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report_text + "\n\n" + section + "\n", encoding="utf-8")


def format_backend_benchmark_section(table: pd.DataFrame, *, source_path: Path) -> str:
    return "\n".join(
        [
            "## GPIS Backend Benchmark",
            "",
            f"Source: `{source_path}`",
            "",
            markdown_table(table),
        ]
    )


def markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "No backend benchmark rows."
    columns = list(table.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(format_value(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    main()
