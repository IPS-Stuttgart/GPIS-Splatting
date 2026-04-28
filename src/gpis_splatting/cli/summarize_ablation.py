from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")


SUMMARY_COLUMNS = [
    "shape",
    "feedback_iterations",
    "feedback_selector",
    "scene",
    "psnr_gpis",
    "psnr_feedback",
    "psnr_delta",
    "rmse_sdf",
    "feedback_rmse_sdf",
    "rmse_delta",
    "iou_inside",
    "feedback_iou_inside",
    "iou_delta",
    "feedback_selected_splats",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize feedback ablation metrics into tables and plots.")
    parser.add_argument("--ablation-root", default="experiments/feedback_ablation")
    parser.add_argument("--metrics-path", default=None, help="Defaults to <ablation-root>/ablation_metrics.csv.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <ablation-root>/summary.")
    parser.add_argument("--view", default="auto", help="View suffix for PSNR columns, or auto.")
    parser.add_argument("--primary-metric", choices=("psnr_delta", "rmse_delta", "iou_delta"), default="psnr_delta")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metrics_path = Path(args.metrics_path) if args.metrics_path else Path(args.ablation_root) / "ablation_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}. Run run_ablation first or pass --metrics-path.")

    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(metrics_path)
    summary, view = build_summary(metrics, view=args.view)
    winners = select_winners(summary, primary_metric=args.primary_metric)

    summary_path = output_dir / "ablation_summary.csv"
    winners_path = output_dir / "ablation_winners.csv"
    markdown_path = output_dir / "ablation_summary.md"
    summary.to_csv(summary_path, index=False)
    winners.to_csv(winners_path, index=False)
    plot_paths = write_summary_plots(summary, output_dir)
    write_markdown_summary(markdown_path, metrics_path, summary_path, winners_path, plot_paths, winners, view, args.primary_metric)

    print(f"Wrote {summary_path}")
    print(f"Wrote {winners_path}")
    print(f"Wrote {markdown_path}")
    for path in plot_paths:
        print(f"Wrote {path}")


def build_summary(metrics: pd.DataFrame, *, view: str = "auto") -> tuple[pd.DataFrame, str]:
    view = _resolve_view(metrics, view)
    data = metrics.copy()
    if "feedback_selector" not in data:
        data["feedback_selector"] = "gate"
        data.loc[data["feedback_iterations"] == 0, "feedback_selector"] = "none"

    psnr_gpis_col = f"psnr_gpis_{view}"
    psnr_feedback_col = f"psnr_feedback_{view}"
    summary = pd.DataFrame(
        {
            "shape": data["shape"],
            "feedback_iterations": data["feedback_iterations"].astype(int),
            "feedback_selector": data["feedback_selector"].fillna("none"),
            "scene": data.get("scene", ""),
            "psnr_gpis": _column_or_nan(data, psnr_gpis_col),
            "psnr_feedback": _column_or_nan(data, psnr_feedback_col),
            "rmse_sdf": _column_or_nan(data, "rmse_sdf"),
            "feedback_rmse_sdf": _column_or_nan(data, "feedback_rmse_sdf"),
            "iou_inside": _column_or_nan(data, "iou_inside"),
            "feedback_iou_inside": _column_or_nan(data, "feedback_iou_inside"),
            "feedback_selected_splats": _column_or_nan(data, "feedback_selected_splats").fillna(0).astype(int),
        }
    )
    summary["psnr_delta"] = summary["psnr_feedback"] - summary["psnr_gpis"]
    summary["rmse_delta"] = summary["feedback_rmse_sdf"] - summary["rmse_sdf"]
    summary["iou_delta"] = summary["feedback_iou_inside"] - summary["iou_inside"]
    baseline = summary["feedback_selector"] == "none"
    summary.loc[baseline, ["psnr_delta", "rmse_delta", "iou_delta"]] = 0.0
    return summary[SUMMARY_COLUMNS], view


def select_winners(summary: pd.DataFrame, *, primary_metric: str = "psnr_delta") -> pd.DataFrame:
    candidates = summary[(summary["feedback_iterations"] > 0) & (summary["feedback_selector"] != "none")].copy()
    candidates = candidates[candidates[primary_metric].notna()]
    if candidates.empty:
        return candidates

    ascending = primary_metric == "rmse_delta"
    candidates = candidates.sort_values(
        ["shape", primary_metric, "feedback_iterations", "feedback_selector"],
        ascending=[True, ascending, True, True],
    )
    if primary_metric != "rmse_delta":
        candidates = candidates.sort_values(
            ["shape", primary_metric, "feedback_iterations", "feedback_selector"],
            ascending=[True, False, True, True],
        )
    winners = candidates.groupby("shape", as_index=False).head(1).reset_index(drop=True)
    winners["winner_metric"] = primary_metric
    return winners


def write_summary_plots(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    feedback = summary[(summary["feedback_iterations"] > 0) & (summary["feedback_selector"] != "none")].copy()
    if feedback.empty:
        return []

    feedback["setting"] = feedback["feedback_selector"] + " fb" + feedback["feedback_iterations"].astype(str)
    plot_specs = [
        ("psnr_delta", "Feedback PSNR delta vs GPIS gate", "PSNR delta (dB)", "psnr_delta_by_selector.png"),
        ("rmse_delta", "Feedback RMSE delta vs base GPIS", "RMSE delta (lower is better)", "rmse_delta_by_selector.png"),
        ("feedback_selected_splats", "Selected pseudo-splats", "Selected splats", "selected_splats_by_selector.png"),
    ]
    paths = []
    for value_col, title, ylabel, filename in plot_specs:
        path = output_dir / filename
        _plot_grouped_bars(feedback, value_col, title, ylabel, path)
        paths.append(path)

    trend_path = output_dir / "feedback_iteration_trend.png"
    _plot_iteration_trend(feedback, trend_path)
    paths.append(trend_path)
    return paths


def write_markdown_summary(
    path: Path,
    metrics_path: Path,
    summary_path: Path,
    winners_path: Path,
    plot_paths: list[Path],
    winners: pd.DataFrame,
    view: str,
    primary_metric: str,
) -> None:
    lines = [
        "# Ablation Summary",
        "",
        f"- Input metrics: `{metrics_path}`",
        f"- Summary CSV: `{summary_path}`",
        f"- Winners CSV: `{winners_path}`",
        f"- PSNR view: `{view}`",
        f"- Winner metric: `{primary_metric}`",
        "",
        "## Generated Plots",
        "",
    ]
    for plot_path in plot_paths:
        lines.append(f"- `{plot_path}`")

    lines.extend(
        [
            "",
            "## Winners",
            "",
            _markdown_table(winners[_unique_columns(["shape", "feedback_iterations", "feedback_selector", primary_metric, "psnr_delta", "rmse_delta", "iou_delta"])])
            if not winners.empty
            else "No feedback rows were available for winner selection.",
            "",
            "## Metric Definitions",
            "",
            "- `psnr_delta = psnr_feedback - psnr_gpis`; higher is better.",
            "- `rmse_delta = feedback_rmse_sdf - rmse_sdf`; lower is better.",
            "- `iou_delta = feedback_iou_inside - iou_inside`; higher is better.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_view(metrics: pd.DataFrame, view: str) -> str:
    if view != "auto":
        if f"psnr_gpis_{view}" not in metrics:
            raise ValueError(f"Missing psnr_gpis_{view} in ablation metrics.")
        return view

    psnr_cols = sorted(column.removeprefix("psnr_gpis_") for column in metrics.columns if column.startswith("psnr_gpis_"))
    if not psnr_cols:
        raise ValueError("Ablation metrics do not contain any psnr_gpis_<view> columns.")
    return "front" if "front" in psnr_cols else psnr_cols[0]


def _column_or_nan(data: pd.DataFrame, column: str) -> pd.Series:
    if column in data:
        return data[column]
    return pd.Series([float("nan")] * len(data), index=data.index)


def _plot_grouped_bars(data: pd.DataFrame, value_col: str, title: str, ylabel: str, path: Path) -> None:
    pivot = data.pivot_table(index="shape", columns="setting", values=value_col, aggfunc="mean").sort_index()
    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Shape")
    ax.set_ylabel(ylabel)
    ax.legend(title="Setting", fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=160)
    plt.close(ax.figure)


def _plot_iteration_trend(data: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for selector, group in data.groupby("feedback_selector"):
        trend = group.groupby("feedback_iterations", as_index=True)[["psnr_delta", "rmse_delta"]].mean().sort_index()
        axes[0].plot(trend.index, trend["psnr_delta"], marker="o", label=selector)
        axes[1].plot(trend.index, trend["rmse_delta"], marker="o", label=selector)

    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Mean PSNR delta")
    axes[1].set_title("Mean RMSE delta")
    axes[0].set_ylabel("PSNR delta (dB)")
    axes[1].set_ylabel("RMSE delta")
    for axis in axes:
        axis.set_xlabel("Feedback iterations")
        axis.legend(title="Selector", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown_table(data: pd.DataFrame) -> str:
    if data.empty:
        return ""

    columns = list(data.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in data.iterrows():
        rows.append("| " + " | ".join(_format_markdown_value(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def _unique_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
