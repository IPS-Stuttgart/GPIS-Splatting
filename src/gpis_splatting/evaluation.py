from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.scenes import available_shapes
from gpis_splatting.serialization import read_json, write_json

EVALUATION_PRESETS: dict[str, dict[str, Any]] = {
    "synthetic_ci": {
        "description": "Small deterministic evaluation for pull requests and local smoke checks.",
        "ablation": {
            "shapes": ["sphere"],
            "feedback_iterations": [0, 1],
            "feedback_selectors": ["gate", "uncertainty"],
            "num_points": 45,
            "noise_std": 0.03,
            "grid_size": 8,
            "lengthscale": 0.8,
            "variance": 1.0,
            "image_size": 32,
            "num_splats": 55,
            "epsilon": 0.11,
            "view": "front",
            "seed": 7,
            "feedback_pseudo_points": 6,
            "feedback_min_gate": 0.0,
        },
        "targets": {
            "max_rmse_sdf": 0.6,
            "min_iou_inside": 0.2,
            "min_psnr_gpis": 5.0,
        },
    },
    "synthetic_quick": {
        "description": "Moderate synthetic benchmark across representative shapes.",
        "ablation": {
            "shapes": ["sphere", "torus", "non_star_convex"],
            "feedback_iterations": [0, 1, 2],
            "feedback_selectors": ["gate", "uncertainty", "uncertainty_diverse"],
            "num_points": 120,
            "noise_std": 0.035,
            "grid_size": 18,
            "lengthscale": 0.8,
            "variance": 1.0,
            "image_size": 72,
            "num_splats": 240,
            "epsilon": 0.09,
            "view": "front",
            "seed": 7,
            "feedback_pseudo_points": 48,
            "feedback_min_gate": 0.35,
        },
        "targets": {
            "max_rmse_sdf": 0.65,
            "min_iou_inside": 0.15,
            "min_psnr_gpis": 5.0,
        },
    },
    "synthetic_full": {
        "description": "Full synthetic benchmark for local research runs.",
        "ablation": {
            "shapes": list(available_shapes()),
            "feedback_iterations": [0, 1, 2],
            "feedback_selectors": ["gate", "uncertainty", "uncertainty_diverse"],
            "num_points": 180,
            "noise_std": 0.035,
            "grid_size": 28,
            "lengthscale": 0.8,
            "variance": 1.0,
            "image_size": 96,
            "num_splats": 420,
            "epsilon": 0.09,
            "view": "front",
            "seed": 7,
            "feedback_pseudo_points": 80,
            "feedback_min_gate": 0.45,
        },
        "targets": {
            "max_rmse_sdf": 0.65,
            "min_iou_inside": 0.15,
            "min_psnr_gpis": 5.0,
        },
    },
}


def preset_names() -> list[str]:
    return sorted(EVALUATION_PRESETS)


def get_evaluation_preset(name: str) -> dict[str, Any]:
    if name not in EVALUATION_PRESETS:
        raise KeyError(f"Unknown evaluation preset {name!r}. Available presets: {', '.join(preset_names())}")
    return EVALUATION_PRESETS[name]


def build_ablation_args(
    preset: dict[str, Any],
    *,
    output_root: str | Path,
    experiment_name: str,
    seed: int | None = None,
) -> list[str]:
    config = dict(preset["ablation"])
    if seed is not None:
        config["seed"] = seed

    args = [
        "--output-root",
        str(output_root),
        "--experiment-name",
        experiment_name,
    ]
    _append_arg(args, "--shapes", config["shapes"])
    _append_arg(args, "--feedback-iterations", config["feedback_iterations"])
    _append_arg(args, "--feedback-selectors", config["feedback_selectors"])
    for key in (
        "num_points",
        "noise_std",
        "grid_size",
        "lengthscale",
        "variance",
        "image_size",
        "num_splats",
        "epsilon",
        "view",
        "seed",
        "feedback_pseudo_points",
        "feedback_min_gate",
        "feedback_pseudo_noise_std",
    ):
        if key in config and config[key] is not None:
            _append_arg(args, f"--{key.replace('_', '-')}", config[key])
    return args


def expected_ablation_rows(preset: dict[str, Any]) -> int:
    config = preset["ablation"]
    feedback_selectors = config["feedback_selectors"]
    rows_per_shape = sum(1 if iteration == 0 else len(feedback_selectors) for iteration in config["feedback_iterations"])
    return rows_per_shape * len(config["shapes"])


def evaluate_ablation_artifacts(
    *,
    ablation_root: str | Path,
    preset_name: str,
    preset: dict[str, Any],
    primary_metric: str,
    benchmark_target: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(ablation_root)
    metrics_path = root / "ablation_metrics.csv"
    summary_path = root / "summary" / "ablation_summary.csv"
    winners_path = root / "summary" / "ablation_winners.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}. Run run_ablation first.")
    if not summary_path.exists() or not winners_path.exists():
        raise FileNotFoundError(f"Missing summary artifacts under {root / 'summary'}. Run summarize_ablation first.")

    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path)
    winners = pd.read_csv(winners_path)
    view = _resolve_view(metrics)
    checks = build_evaluation_checks(metrics, summary, winners, preset=preset, view=view)
    passed = all(bool(check["passed"]) for check in checks)

    status = {
        "preset": preset_name,
        "description": preset["description"],
        "passed": passed,
        "primary_metric": primary_metric,
        "view": view,
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "winners_path": str(winners_path),
        "checks": checks,
        "best": _best_rows(summary, winners, primary_metric=primary_metric),
        "benchmark_target": _load_benchmark_target(benchmark_target),
    }
    return status


def write_evaluation_artifacts(
    *,
    output_dir: str | Path,
    status: dict[str, Any],
    preset: dict[str, Any],
    preset_name: str,
    ablation_args: list[str],
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks_path = out_dir / "evaluation_checks.csv"
    status_path = out_dir / "evaluation_status.json"
    config_path = out_dir / "evaluation_config.json"
    report_path = out_dir / "evaluation_report.md"

    pd.DataFrame(status["checks"]).to_csv(checks_path, index=False)
    write_json(status_path, status)
    write_json(
        config_path,
        {
            "preset": preset_name,
            "description": preset["description"],
            "ablation": preset["ablation"],
            "targets": preset["targets"],
            "ablation_args": ablation_args,
        },
    )
    report_path.write_text(format_evaluation_report(status, checks_path, config_path), encoding="utf-8")
    return {
        "checks": checks_path,
        "status": status_path,
        "config": config_path,
        "report": report_path,
    }


def build_evaluation_checks(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    winners: pd.DataFrame,
    *,
    preset: dict[str, Any],
    view: str,
) -> list[dict[str, Any]]:
    targets = preset["targets"]
    expected_rows = expected_ablation_rows(preset)
    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "expected_row_count",
        len(metrics) == expected_rows,
        len(metrics),
        expected_rows,
        "One metrics row is expected for every shape, feedback depth, and selector case.",
    )

    base_columns = ["rmse_sdf", "iou_inside", "nll_distance", "brier_inside", "ece_inside", f"psnr_gpis_{view}"]
    missing_base = [column for column in base_columns if column not in metrics.columns]
    _add_check(checks, "base_metric_columns_present", not missing_base, len(missing_base), 0, ", ".join(missing_base))
    if not missing_base:
        _add_check(
            checks,
            "base_metrics_not_nan",
            not metrics[base_columns].isna().any().any(),
            int(metrics[base_columns].isna().sum().sum()),
            0,
            "Required base metrics should be present for every row.",
        )

    feedback_rows = metrics[metrics["feedback_iterations"] > 0] if "feedback_iterations" in metrics else pd.DataFrame()
    feedback_columns = ["feedback_rmse_sdf", "feedback_iou_inside", f"psnr_feedback_{view}", "feedback_selected_splats"]
    missing_feedback = [column for column in feedback_columns if column not in metrics.columns]
    _add_check(
        checks,
        "feedback_metric_columns_present",
        feedback_rows.empty or not missing_feedback,
        len(missing_feedback),
        0,
        ", ".join(missing_feedback),
    )
    if not feedback_rows.empty and not missing_feedback:
        _add_check(
            checks,
            "feedback_metrics_not_nan",
            not feedback_rows[feedback_columns].isna().any().any(),
            int(feedback_rows[feedback_columns].isna().sum().sum()),
            0,
            "Feedback rows should include geometry, render, and selection metrics.",
        )

    if "rmse_sdf" in metrics:
        _add_check(checks, "max_rmse_sdf", float(metrics["rmse_sdf"].max()), "<=", targets["max_rmse_sdf"])
    if "iou_inside" in metrics:
        _add_check(checks, "min_iou_inside", float(metrics["iou_inside"].min()), ">=", targets["min_iou_inside"])
    psnr_col = f"psnr_gpis_{view}"
    if psnr_col in metrics:
        _add_check(checks, f"min_{psnr_col}", float(metrics[psnr_col].min()), ">=", targets["min_psnr_gpis"])
    if {"brier_inside", "ece_inside"}.issubset(metrics.columns):
        probability_ok = bool(metrics["brier_inside"].between(0.0, 1.0).all() and metrics["ece_inside"].between(0.0, 1.0).all())
        _add_check(checks, "calibration_metrics_in_unit_interval", probability_ok, int(probability_ok), 1)
    _add_check(
        checks,
        "summary_rows_match_metrics",
        len(summary) == len(metrics),
        len(summary),
        len(metrics),
        "The summary should preserve one row per ablation case.",
    )
    _add_check(
        checks,
        "winner_rows_available",
        not winners.empty,
        len(winners),
        "> 0",
        "At least one feedback setting should be selected as a winner.",
    )
    return checks


def format_evaluation_report(status: dict[str, Any], checks_path: Path, config_path: Path) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- Preset: `{status['preset']}`",
        f"- Description: {status['description']}",
        f"- Passed: `{status['passed']}`",
        f"- Primary metric: `{status['primary_metric']}`",
        f"- Render view: `{status['view']}`",
        f"- Metrics: `{status['metrics_path']}`",
        f"- Summary: `{status['summary_path']}`",
        f"- Winners: `{status['winners_path']}`",
        f"- Checks CSV: `{checks_path}`",
        f"- Config: `{config_path}`",
        "",
        "## Checks",
        "",
        _markdown_table(pd.DataFrame(status["checks"])),
        "",
        "## Best Rows",
        "",
        _markdown_table(pd.DataFrame(status["best"])),
    ]
    if status.get("benchmark_target"):
        target = status["benchmark_target"]
        lines.extend(
            [
                "",
                "## External Target",
                "",
                f"- Name: `{target.get('name', '')}`",
                f"- Dataset: {target.get('dataset', '')}",
                f"- Protocol: {target.get('protocol_url', '')}",
                f"- Train views: `{target.get('train_views', '')}`",
                f"- Primary baseline: `{target.get('primary_baseline', '')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _append_arg(args: list[str], flag: str, value: Any) -> None:
    args.append(flag)
    if isinstance(value, list | tuple):
        args.extend(str(item) for item in value)
    else:
        args.append(str(value))


def _add_check(checks: list[dict[str, Any]], name: str, passed_or_value: bool | float, value: Any, target: Any, details: str = "") -> None:
    if isinstance(passed_or_value, bool):
        passed = passed_or_value
        actual = value
        comparator = "=="
    else:
        comparator = str(value)
        actual = passed_or_value
        if comparator == "<=":
            passed = float(actual) <= float(target)
        elif comparator == ">=":
            passed = float(actual) >= float(target)
        else:
            raise ValueError(f"Unsupported comparator {comparator!r}.")
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "value": _json_scalar(actual),
            "target": _json_scalar(target),
            "details": details,
        }
    )


def _resolve_view(metrics: pd.DataFrame) -> str:
    views = sorted(column.removeprefix("psnr_gpis_") for column in metrics.columns if column.startswith("psnr_gpis_"))
    if not views:
        raise ValueError("Ablation metrics do not contain any psnr_gpis_<view> columns.")
    return "front" if "front" in views else views[0]


def _best_rows(summary: pd.DataFrame, winners: pd.DataFrame, *, primary_metric: str) -> list[dict[str, Any]]:
    if winners.empty:
        return []
    columns = [
        column
        for column in dict.fromkeys(("shape", "feedback_iterations", "feedback_selector", primary_metric, "psnr_delta", "rmse_delta", "iou_delta"))
        if column in winners
    ]
    return [_json_row(row) for row in winners[columns].to_dict(orient="records")]


def _load_benchmark_target(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Missing benchmark target file: {target_path}")
    return read_json(target_path)


def _markdown_table(data: pd.DataFrame) -> str:
    if data.empty:
        return "No rows."
    columns = list(data.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in data.iterrows():
        rows.append("| " + " | ".join(_format_value(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def _json_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_scalar(value) for key, value in row.items()}


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _format_value(value: Any) -> str:
    value = _json_scalar(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
