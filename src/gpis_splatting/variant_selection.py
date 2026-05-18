from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpis_splatting.serialization import write_json

DEFAULT_PARETO_PSNR_DROP_TOLERANCE = 0.2
PARETO_OBJECTIVES = ("min_retained", "max_psnr", "max_ssim", "min_lpips")


def select_psnr_constrained_pareto_variant(
    *,
    comparison_path: str | Path,
    output_dir: str | Path | None = None,
    method_name: str | None = None,
    baseline_variant: str = "baseline",
    psnr_drop_tolerance: float = DEFAULT_PARETO_PSNR_DROP_TOLERANCE,
    objective: str = "min_retained",
) -> dict[str, Any]:
    comparison_file = Path(comparison_path)
    annotated, summary = annotate_psnr_constrained_pareto(
        pd.read_csv(comparison_file),
        baseline_variant=baseline_variant,
        psnr_drop_tolerance=psnr_drop_tolerance,
        objective=objective,
    )
    out_dir = Path(output_dir) if output_dir is not None else comparison_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = method_name or infer_selection_prefix(comparison_file)
    selection_path = out_dir / f"{prefix}_3dgs_pareto_selection.csv"
    status_path = out_dir / f"{prefix}_3dgs_pareto_selection_status.json"
    report_path = out_dir / f"{prefix}_3dgs_pareto_selection_report.md"
    annotated.to_csv(selection_path, index=False)
    status = {
        "schema_version": 1,
        "comparison_path": str(comparison_file),
        "selection_path": str(selection_path),
        "report_path": str(report_path),
        **summary,
    }
    write_json(status_path, status)
    report_path.write_text(format_psnr_constrained_pareto_report(status, annotated), encoding="utf-8")
    return {"selection_path": selection_path, "status_path": status_path, "report_path": report_path, "selection": annotated, "status": status}


def annotate_psnr_constrained_pareto(
    comparison: pd.DataFrame,
    *,
    baseline_variant: str = "baseline",
    psnr_drop_tolerance: float = DEFAULT_PARETO_PSNR_DROP_TOLERANCE,
    objective: str = "min_retained",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    objective = normalize_pareto_objective(objective)
    validate_selection_inputs(comparison, baseline_variant=baseline_variant, psnr_drop_tolerance=psnr_drop_tolerance, objective=objective)
    output = comparison.copy()
    baseline_psnr = resolve_baseline_psnr(output, baseline_variant=baseline_variant)
    psnr = pd.to_numeric(output["mean_psnr"], errors="coerce").to_numpy(dtype=np.float64)
    psnr_drop = psnr_drop_from_baseline(psnr, baseline_psnr)
    feasible = ~np.isnan(psnr_drop) & (psnr_drop <= float(psnr_drop_tolerance))

    output["baseline_variant"] = baseline_variant
    output["baseline_psnr"] = baseline_psnr
    output["psnr_drop_from_baseline"] = psnr_drop
    output["psnr_constraint_tolerance"] = float(psnr_drop_tolerance)
    output["psnr_constraint_satisfied"] = feasible
    output["pareto_frontier"] = False
    output["pareto_selected"] = False
    output["pareto_selection_rank"] = np.nan

    feasible_index = list(output.index[feasible])
    frontier_index = pareto_frontier_indices(output.loc[feasible_index])
    output.loc[frontier_index, "pareto_frontier"] = True
    ranked = rank_selection_candidates(output.loc[frontier_index or feasible_index], objective=objective)
    for rank, index in enumerate(ranked.index, start=1):
        output.loc[index, "pareto_selection_rank"] = rank
    selected_index = ranked.index[0] if not ranked.empty else None
    if selected_index is not None:
        output.loc[selected_index, "pareto_selected"] = True
    return output, build_selection_summary(
        output,
        selected_index=selected_index,
        baseline_variant=baseline_variant,
        baseline_psnr=baseline_psnr,
        psnr_drop_tolerance=psnr_drop_tolerance,
        objective=objective,
    )


def validate_selection_inputs(comparison: pd.DataFrame, *, baseline_variant: str, psnr_drop_tolerance: float, objective: str) -> None:
    missing = sorted({"variant", "mean_psnr"} - set(comparison.columns))
    if missing:
        raise ValueError(f"3DGS render comparison is missing columns required for Pareto selection: {', '.join(missing)}.")
    if "retained_count" not in comparison.columns and "retention_fraction" not in comparison.columns:
        raise ValueError("3DGS render comparison must contain retained_count or retention_fraction for Pareto selection.")
    if comparison.empty:
        raise ValueError("3DGS render comparison is empty.")
    if not baseline_variant:
        raise ValueError("baseline_variant must be non-empty.")
    if float(psnr_drop_tolerance) < 0.0:
        raise ValueError("psnr_drop_tolerance must be non-negative.")
    if objective not in PARETO_OBJECTIVES:
        raise ValueError(f"objective must be one of {PARETO_OBJECTIVES}.")


def resolve_baseline_psnr(comparison: pd.DataFrame, *, baseline_variant: str) -> float:
    baseline = comparison[comparison["variant"].astype(str) == str(baseline_variant)]
    if baseline.empty and "variant_kind" in comparison.columns:
        baseline = comparison[comparison["variant_kind"].astype(str) == str(baseline_variant)]
    if baseline.empty:
        raise ValueError(f"Could not find baseline variant {baseline_variant!r} in render comparison.")
    values = pd.to_numeric(baseline["mean_psnr"], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"Baseline variant {baseline_variant!r} has no valid mean_psnr.")
    value = float(values.iloc[0])
    if np.isneginf(value):
        raise ValueError(f"Baseline variant {baseline_variant!r} has invalid mean_psnr {value!r}.")
    return value


def psnr_drop_from_baseline(psnr: np.ndarray, baseline_psnr: float) -> np.ndarray:
    if np.isposinf(baseline_psnr):
        return np.where(np.isposinf(psnr), 0.0, np.inf)
    return baseline_psnr - psnr


def pareto_frontier_indices(candidates: pd.DataFrame) -> list[Any]:
    frontier: list[Any] = []
    for index, row in candidates.iterrows():
        if not any(index != other_index and dominates(other, row) for other_index, other in candidates.iterrows()):
            frontier.append(index)
    return frontier


def dominates(left: pd.Series, right: pd.Series) -> bool:
    comparisons = [
        smaller_or_equal(metric_value(left, "retained_count", "retention_fraction"), metric_value(right, "retained_count", "retention_fraction")),
        greater_or_equal(metric_value(left, "mean_psnr"), metric_value(right, "mean_psnr")),
    ]
    strict = [
        smaller(metric_value(left, "retained_count", "retention_fraction"), metric_value(right, "retained_count", "retention_fraction")),
        greater(metric_value(left, "mean_psnr"), metric_value(right, "mean_psnr")),
    ]
    if has_valid_metric(left, right, "mean_ssim"):
        comparisons.append(greater_or_equal(metric_value(left, "mean_ssim"), metric_value(right, "mean_ssim")))
        strict.append(greater(metric_value(left, "mean_ssim"), metric_value(right, "mean_ssim")))
    if has_valid_metric(left, right, "mean_lpips_vgg"):
        comparisons.append(smaller_or_equal(metric_value(left, "mean_lpips_vgg"), metric_value(right, "mean_lpips_vgg")))
        strict.append(smaller(metric_value(left, "mean_lpips_vgg"), metric_value(right, "mean_lpips_vgg")))
    return all(comparisons) and any(strict)


def has_valid_metric(left: pd.Series, right: pd.Series, name: str) -> bool:
    return name in left.index and name in right.index and not pd.isna(left[name]) and not pd.isna(right[name])


def metric_value(row: pd.Series, primary: str, fallback: str | None = None) -> float:
    if primary in row.index and not pd.isna(row[primary]):
        return float(row[primary])
    if fallback is not None and fallback in row.index and not pd.isna(row[fallback]):
        return float(row[fallback])
    return float("nan")


def smaller_or_equal(left: float, right: float, *, eps: float = 1e-12) -> bool:
    return np.isfinite(left) and np.isfinite(right) and left <= right + eps


def greater_or_equal(left: float, right: float, *, eps: float = 1e-12) -> bool:
    if np.isposinf(left) and np.isposinf(right):
        return True
    return not pd.isna(left) and not pd.isna(right) and left >= right - eps


def smaller(left: float, right: float, *, eps: float = 1e-12) -> bool:
    return np.isfinite(left) and np.isfinite(right) and left < right - eps


def greater(left: float, right: float, *, eps: float = 1e-12) -> bool:
    if np.isposinf(left) and not np.isposinf(right):
        return True
    return not pd.isna(left) and not pd.isna(right) and left > right + eps


def rank_selection_candidates(candidates: pd.DataFrame, *, objective: str) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    output = candidates.copy()
    if "retained_count" not in output.columns:
        output["retained_count"] = pd.to_numeric(output["retention_fraction"], errors="coerce")
    if "retention_fraction" not in output.columns:
        output["retention_fraction"] = pd.to_numeric(output["retained_count"], errors="coerce")
    for column in ("mean_ssim", "mean_lpips_vgg"):
        if column not in output.columns:
            output[column] = np.nan
    sort_specs = {
        "min_retained": (["retained_count", "retention_fraction", "mean_psnr", "mean_ssim", "mean_lpips_vgg", "variant"], [True, True, False, False, True, True]),
        "max_psnr": (["mean_psnr", "retained_count", "retention_fraction", "mean_ssim", "mean_lpips_vgg", "variant"], [False, True, True, False, True, True]),
        "max_ssim": (["mean_ssim", "mean_psnr", "retained_count", "retention_fraction", "mean_lpips_vgg", "variant"], [False, False, True, True, True, True]),
        "min_lpips": (["mean_lpips_vgg", "mean_psnr", "retained_count", "retention_fraction", "mean_ssim", "variant"], [True, False, True, True, False, True]),
    }
    columns, ascending = sort_specs[objective]
    return output.sort_values(columns, ascending=ascending, na_position="last")


def build_selection_summary(
    annotated: pd.DataFrame,
    *,
    selected_index: Any | None,
    baseline_variant: str,
    baseline_psnr: float,
    psnr_drop_tolerance: float,
    objective: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "baseline_variant": baseline_variant,
        "baseline_psnr": baseline_psnr,
        "psnr_drop_tolerance": float(psnr_drop_tolerance),
        "objective": objective,
        "variant_count": int(len(annotated)),
        "eligible_variant_count": int(annotated["psnr_constraint_satisfied"].sum()),
        "pareto_frontier_count": int(annotated["pareto_frontier"].sum()),
        "pareto_frontier_variants": annotated.loc[annotated["pareto_frontier"], "variant"].astype(str).tolist(),
        "selected_variant": None,
        "selected_variant_kind": None,
        "selected_retained_count": None,
        "selected_retention_fraction": None,
        "selected_mean_psnr": None,
        "selected_psnr_drop_from_baseline": None,
        "selected_mean_ssim": None,
        "selected_mean_lpips_vgg": None,
    }
    if selected_index is not None:
        selected = annotated.loc[selected_index]
        summary.update(
            {
                "selected_variant": optional_string(selected, "variant"),
                "selected_variant_kind": optional_string(selected, "variant_kind"),
                "selected_retained_count": optional_number(selected, "retained_count"),
                "selected_retention_fraction": optional_number(selected, "retention_fraction"),
                "selected_mean_psnr": optional_number(selected, "mean_psnr"),
                "selected_psnr_drop_from_baseline": optional_number(selected, "psnr_drop_from_baseline"),
                "selected_mean_ssim": optional_number(selected, "mean_ssim"),
                "selected_mean_lpips_vgg": optional_number(selected, "mean_lpips_vgg"),
            }
        )
    return summary


def optional_number(row: pd.Series, column: str) -> float | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    return float(row[column])


def optional_string(row: pd.Series, column: str) -> str | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    return str(row[column])


def normalize_pareto_objective(objective: str) -> str:
    return objective.replace("-", "_").strip().lower()


def infer_selection_prefix(comparison_path: Path) -> str:
    stem = comparison_path.stem
    suffix = "_3dgs_render_comparison"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def format_psnr_constrained_pareto_report(status: dict[str, Any], annotated: pd.DataFrame) -> str:
    lines = [
        "# PSNR-Constrained 3DGS Variant Selection",
        "",
        f"- Comparison CSV: `{status['comparison_path']}`",
        f"- Baseline variant: `{status['baseline_variant']}`",
        f"- Baseline PSNR: `{format_number(status['baseline_psnr'])}`",
        f"- Allowed PSNR drop: `{format_number(status['psnr_drop_tolerance'])}` dB",
        f"- Objective: `{status['objective']}`",
        f"- Eligible variants: `{status['eligible_variant_count']}` / `{status['variant_count']}`",
        f"- Pareto frontier size: `{status['pareto_frontier_count']}`",
        f"- Selected variant: `{status['selected_variant'] or 'none'}`",
    ]
    lines.extend(["", "## Selection Table", "", format_selection_table(annotated)])
    return "\n".join(lines) + "\n"


def format_selection_table(annotated: pd.DataFrame) -> str:
    columns = [
        "variant",
        "variant_kind",
        "retained_count",
        "retention_fraction",
        "mean_psnr",
        "psnr_drop_from_baseline",
        "mean_ssim",
        "mean_lpips_vgg",
        "psnr_constraint_satisfied",
        "pareto_frontier",
        "pareto_selected",
        "pareto_selection_rank",
    ]
    present = [column for column in columns if column in annotated.columns]
    rows = annotated.sort_values(["pareto_selected", "pareto_frontier", "pareto_selection_rank", "variant"], ascending=[False, False, True, True], na_position="last")
    lines = ["| " + " | ".join(present) + " |", "| " + " | ".join(["---"] * len(present)) + " |"]
    for _, row in rows[present].iterrows():
        lines.append("| " + " | ".join(format_cell(row[column]) for column in present) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return format_number(float(value))
    escaped = str(value).replace("|", "\\|")
    return f"`{escaped}`"


def format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    numeric = float(value)
    if np.isposinf(numeric):
        return "inf"
    if np.isneginf(numeric):
        return "-inf"
    return f"{numeric:.6g}"
