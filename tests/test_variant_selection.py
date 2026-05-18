from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gpis_splatting.variant_selection import annotate_psnr_constrained_pareto, select_psnr_constrained_pareto_variant


def test_psnr_constrained_pareto_selection_prefers_compression_under_tolerance(tmp_path: Path) -> None:
    comparison_path = tmp_path / "paper_gate_3dgs_render_comparison.csv"
    write_comparison(comparison_path)

    result = select_psnr_constrained_pareto_variant(
        comparison_path=comparison_path,
        output_dir=tmp_path,
        method_name="paper_gate",
        psnr_drop_tolerance=0.2,
        objective="min_retained",
    )

    selection = result["selection"]
    status = result["status"]
    selected = selection[selection["pareto_selected"]].iloc[0]
    rejected = selection[selection["variant"] == "gate_ge_0p5"].iloc[0]

    assert status["selected_variant"] == "gate_ge_0p25"
    assert status["eligible_variant_count"] == 3
    assert selected["retained_count"] == 70
    assert selected["psnr_constraint_satisfied"]
    assert not rejected["psnr_constraint_satisfied"]
    assert (tmp_path / "paper_gate_3dgs_pareto_selection.csv").exists()
    assert (tmp_path / "paper_gate_3dgs_pareto_selection_status.json").exists()
    assert (tmp_path / "paper_gate_3dgs_pareto_selection_report.md").exists()


def test_psnr_constrained_pareto_selection_can_use_photometric_objective() -> None:
    annotated, summary = annotate_psnr_constrained_pareto(write_comparison_frame(), psnr_drop_tolerance=0.2, objective="max-psnr")

    selected = annotated[annotated["pareto_selected"]].iloc[0]
    assert selected["variant"] == "gate_scaled"
    assert summary["selected_variant"] == "gate_scaled"
    assert summary["selected_mean_psnr"] == 30.1


def test_psnr_constrained_pareto_selection_accepts_infinite_perfect_baseline() -> None:
    comparison = pd.DataFrame(
        [
            {"variant": "baseline", "variant_kind": "baseline", "retained_count": 100, "retention_fraction": 1.0, "mean_psnr": np.inf, "mean_ssim": 1.0},
            {"variant": "gate_scaled", "variant_kind": "gate_scaled", "retained_count": 100, "retention_fraction": 1.0, "mean_psnr": np.inf, "mean_ssim": 1.0},
            {"variant": "gate_ge_0p5", "variant_kind": "gate_threshold", "retained_count": 40, "retention_fraction": 0.4, "mean_psnr": 35.0, "mean_ssim": 0.98},
        ]
    )

    annotated, summary = annotate_psnr_constrained_pareto(comparison, psnr_drop_tolerance=0.2)

    assert set(annotated.loc[annotated["psnr_constraint_satisfied"], "variant"]) == {"baseline", "gate_scaled"}
    assert summary["selected_variant"] in {"baseline", "gate_scaled"}


def test_psnr_constrained_pareto_selection_requires_baseline() -> None:
    comparison = write_comparison_frame()
    comparison = comparison[comparison["variant"] != "baseline"]

    try:
        annotate_psnr_constrained_pareto(comparison)
    except ValueError as exc:
        assert "baseline" in str(exc)
    else:
        raise AssertionError("Expected missing baseline to fail.")


def write_comparison(path: Path) -> None:
    write_comparison_frame().to_csv(path, index=False)


def write_comparison_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            comparison_row("baseline", "baseline", 100, 1.0, 30.0, 0.92, 0.10),
            comparison_row("gate_scaled", "gate_scaled", 100, 1.0, 30.1, 0.93, 0.09),
            comparison_row("gate_ge_0p25", "gate_threshold", 70, 0.7, 29.85, 0.91, 0.12),
            comparison_row("gate_ge_0p5", "gate_threshold", 40, 0.4, 29.65, 0.89, 0.14),
        ]
    )


def comparison_row(variant: str, kind: str, retained_count: int, retention_fraction: float, psnr: float, ssim: float, lpips: float) -> dict[str, object]:
    return {
        "variant": variant,
        "variant_kind": kind,
        "retained_count": retained_count,
        "retention_fraction": retention_fraction,
        "mean_psnr": psnr,
        "mean_ssim": ssim,
        "mean_lpips_vgg": lpips,
    }
