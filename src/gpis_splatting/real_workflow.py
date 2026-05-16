from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gpis_splatting.real_benchmark import evaluate_real_renders
from gpis_splatting.real_bootstrap import bootstrap_real_gpis
from gpis_splatting.real_download import download_real_scene
from gpis_splatting.real_pipeline import fit_real_gpis, render_real_splats
from gpis_splatting.real_scene import prepare_real_scene
from gpis_splatting.serialization import write_json


def run_real_evaluation(
    *,
    scene: str = "nerfstudio_poster8_eval",
    prepared_root: str | Path = "real_scenes",
    source_dir: str | Path | None = None,
    download_dataset: bool = True,
    download_root: str | Path | None = None,
    dataset: str = "nerfstudio_poster",
    image_scale: int = 8,
    max_download_images: int | None = None,
    train_view_count: int = 12,
    copy_images: bool = True,
    max_points: int | None = 800,
    max_train_points: int | None = 600,
    seed: int = 7,
    lengthscale: float = 0.35,
    noise_std: float = 0.06,
    splat_sigmas: tuple[float, ...] = (0.025,),
    epsilons: tuple[float, ...] = (0.16,),
    gate_floors: tuple[float, ...] = (0.0,),
    split: str = "test",
    max_frames: int | None = 4,
    require_all: bool = False,
    min_sigma_px: float = 0.8,
    benchmark_target: str | Path | None = None,
) -> dict[str, Any]:
    if not splat_sigmas:
        raise ValueError("At least one splat sigma is required.")
    if not epsilons:
        raise ValueError("At least one epsilon is required.")
    if not gate_floors:
        raise ValueError("At least one gate floor is required.")
    for gate_floor in gate_floors:
        if not 0.0 <= gate_floor <= 1.0:
            raise ValueError("gate floors must be in [0, 1].")

    root = Path(prepared_root).resolve()
    scene_dir = root / scene
    resolved_download_root = Path(download_root) if download_root is not None else root / "_downloads"
    download_result: dict[str, Any] | None = None
    if source_dir is None:
        if not download_dataset:
            raise ValueError("Pass --source-dir or enable dataset download.")
        download_result = download_real_scene(
            dataset=dataset,
            output_root=resolved_download_root,
            image_scale=image_scale,
            max_images=max_download_images,
        )
        source = Path(download_result["output_dir"]).resolve()
    else:
        source = Path(source_dir).resolve()

    if not (scene_dir / "real_scene.json").exists():
        prepare_real_scene(
            input_dir=source,
            output_root=root,
            scene=scene,
            dataset=dataset,
            input_format="transforms",
            image_dir=f"images_{image_scale}",
            train_view_count=train_view_count,
            copy_images=copy_images,
        )

    point_path = (source / "sparse_pc.ply").resolve()
    if not point_path.exists():
        raise FileNotFoundError(f"Expected sparse point cloud at {point_path}.")

    rows: list[dict[str, Any]] = []
    bootstrap_results: dict[float, dict[str, Any]] = {}
    fit_result: dict[str, Any] | None = None
    for sigma_index, splat_sigma in enumerate(splat_sigmas):
        sigma_tag = float_tag(splat_sigma)
        output_prefix = f"real_sigma_{sigma_tag}"
        bootstrap = bootstrap_real_gpis(
            scene_dir=scene_dir,
            point_source="ply",
            point_path=point_path,
            output_prefix=output_prefix,
            max_points=max_points,
            seed=seed,
            free_space_samples_per_point=1,
            add_behind_surface_samples=False,
            splat_sigma=splat_sigma,
        )
        bootstrap_results[splat_sigma] = bootstrap
        if sigma_index == 0:
            fit_result = fit_real_gpis(
                scene_dir=scene_dir,
                samples_path=bootstrap["samples_path"],
                lengthscale=lengthscale,
                noise_std=noise_std,
                max_train_points=max_train_points,
                seed=seed,
            )

        plain_method = f"plain_sigma_{sigma_tag}"
        plain_render = render_real_splats(
            scene_dir=scene_dir,
            splats_path=bootstrap["splats_path"],
            method_name=plain_method,
            split=split,
            use_gpis_gate=False,
            min_sigma_px=min_sigma_px,
            max_frames=max_frames,
        )
        plain_status = evaluate_real_renders(
            scene_dir=scene_dir,
            predictions_dir=plain_render["output_dir"],
            output_dir=scene_dir / "evaluations",
            method_name=plain_method,
            split=split,
            benchmark_target=benchmark_target,
            require_all=require_all,
            allow_diagnostic_proxy=True,
        )
        rows.append(
            comparison_row(
                status=plain_status,
                render=plain_render,
                splat_sigma=splat_sigma,
                epsilon=None,
                gate_floor=None,
                use_gpis_gate=False,
            )
        )

        for epsilon in epsilons:
            for gate_floor in gate_floors:
                method = f"gpis_eps_{float_tag(epsilon)}_floor_{float_tag(gate_floor)}_sigma_{sigma_tag}"
                gated_render = render_real_splats(
                    scene_dir=scene_dir,
                    splats_path=bootstrap["splats_path"],
                    method_name=method,
                    split=split,
                    use_gpis_gate=True,
                    epsilon=epsilon,
                    gate_floor=gate_floor,
                    min_sigma_px=min_sigma_px,
                    max_frames=max_frames,
                )
                gated_status = evaluate_real_renders(
                    scene_dir=scene_dir,
                    predictions_dir=gated_render["output_dir"],
                    output_dir=scene_dir / "evaluations",
                    method_name=method,
                    split=split,
                    benchmark_target=benchmark_target,
                    require_all=require_all,
                    allow_diagnostic_proxy=True,
                )
                rows.append(
                    comparison_row(
                        status=gated_status,
                        render=gated_render,
                        splat_sigma=splat_sigma,
                        epsilon=epsilon,
                        gate_floor=gate_floor,
                        use_gpis_gate=True,
                    )
                )

    comparison = pd.DataFrame(rows)
    eval_dir = scene_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = eval_dir / "real_evaluation_comparison.csv"
    status_path = eval_dir / "real_evaluation_status.json"
    report_path = eval_dir / "real_evaluation_report.md"
    comparison.to_csv(comparison_path, index=False)
    status = {
        "schema_version": 1,
        "scene": scene,
        "scene_dir": str(scene_dir),
        "source_dir": str(source),
        "download": download_result["report"] if download_result is not None else None,
        "fit": fit_result["report"] if fit_result is not None else None,
        "bootstrap_reports": {float_tag(sigma): result["report"] for sigma, result in bootstrap_results.items()},
        "comparison_path": str(comparison_path),
        "report_path": str(report_path),
        "row_count": len(rows),
        "config": {
            "dataset": dataset,
            "image_scale": image_scale,
            "max_download_images": max_download_images,
            "train_view_count": train_view_count,
            "max_points": max_points,
            "max_train_points": max_train_points,
            "seed": seed,
            "lengthscale": lengthscale,
            "noise_std": noise_std,
            "splat_sigmas": list(splat_sigmas),
            "epsilons": list(epsilons),
            "gate_floors": list(gate_floors),
            "split": split,
            "max_frames": max_frames,
            "require_all": require_all,
            "min_sigma_px": min_sigma_px,
        },
        "best_psnr": best_row(comparison, "mean_psnr"),
        "best_ssim": best_row(comparison, "mean_ssim"),
    }
    write_json(status_path, status)
    report_path.write_text(format_real_evaluation_report(status, comparison), encoding="utf-8")
    return {
        "scene_dir": scene_dir,
        "comparison_path": comparison_path,
        "status_path": status_path,
        "report_path": report_path,
        "status": status,
    }


def comparison_row(
    *,
    status: dict[str, Any],
    render: dict[str, Any],
    splat_sigma: float,
    epsilon: float | None,
    gate_floor: float | None,
    use_gpis_gate: bool,
) -> dict[str, Any]:
    summary = status["summary"]
    gate_summary = render["report"].get("gate_summary", {})
    return {
        "scene": status["scene"],
        "method": status["method"],
        "split": status["split"],
        "use_gpis_gate": use_gpis_gate,
        "splat_sigma": splat_sigma,
        "epsilon": epsilon,
        "gate_floor": gate_floor,
        "image_count": summary["image_count"],
        "missing_count": summary["missing_count"],
        "mean_psnr": summary["mean_psnr"],
        "mean_ssim": summary["mean_ssim"],
        "render_backend": summary.get("render_backend"),
        "render_fidelity": summary.get("render_fidelity"),
        "photometric_metrics_policy": summary.get("photometric_metrics_policy"),
        "mean_lpips_vgg": summary["mean_lpips_vgg"],
        "gate_min": gate_summary.get("min"),
        "gate_mean": gate_summary.get("mean"),
        "gate_max": gate_summary.get("max"),
        "raw_gate_min": gate_summary.get("raw_min"),
        "raw_gate_mean": gate_summary.get("raw_mean"),
        "raw_gate_max": gate_summary.get("raw_max"),
        "render_dir": str(render["output_dir"]),
        "render_report": str(render["report_path"]),
        "metrics_path": status["metrics_path"],
        "summary_path": status["summary_path"],
    }


def format_real_evaluation_report(status: dict[str, Any], comparison: pd.DataFrame) -> str:
    lines = [
        "# Real Evaluation Workflow",
        "",
        f"- Scene: `{status['scene']}`",
        f"- Source: `{status['source_dir']}`",
        f"- Comparison CSV: `{status['comparison_path']}`",
        f"- Rows: `{status['row_count']}`",
    ]
    for label, key, metric in (("Best PSNR", "best_psnr", "mean_psnr"), ("Best SSIM", "best_ssim", "mean_ssim")):
        best = status.get(key)
        if best is not None:
            lines.append(f"- {label}: `{best['method']}` (`{best[metric]:.6g}`)")
    lines.extend(["", "## Methods", ""])
    method_table = comparison[["method", "use_gpis_gate", "splat_sigma", "epsilon", "gate_floor", "render_fidelity", "photometric_metrics_policy", "image_count", "missing_count", "mean_psnr", "mean_ssim"]]
    lines.extend(markdown_table(method_table))
    return "\n".join(lines) + "\n"


def best_row(dataframe: pd.DataFrame, metric: str) -> dict[str, Any] | None:
    if dataframe.empty or metric not in dataframe:
        return None
    index = dataframe[metric].astype(float).idxmax()
    row = dataframe.loc[index].to_dict()
    return {key: value for key, value in row.items() if pd.notna(value)}


def float_tag(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def markdown_table(dataframe: pd.DataFrame) -> list[str]:
    columns = list(dataframe.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in dataframe.iterrows():
        values = [_format_table_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _format_table_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
