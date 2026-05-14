from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from gpis_splatting.gpis import GPISPrediction, surface_band_probability
from gpis_splatting.gpis_backends import GPISBackend, GPISBackendName, InducingSelectionName, fit_gpis_backend

Tensor = torch.Tensor
BenchmarkShape = Literal["sphere", "torus", "wavy_plane"]

DEFAULT_BACKENDS: tuple[GPISBackendName, ...] = ("dense_exact", "local_exact", "inducing_points")
KNOWN_BACKENDS: tuple[GPISBackendName, ...] = ("dense_exact", "local_exact", "inducing_points")
KNOWN_SHAPES: tuple[BenchmarkShape, ...] = ("sphere", "torus", "wavy_plane")

CSV_FIELDS: tuple[str, ...] = (
    "benchmark_name",
    "backend",
    "status",
    "skip_reason",
    "shape",
    "n_train",
    "n_query",
    "lengthscale",
    "variance",
    "noise_std",
    "epsilon",
    "batch_size",
    "num_neighbors",
    "num_inducing",
    "inducing_selection",
    "fit_batch_size",
    "effective_training_count",
    "effective_num_inducing",
    "effective_num_neighbors",
    "fit_time_sec",
    "predict_time_sec",
    "total_time_sec",
    "queries_per_sec",
    "model_storage_bytes",
    "model_storage_mib",
    "mean_rmse_vs_dense",
    "variance_rmse_vs_dense",
    "gradient_rmse_vs_dense",
    "gate_rmse_vs_dense",
    "mean_abs_max_vs_dense",
    "variance_abs_max_vs_dense",
    "gradient_abs_max_vs_dense",
)


@dataclass(frozen=True)
class BackendBenchmarkConfig:
    """Configuration for a deterministic GPIS backend scalability benchmark."""

    output_dir: Path
    benchmark_name: str = "gpis_backend_benchmark"
    backends: tuple[GPISBackendName, ...] = DEFAULT_BACKENDS
    shape: BenchmarkShape = "sphere"
    n_train: int = 512
    n_query: int = 256
    seed: int = 17
    lengthscale: float = 0.34
    variance: float = 1.0
    noise_std: float = 0.035
    epsilon: float = 0.08
    batch_size: int = 8192
    num_neighbors: int = 64
    num_inducing: int = 128
    inducing_selection: InducingSelectionName = "farthest"
    fit_batch_size: int = 8192
    max_dense_reference_points: int = 2048
    skip_dense_over_points: int = 4096


def run_backend_benchmark(config: BackendBenchmarkConfig) -> dict[str, Any]:
    """Run the backend benchmark and write CSV, JSON, and Markdown artifacts."""

    validate_backend_benchmark_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = make_benchmark_samples(config.n_train, shape=config.shape, seed=config.seed)
    x_query, _ = make_benchmark_samples(config.n_query, shape=config.shape, seed=config.seed + 1)
    reference_prediction = fit_dense_reference(config, x_train, y_train, x_query)

    rows = [run_one_backend(config, backend_name, x_train, y_train, x_query, reference_prediction) for backend_name in config.backends]
    csv_path = config.output_dir / f"{config.benchmark_name}.csv"
    config_path = config.output_dir / f"{config.benchmark_name}_config.json"
    status_path = config.output_dir / f"{config.benchmark_name}_status.json"
    report_path = config.output_dir / f"{config.benchmark_name}_report.md"

    write_rows_csv(csv_path, rows)
    write_json(config_path, config_to_jsonable(config))
    status = build_status(config, rows, csv_path=csv_path, config_path=config_path, report_path=report_path)
    write_json(status_path, status)
    write_report(report_path, config, rows, status)

    return {
        "rows": rows,
        "csv_path": csv_path,
        "config_path": config_path,
        "status_path": status_path,
        "report_path": report_path,
        "status": status,
    }


def run_one_backend(
    config: BackendBenchmarkConfig,
    backend_name: GPISBackendName,
    x_train: Tensor,
    y_train: Tensor,
    x_query: Tensor,
    reference_prediction: GPISPrediction | None,
) -> dict[str, object]:
    row = base_row(config, backend_name)
    if backend_name == "dense_exact" and config.n_train > config.skip_dense_over_points:
        row.update({"status": "skipped", "skip_reason": f"dense_exact skipped because n_train={config.n_train} exceeds skip_dense_over_points={config.skip_dense_over_points}"})
        return row

    try:
        fit_start = time.perf_counter()
        backend = fit_gpis_backend(
            backend_name,
            x_train,
            y_train,
            lengthscale=config.lengthscale,
            variance=config.variance,
            noise_std=config.noise_std,
            num_neighbors=config.num_neighbors,
            num_inducing=config.num_inducing,
            inducing_selection=config.inducing_selection,
            fit_batch_size=config.fit_batch_size,
        )
        fit_time = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        prediction = backend.predict(x_query, batch_size=config.batch_size)
        predict_time = time.perf_counter() - predict_start
        total_time = fit_time + predict_time

        row.update(
            {
                "status": "success",
                "fit_time_sec": fit_time,
                "predict_time_sec": predict_time,
                "total_time_sec": total_time,
                "queries_per_sec": config.n_query / predict_time if predict_time > 0.0 else math.inf,
                "model_storage_bytes": tensor_storage_bytes(backend),
                "model_storage_mib": tensor_storage_bytes(backend) / float(1024**2),
                "effective_training_count": training_count(backend),
                "effective_num_inducing": getattr(backend, "num_inducing", ""),
                "effective_num_neighbors": getattr(backend, "num_neighbors", ""),
            }
        )
        if reference_prediction is not None:
            row.update(prediction_error_metrics(prediction, reference_prediction, epsilon=config.epsilon))
    except Exception as exc:  # pragma: no cover - exercised when external BLAS/torch errors occur.
        row.update({"status": "failed", "skip_reason": f"{type(exc).__name__}: {exc}"})
    return row


def make_benchmark_samples(n: int, *, shape: BenchmarkShape = "sphere", seed: int = 17) -> tuple[Tensor, Tensor]:
    """Create deterministic 3D points and analytic pseudo-SDF values."""

    if n < 1:
        raise ValueError("n must be positive.")
    if shape not in KNOWN_SHAPES:
        raise ValueError(f"Unknown benchmark shape {shape!r}. Expected one of: {', '.join(KNOWN_SHAPES)}.")
    generator = torch.Generator().manual_seed(int(seed))
    points = torch.rand((int(n), 3), generator=generator, dtype=torch.float64) * 2.0 - 1.0
    if shape == "sphere":
        sdf = torch.linalg.norm(points, dim=-1) - 0.65
    elif shape == "torus":
        major_radius = 0.62
        minor_radius = 0.22
        radial = torch.linalg.norm(points[:, :2], dim=-1) - major_radius
        sdf = torch.linalg.norm(torch.stack((radial, points[:, 2]), dim=-1), dim=-1) - minor_radius
    elif shape == "wavy_plane":
        sdf = points[:, 2] - 0.18 * torch.sin(4.0 * points[:, 0]) * torch.cos(4.0 * points[:, 1])
    else:  # pragma: no cover - kept for type-checker exhaustiveness.
        raise ValueError(f"Unknown benchmark shape {shape!r}.")
    return points, sdf


def fit_dense_reference(config: BackendBenchmarkConfig, x_train: Tensor, y_train: Tensor, x_query: Tensor) -> GPISPrediction | None:
    if config.n_train > config.max_dense_reference_points:
        return None
    backend = fit_gpis_backend(
        "dense_exact",
        x_train,
        y_train,
        lengthscale=config.lengthscale,
        variance=config.variance,
        noise_std=config.noise_std,
        fit_batch_size=config.fit_batch_size,
    )
    return backend.predict(x_query, batch_size=config.batch_size)


def prediction_error_metrics(prediction: GPISPrediction, reference: GPISPrediction, *, epsilon: float) -> dict[str, float]:
    mean_delta = prediction.mean - reference.mean
    variance_delta = prediction.variance - reference.variance
    gradient_delta = prediction.gradient - reference.gradient
    gate_delta = surface_band_probability(prediction, epsilon) - surface_band_probability(reference, epsilon)
    return {
        "mean_rmse_vs_dense": rmse(mean_delta),
        "variance_rmse_vs_dense": rmse(variance_delta),
        "gradient_rmse_vs_dense": rmse(gradient_delta),
        "gate_rmse_vs_dense": rmse(gate_delta),
        "mean_abs_max_vs_dense": max_abs(mean_delta),
        "variance_abs_max_vs_dense": max_abs(variance_delta),
        "gradient_abs_max_vs_dense": max_abs(gradient_delta),
    }


def rmse(values: Tensor) -> float:
    return float(torch.sqrt(torch.mean(values.detach().to(dtype=torch.float64).square())).item())


def max_abs(values: Tensor) -> float:
    return float(torch.max(torch.abs(values.detach())).item())


def base_row(config: BackendBenchmarkConfig, backend_name: GPISBackendName) -> dict[str, object]:
    return {field: "" for field in CSV_FIELDS} | {
        "benchmark_name": config.benchmark_name,
        "backend": backend_name,
        "status": "pending",
        "shape": config.shape,
        "n_train": config.n_train,
        "n_query": config.n_query,
        "lengthscale": config.lengthscale,
        "variance": config.variance,
        "noise_std": config.noise_std,
        "epsilon": config.epsilon,
        "batch_size": config.batch_size,
        "num_neighbors": config.num_neighbors,
        "num_inducing": config.num_inducing,
        "inducing_selection": config.inducing_selection,
        "fit_batch_size": config.fit_batch_size,
    }


def tensor_storage_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_storage_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_storage_bytes(item) for item in value)
    if hasattr(value, "__dict__"):
        return sum(tensor_storage_bytes(item) for item in vars(value).values())
    return 0


def training_count(backend: GPISBackend) -> int | str:
    explicit = getattr(backend, "training_count", None)
    if explicit is not None:
        return int(explicit)
    x_train = getattr(backend, "x_train", None)
    if isinstance(x_train, torch.Tensor):
        return int(x_train.shape[0])
    model = getattr(backend, "model", None)
    model_x_train = getattr(model, "x_train", None)
    if isinstance(model_x_train, torch.Tensor):
        return int(model_x_train.shape[0])
    return ""


def validate_backend_benchmark_config(config: BackendBenchmarkConfig) -> None:
    if config.n_train < 1 or config.n_query < 1:
        raise ValueError("n_train and n_query must be positive.")
    if not config.backends:
        raise ValueError("At least one backend must be requested.")
    unknown_backends = sorted(set(config.backends) - set(KNOWN_BACKENDS))
    if unknown_backends:
        raise ValueError(f"Unknown backend(s): {', '.join(unknown_backends)}.")
    if config.shape not in KNOWN_SHAPES:
        raise ValueError(f"Unknown benchmark shape {config.shape!r}.")
    if config.lengthscale <= 0.0 or config.variance <= 0.0 or config.noise_std <= 0.0 or config.epsilon <= 0.0:
        raise ValueError("lengthscale, variance, noise_std, and epsilon must be positive.")
    if config.batch_size < 1 or config.fit_batch_size < 1:
        raise ValueError("batch_size and fit_batch_size must be positive.")
    if config.num_neighbors < 1 or config.num_inducing < 1:
        raise ValueError("num_neighbors and num_inducing must be positive.")
    if config.max_dense_reference_points < 0 or config.skip_dense_over_points < 0:
        raise ValueError("Dense reference and skip thresholds must be non-negative.")


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{field: format_csv_value(row.get(field, "")) for field in CSV_FIELDS} for row in rows])


def format_csv_value(value: object) -> object:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.8g}"
    return value


def build_status(config: BackendBenchmarkConfig, rows: list[dict[str, object]], *, csv_path: Path, config_path: Path, report_path: Path) -> dict[str, object]:
    successful = [row for row in rows if row["status"] == "success"]
    fastest = min(successful, key=lambda row: float(row["predict_time_sec"]))["backend"] if successful else None
    return {
        "benchmark_name": config.benchmark_name,
        "ok": bool(successful) and not any(row["status"] == "failed" for row in rows),
        "successful_backends": [row["backend"] for row in successful],
        "failed_backends": [row["backend"] for row in rows if row["status"] == "failed"],
        "skipped_backends": [row["backend"] for row in rows if row["status"] == "skipped"],
        "fastest_predict_backend": fastest,
        "csv_path": str(csv_path),
        "config_path": str(config_path),
        "report_path": str(report_path),
    }


def write_report(path: Path, config: BackendBenchmarkConfig, rows: list[dict[str, object]], status: dict[str, object]) -> None:
    lines = [
        f"# {config.benchmark_name}",
        "",
        "This benchmark compares GPIS backends on the same deterministic pseudo-SDF samples.",
        "Dense exact predictions are used as the reference when the requested training size is below the configured dense-reference limit.",
        "",
        "## Configuration",
        "",
        f"- shape: `{config.shape}`",
        f"- training observations: `{config.n_train}`",
        f"- query points: `{config.n_query}`",
        f"- backends: `{', '.join(config.backends)}`",
        f"- dense reference limit: `{config.max_dense_reference_points}`",
        "",
        "## Results",
        "",
        "| Backend | Status | Fit s | Predict s | Queries/s | Storage MiB | Mean RMSE vs dense | Gate RMSE vs dense |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {backend} | {status} | {fit} | {predict} | {qps} | {storage} | {mean_rmse} | {gate_rmse} |".format(
                backend=row["backend"],
                status=row["status"],
                fit=format_report_number(row.get("fit_time_sec")),
                predict=format_report_number(row.get("predict_time_sec")),
                qps=format_report_number(row.get("queries_per_sec")),
                storage=format_report_number(row.get("model_storage_mib")),
                mean_rmse=format_report_number(row.get("mean_rmse_vs_dense")),
                gate_rmse=format_report_number(row.get("gate_rmse_vs_dense")),
            )
        )
    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- ok: `{status['ok']}`",
            f"- fastest prediction backend: `{status['fastest_predict_backend'] or 'none'}`",
            f"- successful backends: `{', '.join(status['successful_backends']) or 'none'}`",
            f"- skipped backends: `{', '.join(status['skipped_backends']) or 'none'}`",
            f"- failed backends: `{', '.join(status['failed_backends']) or 'none'}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_report_number(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.4g}"
    return str(value)


def config_to_jsonable(config: BackendBenchmarkConfig) -> dict[str, object]:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    data["backends"] = list(config.backends)
    return data


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
