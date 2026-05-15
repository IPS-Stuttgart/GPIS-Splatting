from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from gpis_splatting.backend_benchmark import make_benchmark_samples, prediction_error_metrics
from gpis_splatting.gpis_backends import DenseExactGPISBackend
from gpis_splatting.scalable_gpis_backends import ScalableBackendName, fit_scalable_gpis_backend

BACKENDS: tuple[ScalableBackendName, ...] = ("local_exact_scalable", "inducing_points_scalable", "gpu_inducing_points")
CSV_FIELDS = (
    "backend",
    "status",
    "shape",
    "n_train",
    "n_query",
    "num_neighbors",
    "num_inducing",
    "inducing_selection",
    "neighbor_backend",
    "neighbor_train_chunk_size",
    "local_solve_batch_size",
    "compute_device",
    "fit_time_sec",
    "predict_time_sec",
    "queries_per_sec",
    "mean_rmse_vs_dense",
    "variance_rmse_vs_dense",
    "gradient_rmse_vs_dense",
    "gate_rmse_vs_dense",
    "skip_reason",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark scalable GPIS backends on deterministic pseudo-SDF samples.")
    parser.add_argument("--output-dir", default="experiments/scalable_gpis_backend_benchmark")
    parser.add_argument("--benchmark-name", default="scalable_gpis_backend_benchmark")
    parser.add_argument("--backend", dest="backends", action="append", choices=BACKENDS, help="Backend to run. Repeat for multiple backends.")
    parser.add_argument("--shape", choices=("sphere", "torus", "wavy_plane"), default="sphere")
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-query", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lengthscale", type=float, default=0.34)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.035)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--fit-batch-size", type=int, default=8192)
    parser.add_argument("--num-neighbors", type=int, default=64)
    parser.add_argument("--num-inducing", type=int, default=512)
    parser.add_argument("--inducing-selection", choices=("farthest", "uniform", "first", "grid"), default="farthest")
    parser.add_argument("--neighbor-backend", choices=("auto", "cdist", "chunked", "scipy"), default="auto")
    parser.add_argument("--neighbor-train-chunk-size", type=int, default=65536)
    parser.add_argument("--local-solve-batch-size", type=int, default=256)
    parser.add_argument("--compute-device", default="auto")
    parser.add_argument("--max-dense-reference-points", type=int, default=2048)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_benchmark(args)
    csv_path = output_dir / f"{args.benchmark_name}.csv"
    status_path = output_dir / f"{args.benchmark_name}_status.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    status = {
        "ok": bool(rows) and not any(row["status"] == "failed" for row in rows),
        "successful_backends": [row["backend"] for row in rows if row["status"] == "success"],
        "failed_backends": [row["backend"] for row in rows if row["status"] == "failed"],
        "csv_path": str(csv_path),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {status_path}")


def run_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    x_train, y_train = make_benchmark_samples(args.n_train, shape=args.shape, seed=args.seed)
    x_query, _ = make_benchmark_samples(args.n_query, shape=args.shape, seed=args.seed + 1)
    reference = None
    if args.n_train <= args.max_dense_reference_points:
        reference = DenseExactGPISBackend.fit(x_train, y_train, lengthscale=args.lengthscale, variance=args.variance, noise_std=args.noise_std).predict(x_query, batch_size=args.batch_size)
    rows: list[dict[str, object]] = []
    for backend_name in args.backends or BACKENDS:
        row = base_row(args, backend_name)
        try:
            kwargs = {
                "lengthscale": args.lengthscale,
                "variance": args.variance,
                "noise_std": args.noise_std,
                "num_neighbors": args.num_neighbors,
                "neighbor_backend": args.neighbor_backend,
                "neighbor_train_chunk_size": args.neighbor_train_chunk_size,
                "local_solve_batch_size": args.local_solve_batch_size,
            }
            if backend_name != "local_exact_scalable":
                kwargs = {
                    "lengthscale": args.lengthscale,
                    "variance": args.variance,
                    "noise_std": args.noise_std,
                    "num_inducing": args.num_inducing,
                    "inducing_selection": args.inducing_selection,
                    "fit_batch_size": args.fit_batch_size,
                    "compute_device": args.compute_device,
                }
            fit_start = time.perf_counter()
            backend = fit_scalable_gpis_backend(backend_name, x_train, y_train, **kwargs)
            fit_time = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            prediction = backend.predict(x_query, batch_size=args.batch_size)
            predict_time = time.perf_counter() - predict_start
            row.update(
                {
                    "status": "success",
                    "fit_time_sec": format_float(fit_time),
                    "predict_time_sec": format_float(predict_time),
                    "queries_per_sec": format_float(args.n_query / predict_time if predict_time > 0 else math.inf),
                    "compute_device": getattr(backend, "compute_device", "cpu"),
                }
            )
            if reference is not None:
                row.update({key: format_float(value) for key, value in prediction_error_metrics(prediction, reference, epsilon=args.epsilon).items()})
        except Exception as exc:  # pragma: no cover - useful for optional CUDA/SciPy failures.
            row.update({"status": "failed", "skip_reason": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    return rows


def base_row(args: argparse.Namespace, backend_name: str) -> dict[str, object]:
    return {field: "" for field in CSV_FIELDS} | {
        "backend": backend_name,
        "status": "pending",
        "shape": args.shape,
        "n_train": args.n_train,
        "n_query": args.n_query,
        "num_neighbors": args.num_neighbors,
        "num_inducing": args.num_inducing,
        "inducing_selection": args.inducing_selection,
        "neighbor_backend": args.neighbor_backend,
        "neighbor_train_chunk_size": args.neighbor_train_chunk_size,
        "local_solve_batch_size": args.local_solve_batch_size,
        "compute_device": args.compute_device,
    }


def format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.8g}"


if __name__ == "__main__":
    main()
