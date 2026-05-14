from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.backend_benchmark import BackendBenchmarkConfig, DEFAULT_BACKENDS, KNOWN_BACKENDS, KNOWN_SHAPES, run_backend_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark dense, local, and inducing-point GPIS backends on deterministic pseudo-SDF samples.")
    parser.add_argument("--output-dir", default="experiments/gpis_backend_benchmark")
    parser.add_argument("--benchmark-name", default="gpis_backend_benchmark")
    parser.add_argument("--backend", dest="backends", action="append", choices=KNOWN_BACKENDS, help="Backend to run. Repeat to run multiple backends. Defaults to all.")
    parser.add_argument("--shape", choices=KNOWN_SHAPES, default="sphere")
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-query", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lengthscale", type=float, default=0.34)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.035)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-neighbors", type=int, default=64)
    parser.add_argument("--num-inducing", type=int, default=128)
    parser.add_argument("--inducing-selection", choices=("farthest", "uniform", "first"), default="farthest")
    parser.add_argument("--fit-batch-size", type=int, default=8192)
    parser.add_argument("--max-dense-reference-points", type=int, default=2048)
    parser.add_argument("--skip-dense-over-points", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_backend_benchmark(
        BackendBenchmarkConfig(
            output_dir=Path(args.output_dir),
            benchmark_name=args.benchmark_name,
            backends=tuple(args.backends or DEFAULT_BACKENDS),
            shape=args.shape,
            n_train=args.n_train,
            n_query=args.n_query,
            seed=args.seed,
            lengthscale=args.lengthscale,
            variance=args.variance,
            noise_std=args.noise_std,
            epsilon=args.epsilon,
            batch_size=args.batch_size,
            num_neighbors=args.num_neighbors,
            num_inducing=args.num_inducing,
            inducing_selection=args.inducing_selection,
            fit_batch_size=args.fit_batch_size,
            max_dense_reference_points=args.max_dense_reference_points,
            skip_dense_over_points=args.skip_dense_over_points,
        )
    )
    print(f"Wrote {result['csv_path']}")
    print(f"Wrote {result['config_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"fastest_predict_backend: {result['status']['fastest_predict_backend'] or 'none'}")


if __name__ == "__main__":
    main()
