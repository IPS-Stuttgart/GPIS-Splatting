from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.backend_benchmark import BackendBenchmarkConfig, DEFAULT_BACKENDS, KNOWN_BACKENDS, KNOWN_SHAPES, run_backend_benchmark


def parse_ard_lengthscales(value: str | None) -> tuple[float, ...] | None:
    if value is None or value.strip() == "":
        return None
    return tuple(float(part) for part in value.replace(",", ";").split(";") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark dense, local, KD-tree/FAISS, inducing, ARD, SKI-grid, and multiresolution GPIS backends on deterministic pseudo-SDF samples.")
    parser.add_argument("--output-dir", default="experiments/gpis_backend_benchmark")
    parser.add_argument("--benchmark-name", default="gpis_backend_benchmark")
    parser.add_argument("--backend", dest="backends", action="append", choices=KNOWN_BACKENDS, help="Backend to run. Repeat to run multiple backends. Defaults to dependency-free backends.")
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
    parser.add_argument("--leaf-size", type=int, default=32)
    parser.add_argument("--ard-lengthscales", default=None, help="Optional semicolon- or comma-separated ARD lengthscales, e.g. 0.2,0.5,0.3.")
    parser.add_argument("--ski-grid-size", type=int, default=8)
    parser.add_argument("--ski-padding", type=float, default=0.05)
    parser.add_argument("--ski-max-grid-points", type=int, default=2048)
    parser.add_argument("--multires-levels", type=int, default=3)
    parser.add_argument("--multires-lengthscale-decay", type=float, default=0.55)
    parser.add_argument("--multires-inducing-growth", type=float, default=1.5)
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
            leaf_size=args.leaf_size,
            ard_lengthscales=parse_ard_lengthscales(args.ard_lengthscales),
            ski_grid_size=args.ski_grid_size,
            ski_padding=args.ski_padding,
            ski_max_grid_points=args.ski_max_grid_points,
            multires_levels=args.multires_levels,
            multires_lengthscale_decay=args.multires_lengthscale_decay,
            multires_inducing_growth=args.multires_inducing_growth,
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
