"""Compare float16 and optional float8 neighbor ranking with float32.

Run with:
    python scripts/benchmark_sct_neighbor_precision.py

Each precision runs in a fresh subprocess so peak RSS is measured separately.
The production sampler remains float32 regardless of these benchmark results.
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import euclidean_distances


def _get_float8_dtype():
    for name in ("float8_e4m3fn", "float8_e5m2"):
        dtype = getattr(np, name, None)
        if dtype is not None:
            return dtype
    try:
        import ml_dtypes
    except ImportError:
        return None
    return getattr(ml_dtypes, "float8_e4m3fn", None)


def _resolve_dtype(name):
    if name == "float32":
        return np.float32
    if name == "float16":
        return np.float16
    if name == "float8":
        return _get_float8_dtype()
    raise ValueError(f"Unknown precision: {name}")


def _top_indices(distances, n_neighbors):
    indices = np.argpartition(distances, n_neighbors - 1)[:n_neighbors]
    return indices[np.argsort(distances[indices])]


def _float32_distances(query, candidates):
    return euclidean_distances(query, candidates).astype(np.float32, copy=False)


def _reduced_precision_distances(query_low, candidates_low, dtype):
    query_norm = np.sum(query_low * query_low, axis=1, dtype=dtype)
    candidate_norm = np.sum(
        candidates_low * candidates_low,
        axis=1,
        dtype=dtype,
    )
    distances_squared = (
        query_norm[:, None]
        + candidate_norm[None, :]
        - 2 * (query_low @ candidates_low.T)
    )
    np.maximum(distances_squared, 0, out=distances_squared)
    np.sqrt(distances_squared, out=distances_squared)
    return distances_squared


def _peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _run_worker(args):
    dtype = _resolve_dtype(args.precision)
    if dtype is None:
        print(json.dumps({"precision": args.precision, "supported": False}))
        return

    rng = np.random.default_rng(args.seed)
    expression = (
        rng.normal(size=(args.n_cells, args.n_genes))
        .astype(np.float32)
        / np.sqrt(args.n_genes)
    )
    n_keep = min(args.n_neighbors, args.n_cells)
    reference_indices = np.empty((args.n_cells, n_keep), dtype=np.int32)

    start_time = time.perf_counter()
    for start in range(0, args.n_cells, args.block_size):
        stop = min(start + args.block_size, args.n_cells)
        reference_distances = _float32_distances(
            expression[start:stop],
            expression,
        )
        reference_distances[np.arange(stop - start), np.arange(start, stop)] = 0
        for offset, row in enumerate(reference_distances):
            reference_indices[start + offset] = _top_indices(row, n_keep)
        del row, reference_distances
    float32_seconds = time.perf_counter() - start_time

    if args.precision == "float32":
        result = {
            "precision": args.precision,
            "supported": True,
            "top_n_identity": 1.0,
            "max_distance_error": 0.0,
            "mean_distance_error": 0.0,
            "runtime_seconds": float32_seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        print(json.dumps(result))
        return

    observed_indices = np.empty_like(reference_indices)
    max_distance_error = 0.0
    total_distance_error = 0.0
    n_distances = 0
    try:
        start_time = time.perf_counter()
        expression_low = expression.astype(dtype)
        reduced_seconds = time.perf_counter() - start_time
        for start in range(0, args.n_cells, args.block_size):
            stop = min(start + args.block_size, args.n_cells)
            reference_distances = _float32_distances(
                expression[start:stop],
                expression,
            )
            reference_distances[np.arange(stop - start), np.arange(start, stop)] = 0
            start_time = time.perf_counter()
            reduced_distances = _reduced_precision_distances(
                expression_low[start:stop],
                expression_low,
                dtype,
            )
            reduced_distances[np.arange(stop - start), np.arange(start, stop)] = 0
            for offset, row in enumerate(reduced_distances):
                observed_indices[start + offset] = _top_indices(row, n_keep)
            reduced_seconds += time.perf_counter() - start_time
            error = np.abs(
                reduced_distances.astype(np.float32) - reference_distances
            )
            max_distance_error = max(max_distance_error, float(error.max()))
            total_distance_error += float(error.sum(dtype=np.float64))
            n_distances += error.size
            del row, reference_distances, reduced_distances, error
    except (TypeError, ValueError, OverflowError, NotImplementedError) as exc:
        print(json.dumps({
            "precision": args.precision,
            "supported": False,
            "reason": str(exc),
        }))
        return
    same_neighbors = np.all(
        np.sort(reference_indices, axis=1) == np.sort(observed_indices, axis=1),
        axis=1,
    )
    result = {
        "precision": args.precision,
        "supported": True,
        "top_n_identity": float(same_neighbors.mean()),
        "max_distance_error": max_distance_error,
        "mean_distance_error": total_distance_error / n_distances,
        "runtime_seconds": reduced_seconds,
        "float32_reference_seconds": float32_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    print(json.dumps(result))


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cells", type=int, default=2000)
    parser.add_argument("--n-genes", type=int, default=512)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--precision",
        choices=("float32", "float16", "float8"),
        default="float32",
    )
    return parser


def main():
    args = _make_parser().parse_args()
    if args.worker:
        _run_worker(args)
        return

    for precision in ("float32", "float16", "float8"):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--precision",
            precision,
            "--n-cells",
            str(args.n_cells),
            "--n-genes",
            str(args.n_genes),
            "--n-neighbors",
            str(args.n_neighbors),
            "--block-size",
            str(args.block_size),
            "--seed",
            str(args.seed),
        ]
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout.strip())


if __name__ == "__main__":
    main()
