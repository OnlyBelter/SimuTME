"""Bounded all-cell query benchmarks at the six observed pre-cap pool sizes."""

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simutme.simulation.all_cell_neighbors import AllCellNeighborSampler
from simutme.utility import get_neighbor_indices_by_class

POOLS = {"CD4 T": 17909, "CD8 T (GZMK high)": 19429, "CD8 T effector": 21616,
         "Cancer Cells": 119970, "Double-neg-like T": 25077, "Macrophages": 37573}


def worker(args):
    rng = np.random.default_rng(42)
    started = perf_counter()
    expression = rng.normal(size=(args.cells, args.features)).astype(np.float32)
    obs = pd.DataFrame({"cell_type": ["A"] * args.cells},
                       index=[f"r{i}" for i in range(args.cells)])
    draws = rng.integers(args.cells, size=args.samples)
    unique = np.unique(draws)
    measured = unique[:args.query_limit] if args.query_limit else unique
    sampler = AllCellNeighborSampler(expression, obs, "cell_type", "subtype")
    sampler.pool("cell_type", "A")
    preparation = perf_counter() - started
    times = []
    for _ in range(args.repeats):
        sampler.cache.clear()
        sampler.cache_size = 0
        started = perf_counter()
        selected = sampler.neighbors(("cell_type", "A"), measured, args.n_base)
        times.append(perf_counter() - started)
    started = perf_counter()
    sampler.neighbors(("cell_type", "A"), measured, args.n_base)
    cached = perf_counter() - started
    started = perf_counter()
    # Production SCT aggregation averages one selected group at a time.
    aggregated = np.stack([expression[indices].mean(axis=0) for indices in selected])
    aggregation = perf_counter() - started
    with tempfile.TemporaryDirectory(prefix="simutme-all-export-") as directory:
        started = perf_counter()
        pd.DataFrame(aggregated).to_csv(Path(directory) / "profiles.csv", index=False)
        export = perf_counter() - started
    baseline = {}
    for label, cap in (("capped_15000", 15000), ("all_precomputed", None)):
        if cap is None and args.cells > args.full_max_cells:
            baseline[label] = {"skipped": "Pool exceeds configured bounded-baseline size"}
            continue
        started = perf_counter()
        reference = get_neighbor_indices_by_class(
            expression, obs, args.n_base, max_cells_per_class=cap, show_progress=False)
        baseline[label] = {"seconds": perf_counter() - started,
                           "cells": len(reference["A"][1]), "repetitions": 1}
        del reference
    median = float(np.median(times))
    print(json.dumps({
        "cells": args.cells, "features": args.features, "samples_requested": args.samples,
        "unique_anchors": len(unique), "anchors_measured": len(measured), "n_base": args.n_base,
        "preparation_seconds": preparation, "uncached_query_seconds": times,
        "query_median_seconds": median, "query_range_seconds": [min(times), max(times)],
        "cached_query_seconds": cached, "aggregation_seconds": aggregation,
        "export_seconds": export, "cache_stats": sampler.stats,
        "estimated_all_unique_query_seconds": median * len(unique) / len(measured),
        "estimate_is_extrapolation": len(unique) != len(measured),
        "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
            (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2),
        "baseline": baseline,
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--n-base", type=int, default=30)
    parser.add_argument("--query-limit", type=int, default=256,
                        help="Measure at most this many unique anchors; 0 measures all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--full-max-cells", type=int, default=25000)
    parser.add_argument("--pools", nargs="+", choices=list(POOLS), default=list(POOLS))
    parser.add_argument("--cells", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.features, args.samples, args.n_base, args.repeats) < 1 or args.query_limit < 0:
        parser.error("Workload sizes must be positive and query-limit nonnegative")
    if args.cells:
        worker(args)
        return
    results = []
    for label in args.pools:
        print(f"Benchmarking {label}: {POOLS[label]} cells", file=sys.stderr, flush=True)
        output = subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
             "--cells", str(POOLS[label])], text=True)
        result = json.loads(output.strip().splitlines()[-1])
        result["cell_type"] = label
        results.append(result)
    print(json.dumps({
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "source": "synthetic",
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "notes": "Reduced-feature scaling benchmark, not full-gene production timing. "
                 "Baseline timing has one repetition; query timing has configured repetitions. "
                 "RSS is a lifetime high-water mark including baseline preparation. "
                 "Cached, aggregation, and export timings cover only measured anchors.",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
