"""Compare CPU candidate simulation with a pinned pre-optimization Git revision."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simutme.simulation import simulator, simu_helper
from simutme.utility import read_single_cell_type_dataset


def baseline_function(revision, path, name, namespace, class_name=None):
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)
    nodes = ast.parse(source).body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef)
                     and node.name == class_name).body
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


def workload(args):
    rng = np.random.default_rng(42)
    if args.reference:
        obs, source, reader = read_single_cell_type_dataset(args.reference)
        cell_types = sorted(obs[args.cell_type_column].dropna().unique())
    else:
        cell_types = [f"CT{i}" for i in range(args.cell_types)]
        labels = np.repeat(cell_types, args.rows_per_type)
        obs = pd.DataFrame({args.cell_type_column: labels},
                           index=[f"r{i}" for i in range(len(labels))])
        source = pd.DataFrame(
            rng.uniform(1, 1000, (len(obs), args.genes)).astype(np.float32),
            index=obs.index, columns=[f"g{i}" for i in range(args.genes)])
    fractions = pd.DataFrame(rng.dirichlet(np.ones(len(cell_types)), args.samples),
                              index=[f"s{i:06}" for i in range(args.samples)], columns=cell_types)
    tasks = pd.DataFrame({
        "cell_type": np.tile(cell_types, args.samples),
        "class_by": args.cell_type_column,
        "n_cell": 1,
    }, index=np.repeat(fractions.index, len(cell_types)))
    # Fixed replay plan independent of the sampler under measurement.
    pools = {ct: obs.index[obs[args.cell_type_column] == ct].to_numpy() for ct in cell_types}
    tasks["selected_cell_id"] = [
        pools[ct][rng.integers(len(pools[ct]))] for ct in tasks.cell_type]
    generator = object.__new__(simulator.BulkGEPGenerator)
    generator.sct_dataset_df = source
    generator.sct_dataset_reader = None
    generator.cell_type2subtype = {ct: [ct] for ct in cell_types}
    generator.total_rna_coefficient = {ct: 1.0 for ct in cell_types}
    return obs, source, fractions, tasks, generator


def run_worker(args):
    started = time.perf_counter()
    obs, source, fractions, tasks, generator = workload(args)
    setup_seconds = time.perf_counter() - started
    old_map = baseline_function(
        args.baseline, "src/simutme/simulation/simulator.py", "_map_cell_id2exp",
        vars(simulator), "BulkGEPGenerator")
    old_sample = baseline_function(
        args.baseline, "src/simutme/simulation/simu_helper.py", "sample_random",
        vars(simu_helper))

    def aggregate(function):
        return function(generator, selected_cell_id=tasks, cell_frac=fractions,
                        sc_dataset=simu_helper.DatasetType.SCT, simu_method="mul")

    def execute():
        if args.worker == "legacy_direct":
            return [simu_helper.get_sample_id_random(
                obs, 1, row.cell_type, row.n_cell, row.class_by,
                False, simu_helper.DatasetType.SCT, 1)
                for row in tasks.itertuples(index=False)]
        if args.worker in ("legacy_pool", "optimized_sampling"):
            function = old_sample if args.worker == "legacy_pool" else simu_helper.sample_random
            return function(tasks, obs, False, simu_helper.DatasetType.SCT, 1, args.processes)
        if args.worker in ("legacy_generation", "optimized_generation"):
            function = old_sample if args.worker == "legacy_generation" else simu_helper.sample_random
            selection = function(tasks, obs, False, simu_helper.DatasetType.SCT, 1, args.processes)
            saved = tasks["selected_cell_id"].copy()
            tasks["selected_cell_id"] = [ids[0] for ids in selection]
            try:
                return aggregate(old_map if args.worker == "legacy_generation"
                                 else simulator.BulkGEPGenerator._map_cell_id2exp)
            finally:
                tasks["selected_cell_id"] = saved
        return aggregate(old_map if args.worker == "legacy_aggregation"
                         else simulator.BulkGEPGenerator._map_cell_id2exp)

    elapsed = []
    result = execute()  # Warmup, including process startup for pool variants.
    del result
    for _ in range(args.repeats):
        np.random.seed(123)
        started = time.perf_counter()
        result = execute()
        elapsed.append(time.perf_counter() - started)
        if isinstance(result, pd.DataFrame):
            digest = hashlib.sha256(result.to_numpy().tobytes()).hexdigest()
        else:
            digest = None
        del result
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({
        "variant": args.worker, "seconds": elapsed, "median_seconds": float(np.median(elapsed)),
        "range_seconds": [min(elapsed), max(elapsed)], "setup_seconds": setup_seconds,
        "peak_parent_rss_gib": rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2),
        "output_sha256": digest, "samples": args.samples, "genes": source.shape[1],
        "reference_rows": len(obs), "tasks": len(tasks),
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--genes", type=int, default=2000)
    parser.add_argument("--cell-types", type=int, default=16)
    parser.add_argument("--rows-per-type", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--reference", help="Actual SCT H5AD; loads full linear expression into RAM")
    parser.add_argument("--cell-type-column", default="cell_type")
    parser.add_argument("--baseline", default="801f248")
    variants = ("legacy_direct", "legacy_pool", "optimized_sampling",
                "legacy_aggregation", "optimized_aggregation",
                "legacy_generation", "optimized_generation")
    parser.add_argument("--variants", nargs="+", choices=variants, default=list(variants))
    parser.add_argument("--worker", choices=variants, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.samples, args.genes, args.cell_types, args.rows_per_type,
           args.repeats, args.processes) < 1:
        parser.error("Workload sizes, repetitions, and processes must be positive")
    if args.worker:
        run_worker(args)
        return
    results = []
    for variant in args.variants:
        print(f"Benchmarking {variant}", file=sys.stderr, flush=True)
        command = [sys.executable, str(Path(__file__).resolve()),
                   *sys.argv[1:], "--worker", variant]
        output = subprocess.check_output(command, cwd=ROOT, text=True)
        results.append(json.loads(output.strip().splitlines()[-1]))
        print(f"{variant}: median={results[-1]['median_seconds']:.3f}s",
              file=sys.stderr, flush=True)
    by_variant = {result["variant"]: result for result in results}
    if "legacy_aggregation" in by_variant and "optimized_aggregation" in by_variant:
        if (by_variant["legacy_aggregation"]["output_sha256"]
                != by_variant["optimized_aggregation"]["output_sha256"]):
            raise RuntimeError("Legacy and optimized rounded aggregation outputs differ.")
    speedups = {}
    for old, new in (("legacy_pool", "optimized_sampling"),
                     ("legacy_aggregation", "optimized_aggregation"),
                     ("legacy_generation", "optimized_generation")):
        if old in by_variant and new in by_variant:
            speedups[new] = by_variant[old]["median_seconds"] / by_variant[new]["median_seconds"]
    print(json.dumps({
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "logical_cpus": os.cpu_count(), "processes": args.processes,
        "baseline": args.baseline, "reference": args.reference or "synthetic",
        "speedups": speedups,
        "reference_loading": "eager", "rna_coefficients": "all 1.0",
        "noise_enabled": False, "fraction_distribution": "symmetric Dirichlet alpha=1",
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "exclusions": "Fraction/task construction is setup; filtering and file export are excluded. "
                      "Generation timing covers sampling, aggregation, and normalization. "
                      "RSS excludes worker processes and is a process-lifetime high-water mark.",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
