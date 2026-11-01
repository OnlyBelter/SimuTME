"""Benchmark the serial and parallel mGEP candidate paths in temporary storage."""

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import statistics
import sys
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simutme.simulation.filtering import FixedGeneFilter
from simutme.simulation.parallel_candidates import (
    _initialize_worker, _run_batch, _store_shared_matrix,
)
from simutme.simulation.simulator import BulkGEPGenerator
from simutme.utility import ReadExp
from simutme.workflow import (
    SimuConfig, _resolve_configured_segment_prior, _resolve_gep_filtering_params,
)
from simutme.simulation.sampling import (
    estimate_cell_prop_maxima_from_pred_cell_props, flatten_cell_type2subtype,
    merge_cell_prop_prior_with_maxima, restrict_cell_type2subtype_to_output_types,
)


def _memory_monitor(stop, peak):
    try:
        import psutil
    except ImportError:
        return
    root = psutil.Process()
    while not stop.is_set():
        try:
            processes = [root, *root.children(recursive=True)]
            rss = sum(p.memory_info().rss for p in processes if p.is_running())
            peak[0] = max(peak[0] or 0, rss)
        except (psutil.Error, OSError):
            pass
        stop.wait(0.25)


def _workload(args, directory):
    if args.config:
        config = SimuConfig.from_yaml(args.config)
        params = config._dataset_definitions[args.dataset]
        gep = _resolve_gep_filtering_params(args.dataset, params, config)
        if gep.filtering_method != "mgep" or not gep.enabled:
            raise ValueError("Benchmark target must enable mgep filtering.")
        if params.get("sampling_method_prop", "segment") != "segment":
            raise ValueError("Benchmark target must use segment sampling.")
        reference = ReadExp(gep.reference_file, exp_type=gep.ref_exp_type).exp
        if gep.ref_exp_type == "log_space":
            reference = pd.DataFrame(np.expm1(reference.to_numpy() * np.log(2)),
                                     index=reference.index, columns=reference.columns)
        if gep.filtering_ref_types:
            mapping = pd.read_csv(config.tcga2cancer_type_file_path, index_col=0)
            ids = mapping.index[mapping["cancer_type"].isin(gep.filtering_ref_types)]
            reference = reference.loc[reference.index.isin(ids)]
        genes = pd.read_csv(args.gene_list)["gene_name"].tolist()
        cell_types = config.cell_type2subtypes
        output_types = config.output_types
        coefficients = config.total_rna_coefficient
        prior = _resolve_configured_segment_prior(args.dataset, params, config)
        prediction_file = params.get("pred_cell_prop_file_path", config.pred_cell_prop_file_path)
        if prior is None and prediction_file:
            output_mapping = restrict_cell_type2subtype_to_output_types(cell_types, output_types)
            segment_types = flatten_cell_type2subtype(output_mapping)
            maxima, _ = estimate_cell_prop_maxima_from_pred_cell_props(prediction_file, segment_types)
            prior = merge_cell_prop_prior_with_maxima(segment_types, None, maxima)
        sct_path = args.sct
    else:
        import anndata as ad

        rng = np.random.default_rng(42)
        cell_types = {f"CT{i}": [f"CT{i}"] for i in range(4)}
        output_types = list(cell_types)
        coefficients = {name: 1.0 for name in output_types}
        labels = np.repeat(np.arange(4), 50)
        obs = pd.DataFrame(np.eye(4)[labels], columns=output_types,
                           index=[f"sct{i}" for i in range(len(labels))])
        genes = [f"g{i}" for i in range(200)]
        values = rng.uniform(1, 1000, (len(obs), len(genes)))
        sct_path = str(Path(directory) / "synthetic.h5ad")
        ad.AnnData(X=np.log2(values + 1), obs=obs,
                   var=pd.DataFrame(index=genes)).write_h5ad(sct_path)
        reference = pd.DataFrame(rng.uniform(1, 1000, (40, len(genes))), columns=genes)
        gep = SimpleNamespace(
            pca_n_components=1, enabled=True, filtering_by_gene_range=False,
            gene_quantile_range=[0, 0.5, 1], min_percentage_within_gene_range=1,
            filtering_in_pca_space=False, filtering_method="mgep",
            gep_filtering_quantile=(None, 1.0), norm_ord=1,
        )
        prior = None

    generator = BulkGEPGenerator(
        simu_bulk_dir=Path(directory) / "generator", merged_sc_dataset_file_path=None,
        sct_dataset_file_path=sct_path, cell_type2subtype=cell_types,
        sc_dataset_ids=[], output_types=output_types, total_rna_coefficient=coefficients,
        check_basic_info=False, subtype_col_name="cell_type", mixed_sct_loading_mode="eager",
    )
    obs, _, _ = generator._load_sc_data_for_generation("mul")
    space = FixedGeneFilter(genes, reference, gep.reference_file if args.config else "synthetic", None,
                            gep, Path(directory) / "filter", len(genes), len(genes))
    space.prepare_distance_reference()
    if gep.filtering_by_gene_range:
        lower, _, upper = gep.gene_quantile_range
        space.range_bounds = space.reference_linear.quantile([lower, upper])
    space.reference_linear = None
    space.reference_log = None
    attributes = {key: getattr(generator, key) for key in (
        "cell_fraction_type2subtype", "cell_type2subtype", "cell_subtype_used",
        "subtype_col_name", "cell_type_col_name", "total_rna_coefficient",
        "total_cell_number",
    )}
    worker_config = {"seed": args.seed, "batch_size": args.batch_size, "prior": prior,
                     "add_noise": False, "noise_params": (), "quiet": True}
    return generator, obs, space, attributes, worker_config


def _write_result(result, directory):
    for key in ("expression", "fractions", "selected"):
        frame = result[key]
        if not frame.empty:
            if key == "expression":
                frame = np.log2(frame + 1)
            path = directory / f"{key}.csv"
            frame.to_csv(path, mode="a" if path.exists() else "w",
                         header=not path.exists(), float_format="%.17g")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML configuration for a real workload")
    parser.add_argument("--dataset", help="Mixed target in the YAML")
    parser.add_argument("--sct", help="Actual SCT reference H5AD")
    parser.add_argument("--gene-list", help="Actual fixed_filtering_gene_list.csv")
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 4, 8, 16])
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(value < 0 for value in args.workers) or min(args.batch_size, args.batches, args.repeats) < 1:
        parser.error("Workers must be nonnegative; workload sizes must be positive.")
    if args.config and not all((args.dataset, args.sct, args.gene_list)):
        parser.error("--config requires --dataset, --sct, and --gene-list.")
    if not args.config and any((args.dataset, args.sct, args.gene_list)):
        parser.error("Real workload options require --config.")
    thread_vars = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS")
    for key in thread_vars:
        os.environ[key] = "1"

    with tempfile.TemporaryDirectory(prefix="simutme-benchmark-") as temp:
        root = Path(temp)
        setup_start = perf_counter()
        generator, obs, space, attributes, worker_config = _workload(args, root)
        matrix = _store_shared_matrix(generator.sct_dataset_df, root)
        initargs = (str(matrix), generator.sct_dataset_df.index,
                    generator.sct_dataset_df.columns, obs, attributes, worker_config, space)
        setup_seconds = perf_counter() - setup_start
        results = []
        for workers in args.workers:
            peak = [None]
            stop = threading.Event()
            monitor = threading.Thread(target=_memory_monitor, args=(stop, peak), daemon=True)
            monitor.start()
            pool = None
            started = perf_counter()
            try:
                if workers:
                    pool = ProcessPoolExecutor(
                        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                        initializer=_initialize_worker, initargs=initargs,
                    )
                    list(pool.map(_run_batch, range(1000000, 1000000 + workers)))
                else:
                    _initialize_worker(*initargs)
                    _run_batch(1000000)
                warmup_seconds = perf_counter() - started
                repeats, signatures = [], []
                for repeat in range(args.repeats):
                    run_dir = root / f"w{workers}-r{repeat}"
                    run_dir.mkdir()
                    started = perf_counter()
                    batch_ids = range(repeat * args.batches, (repeat + 1) * args.batches)
                    iterator = pool.map(_run_batch, batch_ids) if pool else map(_run_batch, batch_ids)
                    accepted = 0
                    proposals = 0
                    for result in iterator:
                        accepted += result["kept"]
                        proposals += result["fraction_stats"]["attempted"]
                        _write_result(result, run_dir)
                    elapsed = perf_counter() - started
                    digest = hashlib.sha256()
                    for path in sorted(run_dir.glob("*.csv")):
                        digest.update(path.read_bytes())
                    signatures.append(digest.hexdigest())
                    repeats.append({"seconds": elapsed, "candidates": args.batches * args.batch_size,
                                    "accepted": accepted, "fraction_proposals": proposals,
                                    "mgep_acceptance_rate": accepted / (args.batches * args.batch_size),
                                    "fraction_acceptance_rate": (
                                        args.batches * args.batch_size / proposals
                                    )})
            finally:
                if pool:
                    pool.shutdown(wait=True, cancel_futures=True)
                stop.set()
                monitor.join()
            median_seconds = statistics.median(run["seconds"] for run in repeats)
            median_accepted = statistics.median(run["accepted"] for run in repeats)
            results.append({"workers": workers, "warmup_seconds": warmup_seconds,
                            "repeats": repeats, "output_sha256": signatures,
                            "median_seconds": median_seconds,
                            "candidate_profiles_per_second": args.batches * args.batch_size / median_seconds,
                            "accepted_profiles_per_second": median_accepted / median_seconds,
                            "peak_process_tree_rss_bytes": peak[0]})
        signatures = [result["output_sha256"] for result in results]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise RuntimeError("Benchmark variants produced different candidate outputs.")
        baseline = next((result["median_seconds"] for result in results
                         if result["workers"] == 0), None)
        for result in results:
            result["speedup_vs_serial"] = baseline / result["median_seconds"] if baseline else None
        print(json.dumps({
            "platform": platform.platform(), "python": sys.version,
            "numpy": np.__version__, "pandas": pd.__version__,
            "logical_cpus": os.cpu_count(), "sct": args.sct or "synthetic",
            "selected_genes": len(space.genes), "source_shape": generator.sct_dataset_df.shape,
            "batch_size": args.batch_size, "seed": args.seed, "setup_seconds": setup_seconds,
            "thread_limits": {key: os.environ[key] for key in thread_vars},
            "notes": "Timings include candidate generation and CSV writes; exclude checkpoints, "
                     "final H5AD, PCA, and plots. RSS sums processes and double-counts shared pages; "
                     "null means psutil is unavailable. Synthetic results are not server estimates.",
            "results": results,
        }, indent=2))


if __name__ == "__main__":
    main()
