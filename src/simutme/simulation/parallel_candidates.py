"""Deterministic, bounded parallel generation of filtered Mixed candidates."""

import logging
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .sampling import generate_cell_fraction
from .simu_helper import DatasetType
from .simulator import BulkGEPGenerator, _candidate_batch_size

logger = logging.getLogger(__name__)
_WORKER = None
_THREAD_LIMIT = None


def _initialize_worker(matrix_path, source_index, genes, obs, attributes, config, space):
    global _WORKER, _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)
    matrix = np.load(matrix_path, mmap_mode="r")
    generator = object.__new__(BulkGEPGenerator)
    for key, value in attributes.items():
        setattr(generator, key, value)
    generator.sct_dataset_df = pd.DataFrame(matrix, index=source_index, columns=genes, copy=False)
    generator.sct_dataset_reader = None
    generator.filtering_space = space
    generator._reset_runtime_log_state()
    if config.get("quiet"):
        generator._maybe_log_total_rna_coefficients = lambda coefficients: None
    _WORKER = (generator, obs, config)


def _run_batch(batch_id):
    generator, obs, config = _WORKER
    seed = np.random.SeedSequence([config["seed"], batch_id]).generate_state(4)
    np.random.seed(seed)
    space = generator.filtering_space
    space.range_input = space.range_kept = 0
    space.distance_input = space.distance_kept = 0
    space.mgep_distance_input = space.mgep_distance_kept = 0
    started = perf_counter()
    fractions = generate_cell_fraction(
        sampling_method="segment",
        n_cell_frac=config["batch_size"],
        sample_prefix=f"s_segment_{batch_id}",
        cell_prop_prior=config["prior"],
        cell_type2subtype=generator.cell_fraction_type2subtype,
        max_segment_sampling_attempts=(
            max(10000, config["batch_size"] * 500) if config["prior"] else None
        ),
    )
    fraction_stats = fractions.attrs.get("segment_sampling_stats")
    sampled_at = perf_counter()
    selected = generator._sc_sampling(
        cell_frac=fractions, n_threads=1, obs_df=obs, sc_dataset=DatasetType.SCT,
    )
    aggregated_at = perf_counter()
    expression = generator._map_cell_id2exp(
        selected_cell_id=selected, simu_method="mul", sc_dataset=DatasetType.SCT,
        cell_frac=fractions, add_noise=config["add_noise"], noise_params=config["noise_params"],
    )
    filtered_at = perf_counter()
    accepted = space.filter(expression)
    done = perf_counter()
    indices = accepted.index
    return {
        "batch_id": batch_id,
        "expression": accepted,
        "fractions": fractions.loc[indices].copy(),
        "selected": selected.loc[indices].copy(),
        "input": len(expression),
        "kept": len(accepted),
        "stats": space.stats(),
        "fraction_stats": fraction_stats,
        "timings": (
            sampled_at - started, aggregated_at - sampled_at,
            filtered_at - aggregated_at, done - filtered_at,
        ),
    }


def _store_shared_matrix(source, directory):
    path = Path(directory) / "sct-expression.npy"
    rows, columns = source.shape
    dtype = np.result_type(*source.dtypes)
    required = rows * columns * np.dtype(dtype).itemsize
    if shutil.disk_usage(directory).free < required + 1024 * 1024:
        raise OSError(
            f"Temporary storage for shared SCT matrix requires at least {required} bytes; "
            "set TMPDIR to a location with sufficient free space."
        )
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=source.shape)
    for start in range(0, rows, 256):
        matrix[start:start + 256] = source.iloc[start:start + 256].to_numpy(copy=False)
    matrix.flush()
    del matrix
    return path


def run_parallel_candidates(generator, config, obs, space):
    """Run workers with ordered commits and a single parent-side writer."""
    source = generator.sct_dataset_df
    if not source.index.is_unique:
        raise ValueError("Parallel candidates require unique SCT reference row IDs.")
    attributes = {
        key: getattr(generator, key) for key in (
            "cell_fraction_type2subtype", "cell_type2subtype", "cell_subtype_used",
            "subtype_col_name", "cell_type_col_name", "total_rna_coefficient",
            "total_cell_number",
        )
    }
    worker_config = {
        "seed": config.mixed_candidate_seed,
        "batch_size": _candidate_batch_size(config.n_samples, config.filtering),
        "prior": config.cell_prop_prior,
        "add_noise": config.add_noise,
        "noise_params": config.noise_params,
    }
    workers = config.mixed_candidate_workers
    logger.info("Parallel candidates: workers=%d, batch_size=%d, seed=%d.",
                workers, worker_config["batch_size"], config.mixed_candidate_seed)
    with tempfile.TemporaryDirectory(prefix="simutme-candidates-") as directory:
        started = perf_counter()
        matrix_path = _store_shared_matrix(source, directory)
        logger.info("Shared SCT matrix prepared in %.3g seconds at %s.",
                    perf_counter() - started, matrix_path)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_initialize_worker,
            initargs=(str(matrix_path), source.index, source.columns, obs,
                      attributes, worker_config, space),
        ) as executor, tqdm(total=generator.n_samples, initial=generator.generated_bulk_gep_counter) as progress:
            futures = {}
            next_submit = generator.n_round
            thread_vars = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                           "NUMEXPR_NUM_THREADS")
            previous = {key: os.environ.get(key) for key in thread_vars}
            try:
                for key in thread_vars:
                    os.environ[key] = "1"
                for _ in range(workers * 2):
                    futures[next_submit] = executor.submit(_run_batch, next_submit)
                    next_submit += 1
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            while generator.generated_bulk_gep_counter < generator.n_samples:
                batch_id = generator.n_round
                try:
                    result = futures.pop(batch_id).result()
                except Exception as exc:
                    for future in futures.values():
                        future.cancel()
                    raise RuntimeError(f"Parallel candidate batch {batch_id} failed.") from exc
                generator._record_gep_filtering_counts(result["input"], result["kept"])
                for name, value in result["stats"].items():
                    attribute = {
                        "gene_range_input_n_samples": "range_input",
                        "gene_range_kept_n_samples": "range_kept",
                        "pc_distance_input_n_samples": "distance_input",
                        "pc_distance_kept_n_samples": "distance_kept",
                        "mgep_distance_input_n_samples": "mgep_distance_input",
                        "mgep_distance_kept_n_samples": "mgep_distance_kept",
                    }.get(name)
                    if attribute:
                        setattr(generator.filtering_space, attribute,
                                getattr(generator.filtering_space, attribute) + value)
                accepted = result["expression"]
                remaining = generator.n_samples - generator.generated_bulk_gep_counter
                accepted = accepted.iloc[:remaining]
                if not accepted.empty:
                    indices = accepted.index
                    generator._save_simulated_bulk_gep(
                        gep=np.log2(accepted + 1),
                        cell_id=result["selected"].loc[indices],
                        cell_fraction=result["fractions"].loc[indices],
                    )
                    generator.generated_bulk_gep_counter += len(accepted)
                    progress.update(len(accepted))
                generator.n_round += 1
                generator._save_filtering_metadata()
                generator._save_parallel_checkpoint()
                fraction_stats = result["fraction_stats"]
                if fraction_stats:
                    logger.info(
                        "Parallel batch %d fraction proposals: %d/%d accepted.",
                        batch_id, fraction_stats["accepted"], fraction_stats["attempted"],
                    )
                logger.info(
                    "Parallel candidate batch %d: accepted=%d/%d; "
                    "fractions=%.3gs, sampling=%.3gs, aggregation=%.3gs, filtering=%.3gs.",
                    batch_id, result["kept"], result["input"], *result["timings"],
                )
                if generator.generated_bulk_gep_counter < generator.n_samples:
                    futures[next_submit] = executor.submit(_run_batch, next_submit)
                    next_submit += 1
            for future in futures.values():
                future.cancel()
