"""Deterministic, bounded parallel generation of filtered Mixed candidates."""

import copy
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
_CAPPED_WAVE_SIZE = 8


def _initialize_worker(
        matrix_path, source_index, genes, obs, attributes, config, space,
        reference_log_path=None):
    global _WORKER, _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)
    matrix = np.load(matrix_path, mmap_mode="r")
    generator = object.__new__(BulkGEPGenerator)
    for key, value in attributes.items():
        setattr(generator, key, value)
    generator.sct_dataset_df = pd.DataFrame(matrix, index=source_index, columns=genes, copy=False)
    generator.sct_dataset_reader = None
    generator.filtering_space = space
    if reference_log_path is not None:
        generator.filtering_space.reference_log_memmap = np.load(
            reference_log_path, mmap_mode="r",
        )
    generator._reset_runtime_log_state()
    if config.get("quiet"):
        generator._maybe_log_total_rna_coefficients = lambda coefficients: None
    _WORKER = (generator, obs, config)


def _run_batch(batch_id, cancer_type=None):
    generator, obs, config = _WORKER
    seed = np.random.SeedSequence([config["seed"], batch_id]).generate_state(4)
    np.random.seed(seed)
    space = generator.filtering_space
    space.range_input = space.range_kept = 0
    space.distance_input = space.distance_kept = 0
    space.mgep_distance_input = space.mgep_distance_kept = 0
    started = perf_counter()
    sampling_method = config.get("sampling_method_prop", "segment")
    fraction_rng = None
    if sampling_method == "dirichlet":
        fraction_seed = config.get("dirichlet_sampling_seed")
        if fraction_seed is None:
            fraction_seed = config["seed"]
        # Keep the fraction stream distinct from candidate and subgroup streams.
        fraction_rng = np.random.default_rng(
            np.random.SeedSequence([fraction_seed, batch_id, 2])
        )
    fractions = generate_cell_fraction(
        sampling_method=sampling_method,
        n_cell_frac=config["batch_size"],
        sample_prefix=f"s_{sampling_method}_{batch_id}",
        cell_prop_prior=config["prior"] if sampling_method == "segment" else None,
        cell_type2subtype=generator.cell_fraction_type2subtype,
        dirichlet_alpha=config.get("dirichlet_alpha"),
        rng=fraction_rng,
        max_segment_sampling_attempts=(
            max(10000, config["batch_size"] * 500)
            if sampling_method == "segment" and config["prior"] else None
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
    subgroup_seed = None
    if space.uses_subgroup_reference:
        subgroup_seed = int(
            np.random.SeedSequence([config["seed"], batch_id, 1]).generate_state(1)[0]
        )
    filter_options = {} if cancer_type is None else {"cancer_type": cancer_type}
    accepted = space.filter(
        expression, batch_id=batch_id, subgroup_seed=subgroup_seed, **filter_options,
    )
    done = perf_counter()
    if config.get("statistics_only"):
        return {
            "batch_id": batch_id,
            "cancer_type": space.last_batch_cancer_type,
            "input": len(expression),
            "kept": len(accepted),
            "stats": space.stats(),
            "fraction_stats": fraction_stats,
            "pca_n_components": space.last_batch_pca_n_components,
            "timings": (
                sampled_at - started, aggregated_at - sampled_at,
                filtered_at - aggregated_at, done - filtered_at,
            ),
        }
    indices = accepted.index
    return {
        "batch_id": batch_id,
        "expression": accepted,
        "cancer_type": space.last_batch_cancer_type,
        "fractions": fractions.loc[indices].copy(),
        "selected": selected.loc[indices].copy(),
        "input": len(expression),
        "kept": len(accepted),
        "stats": space.stats(),
        "fraction_stats": fraction_stats,
        "pca_n_components": space.last_batch_pca_n_components,
        "timings": (
            sampled_at - started, aggregated_at - sampled_at,
            filtered_at - aggregated_at, done - filtered_at,
        ),
    }


def _store_shared_matrix(source, directory, filename="sct-expression.npy"):
    path = Path(directory) / filename
    rows, columns = source.shape
    dtype = np.result_type(*source.dtypes)
    required = rows * columns * np.dtype(dtype).itemsize
    if shutil.disk_usage(directory).free < required + 1024 * 1024:
        raise OSError(
            f"Temporary storage for shared matrix requires at least {required} bytes; "
            "set TMPDIR to a location with sufficient free space."
        )
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=source.shape)
    for start in range(0, rows, 256):
        matrix[start:start + 256] = source.iloc[start:start + 256].to_numpy(copy=False)
    matrix.flush()
    del matrix
    return path


def run_parallel_candidates(generator, config, obs, space, statistics_only=False,
                            max_batches=None):
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
    resolved_dirichlet_alpha = getattr(generator, "resolved_dirichlet_alpha", None)
    worker_config = {
        "seed": config.mixed_candidate_seed,
        "batch_size": _candidate_batch_size(config.n_samples, config.filtering),
        "sampling_method_prop": config.sampling_method_prop,
        "prior": config.cell_prop_prior,
        "dirichlet_alpha": (
            config.dirichlet_alpha
            if resolved_dirichlet_alpha is None
            else resolved_dirichlet_alpha
        ),
        "dirichlet_sampling_seed": config.dirichlet_sampling_seed,
        "add_noise": config.add_noise,
        "noise_params": config.noise_params,
        "statistics_only": statistics_only,
    }
    workers = config.mixed_candidate_workers
    logger.info(
        "Parallel candidates: workers=%d, batch_size=%d, seed=%d, sampling=%s, "
        "dirichlet_seed=%s.",
        workers, worker_config["batch_size"], config.mixed_candidate_seed,
        config.sampling_method_prop, config.dirichlet_sampling_seed,
    )
    with tempfile.TemporaryDirectory(prefix="simutme-candidates-") as directory:
        started = perf_counter()
        matrix_path = _store_shared_matrix(source, directory)
        logger.info("Shared SCT matrix prepared in %.3g seconds at %s.",
                    perf_counter() - started, matrix_path)
        reference_log_path = None
        worker_space = space
        if space.uses_subgroup_reference:
            if space.reference_log is None:
                raise ValueError(
                    "Parallel subgroup filtering requires the full GEP reference in the coordinator."
                )
            reference_log_path = _store_shared_matrix(
                space.reference_log, directory, filename="gep-reference-log.npy",
            )
            worker_space = copy.copy(space)
            worker_space.reference_log = None
            worker_space.reference_log_memmap = None
            logger.info(
                "Shared GEP reference prepared with %d samples for per-batch subgroups.",
                space.reference_sample_count,
            )
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_initialize_worker,
            initargs=(str(matrix_path), source.index, source.columns, obs,
                      attributes, worker_config, worker_space,
                      str(reference_log_path) if reference_log_path else None),
        ) as executor, tqdm(total=generator.n_samples, initial=generator.generated_bulk_gep_counter) as progress:
            futures = {}
            next_submit = generator.n_round
            capped = not statistics_only and config.max_samples_per_cancer_type is not None

            def submit_capped_wave():
                wave = generator.capped_parallel_wave
                if wave is None or generator.n_round >= wave["end"]:
                    wave = {
                        "start": generator.n_round,
                        "end": generator.n_round + _CAPPED_WAVE_SIZE,
                        "types": generator._eligible_uncapped_types(),
                    }
                    if not wave["types"]:
                        raise ValueError("No reference cancer type has remaining capacity.")
                    generator.capped_parallel_wave = wave
                    generator._save_filtering_metadata()
                    generator._save_parallel_checkpoint()
                for batch in range(generator.n_round, wave["end"]):
                    rng = np.random.default_rng(
                        np.random.SeedSequence([config.mixed_candidate_seed, batch, 3])
                    )
                    selected_type = str(rng.choice(wave["types"]))
                    futures[batch] = executor.submit(_run_batch, batch, selected_type)

            thread_vars = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                           "NUMEXPR_NUM_THREADS")
            previous = {key: os.environ.get(key) for key in thread_vars}
            try:
                for key in thread_vars:
                    os.environ[key] = "1"
                if capped:
                    submit_capped_wave()
                else:
                    for _ in range(workers * 2):
                        if statistics_only and max_batches is not None and next_submit >= max_batches:
                            break
                        futures[next_submit] = executor.submit(_run_batch, next_submit)
                        next_submit += 1
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            while generator.generated_bulk_gep_counter < generator.n_samples and (
                not statistics_only or max_batches is None or generator.n_round < max_batches
            ):
                batch_id = generator.n_round
                try:
                    result = futures.pop(batch_id).result()
                except Exception as exc:
                    for future in futures.values():
                        future.cancel()
                    raise RuntimeError(f"Parallel candidate batch {batch_id} failed.") from exc
                generator._record_gep_filtering_counts(
                    result["input"],
                    result["kept"],
                    cancer_type=result["cancer_type"],
                )
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
                if statistics_only:
                    generator.generated_bulk_gep_counter += result["kept"]
                    progress.update(result["kept"])
                else:
                    accepted = result["expression"]
                    if capped:
                        accepted = generator._retain_capped_batch(accepted, result["cancer_type"])
                    else:
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
                if result["pca_n_components"] is not None:
                    generator._record_gep_filtering_pca_components(
                        result["pca_n_components"]
                    )
                if not statistics_only:
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
                    if capped and not futures:
                        submit_capped_wave()
                    elif not capped and (
                        not statistics_only or max_batches is None or next_submit < max_batches
                    ):
                        futures[next_submit] = executor.submit(_run_batch, next_submit)
                        next_submit += 1
            for future in futures.values():
                future.cancel()
