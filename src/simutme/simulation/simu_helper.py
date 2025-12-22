import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from .nneighbors_shared_memory import parallel_sample_with_nearest_neighbors


class SimuMethod:
    AVERAGE = 'ave'
    MULTIPLY = 'mul'
    RANDOM_REPLACEMENT = 'random_replacement'

class DatasetType:
    MERGED_SC = 'merged_sc_dataset'
    SCT = 'sct_dataset'
    GENERATED = 'generated_sc_dataset'

class SampleType:
    POSITIVE = 'positive'
    NEGATIVE = 'negative'

class SamplingMethodGEP:
    RANDOM = 'random'
    N_NEIGHBORS = 'n_neighbors'


def should_use_neighbor_sampling(
        sc_dataset: Optional[str],
        sampling_method: str
) -> bool:
    """Determine if neighbor-based sampling should be used."""
    return sc_dataset == 'merged_sc_dataset' and sampling_method == 'n_neighbors'

def sample_with_neighbors(
        sampled_cell_ids: pd.DataFrame,
        obs_df: pd.DataFrame,
        cell_distance_dict: Dict[str, Tuple[np.ndarray, List[str]]],
        n_threads: int
) -> List[Tuple[str, ...]]:
    """Sample cells using nearest neighbor method."""
    sampling_tasks = [
        {
            'cell_type': row['cell_type'],
            'n_samples': 1,
            'n_base': row['n_cell'],
            'class_by': row['class_by']
        }
        for _, row in sampled_cell_ids.iterrows()
    ]

    results = parallel_sample_with_nearest_neighbors(
        obs_df=obs_df,
        distance_dict=cell_distance_dict,
        sampling_tasks=sampling_tasks,
        n_processes=n_threads,
    )

    return [results[i] for i in range(len(sampling_tasks))]

def sample_random(
        sampled_cell_ids: pd.DataFrame,
        obs_df: pd.DataFrame,
        sep_by_patient: bool,
        sc_dataset: str,
        minimum_n_base: int,
        n_threads: int
) -> List[Tuple[str, ...]]:
    """Sample cells randomly."""
    tasks = [
        (obs_df, 1, row['cell_type'], row['n_cell'], row['class_by'],
         sep_by_patient, sc_dataset, minimum_n_base)
        for _, row in sampled_cell_ids.iterrows()
    ]

    with multiprocessing.get_context('spawn').Pool(n_threads) as pool:
        return pool.starmap(get_sample_id_random, tasks)


def get_sample_id_random(
        obs_df: pd.DataFrame,
        n_samples: int,
        cell_type: str,
        n_base: int,
        class_by: str = 'leiden',
        sep_by_patient: bool = False,
        sc_dataset: Optional[str] = None,
        minimum_n_base: int = 1,
        glioma_epi_ratio: float = 0.1,
) -> Union[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
    """
    Select sample IDs by random sampling from single-cell metadata.

    Parameters
    ----------
    obs_df : pd.DataFrame
        DataFrame corresponding to `adata.obs` from a single-cell dataset.
        The index contains unique cell IDs.
    n_samples : int
        Number of simulated samples to generate.
    cell_type : str
        Target cell type or subtype to sample from (must be a value in `obs_df[class_by]`).
    n_base : int
        Number of single cells to average per simulated sample. If 0, returns an empty tuple.
    class_by : str, default='leiden'
        Column name in `obs_df` that defines the class for sampling.
    sep_by_patient : bool, default=False
        If True, sample cells for each simulated sample from a single patient
        (based on the 'sample_id' column). Only patients with more than `n_base + 10`
        cells in the target class are considered.
    sc_dataset : str, optional
        Name of the single-cell dataset. If 'sct_dataset' and `cell_type == 'Cancer Cells'`,
        further split into glioma vs epithelial cells according to `glioma_epi_ratio`.
    minimum_n_base : int, default=1
        Minimum number of cells required to form a simulated sample.
    glioma_epi_ratio : float, default=0.1
        Probability of sampling glioma cells (vs epithelial) when splitting cancer cells.

    Returns
    -------
    ids : tuple of str or tuple of tuple of str
        If `n_samples == 1`, returns a single tuple of cell IDs: `('cell1', 'cell2', ...)`.
        If `n_samples > 1`, returns a tuple of tuples: `(('cell1', 'cell2'), ('cell3', 'cell4'), ...)`.

    Raises
    ------
    ValueError
        If fewer than `minimum_n_base` cells are available, or if constraints cannot be satisfied.
    """
    if n_base == 0:
        return tuple()

    # Optional splitting of cancer cells into glioma vs epithelial
    if sc_dataset == 'sct_dataset' and cell_type == 'Cancer Cells':
        available_types = obs_df[class_by].unique()
        if 'Epithelial Cells' in available_types and 'Glioma Cells' in available_types:
            cell_type = 'Glioma Cells' if np.random.random() <= glioma_epi_ratio else 'Epithelial Cells'

    # Filter by target class
    if class_by not in obs_df.columns:
        raise KeyError(f'Column "{class_by}" not found in obs_df.')

    obs_df = obs_df.loc[obs_df[class_by] == cell_type].copy()
    all_samples_id = obs_df.index.to_list()

    if len(all_samples_id) < minimum_n_base:
        raise ValueError(
            f"The number of cells in '{cell_type}' ({len(all_samples_id)}) is less than "
            f"minimum_n_base ({minimum_n_base})."
        )

    # Optional patient-wise sampling
    groupby_patient_count = None
    if sep_by_patient:
        if 'sample_id' not in obs_df.columns:
            raise KeyError('Column "sample_id" required when sep_by_patient=True.')

        groupby_patient_count = obs_df.groupby('sample_id')[class_by].count()
        groupby_patient_count = groupby_patient_count[groupby_patient_count > n_base + 10]

        if groupby_patient_count.empty:
            raise ValueError(
                f"No patient has enough cells (> {n_base + 10}) in '{cell_type}'."
            )

    selected_samples: Dict[Tuple[str, ...], int] = {}
    max_iterations = n_samples * 100  # Prevent infinite loop
    iterations = 0

    # Main sampling loop
    while len(selected_samples) < n_samples:
        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError(
                f"Could not generate {n_samples} unique samples after {max_iterations} iterations. "
                "Consider relaxing constraints or increasing the cell pool."
            )

        # Optionally restrict to one patient per simulated sample
        if groupby_patient_count is not None and len(groupby_patient_count) > 0:
            selected_patient_id = np.random.choice(groupby_patient_count.index)
            all_samples_id = obs_df.loc[obs_df['sample_id'] == selected_patient_id].index.to_list()
        else:
            all_samples_id = obs_df.index.to_list()

        # Sample cells
        if minimum_n_base <= len(all_samples_id) <= n_base:
            sample_ids = tuple(sorted(all_samples_id))
        elif len(all_samples_id) < minimum_n_base:
            continue  # Skip this patient if not enough cells
        else:
            sample_ids = tuple(sorted(np.random.choice(all_samples_id, n_base, replace=False)))

        # Store unique samples
        if sample_ids not in selected_samples:
            selected_samples[sample_ids] = 1

    if n_samples == 1:
        return tuple(selected_samples.keys())[0]  # Single tuple of cell IDs
    return tuple(selected_samples.keys())  # Tuple of tuples


def append_to_csv(
    df: pd.DataFrame,
    filepath: Path,
    float_format: str = '%g',
    index_label: Optional[str] = None
) -> None:
    """Append DataFrame to CSV, creating file if needed."""
    file_exists = filepath.exists()
    df.to_csv(
        filepath,
        mode='a' if file_exists else 'w',
        header=not file_exists,
        float_format=float_format,
        index_label=index_label if not file_exists else None
    )


def validate_generation_params_sct(
        n_sample_each_cell_type: int,
        n_base_for_positive_samples: int,
        sample_type: str,
        simu_method: str,
        sampling_method: str,
        minimum_n_base: int,
) -> None:
    """Validate generation parameters."""
    if n_sample_each_cell_type <= 0:
        raise ValueError(f"n_sample_each_cell_type must be positive, got {n_sample_each_cell_type}")

    if n_base_for_positive_samples <= 0:
        raise ValueError(f"n_base_for_positive_samples must be positive, got {n_base_for_positive_samples}")

    if sample_type not in ('positive', 'negative'):
        raise ValueError(f"sample_type must be 'positive' or 'negative', got '{sample_type}'")

    if simu_method not in ('ave', 'mul', 'random_replacement'):
        raise ValueError(f"simu_method must be 'ave', 'mul', or 'random_replacement', got '{simu_method}'")

    if sampling_method not in ('random', 'n_neighbors'):
        raise ValueError(f"sampling_method must be 'random' or 'n_neighbors', got '{sampling_method}'")

    if minimum_n_base <= 0:
        raise ValueError(f"minimum_n_base must be positive, got {minimum_n_base}")

    if n_base_for_positive_samples < minimum_n_base:
        raise ValueError(
            f"n_base_for_positive_samples ({n_base_for_positive_samples}) "
            f"must be >= minimum_n_base ({minimum_n_base})"
        )