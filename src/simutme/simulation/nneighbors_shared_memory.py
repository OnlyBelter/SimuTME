"""
Shared memory management for multiprocessing in SimuTME.

This module provides classes and functions for efficient data sharing
across multiple processes using Python's shared_memory module.
"""

import os
import pickle
import atexit
import warnings
import logging
import numpy as np
import pandas as pd
from multiprocessing import shared_memory, Pool
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

# Set thread limits before numpy operations
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SampleTaskArgs:
    """Typed arguments for sampling tasks."""
    cell_type: str
    n_samples: int
    n_base: int
    class_by: str = 'leiden'
    sep_by_patient: bool = False
    sc_dataset: str = 'merged_sc_dataset'
    minimum_n_base: int = 1
    glioma_epi_ratio: float = 0.1
    neighbor_sampling_scheme: str = 'single_anchor'

    def to_tuple(self) -> tuple:
        """Convert to tuple for multiprocessing."""
        return (
            self.cell_type,
            self.n_samples,
            self.n_base,
            self.class_by,
            self.sep_by_patient,
            self.sc_dataset,
            self.minimum_n_base,
            self.glioma_epi_ratio,
            self.neighbor_sampling_scheme,
        )

    @classmethod
    def from_tuple(cls, args: tuple) -> 'SampleTaskArgs':
        """Create from tuple."""
        return cls(
            cell_type=args[0],
            n_samples=args[1],
            n_base=args[2],
            class_by=args[3],
            sep_by_patient=args[4],
            sc_dataset=args[5],
            minimum_n_base=args[6],
            glioma_epi_ratio=args[7],
            neighbor_sampling_scheme=args[8],
        )

    @classmethod
    def from_dict(cls, d: dict) -> 'SampleTaskArgs':
        """Create from dictionary."""
        return cls(
            cell_type=d['cell_type'],
            n_samples=d['n_samples'],
            n_base=d['n_base'],
            class_by=d.get('class_by', 'leiden'),
            sep_by_patient=d.get('sep_by_patient', False),
            sc_dataset=d.get('sc_dataset', 'merged_sc_dataset'),
            minimum_n_base=d.get('minimum_n_base', 1),
            glioma_epi_ratio=d.get('glioma_epi_ratio', 0.1),
            neighbor_sampling_scheme=d.get('neighbor_sampling_scheme', 'single_anchor'),
        )


@dataclass
class CompactNeighborData:
    """Top-neighbor indices for each anchor in one ordered candidate pool."""
    neighbor_indices: np.ndarray
    cell_ids: List[str]


NeighborData = Union[Tuple[np.ndarray, List[str]], CompactNeighborData]


@dataclass
class WorkerSharedData:
    """Container for shared data in worker processes."""
    obs_df: pd.DataFrame
    distance_dict: Dict[str, NeighborData]
    shm_handles: List[shared_memory.SharedMemory] = field(default_factory=list)

    def close(self):
        """Close all shared memory handles (don't unlink - main process does that)."""
        for shm in self.shm_handles:
            try:
                shm.close()
            except Exception as e:
                logger.warning(f"Error closing shared memory handle: {e}")
        self.shm_handles.clear()


@dataclass
class SharedMemoryMetadata:
    """Metadata for reconstructing shared memory in worker processes."""
    obs_shm_name: str
    obs_size: int
    distance_shms: Dict[str, str]
    distance_shapes: Dict[str, Tuple[int, ...]]
    distance_dtypes: Dict[str, str]
    distance_cell_ids: Dict[str, List[str]]
    distance_kinds: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for pickling."""
        return {
            'obs_shm_name': self.obs_shm_name,
            'obs_size': self.obs_size,
            'distance_shms': self.distance_shms.copy(),
            'distance_shapes': self.distance_shapes.copy(),
            'distance_dtypes': self.distance_dtypes.copy(),
            'distance_cell_ids': self.distance_cell_ids.copy(),
            'distance_kinds': self.distance_kinds.copy(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SharedMemoryMetadata':
        """Create from dictionary."""
        return cls(**d)


# =============================================================================
# Shared Memory Manager
# =============================================================================

class SharedDataManager:
    """
    Manager for shared memory objects across processes.

    This class handles the creation, tracking, and cleanup of shared memory
    objects used to share data between the main process and worker processes.

    Features:
    - Automatic cleanup via context manager and atexit
    - Proper error handling during creation and cleanup
    - Idempotent cleanup (safe to call multiple times)
    - Support for both numpy arrays and DataFrames

    Example
    -------
    >>> with SharedDataManager(obs_df, distance_dict) as manager:
    ...     metadata = manager.get_metadata()
    ...     with Pool(4, initializer=worker_init, initargs=(metadata,)) as pool:
    ...         results = pool.map(worker_sample, tasks)
    """

    def __init__(
        self,
        obs_df: pd.DataFrame,
        distance_dict: Dict[str, NeighborData],
        dtype: np.dtype = np.float64
    ):
        """
        Initialize shared memory for obs_df and distance matrices.

        Parameters
        ----------
        obs_df : pd.DataFrame
            The observation DataFrame to share.
        distance_dict : dict
            Distance matrices or compact neighbor data keyed by cell type.
        dtype : np.dtype
            Data type for distance matrices (default: float64).
        """
        self._cleaned_up = False
        self._shm_objects: List[shared_memory.SharedMemory] = []
        self._metadata: Optional[SharedMemoryMetadata] = None

        try:
            self._init_obs_df(obs_df)
            self._init_distance_matrices(distance_dict, dtype)
            atexit.register(self.cleanup)
            logger.debug(
                "SharedDataManager initialized with %d neighbor data arrays",
                len(distance_dict),
            )
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize shared memory: {e}") from e

    def _init_obs_df(self, obs_df: pd.DataFrame) -> None:
        """Initialize shared memory for observation DataFrame."""
        obs_bytes = pickle.dumps(obs_df)
        obs_size = len(obs_bytes)

        shm_obs = shared_memory.SharedMemory(create=True, size=obs_size)
        self._shm_objects.append(shm_obs)

        # Copy data to shared memory buffer
        buffer = np.ndarray(obs_size, dtype=np.uint8, buffer=shm_obs.buf)
        np.copyto(buffer, np.frombuffer(obs_bytes, dtype=np.uint8))

        self._obs_shm_name = shm_obs.name
        self._obs_size = obs_size

    def _init_distance_matrices(
        self,
        distance_dict: Dict[str, NeighborData],
        dtype: np.dtype
    ) -> None:
        """Initialize shared memory for distance matrices."""
        self._distance_shms: Dict[str, str] = {}
        self._distance_shapes: Dict[str, Tuple[int, ...]] = {}
        self._distance_dtypes: Dict[str, str] = {}
        self._distance_cell_ids: Dict[str, List[str]] = {}
        self._distance_kinds: Dict[str, str] = {}

        for cell_type, data in distance_dict.items():
            if isinstance(data, CompactNeighborData):
                array = np.asarray(data.neighbor_indices)
                if array.dtype.kind not in 'iu':
                    raise TypeError(
                        f'Neighbor indices for {cell_type!r} must have an '
                        f'integer dtype, got {array.dtype}.'
                    )
                cell_ids = data.cell_ids
                if array.ndim != 2 or array.shape[0] != len(cell_ids):
                    raise ValueError(
                        f'Neighbor indices for {cell_type!r} must have one '
                        f'row per cell ID; got shape {array.shape} for '
                        f'{len(cell_ids)} IDs.'
                    )
                kind = 'neighbor_indices'
            else:
                dist_matrix, cell_ids = data
                array = np.asarray(dist_matrix, dtype=dtype)
                kind = 'distance_matrix'

            shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
            self._shm_objects.append(shm)

            # Create view and copy data
            shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
            np.copyto(shm_array, array)

            self._distance_shms[cell_type] = shm.name
            self._distance_shapes[cell_type] = array.shape
            self._distance_dtypes[cell_type] = array.dtype.name
            self._distance_cell_ids[cell_type] = cell_ids
            self._distance_kinds[cell_type] = kind

    def get_metadata(self) -> dict:
        """
        Get metadata needed by worker processes.

        Returns
        -------
        dict
            Metadata dictionary for worker initialization.

        Raises
        ------
        RuntimeError
            If the manager has already been cleaned up.
        """
        if self._cleaned_up:
            raise RuntimeError("SharedDataManager has been cleaned up")

        return {
            'obs_shm_name': self._obs_shm_name,
            'obs_size': self._obs_size,
            'distance_shms': self._distance_shms.copy(),
            'distance_shapes': self._distance_shapes.copy(),
            'distance_dtypes': self._distance_dtypes.copy(),
            'distance_cell_ids': self._distance_cell_ids.copy(),
            'distance_kinds': self._distance_kinds.copy(),
        }

    def cleanup(self) -> None:
        """Clean up all shared memory objects safely."""
        if self._cleaned_up:
            return

        self._cleaned_up = True

        # Unregister atexit handler
        try:
            atexit.unregister(self.cleanup)
        except Exception:
            pass

        errors = []
        for shm in self._shm_objects:
            try:
                shm.close()
            except Exception as e:
                errors.append(f"Error closing {shm.name}: {e}")

            try:
                shm.unlink()
            except FileNotFoundError:
                pass  # Already unlinked
            except Exception as e:
                errors.append(f"Error unlinking {shm.name}: {e}")

        self._shm_objects.clear()

        if errors:
            warnings.warn(f"Errors during cleanup: {'; '.join(errors)}")

    def __enter__(self) -> 'SharedDataManager':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.cleanup()
        return False

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


# =============================================================================
# Worker Process Functions
# =============================================================================

# Global worker state (module-level for multiprocessing)
_worker_data: Optional[WorkerSharedData] = None


def worker_init(metadata: dict) -> None:
    """
    Initialize worker process with shared memory access.

    This function is called once per worker process by Pool's initializer.

    Parameters
    ----------
    metadata : dict
        Metadata from SharedDataManager.get_metadata()
    """
    global _worker_data

    # Clean up any existing state
    if _worker_data is not None:
        _worker_data.close()
        _worker_data = None

    shm_handles: List[shared_memory.SharedMemory] = []

    try:
        # Reconstruct obs_df from shared memory
        shm_obs = shared_memory.SharedMemory(name=metadata['obs_shm_name'])
        shm_handles.append(shm_obs)

        obs_df = pickle.loads(shm_obs.buf[:metadata['obs_size']])

        # Reconstruct distance or compact neighbor arrays from shared memory.
        distance_dict = {}

        for cell_type, shm_name in metadata['distance_shms'].items():
            shm = shared_memory.SharedMemory(name=shm_name)
            shm_handles.append(shm)

            shape = metadata['distance_shapes'][cell_type]
            dtype = np.dtype(metadata['distance_dtypes'].get(cell_type, 'float64'))

            # Create a VIEW into shared memory (no copy)
            array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            cell_ids = metadata['distance_cell_ids'][cell_type]
            data_kind = metadata.get('distance_kinds', {}).get(
                cell_type, 'distance_matrix'
            )
            if data_kind == 'neighbor_indices':
                distance_dict[cell_type] = CompactNeighborData(array, cell_ids)
            else:
                distance_dict[cell_type] = (array, cell_ids)

        _worker_data = WorkerSharedData(
            obs_df=obs_df,
            distance_dict=distance_dict,
            shm_handles=shm_handles
        )

        logger.debug(f"Worker {os.getpid()} initialized with {len(distance_dict)} distance matrices")

    except Exception as e:
        for shm in shm_handles:
            try:
                shm.close()
            except Exception:
                pass
        raise RuntimeError(f"Failed to initialize worker: {e}") from e


def worker_cleanup() -> None:
    """Clean up worker process shared memory handles."""
    global _worker_data
    if _worker_data is not None:
        _worker_data.close()
        _worker_data = None


def resolve_sampling_class(
    obs_df: pd.DataFrame,
    requested_cell_type: str,
) -> Tuple[str, str]:
    """Resolve a requested label using parent type before subtype metadata."""
    for class_by in ("cell_type", "cell_subtype"):
        if class_by in obs_df.columns and requested_cell_type in set(obs_df[class_by].astype(str)):
            return requested_cell_type, class_by

    available = {
        column: sorted(obs_df[column].dropna().astype(str).unique().tolist())
        for column in ("cell_type", "cell_subtype")
        if column in obs_df.columns
    }
    raise ValueError(
        f"Unknown cell_type '{requested_cell_type}'. "
        f"Not found in cell_type or cell_subtype columns. Available: {available}"
    )


def worker_sample(args: tuple) -> Union[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
    """
    Worker function that uses shared memory to sample cells.

    Parameters
    ----------
    args : tuple
        Task arguments as a tuple (for use with pool.map).

    Returns
    -------
    Union[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]
        Sample IDs with neighbors.

    Raises
    ------
    RuntimeError
        If worker was not initialized properly.
    ValueError
        If cell_type is not found in distance_dict.
    """
    global _worker_data

    if _worker_data is None:
        raise RuntimeError(
            "Worker not initialized. Ensure worker_init() is called via Pool initializer."
        )

    task = SampleTaskArgs.from_tuple(args)

    resolved_cell_type, resolved_class_by = resolve_sampling_class(
        _worker_data.obs_df,
        task.cell_type,
    )
    if resolved_cell_type not in _worker_data.distance_dict:
        available = list(_worker_data.distance_dict.keys())
        raise ValueError(
            f"Resolved cell_type '{resolved_cell_type}' from column "
            f"'{resolved_class_by}', but no distance matrix is available. "
            f"Available: {available}"
        )

    try:
        return get_sample_id_with_neighbors(
            obs_df=_worker_data.obs_df,
            n_samples=task.n_samples,
            cell_type=resolved_cell_type,
            n_base=task.n_base,
            distance_dict=_worker_data.distance_dict,
            class_by=resolved_class_by,
            sep_by_patient=task.sep_by_patient,
            sc_dataset=task.sc_dataset,
            minimum_n_base=task.minimum_n_base,
            glioma_epi_ratio=task.glioma_epi_ratio,
            neighbor_sampling_scheme=task.neighbor_sampling_scheme,
        )
    except Exception as e:
        logger.error(f"Error processing cell_type={task.cell_type}: {e}")
        raise


# =============================================================================
# High-Level Parallel Sampling Function
# =============================================================================

def parallel_sample_with_nearest_neighbors(
    obs_df: pd.DataFrame,
    distance_dict: Dict[str, NeighborData],
    sampling_tasks: List[dict],
    n_processes: int = 4,
    show_progress: bool = True
) -> Dict[int, Any]:
    """
    Run neighbor sampling in parallel using shared-memory arrays.

    This function manages the entire lifecycle of shared memory and worker
    processes, ensuring proper cleanup even if errors occur.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Observation DataFrame containing cell metadata.
    distance_dict : dict
        Pre-computed distance matrices or compact neighbor indices by cell type.
    sampling_tasks : list of dict
        Each dict contains parameters for one sampling task.
        Required keys: 'cell_type', 'n_samples', 'n_base'
        Optional keys: 'class_by', 'sep_by_patient', 'sc_dataset',
                      'minimum_n_base', 'glioma_epi_ratio'
    n_processes : int
        Number of parallel processes.
    show_progress : bool
        Whether to show progress bar.

    Returns
    -------
    dict
        Dictionary mapping task index to sampling results.

    Example
    -------
    >>> tasks = [
    ...     {'cell_type': 'T_cell', 'n_samples': 100, 'n_base': 50},
    ...     {'cell_type': 'B_cell', 'n_samples': 100, 'n_base': 50},
    ... ]
    >>> results = parallel_sample_with_nearest_neighbors(obs_df, dist_dict, tasks)
    """
    if not sampling_tasks:
        return {}

    # Convert tasks to tuples
    args_list = [SampleTaskArgs.from_dict(task).to_tuple() for task in sampling_tasks]

    # For very small task counts, run sequentially
    if len(args_list) < n_processes:
        n_processes = max(1, len(args_list))

    logger.info(f"Running {len(sampling_tasks)} tasks on {n_processes} processes...")

    # Use context manager for automatic cleanup
    with SharedDataManager(obs_df, distance_dict) as manager:
        metadata = manager.get_metadata()

        with Pool(processes=n_processes, initializer=worker_init, initargs=(metadata,)) as pool:
            if show_progress:
                from tqdm import tqdm
                results = list(tqdm(
                    pool.imap(worker_sample, args_list),
                    total=len(args_list),
                    desc="Sampling cells"
                ))
            else:
                results = pool.map(worker_sample, args_list)

    return {i: result for i, result in enumerate(results)}


def get_sample_id_with_neighbors(
        obs_df: pd.DataFrame,
        n_samples: int,
        cell_type: str,
        n_base: int,
        distance_dict: Dict[str, NeighborData],
        class_by: str = 'cell_type',
        sep_by_patient: bool = False,
        sc_dataset: str = 'merged_sc_dataset',
        minimum_n_base: int = 1,
        glioma_epi_ratio: float = 0.1,
        neighbor_sampling_scheme: str = 'single_anchor',
) -> Union[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
    """
    Select sample IDs using pre-computed per-class neighbor data.

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
        Number of single cells to average per simulated sample. Must be >= 1.
    distance_dict : dict
        Dictionary from `compute_distance_matrices_by_class` mapping each class to
        (distance_matrix, cell_ids).
    class_by : str, default='cell_type'
        Column name in `obs_df` that defines the class for sampling.
    sep_by_patient : bool, default=False
        If True, sample cells for each simulated sample from a single patient.
    sc_dataset : str, Name of the single-cell dataset, which will be used to simulate sctGEPs.
    minimum_n_base : int, default=1
        Minimum number of cells required to form a simulated sample.
    glioma_epi_ratio : float, default=0.1
        Probability of sampling glioma cells (vs epithelial) when splitting cancer cells.

    Returns
    -------
    ids : tuple of str or tuple of tuple of str
        If `n_samples == 1`, returns a single tuple of cell IDs.
        If `n_samples > 1`, returns a tuple of tuples.

    Raises
    ------
    ValueError
        If the requested cell type is not found in `distance_dict`, or if constraints
        cannot be satisfied.
    """
    if n_base == 0:
        return tuple()

    cell_type = _resolve_neighbor_sampling_cell_type(
        obs_df=obs_df,
        cell_type=cell_type,
        class_by=class_by,
        sc_dataset=sc_dataset,
        glioma_epi_ratio=glioma_epi_ratio,
    )
    if isinstance(distance_dict.get(cell_type), CompactNeighborData):
        if sep_by_patient or neighbor_sampling_scheme != 'single_anchor':
            raise ValueError(
                'Compact neighbor indices support only single_anchor sampling '
                'with sep_by_patient=False.'
            )

    if neighbor_sampling_scheme == "two_half_anchor":
        sampler = lambda: _sample_ids_two_half_anchor(
            obs_df=obs_df,
            cell_type=cell_type,
            n_base=n_base,
            distance_dict=distance_dict,
            class_by=class_by,
            sep_by_patient=sep_by_patient,
            minimum_n_base=minimum_n_base,
        )
    else:
        sampler = lambda: _sample_single_neighbor_group(
            obs_df=obs_df,
            cell_type=cell_type,
            n_base=n_base,
            distance_dict=distance_dict,
            class_by=class_by,
            sep_by_patient=sep_by_patient,
            minimum_n_base=minimum_n_base,
        )[0]

    selected_samples: Dict[Tuple[str, ...], int] = {}
    max_iterations = n_samples * 100
    iterations = 0

    while len(selected_samples) < n_samples:
        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError(
                f"Could not generate {n_samples} unique samples after {max_iterations} iterations. "
                "Consider relaxing constraints or increasing the cell pool."
            )
        sample_ids = sampler()
        if sample_ids not in selected_samples:
            selected_samples[sample_ids] = 1

    if n_samples == 1:
        return tuple(selected_samples.keys())[0]
    return tuple(selected_samples.keys())


def _resolve_neighbor_sampling_cell_type(
        obs_df: pd.DataFrame,
        cell_type: str,
        class_by: str,
        sc_dataset: str,
        glioma_epi_ratio: float,
) -> str:
    if sc_dataset == 'sct_dataset' and cell_type == 'Cancer Cells':
        available_types = obs_df[class_by].unique()
        if 'Epithelial Cells' in available_types and 'Glioma Cells' in available_types:
            return 'Glioma Cells' if np.random.random() <= glioma_epi_ratio else 'Epithelial Cells'
    return cell_type


def _sample_single_neighbor_group(
        obs_df: pd.DataFrame,
        cell_type: str,
        n_base: int,
        distance_dict: Dict[str, NeighborData],
        class_by: str = 'cell_type',
        sep_by_patient: bool = False,
        minimum_n_base: int = 1,
        required_dataset_id: Optional[str] = None,
        excluded_dataset_id: Optional[str] = None,
) -> Tuple[Tuple[str, ...], Optional[str]]:
    if cell_type not in distance_dict:
        raise ValueError(
            f"Cell type '{cell_type}' not found in distance_dict. "
            f"Available types: {list(distance_dict.keys())}"
        )

    neighbor_data = distance_dict[cell_type]
    if isinstance(neighbor_data, CompactNeighborData):
        all_samples_id = neighbor_data.cell_ids
        distance_matrix = None
    else:
        distance_matrix, all_samples_id = neighbor_data
    cell_id_to_matrix_idx = {cell_id: idx for idx, cell_id in enumerate(all_samples_id)}
    if len(all_samples_id) < minimum_n_base:
        raise ValueError(
            f"The number of cells in '{cell_type}' ({len(all_samples_id)}) is less than "
            f"minimum_n_base ({minimum_n_base})."
        )

    obs_df_filtered = obs_df.loc[obs_df[class_by] == cell_type].copy()
    if obs_df_filtered.empty:
        raise ValueError(f"No cells found for cell type '{cell_type}'.")

    selected_dataset_id = None
    if 'dataset_id' in obs_df_filtered.columns:
        dataset_counts = obs_df_filtered.groupby('dataset_id', observed=True)[class_by].count()
        candidate_datasets = dataset_counts.index.to_list()
        if required_dataset_id is not None:
            if required_dataset_id not in dataset_counts.index:
                raise ValueError(
                    f"Requested dataset_id {required_dataset_id!r} not available for cell type {cell_type!r}."
                )
            selected_dataset_id = required_dataset_id
        elif excluded_dataset_id is not None:
            preferred = [ds_id for ds_id in candidate_datasets if ds_id != excluded_dataset_id]
            feasible = [ds_id for ds_id in preferred if dataset_counts.loc[ds_id] >= minimum_n_base]
            if feasible:
                selected_dataset_id = str(np.random.choice(feasible))
        if selected_dataset_id is not None:
            obs_df_filtered = obs_df_filtered.loc[obs_df_filtered['dataset_id'] == selected_dataset_id].copy()

    groupby_patient_count = None
    if sep_by_patient:
        if 'sample_id' not in obs_df_filtered.columns:
            raise KeyError('Column "sample_id" required when sep_by_patient=True.')
        groupby_patient_count = obs_df_filtered.groupby('sample_id')[class_by].count()
        groupby_patient_count = groupby_patient_count[groupby_patient_count > n_base + 10]
        if groupby_patient_count.empty:
            raise ValueError(
                f"No patient has enough cells (> {n_base + 10}) in '{cell_type}'."
            )

    if groupby_patient_count is not None and len(groupby_patient_count) > 0:
        selected_patient_id = np.random.choice(groupby_patient_count.index)
        patient_mask = obs_df_filtered['sample_id'] == selected_patient_id
        id_pool = obs_df_filtered.loc[patient_mask].index.to_list()
    else:
        id_pool = obs_df_filtered.index.to_list()

    indices_pool = [cell_id_to_matrix_idx[cid] for cid in id_pool if cid in cell_id_to_matrix_idx]
    id_pool = [cid for cid in id_pool if cid in cell_id_to_matrix_idx]

    if len(id_pool) < minimum_n_base:
        raise ValueError(
            f"The restricted pool for '{cell_type}' has only {len(id_pool)} cells, which is less than "
            f"minimum_n_base ({minimum_n_base})."
        )

    if selected_dataset_id is None and 'dataset_id' in obs_df_filtered.columns and id_pool:
        selected_dataset_id = str(obs_df_filtered.loc[id_pool[0], 'dataset_id'])

    if minimum_n_base <= len(id_pool) <= n_base:
        return tuple(sorted(id_pool)), selected_dataset_id

    anchor_local_idx = np.random.randint(0, len(id_pool))
    anchor_id = id_pool[anchor_local_idx]
    anchor_matrix_idx = indices_pool[anchor_local_idx]
    if 'dataset_id' in obs_df_filtered.columns:
        selected_dataset_id = str(obs_df_filtered.loc[anchor_id, 'dataset_id'])

    if n_base == 1:
        return (anchor_id,), selected_dataset_id

    if isinstance(neighbor_data, CompactNeighborData):
        if id_pool != all_samples_id:
            raise ValueError(
                'Compact neighbor indices require the complete, ordered '
                'candidate pool without dataset or patient restrictions.'
            )
        if n_base > neighbor_data.neighbor_indices.shape[1]:
            raise ValueError(
                f'Compact neighbor data retains {neighbor_data.neighbor_indices.shape[1]} '
                f'neighbors, but this sample requests {n_base}.'
            )
        nearest_local_indices = neighbor_data.neighbor_indices[
            anchor_matrix_idx, :n_base
        ]
    else:
        distances = distance_matrix[anchor_matrix_idx, indices_pool]
        nearest_local_indices = np.argpartition(distances, n_base - 1)[:n_base]
        nearest_local_indices = nearest_local_indices[
            np.argsort(distances[nearest_local_indices])
        ]
    neighbor_ids = [id_pool[i] for i in nearest_local_indices]
    return tuple(sorted(neighbor_ids)), selected_dataset_id


def _sample_ids_two_half_anchor(
        obs_df: pd.DataFrame,
        cell_type: str,
        n_base: int,
        distance_dict: Dict[str, NeighborData],
        class_by: str = 'cell_type',
        sep_by_patient: bool = False,
        minimum_n_base: int = 1,
) -> Tuple[str, ...]:
    half_1_size = int(np.ceil(n_base / 2.0))
    half_2_size = int(n_base - half_1_size)

    first_ids, first_dataset_id = _sample_single_neighbor_group(
        obs_df=obs_df,
        cell_type=cell_type,
        n_base=half_1_size,
        distance_dict=distance_dict,
        class_by=class_by,
        sep_by_patient=sep_by_patient,
        minimum_n_base=minimum_n_base,
    )

    if half_2_size == 0:
        return tuple(sorted(first_ids))

    second_ids: Tuple[str, ...]
    used_different_dataset = False
    try:
        second_ids, second_dataset_id = _sample_single_neighbor_group(
            obs_df=obs_df,
            cell_type=cell_type,
            n_base=half_2_size,
            distance_dict=distance_dict,
            class_by=class_by,
            sep_by_patient=sep_by_patient,
            minimum_n_base=minimum_n_base,
            excluded_dataset_id=first_dataset_id,
        )
        used_different_dataset = (
            first_dataset_id is not None and second_dataset_id is not None and first_dataset_id != second_dataset_id
        )
    except ValueError:
        second_ids, _ = _sample_single_neighbor_group(
            obs_df=obs_df,
            cell_type=cell_type,
            n_base=half_2_size,
            distance_dict=distance_dict,
            class_by=class_by,
            sep_by_patient=sep_by_patient,
            minimum_n_base=minimum_n_base,
        )

    if used_different_dataset:
        logger.info("two_half_anchor used different datasets for cell type %s", cell_type)
    elif first_dataset_id is not None:
        logger.info("two_half_anchor fell back to same-dataset half groups for cell type %s", cell_type)

    return tuple(sorted(first_ids + second_ids))
