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
import multiprocessing
from multiprocessing import shared_memory, Pool
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Start method already set
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
        )


@dataclass
class WorkerSharedData:
    """Container for shared data in worker processes."""
    obs_df: pd.DataFrame
    distance_dict: Dict[str, Tuple[np.ndarray, List[str]]]
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

    def to_dict(self) -> dict:
        """Convert to dictionary for pickling."""
        return {
            'obs_shm_name': self.obs_shm_name,
            'obs_size': self.obs_size,
            'distance_shms': self.distance_shms.copy(),
            'distance_shapes': self.distance_shapes.copy(),
            'distance_dtypes': self.distance_dtypes.copy(),
            'distance_cell_ids': self.distance_cell_ids.copy(),
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
        distance_dict: Dict[str, Tuple[np.ndarray, List[str]]],
        dtype: np.dtype = np.float64
    ):
        """
        Initialize shared memory for obs_df and distance matrices.

        Parameters
        ----------
        obs_df : pd.DataFrame
            The observation DataFrame to share.
        distance_dict : dict
            Dictionary mapping cell_type -> (distance_matrix, cell_ids).
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
            logger.debug(f"SharedDataManager initialized with {len(distance_dict)} distance matrices")
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
        distance_dict: Dict[str, Tuple[np.ndarray, List[str]]],
        dtype: np.dtype
    ) -> None:
        """Initialize shared memory for distance matrices."""
        self._distance_shms: Dict[str, str] = {}
        self._distance_shapes: Dict[str, Tuple[int, ...]] = {}
        self._distance_dtypes: Dict[str, str] = {}
        self._distance_cell_ids: Dict[str, List[str]] = {}

        for cell_type, (dist_matrix, cell_ids) in distance_dict.items():
            dist_matrix = np.asarray(dist_matrix, dtype=dtype)

            shm = shared_memory.SharedMemory(create=True, size=dist_matrix.nbytes)
            self._shm_objects.append(shm)

            # Create view and copy data
            shm_array = np.ndarray(dist_matrix.shape, dtype=dtype, buffer=shm.buf)
            np.copyto(shm_array, dist_matrix)

            self._distance_shms[cell_type] = shm.name
            self._distance_shapes[cell_type] = dist_matrix.shape
            self._distance_dtypes[cell_type] = np.dtype(dtype).name
            self._distance_cell_ids[cell_type] = cell_ids

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

        obs_bytes = bytes(shm_obs.buf[:metadata['obs_size']])
        obs_df = pickle.loads(obs_bytes)

        # Reconstruct distance matrices from shared memory
        distance_dict: Dict[str, Tuple[np.ndarray, List[str]]] = {}

        for cell_type, shm_name in metadata['distance_shms'].items():
            shm = shared_memory.SharedMemory(name=shm_name)
            shm_handles.append(shm)

            shape = metadata['distance_shapes'][cell_type]
            dtype = np.dtype(metadata['distance_dtypes'].get(cell_type, 'float64'))

            # Create a VIEW into shared memory (no copy)
            dist_matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            cell_ids = metadata['distance_cell_ids'][cell_type]
            distance_dict[cell_type] = (dist_matrix, cell_ids)

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


def worker_sample(args: tuple) -> Tuple[str, ...]:
    """
    Worker function that uses shared memory to sample cells.

    Parameters
    ----------
    args : tuple
        Task arguments as a tuple (for use with pool.map).

    Returns
    -------
    tuple
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

    if task.cell_type not in _worker_data.distance_dict:
        available = list(_worker_data.distance_dict.keys())
        raise ValueError(f"Unknown cell_type '{task.cell_type}'. Available: {available}")

    try:
        return get_sample_id_with_neighbors(
            obs_df=_worker_data.obs_df,
            n_samples=task.n_samples,
            cell_type=task.cell_type,
            n_base=task.n_base,
            distance_dict=_worker_data.distance_dict,
            class_by=task.class_by,
            sep_by_patient=task.sep_by_patient,
            sc_dataset=task.sc_dataset,
            minimum_n_base=task.minimum_n_base,
            glioma_epi_ratio=task.glioma_epi_ratio,
        )
    except Exception as e:
        logger.error(f"Error processing cell_type={task.cell_type}: {e}")
        raise


# =============================================================================
# High-Level Parallel Sampling Function
# =============================================================================

def parallel_sample_with_nearest_neighbors(
    obs_df: pd.DataFrame,
    distance_dict: Dict[str, Tuple[np.ndarray, List[str]]],
    sampling_tasks: List[dict],
    n_processes: int = 4,
    show_progress: bool = True
) -> Dict[int, Any]:
    """
    Run sampling by finding top (n-1) most similar cells in parallel using shared memory.

    This function manages the entire lifecycle of shared memory and worker
    processes, ensuring proper cleanup even if errors occur.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Observation DataFrame containing cell metadata.
    distance_dict : dict
        Pre-computed distance matrices by cell type.
        Format: {cell_type: (distance_matrix, cell_ids)}
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
        distance_dict: Dict[str, Tuple[np.ndarray, List[str]]],
        class_by: str = 'cell_type',
        sep_by_patient: bool = False,
        sc_dataset: str = 'merged_sc_dataset',
        minimum_n_base: int = 1,
        glioma_epi_ratio: float = 0.1,
) -> Union[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
    """
    Select sample IDs by nearest-neighbor sampling using pre-computed per-class distance matrices.

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

    # Optional splitting of cancer cells into glioma vs epithelial
    if sc_dataset == 'sct_dataset' and cell_type == 'Cancer Cells':
        available_types = obs_df[class_by].unique()
        if 'Epithelial Cells' in available_types and 'Glioma Cells' in available_types:
            cell_type = 'Glioma Cells' if np.random.random() <= glioma_epi_ratio else 'Epithelial Cells'

    # Get pre-computed distance matrix for this cell type
    if cell_type not in distance_dict:
        raise ValueError(
            f"Cell type '{cell_type}' not found in distance_dict. "
            f"Available types: {list(distance_dict.keys())}"
        )

    distance_matrix, all_samples_id = distance_dict[cell_type]

    if len(all_samples_id) < minimum_n_base:
        raise ValueError(
            f"The number of cells in '{cell_type}' ({len(all_samples_id)}) is less than "
            f"minimum_n_base ({minimum_n_base})."
        )

    # Filter obs_df for this cell type
    obs_df_filtered = obs_df.loc[obs_df[class_by] == cell_type].copy()

    # Optional patient-wise sampling
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

    selected_samples: Dict[Tuple[str, ...], int] = {}
    max_iterations = n_samples * 100
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
            patient_mask = obs_df_filtered['sample_id'] == selected_patient_id
            patient_cell_ids = obs_df_filtered.loc[patient_mask].index.to_list()
            # Map to indices in the distance matrix
            patient_indices = [all_samples_id.index(cid) for cid in patient_cell_ids]
            id_pool = patient_cell_ids
            indices_pool = patient_indices
        else:
            id_pool = all_samples_id
            indices_pool = list(range(len(all_samples_id)))

        # Check if enough cells available
        if len(id_pool) < minimum_n_base:
            continue

        # Special case: use all available cells if between minimum_n_base and n_base
        if minimum_n_base <= len(id_pool) <= n_base:
            sample_ids = tuple(sorted(id_pool))
        else:
            # Step 1: Randomly select 1 anchor cell
            anchor_local_idx = np.random.randint(0, len(id_pool))
            anchor_matrix_idx = indices_pool[anchor_local_idx]
            anchor_id = id_pool[anchor_local_idx]

            if n_base == 1:
                sample_ids = (anchor_id,)
            else:
                # Step 2: Find (n_base - 1) nearest neighbors
                # Get distances from anchor to all cells in the pool
                distances = distance_matrix[anchor_matrix_idx, indices_pool]

                # Get top n_base nearest cells (including anchor itself)
                nearest_local_indices = np.argsort(distances)[:n_base]

                # Get cell IDs
                neighbor_ids = [id_pool[i] for i in nearest_local_indices]
                sample_ids = tuple(sorted(neighbor_ids))

        # Store unique samples
        if sample_ids not in selected_samples:
            selected_samples[sample_ids] = 1

    if n_samples == 1:
        return tuple(selected_samples.keys())[0]
    return tuple(selected_samples.keys())
