import os
import json
import warnings
from time import perf_counter
import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from joblib import dump, load
from dataclasses import dataclass, fields
from sklearn.decomposition import PCA
from sklearn.exceptions import InconsistentVersionWarning
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
logger = logging.getLogger(__name__)


from ..utility import get_gene_list_filtered_by_quantile_range, get_quantile, check_dir
from ..utility import ReadExp, ReadH5AD, ExpObj, QueryNeighbors, read_single_cell_type_dataset
from ..utility import get_available_threads, non_log2cpm, log2_transform, get_cell_num, create_h5ad_dataset, print_msg, safe_pca_transform
from .sampling import (
    estimate_dirichlet_alpha_from_pred_cell_props,
    flatten_cell_type2subtype,
    format_segment_fraction_caps,
    generate_cell_fraction,
    restrict_cell_type2subtype_to_output_types,
)
from .simu_helper import (sample_random, get_sample_id_random, sample_with_neighbors,
                          should_use_neighbor_sampling, append_to_csv, SimuMethod, DatasetType,
                          NeighborSamplingScheme)
from .nneighbors_shared_memory import NeighborData
from .filtering import (
    FixedGeneFilter, validate_gep_filtering_method, validate_metadata, write_metadata,
)
# from .nneighbors_shared_memory import parallel_sample_with_nearest_neighbors


def get_cell_type_used(cell_type2subtype: Dict[str, List[str]], output_types: list[str] = None) -> Tuple[List[str], List[str],]:
    """Extracts used cell types and subtypes from the mapping dictionary."""
    cell_type_used = list(cell_type2subtype.keys())
    sub_cell_type_used = [i for v in cell_type2subtype.values() for i in v if i not in cell_type_used]  # exclude cell types without subtypes (same name)
    if output_types is not None:
        cell_type_used = [ct for ct in cell_type_used if ct in output_types]
        sub_cell_type_used = [sub_ct for sub_ct in sub_cell_type_used if sub_ct in output_types]
        all_types = set(cell_type_used + sub_cell_type_used)
        cell_type_not_existed = [ct for ct in output_types if ct not in all_types]
        if len(cell_type_not_existed) > 0:
            raise ValueError(f'The specified output_types contain cell types/subtypes not in the provided cell_type2subtype mapping: {cell_type_not_existed}. '
                             f'Please check the config.yaml file and try again.')
    return cell_type_used, sub_cell_type_used


def _candidate_batch_size(n_samples: int, filtering: bool) -> int:
    """Choose candidate profiles per iteration, oversampling filtered targets."""
    if filtering:
        return min(5000, n_samples * 30)
    return min(1000, n_samples)


@dataclass
class GEPGenerationConfig:
    """Configuration for GEP generation."""
    n_samples: int
    sampling_method_prop: str = 'segment'
    total_cell_number: int = 100
    n_threads: int = 10
    filtering: bool = True
    reference_file: Optional[Union[str, pd.DataFrame, Path]] = None
    ref_exp_type: Optional[str] = None
    gep_filtering_quantile: Tuple[Optional[float], float] = (None, 0.95)
    log_file_path: Optional[str] = None
    n_top: int = 20
    simu_method: str = 'mul'
    filtering_method: str = 'median_gep'
    add_noise: bool = False
    noise_params: tuple = ()
    filtering_ref_types: Optional[List[str]] = None
    show_filtering_info: bool = False
    cell_prop_prior: Optional[Dict] = None
    high_corr_gene_list: Optional[List[str]] = None
    filtering_by_gene_range: bool = False
    min_percentage_within_gene_range: float = 0.95
    gene_quantile_range: Optional[List[float]] = None
    filtering_in_pca_space: bool = False
    pca_n_components: Union[int, float] = 0.9
    norm_ord: int = 1
    dirichlet_alpha: Optional[Dict[str, float]] = None
    pred_cell_prop_file_path: Optional[Union[str, Path]] = None
    dirichlet_sampling_seed: Optional[int] = None
    segment_fraction_max_caps: Optional[Dict[str, float]] = None
    segment_fraction_cap_metadata: Optional[Dict[str, Any]] = None
    filtering_space: Optional[FixedGeneFilter] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.simu_method not in ('ave', 'mul'):
            raise ValueError(f"Invalid simu_method: {self.simu_method}")
        validate_gep_filtering_method(self.filtering_method)
        if self.dirichlet_sampling_seed is not None and (
            isinstance(self.dirichlet_sampling_seed, bool)
            or not isinstance(self.dirichlet_sampling_seed, (int, np.integer))
            or self.dirichlet_sampling_seed < 0
        ):
            raise ValueError(
                '"dirichlet_sampling_seed" must be a non-negative integer or None.'
            )
        if self.pred_cell_prop_file_path not in (None, ""):
            self.pred_cell_prop_file_path = Path(self.pred_cell_prop_file_path)
            if not self.pred_cell_prop_file_path.exists():
                raise FileNotFoundError(
                    f'"pred_cell_prop_file_path" does not exist: {self.pred_cell_prop_file_path}'
                )
            if self.sampling_method_prop == "segment":
                if not self.filtering:
                    raise ValueError(
                        "Segment caps from pred_cell_prop_file_path require GEP filtering."
                    )
                if not self.segment_fraction_max_caps or not self.segment_fraction_cap_metadata:
                    raise ValueError(
                        "Segment prediction-file caps were not resolved before GEP generation."
                    )
            elif self.sampling_method_prop != "dirichlet":
                raise ValueError(
                    "pred_cell_prop_file_path supports Dirichlet alpha estimation or "
                    "GEP-filtered segment maximum caps only."
                )
        else:
            self.pred_cell_prop_file_path = None
            if self.segment_fraction_max_caps or self.segment_fraction_cap_metadata:
                raise ValueError("Segment fraction caps require pred_cell_prop_file_path.")
        if self.segment_fraction_max_caps and self.sampling_method_prop != "segment":
            raise ValueError("Empirical maximum caps are supported only with segment sampling.")


class BulkGEPGenerator:
    """
Generate simulated bulk gene expression profiles (GEPs) by aggregating single-cell datasets.

    Parameters
    ----------
    simu_bulk_dir : str or pathlib.Path
        Directory where simulated bulk GEPs and related outputs will be saved.
    merged_sc_dataset_file_path : str or pathlib.Path
        Path to the merged single-cell dataset (.h5ad).
    sct_dataset_file_path : str or pathlib.Path, optional
        Path to the single-cell tumor (sct) dataset file, if available. If None,
        only merged_sc_dataset_file_path will be used.
    cell_type2subtype : dict[str, list[str]]
        Mapping from major cell types to their subtypes.
    sc_dataset_ids : list[str]
        List of dataset IDs (batch labels) to select from the merged single-cell file.
    bulk_dataset_name : str, default="simulated_bulk"
        Name used to label the simulated bulk dataset (e.g., in file names or metadata).
    check_basic_info : bool, default=True
        If True, run basic consistency checks on the input single-cell datasets
        (e.g., gene overlap, cell type annotations).
    zero_ratio_threshold : float, default=0.97
        Threshold on the fraction of zero entries used to filter genes from the
        single-cell GEPs before simulation. Genes with a higher zero ratio may be removed.
    sc_dataset_gep_type : str, default="log_space"
        Type/scale of the single-cell GEP data. Typically "log_space" for log-normalized
        expression or "count" for raw counts.
    tcga2cancer_type_file_path : str or pathlib.Path, optional
        Path to a file mapping TCGA project IDs to higher-level cancer types.
        Used when simulating TCGA-like bulk cohorts. If None, TCGA-specific
        annotations will be skipped.
    total_rna_coefficient : dict[str, float], optional
        Optional mapping from cell type (or subtype) to a scaling coefficient
        representing relative total RNA content. If provided, these coefficients
        are used to weight cells when aggregating single-cell GEPs into bulk GEPs.
    subtype_col_name : str, optional
        Name of the column in the single-cell AnnData object that stores subtype labels.
        If None, only the major cell type labels are used.
    cell_type_col_name : str, default="cell_type"
        Name of the column in the single-cell AnnData object that stores major
        cell type labels.
    output_types : list[str], optional
        Types and subtypes of cells to generate. This list should be a sublist of all cell types and subtypes.
        If None, all cell types and their subtypes will be included.
    """

    def __init__(
        self,
        simu_bulk_dir: str | Path,
        merged_sc_dataset_file_path: str | Path,
        sct_dataset_file_path: Optional[Union[str, Path]],
        cell_type2subtype: Dict[str, List[str]],
        sc_dataset_ids: List[str],
        cancer_cell_dataset_ids: Optional[List[str]] = None,
        non_cancer_cell_dataset_ids: Optional[List[str]] = None,
        stratify_by_dataset: str = "none",
        selected_subtypes_by_cell_type: Optional[Dict[str, List[str]]] = None,
        stratify_seed: int = 123,
        bulk_dataset_name: str = "simulated_bulk",
        check_basic_info: bool = True,
        zero_ratio_threshold: float = 0.97,
        sc_dataset_gep_type: str = 'log_space',
        tcga2cancer_type_file_path: str | Path = None,
        total_rna_coefficient: Optional[Dict[str, float]] = None,
        subtype_col_name: Optional[str] = 'cell_subtype',
        cell_type_col_name: str = 'cell_type',
        output_types: Optional[List[str]] = None,
        sampling_method_gep: str = 'random',
        mixed_sct_loading_mode: str = "auto",
        mixed_sct_auto_eager_max_gb: float = 4.0,
    ):
        allowed_stratify = {"none", "balanced_quota"}
        if stratify_by_dataset not in allowed_stratify:
            raise ValueError(
                f"Invalid stratify_by_dataset={stratify_by_dataset!r}; expected one of {sorted(allowed_stratify)}"
            )
        allowed_loading_modes = {"auto", "eager", "lazy"}
        if mixed_sct_loading_mode not in allowed_loading_modes:
            raise ValueError(
                f"Invalid mixed_sct_loading_mode={mixed_sct_loading_mode!r}; "
                f"expected one of {sorted(allowed_loading_modes)}"
            )
        if float(mixed_sct_auto_eager_max_gb) <= 0:
            raise ValueError("mixed_sct_auto_eager_max_gb must be positive")

        self.simu_bulk_dir = Path(simu_bulk_dir)
        self.simu_bulk_dir.mkdir(parents=True, exist_ok=True)
        # check_dir(simu_bulk_dir)

        self.merged_sc_fp = merged_sc_dataset_file_path
        self.sct_dataset_file_path = sct_dataset_file_path
        self.cell_type2subtype = cell_type2subtype
        self.sc_dataset_used = sc_dataset_ids
        self.cancer_cell_dataset_ids = cancer_cell_dataset_ids
        self.non_cancer_cell_dataset_ids = non_cancer_cell_dataset_ids
        self.stratify_by_dataset = stratify_by_dataset
        self.subtype_col_name = subtype_col_name
        self.cell_type_col_name = cell_type_col_name
        self.selected_subtypes_by_cell_type = self._normalize_selected_subtypes_by_cell_type(
            selected_subtypes_by_cell_type
        )
        self.stratify_seed = int(stratify_seed)
        self.bulk_dataset_name = bulk_dataset_name
        self.zero_ratio_threshold = zero_ratio_threshold
        self.sc_dataset_gep_type = sc_dataset_gep_type
        self.tcga2cancer_type_file_path = tcga2cancer_type_file_path
        self.total_rna_coefficient = total_rna_coefficient
        self.mixed_sct_loading_mode = mixed_sct_loading_mode
        self.mixed_sct_auto_eager_max_gb = float(mixed_sct_auto_eager_max_gb)

        # Initialize state variables
        self.cell_type_in_sc = []
        self.cell_subtype_in_sc = []
        self.dataset_in_sc = []
        self.merged_sc_dataset = None
        self.merged_sc_dataset_obs = None
        self.merged_sc_dataset_df = None
        self.sct_dataset_obs = None
        self.sct_dataset_df = None
        self.sct_dataset_reader = None
        self.generated_sc_dataset_df = None
        self.generated_sc_fp = None

        # Filtering and Reference stats
        self.m_gep_ref = None
        self.marker_ratio_ref = None
        self.q_dis_nn_ref_upper = 0.0
        self.q_dis_nn_ref_lower = 0.0
        self.filtering_quantile_upper = 0.0
        self.filtering_quantile_lower = None
        self.ref_neighbor_counter = {}
        self.n_neighbors_each_ref = 1

        # Counters
        self.generated_bulk_gep_counter = 0
        self.n_round = 0
        self.n_samples = 0
        self.total_cell_number = 0
        self.gep_filtering_input_n_samples = None
        self.gep_filtering_kept_n_samples = None
        self.gep_filtering_removed_n_samples = None
        self.gep_filtering_pca_target = None
        self.gep_filtering_n_pcs = None

        # Process cell types
        self.cell_type_used, self.cell_subtype_used = get_cell_type_used(cell_type2subtype, output_types=output_types)
        self.cell_fraction_type2subtype = restrict_cell_type2subtype_to_output_types(
            cell_type2subtype,
            output_types,
        )

        # Define file paths
        prefix = f'simu_gep_{self.bulk_dataset_name}_gepsamp-{sampling_method_gep}'
        self.generated_cell_fraction_fp = (
            self.simu_bulk_dir / f'cell_fractions_{self.bulk_dataset_name}_gepsamp-{sampling_method_gep}.csv'
        )
        self.ref_neighbor_counter_fp = (
            self.simu_bulk_dir / f'ref2n_neighbors_{self.bulk_dataset_name}_gepsamp-{sampling_method_gep}.csv'
        )
        self.generated_bulk_gep_csv_fp = self.simu_bulk_dir / f'{prefix}_log2cpm1p.csv'
        self.sampled_sc_cell_id_file_path = self.simu_bulk_dir / f'{prefix}_sampled_sc_cell_id.csv'
        self.generated_bulk_gep_fp = self.generated_bulk_gep_csv_fp.parent / self.generated_bulk_gep_csv_fp.name.replace('.csv', '.h5ad')
        self._reset_runtime_log_state()

        if check_basic_info and not self.generated_bulk_gep_fp.exists():  # Put this checker the last one
            self._check_basic_info()

    def _reset_runtime_log_state(self) -> None:
        self._logged_total_rna_coefficients = set()
        self._logged_pca_model_loads = set()
        self._logged_pca_variance = set()

    @staticmethod
    def _estimate_sct_matrix_bytes(reader: ReadH5AD) -> int:
        h5ad = reader.get_h5ad()
        return int(h5ad.n_obs) * int(h5ad.n_vars) * np.dtype(np.float32).itemsize

    def _resolve_sct_loading_mode(self, reader: ReadH5AD) -> str:
        if self.mixed_sct_loading_mode != "auto":
            return self.mixed_sct_loading_mode
        estimated_bytes = self._estimate_sct_matrix_bytes(reader)
        max_bytes = int(self.mixed_sct_auto_eager_max_gb * (1024 ** 3))
        return "eager" if estimated_bytes <= max_bytes else "lazy"

    @staticmethod
    def _close_backed_reader(reader: ReadH5AD) -> None:
        h5ad = reader.get_h5ad()
        file_manager = getattr(h5ad, "file", None)
        if file_manager is not None:
            file_manager.close()

    def _reset_gep_filtering_stats(self, enabled: bool) -> None:
        self.gep_filtering_pca_target = None
        self.gep_filtering_n_pcs = None
        if enabled:
            self.gep_filtering_input_n_samples = 0
            self.gep_filtering_kept_n_samples = 0
            self.gep_filtering_removed_n_samples = 0
        else:
            self.gep_filtering_input_n_samples = None
            self.gep_filtering_kept_n_samples = None
            self.gep_filtering_removed_n_samples = None

    def _record_gep_filtering_counts(self, input_n_samples: int, kept_n_samples: int) -> None:
        if self.gep_filtering_input_n_samples is None:
            return
        removed_n_samples = max(0, int(input_n_samples) - int(kept_n_samples))
        self.gep_filtering_input_n_samples += int(input_n_samples)
        self.gep_filtering_kept_n_samples += int(kept_n_samples)
        self.gep_filtering_removed_n_samples += removed_n_samples

    @staticmethod
    def _format_total_rna_coefficient_message(ct2rna_coefficient: Dict[str, float]) -> str:
        if not ct2rna_coefficient:
            return 'Total RNA coefficients: none applied.'

        n_cell_types = len(ct2rna_coefficient)
        non_default = {
            cell_type: float(value)
            for cell_type, value in sorted(ct2rna_coefficient.items())
            if not np.isclose(float(value), 1.0)
        }
        if not non_default:
            return f'Total RNA coefficients: all {n_cell_types} cell types use 1.0.'

        coeff_summary = ', '.join(f'{cell_type}={value:g}' for cell_type, value in non_default.items())
        return f'Total RNA coefficients: {len(non_default)}/{n_cell_types} non-default ({coeff_summary}).'

    def _maybe_log_total_rna_coefficients(self, ct2rna_coefficient: Dict[str, float]) -> None:
        if not hasattr(self, '_logged_total_rna_coefficients'):
            self._reset_runtime_log_state()
        signature = tuple((cell_type, float(value)) for cell_type, value in sorted(ct2rna_coefficient.items()))
        if signature in self._logged_total_rna_coefficients:
            return
        self._logged_total_rna_coefficients.add(signature)
        tqdm.write(f'   > {self._format_total_rna_coefficient_message(ct2rna_coefficient)}')

    def _maybe_log_pca_model_loaded(
        self,
        pca_model_path: Path,
        gene_list_in_pca_file_path: Path,
        n_genes: int,
    ) -> None:
        if not hasattr(self, '_logged_pca_model_loads'):
            self._reset_runtime_log_state()
        signature = (str(pca_model_path), str(gene_list_in_pca_file_path), int(n_genes))
        if signature in self._logged_pca_model_loads:
            return
        self._logged_pca_model_loads.add(signature)
        logger.info('PCA model loaded: %s (%d genes).', pca_model_path, n_genes)

    def _maybe_log_pca_variance_summary(self, pca_model_path: Path, n_components: int, explained_variance_pct: float) -> None:
        if not hasattr(self, '_logged_pca_variance'):
            self._reset_runtime_log_state()
        signature = (str(pca_model_path), int(n_components), round(float(explained_variance_pct), 2))
        if signature in self._logged_pca_variance:
            return
        self._logged_pca_variance.add(signature)
        logger.info('PCA reference space: %d PCs explain %.2f%% variance.', n_components, explained_variance_pct)

    def generate_gep(self, config: GEPGenerationConfig) -> None:
        """
        Main method to generate simulated bulk GEPs.
        """
        self._reset_runtime_log_state()
        self._reset_gep_filtering_stats(config.filtering)
        self.n_samples = config.n_samples
        self.total_cell_number = config.total_cell_number
        self.filtering_quantile_lower, self.filtering_quantile_upper = config.gep_filtering_quantile

        # Adjust threads
        n_total_cpus = multiprocessing.cpu_count()
        n_threads = min(n_total_cpus - 1, config.n_threads)

        if 'linear_mmd' in config.filtering_method:
            norm_ord = 2

        self.filtering_space = config.filtering_space
        if self.filtering_space is not None:
            self._prepare_filtering_artifacts(config)

        if self.generated_bulk_gep_fp.exists():
            logger.info(f'Previous result existed: {self.generated_bulk_gep_fp}')
            logger.info(self.__str__())
            return

        # Resume check
        self._check_intermediate_generated_gep()

        # Load Data
        obs_df, gene_list_in_sc_ds, sc_dataset_key = self._load_sc_data_for_generation(config.simu_method)
        candidate_batch_size = _candidate_batch_size(
            config.n_samples,
            config.filtering,
        )

        # Setup Filtering Reference
        sample_id_for_filtering = []
        if config.filtering and config.filtering_ref_types and self.filtering_space is None:
            s2c = pd.read_csv(self.tcga2cancer_type_file_path, index_col=0)
            sample_id_for_filtering = s2c.loc[s2c['cancer_type'].isin(config.filtering_ref_types)].index.to_list()
            if sample_id_for_filtering:
                logger.info(f'{len(sample_id_for_filtering)} samples used for filtering.')
            else:
                logger.warning(
                    'No TCGA reference samples matched filtering_ref_types=%s; using the full reference cohort.',
                    config.filtering_ref_types,
                )

        # Main Generation Loop
        dirichlet_alpha = config.dirichlet_alpha
        if config.sampling_method_prop == 'dirichlet' and config.pred_cell_prop_file_path is not None:
            cell_type_with_subtype = flatten_cell_type2subtype(self.cell_fraction_type2subtype)
            logger.info(
                'Estimating Dirichlet alpha from predicted cell proportions: %s',
                config.pred_cell_prop_file_path,
            )
            dirichlet_alpha = estimate_dirichlet_alpha_from_pred_cell_props(
                pred_cell_prop_file_path=config.pred_cell_prop_file_path,
                cell_types=cell_type_with_subtype,
            )
            logger.info('Estimated Dirichlet alpha for %d cell types.', len(dirichlet_alpha))
        if config.segment_fraction_max_caps:
            logger.info(
                "Segment empirical maximum caps from %s: %s",
                config.segment_fraction_cap_metadata["segment_fraction_cap_source"],
                format_segment_fraction_caps(config.segment_fraction_max_caps),
            )

        dirichlet_rng = None
        if (
            config.sampling_method_prop == 'dirichlet'
            and config.dirichlet_sampling_seed is not None
        ):
            dirichlet_rng = np.random.default_rng(config.dirichlet_sampling_seed)

        with tqdm(total=self.n_samples) as pbar:
            if self.generated_bulk_gep_counter != 0:
                pbar.update(self.generated_bulk_gep_counter)

            while self.generated_bulk_gep_counter < self.n_samples:
                # 1. Generate Fractions
                batch_started = perf_counter()
                generated_cell_frac = generate_cell_fraction(
                    sampling_method=config.sampling_method_prop,
                    n_cell_frac=candidate_batch_size,
                    sample_prefix=f's_{config.sampling_method_prop}_{self.n_round}',
                    cell_prop_prior=config.cell_prop_prior if config.filtering else None,
                    cell_type2subtype=self.cell_fraction_type2subtype,
                    dirichlet_alpha=dirichlet_alpha,
                    rng=dirichlet_rng,
                    max_segment_sampling_attempts=(
                        max(10000, candidate_batch_size * 500)
                        if config.segment_fraction_max_caps else None
                    ),
                )
                fractions_finished = perf_counter()
                fraction_stats = generated_cell_frac.attrs.get("segment_sampling_stats")
                if config.segment_fraction_max_caps and fraction_stats:
                    logger.info(
                        "Segment fraction cap proposals: requested=%d, attempted=%d, "
                        "rejected=%d, acceptance=%.3g%%.",
                        fraction_stats["requested"], fraction_stats["attempted"],
                        fraction_stats["rejected"],
                        100 * fraction_stats["accepted"] / fraction_stats["attempted"],
                    )

                # 2. Sample Cells
                selected_cell_ids = self._sc_sampling(
                    cell_frac=generated_cell_frac,
                    n_threads=n_threads,
                    obs_df=obs_df,
                    sc_dataset=sc_dataset_key
                )
                sampling_finished = perf_counter()

                # 3. Aggregate Expression
                simulated_gep = self._map_cell_id2exp(
                    selected_cell_id=selected_cell_ids,
                    simu_method=config.simu_method,
                    sc_dataset=sc_dataset_key,
                    cell_frac=generated_cell_frac,
                    add_noise=config.add_noise,
                    noise_params=config.noise_params
                )
                aggregation_finished = perf_counter()

                simulated_gep_bak = (
                    simulated_gep.copy() if self.filtering_space is None else None
                )

                # 4. Filtering (GEP-level filtering during generation)
                if self.filtering_space is not None:
                    pre_filter_n_samples = len(simulated_gep)
                    simulated_gep = self.filtering_space.filter(simulated_gep)
                    if config.filtering:
                        self._record_gep_filtering_counts(pre_filter_n_samples, len(simulated_gep))
                    if self.filtering_space.pca is not None:
                        self.gep_filtering_pca_target = self.filtering_space.pca.n_components
                        self.gep_filtering_n_pcs = int(self.filtering_space.pca.n_components_)
                elif config.filtering:
                    pre_filter_n_samples = int(simulated_gep.shape[0])
                    simulated_gep = self._filter_simulated_geps(
                        simulated_gep=simulated_gep,
                        simulated_gep_bak=simulated_gep_bak,
                        reference_file=config.reference_file,
                        ref_exp_type=config.ref_exp_type,
                        filtering_method=config.filtering_method,
                        sample_id_for_filtering=sample_id_for_filtering,
                        gene_list_in_sc_ds=gene_list_in_sc_ds,
                        high_corr_gene_list=config.high_corr_gene_list,
                        filtering_by_gene_range=config.filtering_by_gene_range,
                        min_percentage_within_gene_range=config.min_percentage_within_gene_range,
                        gene_quantile_range=config.gene_quantile_range,
                        filtering_in_pca_space=config.filtering_in_pca_space,
                        pca_n_components=config.pca_n_components,
                        norm_ord=config.norm_ord,
                        n_top=config.n_top,
                        show_filtering_info=config.show_filtering_info
                    )
                    kept_n_samples = 0 if simulated_gep is None else int(simulated_gep.shape[0])
                    self._record_gep_filtering_counts(
                        input_n_samples=pre_filter_n_samples,
                        kept_n_samples=kept_n_samples,
                    )

                # 5. Save Results
                filtering_finished = perf_counter()
                if simulated_gep is not None and not simulated_gep.empty:
                    # Ensure we don't exceed requested n_samples
                    remaining_needed = self.n_samples - self.generated_bulk_gep_counter
                    if simulated_gep.shape[0] > remaining_needed:
                        simulated_gep = simulated_gep.iloc[:remaining_needed]

                    # Convert back to log2(TPM+1) for saving
                    simulated_gep_log = (
                        np.log2(simulated_gep + 1) if self.filtering_space is not None
                        else log2_transform(simulated_gep)
                    )

                    count = simulated_gep.shape[0]
                    self.generated_bulk_gep_counter += count
                    pbar.update(count)

                    # Sync indices
                    valid_indices = simulated_gep.index
                    generated_cell_frac = generated_cell_frac.loc[valid_indices].copy()
                    selected_cell_ids = selected_cell_ids.loc[valid_indices].copy()

                    self._save_simulated_bulk_gep(gep=simulated_gep_log,
                                                  cell_id=selected_cell_ids,
                                                  cell_fraction=generated_cell_frac)

                self.n_round += 1
                if self.filtering_space is not None:
                    self._save_filtering_metadata()
                logger.info(
                    "Candidate batch timing: fractions=%.3fs, sampling=%.3fs, "
                    "aggregation=%.3fs, filtering=%.3fs, saving=%.3fs, total=%.3fs.",
                    fractions_finished - batch_started,
                    sampling_finished - fractions_finished,
                    aggregation_finished - sampling_finished,
                    filtering_finished - aggregation_finished,
                    perf_counter() - filtering_finished,
                    perf_counter() - batch_started,
                )

        # Finalize
        self._finalize_generation(config.sampling_method_prop, config.filtering_method, config.log_file_path)
        if self.filtering_space is not None:
            self._save_filtering_metadata()

    def _prepare_filtering_artifacts(self, config):
        space = self.filtering_space
        generation = {
            item.name: getattr(config, item.name)
            for item in fields(config)
            if item.name not in ("filtering_space", "log_file_path", "n_threads", "show_filtering_info")
        }
        sct_file = Path(self.sct_dataset_file_path)
        generation.update(
            sct_file=str(sct_file.resolve()),
            sct_size=sct_file.stat().st_size,
            sct_mtime_ns=sct_file.stat().st_mtime_ns,
            cell_type2subtype=self.cell_fraction_type2subtype,
            total_rna_coefficient=self.total_rna_coefficient,
        )
        space.identity["generation"] = json.loads(json.dumps(generation, default=str))
        self.filtering_metadata_path = space.result_dir / "filtering_metadata.json"
        artifacts = [
            self.generated_bulk_gep_fp, self.generated_bulk_gep_csv_fp,
            self.generated_cell_fraction_fp, self.sampled_sc_cell_id_file_path,
            self.ref_neighbor_counter_fp, *space.additional_artifacts,
        ]
        result_artifacts = list(space.result_dir.iterdir()) if space.result_dir.exists() else []
        if any(path.exists() for path in artifacts) or result_artifacts:
            metadata = validate_metadata(self.filtering_metadata_path, space.identity)
            space.restore_stats(metadata.get("stats", {}))
            for key, value in metadata.get("generator_stats", {}).items():
                setattr(self, key, value)
            intermediate = [
                self.generated_bulk_gep_csv_fp, self.generated_cell_fraction_fp,
                self.sampled_sc_cell_id_file_path,
            ]
            if not self.generated_bulk_gep_fp.exists() and any(p.exists() for p in intermediate):
                if not all(p.exists() for p in intermediate):
                    raise ValueError(
                        "Intermediate generation files are incomplete. Use a new dataset name/output location."
                    )
            if self.generated_bulk_gep_csv_fp.exists():
                columns = pd.read_csv(self.generated_bulk_gep_csv_fp, nrows=0, index_col=0).columns.tolist()
                if columns != space.genes:
                    raise ValueError("Intermediate gene order differs from G. Use a new output location.")
            if self.generated_bulk_gep_fp.exists():
                reader = ReadH5AD(self.generated_bulk_gep_fp, backed="r")
                try:
                    if reader.get_h5ad().var_names.tolist() != space.genes:
                        raise ValueError("Accepted artifact gene order differs from G. Use a new output location.")
                    if not self.generated_cell_fraction_fp.exists():
                        raise ValueError("Accepted artifact has no cell fractions. Use a new output location.")
                    fractions = pd.read_csv(self.generated_cell_fraction_fp, index_col=0)
                    if reader.get_h5ad().obs_names.tolist() != fractions.index.tolist():
                        raise ValueError("Accepted artifact and cell-fraction sample IDs differ. Use a new output location.")
                finally:
                    reader.get_h5ad().file.close()
            for artifact in space.additional_artifacts:
                if artifact.exists():
                    validate_metadata(artifact.with_suffix(".metadata.json"), space.identity)
        space.validate_model()
        space.persist_genes()
        if space.pca is not None:
            self.gep_filtering_pca_target = space.pca.n_components
            self.gep_filtering_n_pcs = int(space.pca.n_components_)
        self._save_filtering_metadata()

    def _save_filtering_metadata(self):
        generator_stats = {
            key: getattr(self, key)
            for key in (
                "gep_filtering_input_n_samples", "gep_filtering_kept_n_samples",
                "gep_filtering_removed_n_samples", "gep_filtering_pca_target",
                "gep_filtering_n_pcs",
            )
        }
        write_metadata(
            self.filtering_metadata_path, self.filtering_space.identity,
            stats=self.filtering_space.stats(), generator_stats=generator_stats,
        )

    def _load_sc_data_for_generation(
            self,
            simu_method: str
    ) -> Tuple[pd.DataFrame, List[str], str]:
        """
        Load and prepare single-cell dataframes for generation.

        Parameters
        ----------
        simu_method : str
            Simulation method ('ave' or 'mul').

        Returns
        -------
        tuple
            (obs_df, gene_list, sc_dataset_key)
        """
        if (self.merged_sc_fp is None) and (self.sct_dataset_file_path is None):
            raise FileNotFoundError('Either "merged_sc_dataset_file_path" or "sct_dataset_file_path" is required.')
        obs_df = pd.DataFrame()
        sc_dataset = ''
        gene_list = []

        if simu_method == SimuMethod.AVERAGE: # generate average expression profiles for sctGEPs
            obs_df = self.merged_sc_dataset_obs
            sc_dataset = DatasetType.MERGED_SC

            if (self.merged_sc_fp is not None) and (self.merged_sc_dataset_df is None):
                reader = ReadH5AD(self.merged_sc_fp)  # TODO, improve to read faster??
                self.merged_sc_dataset_df = reader.get_df(
                    convert_to_tpm=True) if self.sc_dataset_gep_type == 'log_space' else reader.get_df()

            gene_list = self.merged_sc_dataset_df.columns.to_list()

        elif simu_method == SimuMethod.MULTIPLY:  # generate multi-cell-type bulk GEPs by matrix multiplication
            self.total_cell_number = 1
            if (self.sct_dataset_file_path is not None) and (self.sct_dataset_reader is None):
                probe_reader = ReadH5AD(self.sct_dataset_file_path, backed='r')
                resolved_loading_mode = self._resolve_sct_loading_mode(probe_reader)
                estimated_bytes = self._estimate_sct_matrix_bytes(probe_reader)
                self._close_backed_reader(probe_reader)

                if resolved_loading_mode == "eager":
                    self.sct_dataset_obs, self.sct_dataset_df, self.sct_dataset_reader = self._read_sct_dataset(
                        load_expression=True,
                        backed=None,
                    )
                else:
                    self.sct_dataset_obs, self.sct_dataset_df, self.sct_dataset_reader = self._read_sct_dataset(
                        load_expression=False,
                        backed='r',
                    )
                logger.info(
                    'Using Mixed SCT loading mode "%s" for %s (%d rows x %d genes, estimated %.2f GiB; auto eager threshold %.2f GiB).',
                    resolved_loading_mode,
                    self.sct_dataset_file_path,
                    int(self.sct_dataset_reader.get_h5ad().n_obs),
                    int(self.sct_dataset_reader.get_h5ad().n_vars),
                    estimated_bytes / float(1024 ** 3),
                    self.mixed_sct_auto_eager_max_gb,
                )
                sc_dataset = DatasetType.SCT
                obs_df = self.sct_dataset_obs
                gene_list = self.sct_dataset_reader.get_h5ad().var.index.to_list()

        else:
            supported = ", ".join(
                [
                    v
                    for k, v in SimuMethod.__dict__.items()
                    if (not k.startswith("__")) and isinstance(v, str)
                ]
            )
            raise TypeError(
                f'Invalid simulation method "{simu_method}", only {supported} are supported.'
            )

        return obs_df, gene_list, sc_dataset

    def _filter_simulated_geps(self, simulated_gep, simulated_gep_bak, reference_file, ref_exp_type,
                               filtering_method, sample_id_for_filtering, gene_list_in_sc_ds,
                               high_corr_gene_list, filtering_by_gene_range, min_percentage_within_gene_range,
                               gene_quantile_range, filtering_in_pca_space, pca_n_components, norm_ord, n_top,
                               show_filtering_info):
        """
        Encapsulates all filtering logic (Marker Ratio, Distance, PCA, Gene Range).
        Returns filtered TPM DataFrame or None.
        """
        exp_ref_df = None
        exp_obj_ref = None

        def _load_reference_df() -> pd.DataFrame:
            reference_obj = ExpObj(exp_file=reference_file, exp_type=ref_exp_type)
            reference_df = reference_obj.get_exp()
            if sample_id_for_filtering:
                filtered_ref_df = reference_df.loc[reference_df.index.isin(sample_id_for_filtering)]
                if filtered_ref_df.empty:
                    logger.warning(
                        'No reference samples matched the requested filtering_ref_types; '
                        'using the full reference cohort instead.'
                    )
                else:
                    reference_df = filtered_ref_df
            return reference_df

        def _pca_input_matrix(df: pd.DataFrame, label: str) -> np.ndarray:
            values = df.to_numpy(dtype=np.float64, copy=False)
            if not np.isfinite(values).all():
                raise ValueError(f'{label} contains non-finite values before PCA.')
            return values

        # 1. Pre-processing for filtering
        if high_corr_gene_list is not None:
            gene_list_in_sc_ds = high_corr_gene_list
            simulated_gep = simulated_gep.loc[:, gene_list_in_sc_ds]
            simulated_gep = non_log2cpm(simulated_gep)  # Ensure TPM

        # Load Reference if needed
        if self.m_gep_ref is None or filtering_method == 'marker_ratio':
            exp_obj_ref = ExpObj(exp_file=reference_file, exp_type=ref_exp_type)
            exp_ref_df = _load_reference_df()

        # 2. Marker Ratio Filtering
        if filtering_method == 'marker_ratio':
            if self.marker_ratio_ref is None:
                exp_obj_ref.cal_marker_gene_ratio(agg_methods={'CD4 T': 'max', 'B Cells': 'max'},
                                                  cell_types=self.cell_type_used)
                self.marker_ratio_ref = exp_obj_ref.get_marker_ratios()

                # Dynamic neighbor adjustment
                n_ref = self.marker_ratio_ref.shape[0]
                if n_ref < self.n_samples:
                    self.n_neighbors_each_ref = int(np.ceil(self.n_samples / max(1, n_ref - 1000)))
                if not self.ref_neighbor_counter:
                    self.ref_neighbor_counter = {i: 0 for i in self.marker_ratio_ref.index}
                n_top = min(n_top, self.n_neighbors_each_ref)

            simulated_gep = self._filter_gep_by_reference(simulated_gep, n_top=n_top)

            # Adaptive Quantile Adjustment
            if (simulated_gep is None) or (simulated_gep.shape[0] < 50):
                increment = 0.001 if self.filtering_quantile_upper < 0.999 else 0.0001
                self.filtering_quantile_upper += increment
                qn1 = QueryNeighbors(df_file=self.marker_ratio_ref)
                self.q_dis_nn_ref_upper = qn1.get_quantile_of_nn_distance(quantile=self.filtering_quantile_upper)
                print(
                    f'   > Adjusting filtering quantile to {self.filtering_quantile_upper:.4f} (Dist: {self.q_dis_nn_ref_upper:.4f})')

        # 3. Gene Range Filtering
        if filtering_by_gene_range and simulated_gep is not None:
            if exp_ref_df is None:
                exp_ref_df = _load_reference_df()
            # if tcga_gene_info is None:
            if gene_quantile_range is None:
                quantile_range = [0.005, 0.5, 0.995]
            else:
                quantile_range = gene_quantile_range
            q_col_name = ['q_' + str(int(q * 1000) / 10) for q in quantile_range]
            tcga_gene_info = get_quantile(exp_ref_df, quantile_range=quantile_range,
                                          col_name=q_col_name)
            valid_gep_list = []
            for inx, row in simulated_gep.iterrows():
                valid = True
                current_gene_list = \
                    get_gene_list_filtered_by_quantile_range(bulk_exp=row, tcga_exp=exp_ref_df,
                                                             tcga_gene_info=tcga_gene_info,
                                                             quantile_range=quantile_range,
                                                             q_col_name=q_col_name)
                if len(current_gene_list) / exp_ref_df.shape[1] < min_percentage_within_gene_range:
                    valid = False
                valid_gep_list.append(valid)
            if show_filtering_info:
                logger.info(f'{np.sum(valid_gep_list)} were kept after filtering by gene range.')
            simulated_gep = simulated_gep.loc[valid_gep_list, :].copy()

        if simulated_gep is None or simulated_gep.empty:
            return None

            # 4. Distance / PCA Filtering
        if (filtering_method in ['median_gep', 'mean_gep', 'linear_mmd']) and (simulated_gep is not None):

            # PCA Transformation logic
            if filtering_in_pca_space:
                if exp_ref_df is None:
                    exp_ref_df = _load_reference_df()
                # ... (Load or Train PCA model, transform ref and simu GEPs) ...
                # Note: This block in original code is complex. Ensure gene lists align perfectly.
                # TODO, this gene list should be the same as the gene list in PCA model, not the gene list in sc_ds
                pca_model_dir = os.path.dirname(os.path.dirname(reference_file))
                pca_model_dir = os.path.join(pca_model_dir, f'pca_model_{pca_n_components}')
                check_dir(pca_model_dir)
                pca_model_path = os.path.join(pca_model_dir, 'tcga_pca_model_for_gep_filtering.pkl')
                gene_list_in_pca_file_path = os.path.join(pca_model_dir, 'gene_list_for_pca.csv')
                if os.path.exists(pca_model_path) and os.path.exists(gene_list_in_pca_file_path):
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        warnings.simplefilter('always', InconsistentVersionWarning)
                        pca_model = load(pca_model_path)
                    version_warning = any(
                        issubclass(current_warning.category, InconsistentVersionWarning)
                        for current_warning in caught_warnings
                    )
                    if version_warning:
                        logger.warning(
                            'Stored PCA model %s was created with a different scikit-learn version; '
                            'retraining it with the current environment.',
                            pca_model_path,
                        )
                        pca_model = None
                    else:
                        gene_list_in_pca = pd.read_csv(gene_list_in_pca_file_path,
                                                       index_col='0').index.to_list()
                        self._maybe_log_pca_model_loaded(
                            pca_model_path=pca_model_path,
                            gene_list_in_pca_file_path=gene_list_in_pca_file_path,
                            n_genes=len(gene_list_in_pca),
                        )
                        # align reference GEPs with the gene list in the PCA model
                        exp_obj_ref = ReadExp(exp_file=exp_ref_df, exp_type=ref_exp_type)
                        exp_obj_ref.align_with_gene_list(gene_list=gene_list_in_pca, fill_not_exist=True)
                        exp_ref_df = exp_obj_ref.get_exp()  # TPM
                else:
                    pca_model = None

                if pca_model is None:
                    pca_model = PCA(n_components=pca_n_components, random_state=42)
                    # using the intersection of gene list in sc_ds and gene list in TCGA
                    gene_list_in_pca = list(set(gene_list_in_sc_ds) & set(exp_ref_df.columns))
                    exp_ref_df = exp_ref_df.loc[:, gene_list_in_pca].copy()
                    if exp_ref_df.shape[0] == 0:
                        raise ValueError(
                            'Reference expression matrix is empty after filtering and cannot be used for PCA.'
                        )
                    exp_ref_df = non_log2cpm(exp_ref_df)  # TPM
                    exp_ref_df_log = log2_transform(exp_ref_df)  # using log2(TPM+1) for PCA
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', RuntimeWarning)
                        pca_model.fit(_pca_input_matrix(exp_ref_df_log, 'Reference expression matrix'))
                    dump(pca_model, pca_model_path)
                    # save the gene list for PCA
                    pd.DataFrame(gene_list_in_pca).to_csv(gene_list_in_pca_file_path)
                self.gep_filtering_pca_target = pca_model.n_components
                self.gep_filtering_n_pcs = int(pca_model.n_components_)
                if exp_ref_df.shape[0] == 0:
                    raise ValueError('Reference expression matrix is empty after filtering and cannot be used for PCA.')
                assert np.all(exp_ref_df.columns == gene_list_in_pca)
                exp_ref_df = log2_transform(exp_ref_df)
                exp_ref_df = safe_pca_transform(pca_model, exp_ref_df, label='Reference expression matrix')
                exp_ref_df = pd.DataFrame(exp_ref_df, index=range(exp_ref_df.shape[0]),
                                          columns=range(exp_ref_df.shape[1]))
                if not os.path.exists(os.path.join(pca_model_dir, 'tcga_pca_ref.csv')):
                    exp_ref_df.to_csv(os.path.join(pca_model_dir, 'tcga_pca_ref.csv'))
                cumsum = np.cumsum(pca_model.explained_variance_ratio_)
                d = len(cumsum)
                self._maybe_log_pca_variance_summary(
                    pca_model_path=pca_model_path,
                    n_components=d,
                    explained_variance_pct=float(cumsum.max() * 100),
                )
                # exp_ref_df = exp_ref_df.iloc[:, :d].copy()
                # align simulated GEP with the gene list in the PCA model
                simulated_gep_obj = ReadExp(exp_file=simulated_gep, exp_type='TPM')
                simulated_gep_obj.align_with_gene_list(gene_list=gene_list_in_pca, fill_not_exist=True)
                simulated_gep = simulated_gep_obj.get_exp()  # TPM
                assert np.all(simulated_gep.columns == gene_list_in_pca)
                simulated_gep = log2_transform(simulated_gep)
                surviving_sample_ids = simulated_gep.index.copy()
                simulated_gep = safe_pca_transform(pca_model, simulated_gep, label='Simulated expression matrix')
                simulated_gep = pd.DataFrame(simulated_gep, index=surviving_sample_ids,
                                             columns=range(simulated_gep.shape[1]))
                # simulated_gep = simulated_gep.iloc[:, :d].copy()
            assert np.all(exp_ref_df.columns == simulated_gep.columns)

                # Calculate Distances
            if self.m_gep_ref is None:
                if 'mean_gep' in filtering_method or 'linear_mmd' in filtering_method:
                    self.m_gep_ref = exp_ref_df.mean(axis=0).values.reshape(1, -1)
                else:
                    self.m_gep_ref = exp_ref_df.median(axis=0).values.reshape(1, -1)

                dist_ref = np.linalg.norm(exp_ref_df - self.m_gep_ref, ord=norm_ord, axis=1)
                if 'linear_mmd' in filtering_method: dist_ref = dist_ref ** 2

                self.q_dis_nn_ref_upper = np.quantile(dist_ref, self.filtering_quantile_upper)
                if self.filtering_quantile_lower is not None:
                    self.q_dis_nn_ref_lower = np.quantile(dist_ref, self.filtering_quantile_lower)

            # Filter Simulation
            dis_simu = np.linalg.norm(simulated_gep.values - self.m_gep_ref, ord=norm_ord, axis=1)
            if 'linear_mmd' in filtering_method: dis_simu = dis_simu ** 2

            mask = dis_simu <= self.q_dis_nn_ref_upper
            if self.filtering_quantile_lower is not None:
                mask &= (dis_simu >= self.q_dis_nn_ref_lower)

            simulated_gep = simulated_gep.loc[mask]

        # Return the original full-gene TPMs for the surviving samples
        if simulated_gep is not None and not simulated_gep.empty:
            return simulated_gep_bak.loc[simulated_gep.index].copy()
        return None

    def _sc_sampling(
            self,
            cell_frac: pd.DataFrame,
            obs_df: pd.DataFrame,
            n_threads: int = 10,
            total_cell_number: Optional[int] = None,
            sep_by_patient: bool = False,
            sc_dataset: Optional[str] = None,
            minimum_n_base: int = 1,
            cell_distance_dict: Optional[Dict[str, NeighborData]] = None,
            sampling_method_gep: str = 'random',
            neighbor_sampling_scheme: str = NeighborSamplingScheme.SINGLE_ANCHOR,
    ) -> pd.DataFrame:
        """Select single cells to construct bulk samples."""
        if total_cell_number is not None:
            self.total_cell_number = total_cell_number

        # Prepare sampling tasks by getting the required number of cells per type to be sampled
        sampled_cell_ids = self._prepare_sampling_tasks(cell_frac)

        # Adjust thread count
        n_threads = get_available_threads(n_threads)

        # Execute sampling based on method
        if should_use_neighbor_sampling(sc_dataset, sampling_method_gep):
            results = sample_with_neighbors(
                sampled_cell_ids,
                obs_df,
                cell_distance_dict,
                n_threads,
                minimum_n_base=minimum_n_base,
                sc_dataset=sc_dataset or DatasetType.MERGED_SC,
                neighbor_sampling_scheme=neighbor_sampling_scheme,
            )
        else:
            results = sample_random(
                sampled_cell_ids, obs_df, sep_by_patient,
                sc_dataset, minimum_n_base, n_threads
            )

        # Format results
        sampled_cell_ids['selected_cell_id'] = [';'.join(r) for r in results]
        sampled_cell_ids.index.name = 'sample_id'
        sampled_cell_ids.sort_values(by=['sample_id', 'cell_type'], inplace=True)

        return sampled_cell_ids

    def _map_cell_id2exp(self, selected_cell_id: pd.DataFrame, sc_dataset: str = DatasetType.MERGED_SC,
                         simu_method: str = SimuMethod.AVERAGE, cell_frac: pd.DataFrame = None,
                         gep_type='MCT', add_noise: bool = False, noise_params: tuple = ()) -> pd.DataFrame:
        """
        Aggregates single-cell expression to bulk.
        """
        # Select source dataframe
        lazy_sct_reader = None
        if sc_dataset == DatasetType.SCT:
            sc_ds_df = self.sct_dataset_df
            lazy_sct_reader = self.sct_dataset_reader
        elif sc_dataset == DatasetType.MERGED_SC:
            sc_ds_df = self.merged_sc_dataset_df
        else:
            sc_ds_df = self.generated_sc_dataset_df

        def _get_expression_rows(cell_ids: List[str]) -> pd.DataFrame:
            if sc_ds_df is not None:
                return sc_ds_df.loc[cell_ids, :].copy()
            if lazy_sct_reader is not None:
                return lazy_sct_reader.get_rows_df(cell_ids, convert_to_tpm=True)
            raise RuntimeError('Single-cell expression data is not loaded.')

        simulated_exp = {}

        if gep_type == 'SCT':
            selected_cell_id = selected_cell_id.loc[selected_cell_id['n_cell'] > 1, :].copy()
            # n_non_zero = 1000
            n_genes = sc_ds_df.shape[1]
            for cell_type, group in selected_cell_id.groupby('cell_type'):
                for sample_id, row in group.iterrows():
                    cell_ids = row['selected_cell_id'].split(';')
                    current_gene_exp = _get_expression_rows(cell_ids).mean(axis=0)
                    if simu_method == 'random_replacement':
                        # long_tail_noise_non_zero = np.random.random(n_non_zero) * 2
                        # n_zero = np.random.randint(n_non_zero/10, n_non_zero)
                        # long_tail_noise = np.append(long_tail_noise_non_zero, np.zeros(n_zero))
                        long_tail_noise = np.random.random(size=n_genes)
                        # replace the values < 1 with random selected values
                        mask = (current_gene_exp < 1).values.astype(int)
                        current_gene_exp = current_gene_exp + long_tail_noise * mask
                    simulated_exp[sample_id] = pd.Series(current_gene_exp.values, index=current_gene_exp.index)
        else:
            # Logic for Multi Cell Type (Bulk) generation
            assert simu_method == 'mul', 'Only support matrix multiplication for generating MCT'
            subtype2cell_type = {i: k for k, v in self.cell_type2subtype.items() for i in v}
            # Optimization: Pre-calculate RNA coefficients if needed
            ct2rna_coefficient = {ct: 1.0 for ct in selected_cell_id['cell_type'].unique()}
            if self.total_rna_coefficient:
                # Consider the total RNA amount of each cell type
                all_cell_types = selected_cell_id['cell_type'].unique()
                for cell_type in all_cell_types:
                    if cell_type in self.total_rna_coefficient:
                        ct2rna_coefficient[cell_type] = self.total_rna_coefficient[cell_type]
                    elif (cell_type in subtype2cell_type) and \
                            (subtype2cell_type[cell_type] in self.total_rna_coefficient):
                        ct2rna_coefficient[cell_type] = self.total_rna_coefficient[subtype2cell_type[cell_type]]
                    else:
                        ct2rna_coefficient[cell_type] = 1.0
                self._maybe_log_total_rna_coefficients(ct2rna_coefficient)

            single_row_ids = not selected_cell_id['selected_cell_id'].astype(str).str.contains(';', regex=False).any()
            if sc_ds_df is not None and sc_ds_df.index.is_unique and single_row_ids:
                simulated_exp_df = self._aggregate_eager_bulk(
                    selected_cell_id, cell_frac, sc_ds_df, ct2rna_coefficient,
                    add_noise, noise_params,
                )
                return non_log2cpm(simulated_exp_df, sum_exp=1e6).round(3)
            if lazy_sct_reader is not None and single_row_ids:
                simulated_exp = self._aggregate_lazy_sct_bulk(
                    selected_cell_id=selected_cell_id,
                    cell_frac=cell_frac,
                    ct2rna_coefficient=ct2rna_coefficient,
                    add_noise=add_noise,
                    noise_params=noise_params,
                )
            else:
                # Group by sample to perform aggregation
                for sample_id, group in selected_cell_id.groupby(level=0):
                    cell_ids = group['selected_cell_id'].to_list()

                    # Extract expression for these cells
                    current_merged = _get_expression_rows(cell_ids)

                    # Perform aggregation in float64 to avoid overflow when large TPM
                    # values are combined with RNA scaling coefficients.
                    expression_matrix = current_merged.to_numpy(dtype=np.float64, copy=True)

                    # Align cell types
                    _cell_types = group['cell_type'].to_list()
                    current_cell_frac = cell_frac.loc[sample_id, _cell_types].to_numpy(dtype=np.float64).reshape(-1, 1)

                    if not np.isfinite(expression_matrix).all():
                        raise ValueError(
                            f'Encountered non-finite single-cell expression values for sample "{sample_id}" '
                            'before aggregation.'
                        )
                    if not np.isfinite(current_cell_frac).all():
                        raise ValueError(
                            f'Encountered non-finite cell fractions for sample "{sample_id}" before aggregation.'
                        )

                    if self.total_rna_coefficient:
                        rna_coeffs = np.array(
                            [ct2rna_coefficient.get(ct, 1.0) for ct in _cell_types],
                            dtype=np.float64,
                        ).reshape(-1, 1)
                        expression_matrix *= rna_coeffs

                    # Weighted sum
                    # Broadcasting the cell fractions across rows is equivalent to
                    # expression_matrix.T @ current_cell_frac, but avoids backend-specific
                    # matmul floating-point warnings seen in some environments.
                    with np.errstate(over='raise', invalid='raise'):
                        try:
                            matrix_mul = np.sum(
                                expression_matrix * current_cell_frac,
                                axis=0,
                                dtype=np.float64,
                            ).reshape(-1, 1)
                        except FloatingPointError as exc:
                            raise ValueError(
                                f'Failed to aggregate expression for sample "{sample_id}" because the '
                                'expression matrix contains extreme or non-finite values.'
                            ) from exc

                    if not np.isfinite(matrix_mul).all():
                        raise ValueError(
                            f'Encountered non-finite aggregated expression for sample "{sample_id}". '
                            'Please check the single-cell expression scale and total_rna_coefficient values.'
                        )

                    if self.total_rna_coefficient:
                        # Normalize to TPM (1e6)
                        total_expression = float(np.sum(matrix_mul, dtype=np.float64))
                        if not np.isfinite(total_expression) or total_expression <= 0:
                            raise ValueError(
                                f'Invalid total expression ({total_expression}) for sample "{sample_id}" '
                                'after RNA coefficient weighting.'
                            )
                        matrix_mul = matrix_mul / total_expression * 1e6

                    simulated_exp[sample_id] = pd.Series(matrix_mul.flatten(), index=current_merged.columns)

                    # Add Noise
                    if add_noise:
                        noise = self._sample_noise(n_samples=len(simulated_exp[sample_id]), f=noise_params[0])
                        if noise.sum() > noise_params[1]:
                            noise = noise / np.sum(noise) * noise_params[1]
                        simulated_exp[sample_id] += noise

        simulated_exp_df = pd.DataFrame.from_dict(simulated_exp, orient='index')
        # Ensure TPM format (if not already)
        simulated_exp_df = non_log2cpm(simulated_exp_df, sum_exp=1e6)
        return simulated_exp_df.round(3)

    @staticmethod
    def _lazy_sct_chunk_size() -> int:
        return 512

    def _aggregate_eager_bulk(
        self, selected_cell_id, cell_frac, source, coefficients,
        add_noise=False, noise_params=(),
    ):
        """Gather bounded chunks once, preserving legacy sample/reduction order."""
        groups = selected_cell_id.groupby(level=0).indices
        sample_ids = list(groups)
        if not sample_ids:
            return pd.DataFrame(index=sample_ids, columns=source.columns, dtype=np.float64)
        positions = np.concatenate(list(groups.values()))
        tasks = selected_cell_id.iloc[positions]
        source_rows = source.index.get_indexer(tasks['selected_cell_id'])
        fraction_rows = cell_frac.index.get_indexer(tasks.index)
        fraction_columns = cell_frac.columns.get_indexer(tasks['cell_type'])
        if (source_rows < 0).any():
            raise ValueError('Selected cell IDs are absent from the expression reference.')
        if (fraction_rows < 0).any() or (fraction_columns < 0).any():
            raise ValueError('Selected sample IDs or cell types are absent from cell fractions.')
        fractions = cell_frac.to_numpy(dtype=np.float64)[fraction_rows, fraction_columns]
        rna = np.array([coefficients[ct] for ct in tasks['cell_type']], dtype=np.float64)
        if not np.isfinite(fractions).all() or not np.isfinite(rna).all():
            raise ValueError('Encountered non-finite aggregation weights.')
        offsets = np.concatenate(([0], np.cumsum([len(group) for group in groups.values()])))
        n_genes = source.shape[1]
        # Allow two float64 expression temporaries within a 64 MiB budget.
        max_rows = max(1, (64 * 1024 ** 2) // max(1, n_genes * 8 * 2))
        output = np.empty((len(sample_ids), n_genes), dtype=np.float64)
        first = 0
        while first < len(sample_ids):
            last = first + 1
            while last < len(sample_ids) and offsets[last + 1] - offsets[first] <= max_rows:
                last += 1
            start, stop = offsets[first], offsets[last]
            expression = source.iloc[source_rows[start:stop]].to_numpy(dtype=np.float64, copy=True)
            if not np.isfinite(expression).all():
                raise ValueError('Encountered non-finite single-cell expression before aggregation.')
            with np.errstate(over='raise', invalid='raise'):
                try:
                    if self.total_rna_coefficient:
                        expression *= rna[start:stop, None]
                    expression *= fractions[start:stop, None]
                    for sample in range(first, last):
                        lo, hi = offsets[sample:sample + 2] - start
                        values = expression[lo:hi].sum(axis=0, dtype=np.float64)
                        if self.total_rna_coefficient:
                            total = values.sum(dtype=np.float64)
                            if not np.isfinite(total) or total <= 0:
                                raise ValueError('Invalid total expression after RNA coefficient weighting.')
                            values = values / total * 1e6
                        if add_noise:
                            noise = self._sample_noise(n_samples=n_genes, f=noise_params[0])
                            if noise.sum() > noise_params[1]:
                                noise = noise / noise.sum() * noise_params[1]
                            values += noise
                        output[sample] = values
                except FloatingPointError as exc:
                    raise ValueError('Failed to aggregate extreme or non-finite expression values.') from exc
            first = last
        if not np.isfinite(output).all():
            raise ValueError('Encountered non-finite aggregated expression.')
        return pd.DataFrame(output, index=sample_ids, columns=source.columns)

    def _aggregate_lazy_sct_bulk(
        self,
        selected_cell_id: pd.DataFrame,
        cell_frac: pd.DataFrame,
        ct2rna_coefficient: Dict[str, float],
        add_noise: bool = False,
        noise_params: tuple = (),
    ) -> Dict[str, pd.Series]:
        if self.sct_dataset_reader is None:
            raise RuntimeError('Lazy SCT reader is not available for aggregation.')

        chunk_size = self._lazy_sct_chunk_size()
        sample_order = list(dict.fromkeys(selected_cell_id.index.to_list()))
        sample_accumulators: Dict[str, np.ndarray] = {sample_id: None for sample_id in sample_order}
        gene_index = None

        for start in range(0, len(selected_cell_id), chunk_size):
            chunk_meta = selected_cell_id.iloc[start:start + chunk_size].copy()
            chunk_ids = chunk_meta['selected_cell_id'].astype(str).to_list()
            chunk_df = self.sct_dataset_reader.get_rows_df(chunk_ids, convert_to_tpm=True)
            if gene_index is None:
                gene_index = chunk_df.columns

            expression_matrix = chunk_df.to_numpy(dtype=np.float64, copy=False)
            if not np.isfinite(expression_matrix).all():
                raise ValueError('Encountered non-finite lazy SCT expression values before aggregation.')

            chunk_sample_ids = chunk_meta.index.to_list()
            chunk_cell_types = chunk_meta['cell_type'].to_list()
            weights = np.array(
                [
                    float(cell_frac.loc[sample_id, cell_type]) * float(ct2rna_coefficient.get(cell_type, 1.0))
                    for sample_id, cell_type in zip(chunk_sample_ids, chunk_cell_types)
                ],
                dtype=np.float64,
            )
            if not np.isfinite(weights).all():
                raise ValueError('Encountered non-finite lazy SCT aggregation weights.')

            weighted_expression = expression_matrix * weights.reshape(-1, 1)
            chunk_sample_index = pd.Index(chunk_sample_ids)
            for sample_id in chunk_sample_index.unique():
                sample_values = np.sum(
                    weighted_expression[chunk_sample_index == sample_id],
                    axis=0,
                    dtype=np.float64,
                )
                if sample_accumulators[sample_id] is None:
                    sample_accumulators[sample_id] = sample_values
                else:
                    sample_accumulators[sample_id] += sample_values

        simulated_exp: Dict[str, pd.Series] = {}
        for sample_id in sample_order:
            aggregated = sample_accumulators[sample_id]
            if aggregated is None or not np.isfinite(aggregated).all():
                raise ValueError(f'Encountered invalid lazy SCT aggregated expression for sample "{sample_id}".')
            current_series = pd.Series(aggregated, index=gene_index)
            if add_noise:
                noise = self._sample_noise(n_samples=len(current_series), f=noise_params[0])
                if noise.sum() > noise_params[1]:
                    noise = noise / np.sum(noise) * noise_params[1]
                current_series = current_series + noise
            simulated_exp[sample_id] = current_series

        return simulated_exp

    def _finalize_generation(self, sampling_method, filtering_method, log_file_path):
        """Prints summary and creates final h5ad."""
        msg = f'   > Generated {self.generated_bulk_gep_counter} samples.'
        print(msg)

        data_info = f'Simulated {self.generated_bulk_gep_counter} bulk GEPs by {sampling_method}, log2(TPM+1)'
        create_h5ad_dataset(simulated_bulk_exp_file_path=self.generated_bulk_gep_csv_fp,
                            cell_fraction_file_path=self.generated_cell_fraction_fp,
                            dataset_info=data_info,
                            result_file_path=self.generated_bulk_gep_fp)

        if log_file_path:
            print_msg(str(self), log_file_path=log_file_path)

    def _prepare_sampling_tasks(self, cell_frac: pd.DataFrame) -> pd.DataFrame:
        """Prepare sampling task DataFrame from cell fractions."""
        cell_num = get_cell_num(cell_type_frac=cell_frac, total_num=self.total_cell_number)

        tasks = []
        for cell_type in cell_num.columns:
            part = pd.DataFrame({'n_cell': cell_num[cell_type]})
            part = part.loc[part['n_cell'] > 0]
            if part.empty:
                continue
            part['cell_type'] = cell_type
            part['class_by'] = (
                self.subtype_col_name
                if cell_type in self.cell_subtype_used
                else self.cell_type_col_name
            )
            tasks.append(part)

        sampled_tasks = pd.concat(tasks)
        overrides = cell_frac.attrs.get("sampling_task_overrides", {})
        for sample_id, override in overrides.items():
            if sample_id not in sampled_tasks.index:
                continue
            sampled_tasks.loc[sample_id, ["cell_type", "class_by"]] = override
        return sampled_tasks

    @staticmethod
    def _sample_noise(miu=0, s=566.1, f=0.25, n_samples=10000) -> np.ndarray:
        """Generate log-normal noise."""
        sigma = f * np.log2(s)
        return np.power(2, stats.norm(miu, sigma).rvs(size=n_samples))

    def _check_basic_info(self):
        """Validates that requested cell types/datasets exist in the source file."""
        self.get_info_in_merged_single_cell_dataset(zero_ratio_threshold=self.zero_ratio_threshold)

        # Check Cell Types
        missing_ct = set(self.cell_type_used) - set(self.cell_type_in_sc)
        if missing_ct:
            raise ValueError(f"Invalid cell types: {missing_ct}. Available: {self.cell_type_in_sc}")

        # Check Datasets
        missing_ds = set(self.sc_dataset_used) - set(self.dataset_in_sc)
        if missing_ds:
            raise ValueError(f"Invalid datasets: {missing_ds}. Available: {self.dataset_in_sc}")

        print(f"   > Using {len(self.cell_type_used)} cell types, "
              f"{len(self.cell_subtype_used)} subtypes and {len(self.sc_dataset_used)} datasets.")

    def _normalize_selected_subtypes_by_cell_type(
        self,
        selected_subtypes_by_cell_type: Optional[Dict[str, List[str] | str]],
    ) -> Dict[str, List[str] | str] | None:
        """Validate and normalize optional SCT parent-type subtype restrictions."""
        if selected_subtypes_by_cell_type in (None, ""):
            return None
        if not isinstance(selected_subtypes_by_cell_type, dict):
            raise ValueError(
                '"selected_subtypes_by_cell_type" must be a mapping from parent cell type to subtype list.'
            )
        if self.subtype_col_name in (None, ""):
            raise ValueError(
                '"selected_subtypes_by_cell_type" requires a valid subtype column name.'
            )

        normalized: Dict[str, List[str]] = {}
        for parent_cell_type, subtypes in selected_subtypes_by_cell_type.items():
            parent_name = str(parent_cell_type).strip()
            if parent_name == "":
                raise ValueError('Invalid empty parent cell type in "selected_subtypes_by_cell_type".')
            if parent_name not in self.cell_type2subtype:
                raise ValueError(
                    f'Unknown parent cell type in "selected_subtypes_by_cell_type": {parent_name!r}. '
                    f'Available parent cell types: {sorted(self.cell_type2subtype)}'
                )
            if isinstance(subtypes, str) and subtypes.strip().lower() == "auto":
                normalized[parent_name] = "auto"
                continue
            if isinstance(subtypes, str) or subtypes in (None, ""):
                raise ValueError(
                    f'"selected_subtypes_by_cell_type[{parent_name}]" must be a non-empty list of subtype names.'
                )
            cleaned_subtypes = [str(subtype).strip() for subtype in list(subtypes)]
            if any(subtype == "" for subtype in cleaned_subtypes):
                raise ValueError(
                    f'Invalid empty subtype name found in "selected_subtypes_by_cell_type[{parent_name}]".'
                )
            deduped_subtypes = list(dict.fromkeys(cleaned_subtypes))
            if not deduped_subtypes:
                raise ValueError(
                    f'"selected_subtypes_by_cell_type[{parent_name}]" must contain at least one subtype.'
                )
            allowed_subtypes = set(self.cell_type2subtype[parent_name])
            invalid_subtypes = [subtype for subtype in deduped_subtypes if subtype not in allowed_subtypes]
            if invalid_subtypes:
                raise ValueError(
                    f'Invalid subtype(s) for parent cell type {parent_name!r}: {invalid_subtypes}. '
                    f'Allowed subtypes: {self.cell_type2subtype[parent_name]}'
                )
            normalized[parent_name] = deduped_subtypes
        return normalized

    def _validate_dataset_id_subset(self, dataset_ids: Optional[List[str]], label: str) -> None:
        """Validate configured dataset_id filters against the merged scRNA-seq obs."""
        if not dataset_ids:
            return
        cleaned = [str(i).strip() for i in dataset_ids]
        if any(i == '' for i in cleaned):
            raise ValueError(f'Invalid empty dataset_id found in {label}.')
        missing_ds = set(cleaned) - set(self.dataset_in_sc)
        if missing_ds:
            raise ValueError(f"Invalid datasets in {label}: {missing_ds}. Available: {self.dataset_in_sc}")

    def _stratify_obs_by_dataset_balanced_quota(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        """Down-sample obs so each (cell_type, dataset_id) group has equal quota per cell type.

        For each cell_type c present in obs_df:
          - group rows by dataset_id -> n_d(c) rows per dataset
          - K(c) = min_d n_d(c) over datasets with cell_type c
          - from each dataset, sample min(n_d(c), K(c)) rows without replacement.

        Preserves existing DataFrame columns. Rows within each group are sampled with an
        RNG seeded from self.stratify_seed XOR a stable hash of the cell_type name, so
        results are reproducible across runs with identical inputs / stratify_seed.

        If obs_df is empty, returns unchanged. If stratify_by_dataset is not
        "balanced_quota", returns obs_df unchanged.
        """
        if self.stratify_by_dataset != "balanced_quota":
            return obs_df
        if obs_df is None or len(obs_df) == 0:
            return obs_df

        cell_type_col = self.cell_type_col_name
        if cell_type_col not in obs_df.columns:
            logger.warning(
                "stratify_by_dataset=balanced_quota skipped: obs_df has no column %r",
                cell_type_col,
            )
            return obs_df

        per_cell_type_parts = []
        for cell_type, ct_df in obs_df.groupby(cell_type_col, sort=False, observed=True):
            if len(ct_df) == 0:
                continue
            per_ds = ct_df.groupby("dataset_id", sort=False, observed=True)
            counts = per_ds.size()
            if counts.min() <= 0:
                # No-op if any dataset contributes zero rows for this type (groupby already handled).
                per_cell_type_parts.append(ct_df.iloc[0:0])
                continue
            K = int(counts.min())
            bottleneck_ds = str(counts.idxmin())
            before_rows = len(ct_df)

            if K < 10:
                logger.warning(
                    "Balanced quota for cell_type %s is only K=%s rows per dataset "
                    "(bottleneck dataset: %s). Consider widening the source dataset pool.",
                    cell_type, K, bottleneck_ds,
                )

            ct_rows_kept_indices: list = []
            # Stable per-(cell_type, seed) RNG so repeated runs produce identical draws.
            seed_offset = (int(self.stratify_seed) ^ hash(str(cell_type)) & 0x7FFFFFFF)
            rng = np.random.default_rng(seed_offset)
            for ds_id, ds_df in per_ds:
                ds_df = ds_df.copy()
                n_ds = len(ds_df)
                n_keep = min(K, n_ds)
                if n_keep <= 0:
                    continue
                choice_idx = rng.choice(n_ds, size=n_keep, replace=False)
                ct_rows_kept_indices.extend(ds_df.index.take(choice_idx).tolist())

            stratified_ct = ct_df.loc[ct_rows_kept_indices]
            after_rows = len(stratified_ct)
            n_datasets = int(counts.shape[0])
            logger.info(
                "Stratified %s: K=%d (bottleneck dataset: %s, %d datasets total), "
                "kept %d rows per dataset × %d datasets = %d rows (was: %d rows before)",
                cell_type, K, bottleneck_ds, n_datasets,
                K, n_datasets, after_rows, before_rows,
            )
            per_cell_type_parts.append(stratified_ct)

        if not per_cell_type_parts:
            return obs_df.iloc[0:0]
        return pd.concat(per_cell_type_parts, axis=0)

    def get_info_in_merged_single_cell_dataset(self, check_zero_ratio: bool = True, zero_ratio_threshold: float = 0.95):
        """Reads metadata from the merged single-cell dataset."""
        reader = ReadH5AD(self.merged_sc_fp)
        if self.merged_sc_dataset is None:
            self.merged_sc_dataset = reader.get_h5ad()

        self.cell_type_in_sc = [str(i) for i in self.merged_sc_dataset.obs[self.cell_type_col_name].unique() if
                                str(i) != 'nan']
        self.dataset_in_sc = list(self.merged_sc_dataset.obs['dataset_id'].unique())

        self._validate_dataset_id_subset(self.cancer_cell_dataset_ids, 'cancer_cell_dataset_ids')
        self._validate_dataset_id_subset(self.non_cancer_cell_dataset_ids, 'non_cancer_cell_dataset_ids')

        if self.cancer_cell_dataset_ids is not None or self.non_cancer_cell_dataset_ids is not None:
            cancer_dataset_ids = self.cancer_cell_dataset_ids or self.sc_dataset_used
            non_cancer_dataset_ids = self.non_cancer_cell_dataset_ids or self.sc_dataset_used
            is_cancer = self.merged_sc_dataset.obs[self.cell_type_col_name].astype(str) == 'Cancer Cells'
            cancer_mask = is_cancer & self.merged_sc_dataset.obs['dataset_id'].isin(cancer_dataset_ids)
            non_cancer_mask = (~is_cancer) & self.merged_sc_dataset.obs['dataset_id'].isin(non_cancer_dataset_ids)
            self.merged_sc_dataset_obs = self.merged_sc_dataset.obs.loc[cancer_mask | non_cancer_mask].copy()
        else:
            # Filter obs by selected datasets
            self.merged_sc_dataset_obs = self.merged_sc_dataset.obs.loc[
                self.merged_sc_dataset.obs['dataset_id'].isin(self.sc_dataset_used)
            ].copy()

        # Dataset-balanced stratified down-sampling (applied after cancer/non-cancer +
        # dataset_id filters but before zero_ratio filtering, which further reduces the pool
        # uniformly across rows).
        self.merged_sc_dataset_obs = self._stratify_obs_by_dataset_balanced_quota(self.merged_sc_dataset_obs)

        # Zero ratio check
        if check_zero_ratio:
            if self.merged_sc_dataset_df is None:
                self.merged_sc_dataset_df = reader.get_df(
                    convert_to_tpm=True) if self.sc_dataset_gep_type == 'log_space' else reader.get_df()

            zero_ratios = (self.merged_sc_dataset_df < 1).sum(axis=1) / self.merged_sc_dataset_df.shape[1]
            high_zero_cells = zero_ratios[zero_ratios > zero_ratio_threshold].index

            if not high_zero_cells.empty:
                logger.info(f'Removing {len(high_zero_cells)} cells with zero ratio > {zero_ratio_threshold}')
                self.merged_sc_dataset_obs = self.merged_sc_dataset_obs.drop(high_zero_cells, errors='ignore')
        output_types = self.cell_type_used + self.cell_subtype_used
        cond_a = self.merged_sc_dataset_obs[self.subtype_col_name].isin(output_types)
        cond_b = self.merged_sc_dataset_obs[self.cell_type_col_name].isin(output_types)
        choices = [self.merged_sc_dataset_obs[self.subtype_col_name], self.merged_sc_dataset_obs[self.cell_type_col_name]]
        self.merged_sc_dataset_obs['output_types'] = np.select(
            condlist=[cond_a, cond_b],
            choicelist=choices,
            default='Unknown'
        )
        self.merged_sc_dataset_obs = self.merged_sc_dataset_obs[
            self.merged_sc_dataset_obs['output_types'] != 'Unknown'].copy()
        self.merged_sc_dataset_df = self.merged_sc_dataset_df.loc[self.merged_sc_dataset_obs.index, :].copy()
        self.merged_sc_dataset = self.merged_sc_dataset[self.merged_sc_dataset_obs.index, :]

    def _check_intermediate_generated_gep(self):
        """
        if the generation process broke accidentally,
        generated intermediate result can be used to recovery generation process
        """
        if self.generated_cell_fraction_fp.exists():
            gen_cell_frac = pd.read_csv(self.generated_cell_fraction_fp, index_col=0)
            gen_cell_frac = gen_cell_frac[~gen_cell_frac.index.duplicated(keep='first')].copy()
            if self.generated_bulk_gep_csv_fp.exists():
                gen_bulk_gep = pd.read_csv(self.generated_bulk_gep_csv_fp, index_col=0, usecols=[0, 1])
                gen_bulk_gep = gen_bulk_gep[~gen_bulk_gep.index.duplicated(keep='first')].copy()
                common_inx = [i for i in gen_cell_frac.index if i in gen_bulk_gep.index]
                if common_inx and (len(common_inx) == gen_bulk_gep.shape[0]) and (len(common_inx) == gen_cell_frac.shape[0]):
                    self.generated_bulk_gep_counter = len(common_inx)
                    if 'seg_random' in gen_cell_frac.iloc[-1].name:
                        n_round = int(gen_cell_frac.iloc[-1].name.split('_')[3])
                    else:
                        n_round = int(gen_cell_frac.iloc[-1].name.split('_')[2])
                    self.n_round = n_round + 1
                    logger.info(f'The following intermediate generated result will be reused: \n'
                          f'{self.generated_cell_fraction_fp}, {self.generated_bulk_gep_csv_fp}.\n'
                          f'self.n_round will be reset to {self.n_round}, '
                          f'self.generated_bulk_gep_counter will be reset to {self.generated_bulk_gep_counter}')
                    if self.ref_neighbor_counter_fp.exists():
                        ref2n_neighbors = pd.read_csv(self.ref_neighbor_counter_fp, index_col=0).to_dict()['0']
                        self.ref_neighbor_counter = ref2n_neighbors.copy()
                else:
                    if getattr(self, "filtering_space", None) is not None:
                        raise ValueError(
                            "Intermediate expression and cell-fraction samples do not match. "
                            "Use a new dataset name/output location; artifacts are not deleted."
                        )
                    files_to_remove = [
                        self.generated_cell_fraction_fp,
                        self.generated_bulk_gep_csv_fp,
                        self.sampled_sc_cell_id_file_path,
                    ]
                    for fp in files_to_remove:
                        try:
                            Path(fp).unlink(missing_ok=True)
                        except Exception as e:
                            logger.warning(f"Failed to remove {fp}: {e}")

    def _save_simulated_bulk_gep(
            self,
            gep: pd.DataFrame,
            cell_id: pd.DataFrame,
            cell_fraction: Optional[pd.DataFrame] = None
    ) -> None:
        """Save generated GEPs and metadata."""
        precision = "%.17g" if getattr(self, "filtering_space", None) is not None else "%g"
        append_to_csv(gep, self.generated_bulk_gep_csv_fp, float_format=precision)
        append_to_csv(cell_id, self.sampled_sc_cell_id_file_path)

        if cell_fraction is not None:
            append_to_csv(
                cell_fraction,
                self.generated_cell_fraction_fp,
                index_label='sample_id'
            )

        if self.ref_neighbor_counter:
            pd.DataFrame.from_dict(
                self.ref_neighbor_counter, orient='index'
            ).to_csv(self.ref_neighbor_counter_fp)

    def _filter_gep_by_reference(self, simulated_gep, n_top: int = None) -> Union[pd.DataFrame, None]:
        """
        Filtering generated GEP by reference dataset, such as TCGA

        :param simulated_gep: simulated GEPs that need to filter by reference, TPM
        :param n_top: if too many neighbors were founded for one single sample, only keep n_top neighbors
        """

        # print(marker_ratio_ref.head(2))
        exp_obj_simu_gep = ExpObj(exp_file=simulated_gep, exp_type='TPM')
        exp_obj_simu_gep.cal_marker_gene_ratio(agg_methods={'CD4 T': 'max', 'B Cells': 'max'},
                                               cell_types=self.cell_type_used)
        marker_ratio_simu_gep = exp_obj_simu_gep.get_marker_ratios()

        # the distance of nearest neighbor for each sample
        # quantile = 0.999
        if self.q_dis_nn_ref_upper == 0:
            qn1 = QueryNeighbors(df_file=self.marker_ratio_ref)
            # quantile of distance
            self.q_dis_nn_ref_upper = qn1.get_quantile_of_nn_distance(quantile=self.filtering_quantile_upper)
            logger.info(f'Quantile distance of {self.filtering_quantile_upper * 100}% is: {self.q_dis_nn_ref_upper}')
        # nn_dis = qn1.get_nn()

        qn2 = QueryNeighbors(df_file=marker_ratio_simu_gep)
        current_ref_nc = sorted(self.ref_neighbor_counter.items(), key=lambda x: x[1])
        _keep_ref = [i for i, j in current_ref_nc if j < self.n_neighbors_each_ref]
        ref_neighbors_within_radius = qn2.get_neighbors_by_radius(
            radius=self.q_dis_nn_ref_upper, n_top=n_top, share_neighbors=False,
            q_df_file=self.marker_ratio_ref.loc[_keep_ref, :].copy(),
            )
        ref2n_neighbors = ref_neighbors_within_radius.groupby(ref_neighbors_within_radius.index).count()
        _keep_neighbors = []
        for i, n_n in ref2n_neighbors.iterrows():
            self.ref_neighbor_counter[i] += n_n['nn']
        if ref_neighbors_within_radius.shape[0] > 0:
            return simulated_gep.loc[ref_neighbors_within_radius['nn'], :].copy()
        else:
            return None

    def _read_sct_dataset(self, latent_z_nn_info_file=None, load_expression: bool = True, backed: Optional[str] = None):
        """
        positive samples of SCT (single cell type), generated by BulkGEPGeneratorSCT
        :param latent_z_nn_info_file: neighbor information of latent z for all samples, used for QC
        """
        sct_dataset_obs, sct_dataset_df, sct_dataset_obj = \
            read_single_cell_type_dataset(sct_dataset_file_path=self.sct_dataset_file_path,
                                          latent_z_nn_info_file=latent_z_nn_info_file,
                                          load_expression=load_expression,
                                          backed=backed)
        return sct_dataset_obs, sct_dataset_df, sct_dataset_obj

    def _validate_inputs(self) -> None:
        """Validate initialization inputs."""
        if not self.merged_sc_fp and not self.sct_dataset_file_path:
            raise ValueError(
                "At least one of 'merged_sc_dataset_file_path' or "
                "'sct_dataset_file_path' must be provided."
            )

        if self.merged_sc_fp and not Path(self.merged_sc_fp).exists():
            raise FileNotFoundError(f"Merged SC dataset not found: {self.merged_sc_fp}")

        if self.sct_dataset_file_path and not Path(self.sct_dataset_file_path).exists():
            raise FileNotFoundError(f"SCT dataset not found: {self.sct_dataset_file_path}")

        if not self.cell_type2subtype:
            raise ValueError("cell_type2subtype mapping cannot be empty")

        if self.zero_ratio_threshold < 0 or self.zero_ratio_threshold > 1:
            raise ValueError(f"zero_ratio_threshold must be in [0, 1], got {self.zero_ratio_threshold}")

    def __str__(self):
        """JSON representation of the object configuration."""
        config = {k: v for k, v in self.__dict__.items() if isinstance(v, (list, str, int, float, dict))}
        return f"{self.__class__.__name__}:\n{json.dumps(config, indent=2)}"
