import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from joblib import dump, load
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
logger = logging.getLogger(__name__)


from ..utility import get_gene_list_filtered_by_quantile_range, get_quantile, check_dir
from ..utility import ReadExp, ReadH5AD, ExpObj, QueryNeighbors, read_single_cell_type_dataset
from ..utility import get_available_threads, non_log2cpm, log2_transform, get_cell_num, create_h5ad_dataset, print_msg
from .sampling import generate_cell_fraction
from .simu_helper import (sample_random, get_sample_id_random, sample_with_neighbors,
                          should_use_neighbor_sampling, append_to_csv, SimuMethod, DatasetType)
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

    def __post_init__(self):
        """Validate configuration."""
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.simu_method not in ('ave', 'mul'):
            raise ValueError(f"Invalid simu_method: {self.simu_method}")


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
    ):

        self.simu_bulk_dir = Path(simu_bulk_dir)
        self.simu_bulk_dir.mkdir(parents=True, exist_ok=True)
        # check_dir(simu_bulk_dir)

        self.merged_sc_fp = merged_sc_dataset_file_path
        self.sct_dataset_file_path = sct_dataset_file_path
        self.cell_type2subtype = cell_type2subtype
        self.sc_dataset_used = sc_dataset_ids
        self.bulk_dataset_name = bulk_dataset_name
        self.zero_ratio_threshold = zero_ratio_threshold
        self.sc_dataset_gep_type = sc_dataset_gep_type
        self.tcga2cancer_type_file_path = tcga2cancer_type_file_path
        self.total_rna_coefficient = total_rna_coefficient
        self.subtype_col_name = subtype_col_name
        self.cell_type_col_name = cell_type_col_name

        # Initialize state variables
        self.cell_type_in_sc = []
        self.cell_subtype_in_sc = []
        self.dataset_in_sc = []
        self.merged_sc_dataset = None
        self.merged_sc_dataset_obs = None
        self.merged_sc_dataset_df = None
        self.sct_dataset_obs = None
        self.sct_dataset_df = None
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

        # Process cell types
        self.cell_type_used, self.cell_subtype_used = get_cell_type_used(cell_type2subtype, output_types=output_types)

        # Define file paths
        prefix = f'simu_bulk_exp_{self.bulk_dataset_name}' + f'_by_{sampling_method_gep}SCT'
        self.generated_cell_fraction_fp = self.simu_bulk_dir / f'generated_frac_{self.bulk_dataset_name}.csv'
        self.ref_neighbor_counter_fp = self.simu_bulk_dir / f'ref2n_neighbors_{self.bulk_dataset_name}.csv'
        self.generated_bulk_gep_csv_fp = self.simu_bulk_dir / f'{prefix}_log2cpm1p.csv'
        self.sampled_sc_cell_id_file_path = self.simu_bulk_dir / f'{prefix}_sampled_sc_cell_id.csv'
        self.generated_bulk_gep_fp = self.generated_bulk_gep_csv_fp.parent / self.generated_bulk_gep_csv_fp.name.replace('.csv', '.h5ad')

        if check_basic_info and not self.generated_bulk_gep_fp.exists():  # Put this checker the last one
            self._check_basic_info()

    def generate_gep(self, config: GEPGenerationConfig) -> None:
        """
        Main method to generate simulated bulk GEPs.
        """
        self.n_samples = config.n_samples
        self.total_cell_number = config.total_cell_number
        self.filtering_quantile_lower, self.filtering_quantile_upper = config.gep_filtering_quantile

        # Adjust threads
        n_total_cpus = multiprocessing.cpu_count()
        n_threads = min(n_total_cpus - 1, config.n_threads)

        if 'linear_mmd' in config.filtering_method:
            norm_ord = 2

        # Determine batch size for generation
        min_n_cell_frac = min(5000, config.n_samples) if config.filtering else min(1000, config.n_samples)

        if self.generated_bulk_gep_fp.exists():
            logger.info(f'Previous result existed: {self.generated_bulk_gep_fp}')
            logger.info(self.__str__())
            return

        # Resume check
        self._check_intermediate_generated_gep()

        # Load Data
        obs_df, gene_list_in_sc_ds, sc_dataset_key = self._load_sc_data_for_generation(config.simu_method)
        if sc_dataset_key == DatasetType.MERGED_SC:
            min_n_cell_frac = 300  # Smaller batch for merged dataset due to sparsity

        # Setup Filtering Reference
        sample_id_for_filtering = []
        if config.filtering and config.filtering_ref_types:
            s2c = pd.read_csv(self.tcga2cancer_type_file_path, index_col=0)
            sample_id_for_filtering = s2c.loc[s2c['cancer_type'].isin(config.filtering_ref_types)].index.to_list()
            logger.info(f'{len(sample_id_for_filtering)} samples used for filtering.')

        # Main Generation Loop
        with tqdm(total=self.n_samples) as pbar:
            if self.generated_bulk_gep_counter != 0:
                pbar.update(self.generated_bulk_gep_counter)

            while self.generated_bulk_gep_counter < self.n_samples:
                # 1. Generate Fractions
                generated_cell_frac = generate_cell_fraction(
                    sampling_method=config.sampling_method_prop,
                    n_cell_frac=min_n_cell_frac,
                    sample_prefix=f's_{config.sampling_method_prop}_{self.n_round}',
                    cell_prop_prior=config.cell_prop_prior if config.filtering else None,
                    cell_type2subtype=self.cell_type2subtype
                )

                # 2. Sample Cells (Multiprocessing)
                selected_cell_ids = self._sc_sampling(
                    cell_frac=generated_cell_frac,
                    n_threads=n_threads,
                    obs_df=obs_df,
                    sc_dataset=sc_dataset_key
                )

                # 3. Aggregate Expression
                simulated_gep = self._map_cell_id2exp(
                    selected_cell_id=selected_cell_ids,
                    simu_method=config.simu_method,
                    sc_dataset=sc_dataset_key,
                    cell_frac=generated_cell_frac,
                    add_noise=config.add_noise,
                    noise_params=config.noise_params
                )

                simulated_gep_bak = simulated_gep.copy()  # Keep TPM version

                # 4. Filtering (GEP-level filtering during generation)
                if config.filtering:
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

                # 5. Save Results
                if simulated_gep is not None and not simulated_gep.empty:
                    # Ensure we don't exceed requested n_samples
                    remaining_needed = self.n_samples - self.generated_bulk_gep_counter
                    if simulated_gep.shape[0] > remaining_needed:
                        simulated_gep = simulated_gep.iloc[:remaining_needed]

                    # Convert back to log2(TPM+1) for saving
                    simulated_gep_log = log2_transform(simulated_gep)

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

        # Finalize
        self._finalize_generation(config.sampling_method_prop, config.filtering_method, config.log_file_path)

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
            if (self.sct_dataset_file_path is not None) and (self.sct_dataset_df is None):
                self.sct_dataset_obs, self.sct_dataset_df = self._read_sct_dataset()
                sc_dataset = DatasetType.SCT
                obs_df = self.sct_dataset_obs
                gene_list = self.sct_dataset_df.columns.to_list()

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

        # 1. Pre-processing for filtering
        if high_corr_gene_list is not None:
            gene_list_in_sc_ds = high_corr_gene_list
            simulated_gep = simulated_gep.loc[:, gene_list_in_sc_ds]
            simulated_gep = non_log2cpm(simulated_gep)  # Ensure TPM

        # Load Reference if needed
        if self.m_gep_ref is None or filtering_method == 'marker_ratio':
            exp_obj_ref = ExpObj(exp_file=reference_file, exp_type=ref_exp_type)
            exp_ref_df = exp_obj_ref.get_exp()
            exp_ref_df = exp_ref_df.loc[exp_ref_df.index.isin(sample_id_for_filtering)]

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

            # 4. Distance / PCA Filtering
        if (filtering_method in ['median_gep', 'mean_gep', 'linear_mmd']) and (simulated_gep is not None):

            # PCA Transformation logic
            if filtering_in_pca_space:
                # ... (Load or Train PCA model, transform ref and simu GEPs) ...
                # Note: This block in original code is complex. Ensure gene lists align perfectly.
                # TODO, this gene list should be the same as the gene list in PCA model, not the gene list in sc_ds
                pca_model_dir = os.path.dirname(os.path.dirname(reference_file))
                pca_model_dir = os.path.join(pca_model_dir, f'pca_model_{pca_n_components}')
                check_dir(pca_model_dir)
                pca_model_path = os.path.join(pca_model_dir, 'tcga_pca_model_for_gep_filtering.pkl')
                gene_list_in_pca_file_path = os.path.join(pca_model_dir, 'gene_list_for_pca.csv')
                if os.path.exists(pca_model_path) and os.path.exists(gene_list_in_pca_file_path):
                    pca_model = load(pca_model_path)
                    gene_list_in_pca = pd.read_csv(gene_list_in_pca_file_path,
                                                   index_col='0').index.to_list()
                    logger.info(f'PCA model was loaded from {pca_model_path}, '
                          f'and gene list was loaded from {gene_list_in_pca_file_path}')
                    # align reference GEPs with the gene list in the PCA model
                    exp_obj_ref = ReadExp(exp_file=exp_ref_df, exp_type=ref_exp_type)
                    exp_obj_ref.align_with_gene_list(gene_list=gene_list_in_pca, fill_not_exist=True)
                    exp_ref_df = exp_obj_ref.get_exp()  # TPM
                else:
                    pca_model = PCA(n_components=pca_n_components, random_state=42)
                    # using the intersection of gene list in sc_ds and gene list in TCGA
                    gene_list_in_pca = list(set(gene_list_in_sc_ds) & set(exp_ref_df.columns))
                    exp_ref_df = exp_ref_df.loc[:, gene_list_in_pca].copy()
                    exp_ref_df = non_log2cpm(exp_ref_df)  # TPM
                    exp_ref_df_log = log2_transform(exp_ref_df)  # using log2(TPM+1) for PCA
                    pca_model.fit(exp_ref_df_log)
                    dump(pca_model, pca_model_path)
                    # save the gene list for PCA
                    pd.DataFrame(gene_list_in_pca).to_csv(gene_list_in_pca_file_path)
                assert np.all(exp_ref_df.columns == gene_list_in_pca)
                exp_ref_df = log2_transform(exp_ref_df)
                exp_ref_df = pca_model.transform(exp_ref_df)
                exp_ref_df = pd.DataFrame(exp_ref_df, index=range(exp_ref_df.shape[0]),
                                          columns=range(exp_ref_df.shape[1]))
                if not os.path.exists(os.path.join(pca_model_dir, 'tcga_pca_ref.csv')):
                    exp_ref_df.to_csv(os.path.join(pca_model_dir, 'tcga_pca_ref.csv'))
                cumsum = np.cumsum(pca_model.explained_variance_ratio_)
                d = len(cumsum)
                logger.info(f'{d} dimensions are needed to explain '
                      f'{cumsum.max() * 100}% variance.')  # 1515
                # exp_ref_df = exp_ref_df.iloc[:, :d].copy()
                # align simulated GEP with the gene list in the PCA model
                simulated_gep_obj = ReadExp(exp_file=simulated_gep, exp_type='TPM')
                simulated_gep_obj.align_with_gene_list(gene_list=gene_list_in_pca, fill_not_exist=True)
                simulated_gep = simulated_gep_obj.get_exp()  # TPM
                assert np.all(simulated_gep.columns == gene_list_in_pca)
                simulated_gep = log2_transform(simulated_gep)
                simulated_gep = pca_model.transform(simulated_gep)
                simulated_gep = pd.DataFrame(simulated_gep, index=simulated_gep_bak.index,
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
            cell_distance_dict: Optional[Dict[str, Tuple[np.ndarray, List[str]]]] = None,
            sampling_method_gep: str = 'random',
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
                sampled_cell_ids, obs_df, cell_distance_dict, n_threads
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
        if sc_dataset == DatasetType.SCT:
            sc_ds_df = self.sct_dataset_df
        elif sc_dataset == DatasetType.MERGED_SC:
            sc_ds_df = self.merged_sc_dataset_df
        else:
            sc_ds_df = self.generated_sc_dataset_df

        simulated_exp = {}

        if gep_type == 'SCT':
            selected_cell_id = selected_cell_id.loc[selected_cell_id['n_cell'] > 1, :].copy()
            # n_non_zero = 1000
            n_genes = sc_ds_df.shape[1]
            for cell_type, group in selected_cell_id.groupby('cell_type'):
                for sample_id, row in group.iterrows():
                    cell_ids = row['selected_cell_id'].split(';')
                    # simulated_exp[sample_id] = sc_ds_df.loc[cell_ids, :].mean(axis=0)  # average
                    current_gene_exp = sc_ds_df.loc[cell_ids, :].mean(axis=0)  # average
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
                print('   > The following total RNA coefficient will be used: ', ct2rna_coefficient)

            # Group by sample to perform aggregation
            for sample_id, group in selected_cell_id.groupby(level=0):
                cell_ids = group['selected_cell_id'].to_list()

                # Extract expression for these cells
                # Note: This line is memory intensive if cell_ids is large
                current_merged = sc_ds_df.loc[cell_ids, :].copy()

                # Align cell types
                _cell_types = group['cell_type'].to_list()
                current_cell_frac = cell_frac.loc[sample_id, _cell_types].values.reshape(-1, 1)

                if self.total_rna_coefficient:
                    rna_coeffs = np.array([ct2rna_coefficient.get(ct, 1.0) for ct in _cell_types]).reshape(-1, 1)
                    # Apply coefficient to the expression matrix
                    current_merged = current_merged * rna_coeffs

                    # Weighted sum
                    # (Cells x Genes).T @ (Cells x 1) -> (Genes x 1)
                    matrix_mul = current_merged.values.T @ current_cell_frac

                    # Normalize to TPM (1e6)
                    matrix_mul = matrix_mul / np.sum(matrix_mul) * 1e6
                else:
                    # Standard mix
                    matrix_mul = current_merged.values.T @ current_cell_frac

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
            part['cell_type'] = cell_type
            part['class_by'] = (
                self.subtype_col_name
                if cell_type in self.cell_subtype_used
                else self.cell_type_col_name
            )
            tasks.append(part)

        return pd.concat(tasks)

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

    def get_info_in_merged_single_cell_dataset(self, check_zero_ratio: bool = True, zero_ratio_threshold: float = 0.95):
        """Reads metadata from the merged single-cell dataset."""
        reader = ReadH5AD(self.merged_sc_fp)
        if self.merged_sc_dataset is None:
            self.merged_sc_dataset = reader.get_h5ad()

        self.cell_type_in_sc = [str(i) for i in self.merged_sc_dataset.obs[self.cell_type_col_name].unique() if
                                str(i) != 'nan']
        self.dataset_in_sc = list(self.merged_sc_dataset.obs['dataset_id'].unique())

        # Filter obs by selected datasets
        self.merged_sc_dataset_obs = self.merged_sc_dataset.obs.loc[
            self.merged_sc_dataset.obs['dataset_id'].isin(self.sc_dataset_used)
        ].copy()

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
        append_to_csv(gep, self.generated_bulk_gep_csv_fp)
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

    def _read_sct_dataset(self, latent_z_nn_info_file=None):
        """
        positive samples of SCT (single cell type), generated by BulkGEPGeneratorSCT
        :param latent_z_nn_info_file: neighbor information of latent z for all samples, used for QC
        """
        sct_dataset_obs, sct_dataset_df = \
            read_single_cell_type_dataset(sct_dataset_file_path=self.sct_dataset_file_path,
                                          latent_z_nn_info_file=latent_z_nn_info_file)
        return sct_dataset_obs, sct_dataset_df

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
