import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import json
import pickle
import atexit
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from joblib import dump, load
from sklearn.decomposition import PCA
multiprocessing.set_start_method('spawn', force=True)
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
logger = logging.getLogger(__name__)


from ..utility import get_gene_list_filtered_by_quantile_range, get_quantile, check_dir
from ..utility import ReadExp, ReadH5AD, ExpObj, QueryNeighbors, read_single_cell_type_dataset
from ..utility import log_exp2cpm, non_log2cpm, log2_transform, get_cell_num, get_sample_id_random, create_h5ad_dataset, print_msg
from ..utility import get_distance_matrices_by_class
from .sampling import generate_cell_fraction
from .nneighbors_shared_memory import parallel_sample_with_nearest_neighbors


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

    def __init__(self,
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
                 output_types: Optional[List[str]] = None):

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
        prefix = f'simu_bulk_exp_{self.bulk_dataset_name}'
        self.generated_cell_fraction_fp = self.simu_bulk_dir / f'generated_frac_{self.bulk_dataset_name}.csv'
        self.ref_neighbor_counter_fp = self.simu_bulk_dir / f'ref2n_neighbors_{self.bulk_dataset_name}.csv'
        self.generated_bulk_gep_csv_fp = self.simu_bulk_dir / f'{prefix}_log2cpm1p.csv'
        self.sampled_sc_cell_id_file_path = self.simu_bulk_dir / f'{prefix}_sampled_sc_cell_id.csv'
        self.generated_bulk_gep_fp = self.generated_bulk_gep_csv_fp.parent / self.generated_bulk_gep_csv_fp.name.replace('.csv', '.h5ad')

        if check_basic_info and not self.generated_bulk_gep_fp.exists():  # Put this checker the last one
            self._check_basic_info()

    def generate_gep(self, n_samples: int,
                     sampling_method_prop: str = 'segment',
                     total_cell_number: int = 100,
                     n_threads: int = 10,
                     filtering: bool = True,
                     reference_file: str | pd.DataFrame | Path = None,
                     ref_exp_type: str = None,
                     gep_filtering_quantile: tuple = (None, 0.95),
                     log_file_path: str = None,
                     n_top: int = 20,
                     simu_method: str = 'mul',
                     filtering_method: str = 'media_gep',
                     add_noise: bool = False,
                     noise_params: tuple = (),
                     filtering_ref_types: List[str] = None,
                     show_filtering_info: bool = False,
                     cell_prop_prior: Dict = None,
                     high_corr_gene_list: List[str] = None,
                     filtering_by_gene_range: bool = False,
                     min_percentage_within_gene_range: float = 0.95,
                     gene_quantile_range: List[float] = None,
                     filtering_in_pca_space: bool = False,
                     pca_n_components: Union[int, float] = 0.9,
                     norm_ord: int = 1):
        """
        Main method to generate simulated bulk GEPs.
        """
        self.n_samples = n_samples
        self.total_cell_number = total_cell_number
        self.filtering_quantile_lower, self.filtering_quantile_upper = gep_filtering_quantile

        # Adjust threads
        n_total_cpus = multiprocessing.cpu_count()
        n_threads = min(n_total_cpus - 1, n_threads)

        if 'linear_mmd' in filtering_method:
            norm_ord = 2

        # Determine batch size for generation
        min_n_cell_frac = min(5000, n_samples) if filtering else min(1000, n_samples)

        if os.path.exists(self.generated_bulk_gep_fp):
            print(f'   Previous result existed: {self.generated_bulk_gep_fp}')
            print(self.__str__())
            return

        # Resume check
        self._check_intermediate_generated_gep()

        # Load Data
        obs_df, gene_list_in_sc_ds, sc_dataset_key = self._load_sc_data_for_generation(simu_method)
        if sc_dataset_key == 'merged_sc_dataset':
            min_n_cell_frac = 300  # Smaller batch for merged dataset due to sparsity

        # Setup Filtering Reference
        sample_id_for_filtering = []
        if filtering and filtering_ref_types:
            s2c = pd.read_csv(self.tcga2cancer_type_file_path, index_col=0)
            sample_id_for_filtering = s2c.loc[s2c['cancer_type'].isin(filtering_ref_types)].index.to_list()
            print(f'   > {len(sample_id_for_filtering)} samples used for filtering.')

        # Main Generation Loop
        with tqdm(total=self.n_samples) as pbar:
            if self.generated_bulk_gep_counter != 0:
                pbar.update(self.generated_bulk_gep_counter)

            while self.generated_bulk_gep_counter < self.n_samples:
                # 1. Generate Fractions
                generated_cell_frac = generate_cell_fraction(
                    sampling_method=sampling_method_prop,
                    n_cell_frac=min_n_cell_frac,
                    sample_prefix=f's_{sampling_method_prop}_{self.n_round}',
                    cell_prop_prior=cell_prop_prior if filtering else None,
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
                    simu_method=simu_method,
                    sc_dataset=sc_dataset_key,
                    cell_frac=generated_cell_frac,
                    add_noise=add_noise,
                    noise_params=noise_params
                )

                simulated_gep_bak = simulated_gep.copy()  # Keep TPM version

                # 4. Filtering (GEP-level filtering during generation)
                if filtering:
                    simulated_gep = self._filter_simulated_geps(
                        simulated_gep=simulated_gep,
                        simulated_gep_bak=simulated_gep_bak,
                        reference_file=reference_file,
                        ref_exp_type=ref_exp_type,
                        filtering_method=filtering_method,
                        sample_id_for_filtering=sample_id_for_filtering,
                        gene_list_in_sc_ds=gene_list_in_sc_ds,
                        high_corr_gene_list=high_corr_gene_list,
                        filtering_by_gene_range=filtering_by_gene_range,
                        min_percentage_within_gene_range=min_percentage_within_gene_range,
                        gene_quantile_range=gene_quantile_range,
                        filtering_in_pca_space=filtering_in_pca_space,
                        pca_n_components=pca_n_components,
                        norm_ord=norm_ord,
                        n_top=n_top,
                        show_filtering_info=show_filtering_info
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
        self._finalize_generation(sampling_method_prop, filtering_method, log_file_path)

    def _load_sc_data_for_generation(self, simu_method):
        """Helper to load and prepare single cell dataframes."""
        if (self.merged_sc_fp is None) and (self.sct_dataset_file_path is None):
            raise FileNotFoundError('Either "merged_sc_dataset_file_path" or "sct_dataset_file_path" is required.')
        obs_df = pd.DataFrame()
        sc_dataset = ''
        gene_list = []

        if simu_method == 'ave': # generate average expression profiles for sctGEPs
            obs_df = self.merged_sc_dataset_obs
            sc_dataset = 'merged_sc_dataset'

            if (self.merged_sc_fp is not None) and (self.merged_sc_dataset_df is None):
                reader = ReadH5AD(self.merged_sc_fp)  # TODO, improve to read faster??
                self.merged_sc_dataset_df = reader.get_df(
                    convert_to_tpm=True) if self.sc_dataset_gep_type == 'log_space' else reader.get_df()

            gene_list = self.merged_sc_dataset_df.columns.to_list()

        elif simu_method == 'mul':
            self.total_cell_number = 1
            if (self.sct_dataset_file_path is not None) and (self.sct_dataset_df is None):
                self.sct_dataset_obs, self.sct_dataset_df = self._read_sct_dataset()
                sc_dataset = 'sct_dataset'
                obs_df = self.sct_dataset_obs
                gene_list = self.sct_dataset_df.columns.to_list()

        else:
            raise TypeError('Invalid simulation method "{}", only "ave" and "mul" are supported.'.format(simu_method))

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
                print(f'   > {np.sum(valid_gep_list)} were kept after filtering by gene range.')
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
                    print(f'   > PCA model was loaded from {pca_model_path}, '
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
                print(f'   > {d} dimensions are needed to explain '
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

    def _sc_sampling(self, cell_frac: pd.DataFrame, obs_df: pd.DataFrame, n_threads: int = 10,
                     total_cell_number: int = None, sep_by_patient=False, sc_dataset=None, minimum_n_base: int = 1,
                     cell_distance_dict: Dict[str, Tuple[np.ndarray, List[str]]] = None, sampling_method_gep: str = 'random',
                     ) -> pd.DataFrame:
        """
        Select specific single cells to construct the bulk samples.
        Parameters
        ----------
        cell_frac : pd.DataFrame
            DataFrame where each row corresponds to a sample and each column to a cell type,
            with values indicating the fraction of each cell type in that sample.
        obs_df : pd.DataFrame
            DataFrame containing metadata for single cells or sctGEPs, including cell type annotations.
        n_threads : int
            Number of threads to use for parallel processing.
        total_cell_number : int, optional
            Total number of cells to sample for each bulk sample. If None, uses the instance's total_cell_number.
        sep_by_patient : bool
            Whether to separate sampling by patient (if patient info is available in obs_df).
        sc_dataset : str
            Identifier for the single-cell dataset to use (e.g., 'merged_sc_dataset' or 'sct_dataset').
        minimum_n_base : int
            Minimum number of base cells to sample per cell type.
        sampling_method_gep: str
            Method for sampling cells to simulate bulk sctGEPs, either 'random' or 'n_neighbors'.
        cell_distance_dict : Dict[str, Tuple[np.ndarray, List[str]]], optional
            Cell distance matrices for each cell type, if precomputed. If provided, sampling cells by n_neighbors will use these distances.
            n_neighbors will consider cell similarity and find the nearest (n_base - 1) cells to form a sctGEP.
        Returns
        -------
        pd.DataFrame
            DataFrame with sampled cell IDs for each bulk sample and cell type.
        """
        if total_cell_number is not None:
            self.total_cell_number = total_cell_number

        # Calculate how many cells of each type are needed per sample
        cell_num = get_cell_num(cell_type_frac=cell_frac, total_num=self.total_cell_number)

        # Prepare sampling tasks
        cell_num_flatten = []
        for cell_type in cell_num.columns:
            _part = pd.DataFrame({'n_cell': cell_num[cell_type]})
            _part['cell_type'] = cell_type
            _part['class_by'] = self.subtype_col_name if cell_type in self.cell_subtype_used else self.cell_type_col_name
            cell_num_flatten.append(_part)

        sampled_cell_ids: pd.DataFrame = pd.concat(cell_num_flatten)

        # Prepare arguments for multiprocessing
        # Note: passing large 'obs_df' to starmap can be slow due to pickling.
        # If obs_df is huge, consider using shared memory or passing only necessary columns.
        n_threads = min(multiprocessing.cpu_count() - 2, n_threads)
        if sc_dataset == 'merged_sc_dataset' and sampling_method_gep == 'n_neighbors':
            sampling_tasks = [{'cell_type': row['cell_type'], 'n_samples': 1,
                      'n_base': row['n_cell'], 'class_by': row['class_by']} for i, row in sampled_cell_ids.iterrows()]
            results = parallel_sample_with_nearest_neighbors(
                obs_df=obs_df,
                distance_dict=cell_distance_dict,
                sampling_tasks=sampling_tasks,
                n_processes=n_threads,
            )
            sampled_cell_ids['selected_cell_id'] = [';'.join(sample_ids) for _, sample_ids in results.items()]

        else:
            sampling_tasks = [(obs_df, 1, row['cell_type'], row['n_cell'], row['class_by'],
                      sep_by_patient, sc_dataset, minimum_n_base)
                     for i, row in sampled_cell_ids.iterrows()]
            with multiprocessing.get_context('spawn').Pool(n_threads) as p:
                results = p.starmap(get_sample_id_random, sampling_tasks)

            sampled_cell_ids['selected_cell_id'] = [';'.join(i) for i in results]
        sampled_cell_ids.index.name = 'sample_id'
        sampled_cell_ids.sort_values(by=['sample_id', 'cell_type'], inplace=True)

        return sampled_cell_ids

    def _map_cell_id2exp(self, selected_cell_id: pd.DataFrame, sc_dataset: str = 'merged_sc_dataset',
                         simu_method: str = 'ave', cell_frac: pd.DataFrame = None,
                         gep_type='MCT', add_noise: bool = False, noise_params: tuple = ()) -> pd.DataFrame:
        """
        Aggregates single-cell expression to bulk.
        """
        # Select source dataframe
        if sc_dataset == 'sct_dataset':
            sc_ds_df = self.sct_dataset_df
        elif sc_dataset == 'merged_sc_dataset':
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
                print(f'   > Removing {len(high_zero_cells)} cells with zero ratio > {zero_ratio_threshold}')
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
        if os.path.exists(self.generated_cell_fraction_fp):
            gen_cell_frac = pd.read_csv(self.generated_cell_fraction_fp, index_col=0)
            gen_cell_frac = gen_cell_frac[~gen_cell_frac.index.duplicated(keep='first')].copy()
            if os.path.exists(self.generated_bulk_gep_csv_fp):
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
                    print(f'   The following intermediate generated result will be reused: \n'
                          f'{self.generated_cell_fraction_fp}, {self.generated_bulk_gep_csv_fp}.\n'
                          f'self.n_round will be reset to {self.n_round}, '
                          f'self.generated_bulk_gep_counter will be reset to {self.generated_bulk_gep_counter}')
                    if os.path.exists(self.ref_neighbor_counter_fp):
                        ref2n_neighbors = pd.read_csv(self.ref_neighbor_counter_fp, index_col=0).to_dict()['0']
                        self.ref_neighbor_counter = ref2n_neighbors.copy()
                else:
                    os.remove(self.generated_cell_fraction_fp)
                    os.remove(self.generated_bulk_gep_csv_fp)
                    os.remove(self.sampled_sc_cell_id_file_path)

    def _save_simulated_bulk_gep(self, gep: pd.DataFrame, cell_id: pd.DataFrame, cell_fraction: pd.DataFrame = None):

        if not os.path.exists(self.generated_bulk_gep_csv_fp):
            gep.to_csv(self.generated_bulk_gep_csv_fp, float_format='%g')
        else:
            gep.to_csv(self.generated_bulk_gep_csv_fp, header=False, mode='a', float_format='%g')

        if not os.path.exists(self.sampled_sc_cell_id_file_path):
            cell_id.to_csv(self.sampled_sc_cell_id_file_path)
        else:
            cell_id.to_csv(self.sampled_sc_cell_id_file_path, header=False, mode='a')

        if cell_fraction is not None:
            if not os.path.exists(self.generated_cell_fraction_fp):
                cell_fraction.to_csv(self.generated_cell_fraction_fp, float_format='%g', index_label='sample_id')
            else:
                cell_fraction.to_csv(self.generated_cell_fraction_fp, header=False, mode='a', float_format='%g')

        if self.ref_neighbor_counter:
            pd.DataFrame.from_dict(self.ref_neighbor_counter, orient='index').to_csv(self.ref_neighbor_counter_fp)

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
            print(f'   > Quantile distance of {self.filtering_quantile_upper * 100}% is: {self.q_dis_nn_ref_upper}')
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
        # print(f'   There are {ref_neighbors_within_radius.shape[0]} simulated GEPs left after filtering.')
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

    def __str__(self):
        """JSON representation of the object configuration."""
        config = {k: v for k, v in self.__dict__.items() if isinstance(v, (list, str, int, float, dict))}
        return f"{self.__class__.__name__}:\n{json.dumps(config, indent=2)}"


class SingleCellTypeGEPGenerator(BulkGEPGenerator):
    """
    Generator for Single Cell Type (SCT) GEPs or specific mixtures.
    Inherits from BulkGEPGenerator.
    """

    def __init__(self,
                 merged_sc_dataset_file_path: str | Path,
                 cell_type2subtype: Dict[str, List[str]],
                 sc_dataset_ids: List[str],
                 simu_bulk_dir: str | Path,
                 bulk_dataset_name: str,
                 zero_ratio_threshold: float = 0.97,
                 sc_dataset_gep_type: str = 'log_space',
                 subtype_col_name: Optional[str] = None,
                 cell_type_col_name: str = 'cell_type',
                 output_types: Optional[List[str]] = None):

        super().__init__(
            merged_sc_dataset_file_path=merged_sc_dataset_file_path,
            simu_bulk_dir=simu_bulk_dir,
            cell_type2subtype=cell_type2subtype,
            sc_dataset_ids=sc_dataset_ids,
            bulk_dataset_name=bulk_dataset_name,
            zero_ratio_threshold=zero_ratio_threshold,
            sct_dataset_file_path=None,
            sc_dataset_gep_type=sc_dataset_gep_type,
            subtype_col_name=subtype_col_name,
            cell_type_col_name=cell_type_col_name,
            output_types=output_types,
        )

        # Cell distance matrix
        self.cell_distance_dict: Dict[str, Tuple[np.ndarray, List[str]]] = {}

    def generate_samples(self,
                         n_sample_each_cell_type: int = 10000,
                         n_base_for_positive_samples: int = 100,
                         sample_type: str = 'positive',
                         sep_by_patient: bool = False,
                         simu_method: str = 'ave',
                         test_set: bool = False,
                         minimum_n_base: int = 1,
                         ref_gene_list_file_path: Optional[str] = None,
                         sampling_method: str = 'random',):
        """
        Generate simulated samples (Positive: pure cell types; Negative: mixtures).

        :param n_sample_each_cell_type: Number of samples per cell type.
        :param n_base_for_positive_samples: Number of single cells to average for 'positive' samples.
        :param sample_type: 'positive' (1 cell type) or 'negative' (mixed types).
        :param sep_by_patient: If True, sample from a single patient's cells.
        :param simu_method: 'ave' (average) or other simulation methods.
        :param test_set: If True, restricts cell types to major types only.
        :param minimum_n_base: Minimum required cells to form a sample.
        :param ref_gene_list_file_path: Path to a CSV containing the gene list to filter by.
        :param sampling_method: 'random' or 'n_neighbors' for cell selection.
            n_neighbors considers cell similarity to find (n_base - 1) nearest neighbors around 1 randomly selected cell within the same cell type.
        """
        if self.generated_bulk_gep_fp.exists():
            print(f'   > Previous result exists: {self.generated_bulk_gep_fp}')
            return

        # 1. Setup Configuration
        if test_set:
            self.cell_type_used = [k for k, v in self.cell_type2subtype.items() if k == v[0]]

        total_types = len(self.cell_type_used) + len(self.cell_subtype_used)
        self.n_samples = n_sample_each_cell_type * total_types

        # 2. Prepare Fractions (Manager Step)
        self._prepare_fractions(sample_type, n_sample_each_cell_type)

        # 3. Prepare Gene Expression Data (Manager Step)
        self._prepare_gene_expression(ref_gene_list_file_path)

        # 4. Calculate cell distance dict if using n_neighbors sampling
        if sampling_method == 'n_neighbors' and not self.cell_distance_dict:
            self._get_cell_distance_dict()

        # 5. Main Generation Loop
        chunk_size = max(100, int(n_sample_each_cell_type / 10)) if n_sample_each_cell_type > 1000 else n_sample_each_cell_type

        # Check for existing partial results
        existed_ids = pd.Index([])
        if self.generated_bulk_gep_csv_fp.exists():
            existed_ids = pd.read_csv(self.generated_bulk_gep_csv_fp, index_col=0, usecols=[0]).index

        print(f"   > Starting generation: {self.n_samples} samples, chunk size {chunk_size}")

        with pd.read_csv(self.generated_cell_fraction_fp, chunksize=chunk_size, index_col=0) as reader:
            for chunk_rows in tqdm(reader, desc="Processing Chunks"):
                # Filter already generated samples
                if not existed_ids.empty:
                    chunk_rows = chunk_rows.loc[~chunk_rows.index.isin(existed_ids)].copy()
                    # Update existed_ids by removing interactions
                    existed_ids = existed_ids.difference(chunk_rows.index)

                if not chunk_rows.empty:
                    self._process_chunk(
                        rows=chunk_rows,
                        sample_type=sample_type,
                        n_base=n_base_for_positive_samples,
                        min_n_base=minimum_n_base,
                        sep_by_patient=sep_by_patient,
                        simu_method=simu_method,
                        sampling_method_gep=sampling_method,
                    )

        # 6. Finalize
        self._finalize_dataset()

    def _prepare_fractions(self, sample_type: str, n_sample_each_cell_type: int):
        """Helper to generate or verify cell fraction file."""
        if not os.path.exists(self.generated_cell_fraction_fp):
            print(f'   > Generating cell proportions for {self.bulk_dataset_name} ({sample_type})...')
            generated_cell_frac = self.generate_frac_sc(
                sample_type=sample_type,
                sample_prefix=f'{self.bulk_dataset_name}_{sample_type[:3]}'
            )
            generated_cell_frac.to_csv(self.generated_cell_fraction_fp, float_format='%g', index_label='sample_id')
        else:
            print(f'   > Using existing cell fractions: {self.generated_cell_fraction_fp}')

    def _prepare_gene_expression(self, ref_gene_list_fp: Optional[str]):
        """Helper to align expression data with reference gene list."""
        if ref_gene_list_fp:
            print(f'   > Aligning with reference genes from: {ref_gene_list_fp}')
            ref_gene_list = pd.read_csv(ref_gene_list_fp, index_col=0).index.to_list()

            # Use ReadExp for safe handling
            sc_df_obj = ReadExp(self.merged_sc_dataset_df, exp_type='TPM')
            sc_df_obj.align_with_gene_list(ref_gene_list)
            # Update the main dataframe in memory
            self.merged_sc_dataset_df = sc_df_obj.get_exp()
            print(f'   > Dataset aligned. Genes: {self.merged_sc_dataset_df.shape[1]}')

    def _get_cell_distance_dict(self):
        """Calculate cell distance dictionary for n_neighbors sampling for all cell types in "output_types"."""
        print('   > Calculating cell distance dictionary for n_neighbors sampling...')
        self.cell_distance_dict = get_distance_matrices_by_class(
            adata_X=self.merged_sc_dataset.X.toarray(),
            obs_df=self.merged_sc_dataset_obs,
            class_by='output_types',
            metric='euclidean'
        )

    def _process_chunk(self, rows: pd.DataFrame, sample_type: str, n_base: int,
                       min_n_base: int, sep_by_patient: bool, simu_method: str = 'ave',
                       sampling_method_gep: str = 'random'):
        """
        Process a single chunk of samples: select cells -> aggregate -> save.
        """
        # Determine total cells to aggregate
        if sample_type == 'positive' and n_base > 1:
            total_cell_number = n_base
        else:
            # For negative samples or single-cell positive samples
            total_cell_number = 0

            # Filter Observation DataFrame based on sample type
        # current_obs_df = self.merged_sc_dataset_obs
        current_obs_df, gene_list_in_sc_ds, sc_dataset_key = self._load_sc_data_for_generation(simu_method)

        if sample_type == 'positive':
            # Identify the single active cell type in this chunk
            # Note: This assumes the chunk is homogeneous or we process row by row.
            # If chunk has mixed types, we must be careful.
            # The original code assumed 1 chunk = 1 type or handled it via argmax.
            # Here we iterate to be safe if a chunk contains multiple types (rare but possible).

            # Optimization: If all rows in chunk target the same cell type
            unique_targets = rows.idxmax(axis=1).unique()

            if len(unique_targets) == 1:
                target_type = unique_targets[0]
                current_obs_df = self._get_obs_for_type(target_type, min_n_base, n_base)
            else:
                # Mixed chunk: This is complex. For performance, we pass full obs
                # and let _sc_sampling handle filtering, or we split the chunk.
                # Passing full obs is safer here.
                pass

        # Sampling
        selected_cell_ids = self._sc_sampling(
            cell_frac=rows,
            total_cell_number=total_cell_number,
            obs_df=current_obs_df,
            sep_by_patient=sep_by_patient,
            minimum_n_base=min_n_base,
            sc_dataset=sc_dataset_key,
            cell_distance_dict=self.cell_distance_dict,
            sampling_method_gep=sampling_method_gep,
        )

        # Aggregation
        simulated_gep = self._map_cell_id2exp(
            selected_cell_id=selected_cell_ids,
            simu_method=simu_method,
            cell_frac=rows,
            sc_dataset='merged_sc_dataset',
            gep_type='SCT'
        )

        # Save
        if simulated_gep is not None:
            simulated_gep = log2_transform(simulated_gep)
            self.generated_bulk_gep_counter += simulated_gep.shape[0]

            # Ensure alignment
            valid_indices = simulated_gep.index
            selected_cell_ids = selected_cell_ids.loc[valid_indices].copy()

            self._save_simulated_bulk_gep(gep=simulated_gep, cell_id=selected_cell_ids)

    def _get_obs_for_type(self, cell_type: str, min_n: int, warn_n: int) -> pd.DataFrame:
        """Helper to filter obs for a specific cell type."""
        col_name = self.subtype_col_name if cell_type in self.cell_subtype_used else self.cell_type_col_name

        if col_name not in self.merged_sc_dataset_obs.columns:
            raise ValueError(f"Column '{col_name}' not found in obs.")

        filtered_obs = self.merged_sc_dataset_obs.loc[
            self.merged_sc_dataset_obs[col_name] == cell_type
            ].copy()

        if filtered_obs.shape[0] < min_n:
            raise ValueError(f"Insufficient cells for {cell_type}: {filtered_obs.shape[0]} < {min_n}")
        elif filtered_obs.shape[0] <= warn_n:
            print(f"   > Warning: {cell_type} has only {filtered_obs.shape[0]} cells (requested {warn_n}).")

        return filtered_obs

    def _finalize_dataset(self):
        """Create final H5AD."""
        data_info = f'Simulated {self.generated_bulk_gep_counter} SCT GEPs, log2(TPM + 1)'
        create_h5ad_dataset(
            simulated_bulk_exp_file_path=self.generated_bulk_gep_csv_fp,
            cell_fraction_file_path=self.generated_cell_fraction_fp,
            dataset_info=data_info,
            result_file_path=self.generated_bulk_gep_fp
        )

    def generate_frac_sc(self, sample_prefix: str = None, sample_type: str = 'positive') -> pd.DataFrame:
        """
        Generate cell fractions for single cell samples.

        Positive: 100% of one cell type.
        Negative: Mixture of >= 2 cell types.

        :param sample_prefix: Prefix for sample names.
        :param sample_type: 'positive' or 'negative'.
        :return: DataFrame (samples x cell_types).
        """
        if sample_prefix is None:
            sample_prefix = f's_sc_{sample_type}'

        cols = self.cell_type_used + self.cell_subtype_used
        n_cell_types = len(cols)

        # Initialize empty DataFrame
        # Note: creating a huge zero matrix can be memory intensive.
        # If n_samples is huge, consider generating in chunks.
        generated_frac_df = pd.DataFrame(
            0.0,
            index=[f'{sample_prefix}_{i}' for i in range(self.n_samples)],
            columns=cols
        )

        if sample_type == 'positive':
            # Evenly distribute samples across all cell types
            samples_per_type = self.n_samples // n_cell_types

            for i, cell_type in enumerate(cols):
                start_idx = i * samples_per_type
                # Handle remainder for the last cell type
                end_idx = self.n_samples if i == n_cell_types - 1 else (i + 1) * samples_per_type

                generated_frac_df.iloc[start_idx:end_idx, i] = 1.0

        else:  # Negative samples (Mixtures)
            # Distribute samples across different "complexity levels" (2 types, 3 types ... N types)
            # Range: 2 to n_cell_types
            complexity_levels = list(range(2, n_cell_types + 1))
            n_levels = len(complexity_levels)
            samples_per_level = self.n_samples // n_levels

            current_idx = 0

            for level_i, n_ct in enumerate(complexity_levels):
                # Determine range for this complexity level
                end_idx = self.n_samples if level_i == n_levels - 1 else current_idx + samples_per_level

                n_rows = end_idx - current_idx
                if n_rows <= 0: break

                frac_val = 1.0 / n_ct

                # Vectorized generation for this block isn't easy due to random choice per row,
                # but we can optimize the loop.
                for row_idx in range(current_idx, end_idx):
                    if n_ct < n_cell_types:
                        chosen_indices = np.random.choice(n_cell_types, size=n_ct, replace=False)
                    else:
                        chosen_indices = np.arange(n_cell_types)

                    generated_frac_df.iloc[row_idx, chosen_indices] = frac_val

                current_idx = end_idx

        return generated_frac_df.round(4)
