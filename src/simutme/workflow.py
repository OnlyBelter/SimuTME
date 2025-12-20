import os
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump, load
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from .utility import do_pca_analysis, plot_pca
from .utility import cal_corr_gene_exp_with_cell_frac
# Assuming these are existing imports in your project
from .utility import ReadH5AD, ReadExp, create_h5ad_dataset
from .utility import check_dir, non_log2cpm, log2_transform, get_gene_list_filtered_by_quantile_range
from .simulation import BulkGEPGenerator, SingleCellTypeGEPGenerator


# --- 1. Configuration Management ---
@dataclass
class SimuConfig:
    """Stores all simulation-related paths and global parameters."""
    # Path configuration
    simu_bulk_dir: Path
    sct_dataset_file_path: Path
    tcga2cancer_type_file_path: Path
    tcga_merged_tpm_file_path: Path
    high_corr_gene_file_path: Path
    merged_sc_dataset_file_path: Path

    # Core constants
    sc_dataset_ids: List[str]
    cell_type2subtypes: Dict[str, List[str]]
    total_rna_coefficient: Dict[str, float]
    n_base: int
    output_types: List[str]

    # Filtering parameters
    gene_quantile_range: List[float]
    gep_filtering_quantile: Tuple[float, float]
    pca_n_components: float
    gene_list_type: str

    # Cached attributes
    _high_corr_gene_list: List[str] = field(init=False, default=None)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Flatten the YAML structure to match the Dataclass fields
        # Combining paths, constants, and parameters into one dictionary
        config_data = {
            **data['paths'],
            **data['constants'],
            **data['parameters']
        }

        # Convert list to tuple for gep_filtering_quantile if necessary
        if 'gep_filtering_quantile' in config_data:
            config_data['gep_filtering_quantile'] = tuple(config_data['gep_filtering_quantile'])

        return cls(**config_data)

    @property
    def high_corr_genes(self) -> List[str]:
        """Lazy load high correlation genes"""
        if self._high_corr_gene_list is None:
            if os.path.exists(self.high_corr_gene_file_path):
                df = pd.read_csv(self.high_corr_gene_file_path)
                self._high_corr_gene_list = df['gene_name'].to_list()
            else:
                logging.warning(f"Warning: High corr gene file not found: {self.high_corr_gene_file_path}")
                self._high_corr_gene_list = []
        return self._high_corr_gene_list


# --- 2. Main Workflow Function ---
def run_bulk_simulation(dataset_name: str,
                        params: Dict[str, Any],
                        config: SimuConfig,
                        log_file_path: str = None) -> str:
    """
    Executes the complete simulation workflow for a single dataset: Generation -> Filtering -> PCA Analysis.

    :param dataset_name: Name of the dataset (e.g., 'Mixed_N10K_segment')
    :param params: Dataset-specific parameters (sc_dataset_ids, n_samples, etc.)
    :param config: Global configuration object (paths, constants)
    :param log_file_path: Path to the log file
    :return: The file path of the final generated .h5ad file
    """
    logging.info(f"\n=== Starting Simulation for {dataset_name} ===")

    # 1. Initialize generator
    simu_bulk_dir = Path(config.simu_bulk_dir)
    n_sc_datasets = len(params.get('sc_dataset_ids', config.sc_dataset_ids))
    n_base = config.n_base
    sc_dataset_dir = simu_bulk_dir / f'generated_sc_dataset_{n_sc_datasets}ds_n_base{n_base}_all_subtypes'
    if params.get('dataset_type', 'mixed').lower() == 'sct':  # single-cell-type dataset
        logging.info(f"   > Using Single-Cell-Type GEP Generation for {dataset_name}")
        sct_generator = SingleCellTypeGEPGenerator(
            simu_bulk_dir=sc_dataset_dir,
            cell_type2subtype=config.cell_type2subtypes,
            sc_dataset_ids=config.sc_dataset_ids,
            bulk_dataset_name=dataset_name,
            merged_sc_dataset_file_path=config.merged_sc_dataset_file_path,
            zero_ratio_threshold=params.get('zero_ratio_threshold', 0.97),
            cell_type_col_name=params.get('cell_type_col', 'cell_type'),
            subtype_col_name=params.get('cell_subtype_col', 'cell_subtype'),
            output_types=config.output_types,
        )
        generated_sct_fp = sct_generator.generated_bulk_gep_fp
        if not generated_sct_fp.exists():
            sct_generator.generate_samples(n_sample_each_cell_type=params['n_samples'],
                                           sample_type='positive',
                                           simu_method='ave',  # average multiple cells to get one sct GEP
                                           sep_by_patient=False,
                                           n_base_for_positive_samples=n_base,
                                           sampling_method=params.get('sampling_method_gep', 'random'), # sampling method for finding single cells
                                           )
            logging.info(f"   > Single-Cell-Type Dataset Generation Completed for {dataset_name}")
        else:
            logging.info(f"   > Single-Cell-Type GEP file exists: {generated_sct_fp}")
        return str(generated_sct_fp)
    else:  # mixed bulk GEPs
        logging.info(f"   > Using Bulk GEP (with Mixed Cell Type) Generation for Dataset {dataset_name}")
        if params['filtering']:
            filtering_str = f'with_filtering_{config.gep_filtering_quantile[1]}'
        else:
            filtering_str = 'no_filtering'
        sampling_method = params['sampling_method_prop']  # sampling method for cell proportions
        simu_bulk_dir = simu_bulk_dir / f'{sampling_method}_{n_sc_datasets}ds_n_base{n_base}_{filtering_str}_median_gep'
        bulk_generator = BulkGEPGenerator(
            simu_bulk_dir=simu_bulk_dir,
            merged_sc_dataset_file_path=config.merged_sc_dataset_file_path,
            cell_type2subtype=params.get('cell_type2subtypes', config.cell_type2subtypes),
            sc_dataset_ids=config.sc_dataset_ids,
            bulk_dataset_name=dataset_name,
            sct_dataset_file_path=config.sct_dataset_file_path,
            check_basic_info=False,
            tcga2cancer_type_file_path=config.tcga2cancer_type_file_path,
            total_rna_coefficient=config.total_rna_coefficient,
            cell_type_col_name='cell_type',
            subtype_col_name='cell_type'
        )

        generated_bulk_gep_fp = bulk_generator.generated_bulk_gep_fp

        # 2. Generate GEP (if it doesn't exist)
        if not os.path.exists(generated_bulk_gep_fp):
            bulk_generator.generate_gep(
                n_samples=params['n_samples'],
                simu_method='mul',  # matrix multiplication by cell proportions and sampled GEPs to get bulk GEPs
                sampling_method_prop=params['sampling_method_prop'],
                reference_file=config.tcga_merged_tpm_file_path,
                ref_exp_type='TPM',
                filtering=params.get('filtering', False),
                filtering_ref_types=params.get('filtering_ref_types', None),
                gep_filtering_quantile=config.gep_filtering_quantile,
                n_threads=5,
                log_file_path=log_file_path,
                show_filtering_info=False,
                filtering_method='median_gep',
                cell_prop_prior=params.get('cell_prop_prior', None),
                filtering_in_pca_space=params.get('filtering_in_pca_space', True),
                norm_ord=1,
                pca_n_components=config.pca_n_components
            )
        else:
            logging.info(f"   > GEP file exists: {generated_bulk_gep_fp}")

        # 3. Post-processing: Gene-level filtering and PCA (if needed)
        if params.get('filtering', False):
            _run_post_processing(dataset_name, generated_bulk_gep_fp, config)

        return generated_bulk_gep_fp


# --- 3. Internal Helper Functions ---
def _run_post_processing(dataset_name: str,
                         generated_bulk_gep_fp: str,
                         config: SimuConfig):
    """Handles complex gene filtering, intersection logic, and PCA plotting."""

    # Define output filenames
    replace_by = f'_filtered_by_{config.gene_list_type}.h5ad'
    filtered_file_path = generated_bulk_gep_fp.replace('.h5ad', replace_by)

    if os.path.exists(filtered_file_path):
        logging.info(f"   > Filtered file exists: {filtered_file_path}")
        return

    logging.info(f"   > Running post-processing (Gene Filtering & PCA) for {dataset_name}...")
    current_result_dir = os.path.join(config.simu_bulk_dir, dataset_name)
    check_dir(current_result_dir)

    # A. Get/Generate Quantile Gene List
    gene_list = config.high_corr_genes.copy()  # Start with high corr genes

    if 'quantile_range' in config.gene_list_type:
        q_names = [str(x) for x in config.gene_quantile_range]  # Simplified naming logic
        # Note: Filename generation logic is simplified here; you may need to adjust q_names based on original code.
        csv_name = f'gene_list_filtered_by_{config.gene_list_type}_{q_names[0]}_{q_names[-1]}.csv'
        gene_list_file_path = os.path.join(current_result_dir, csv_name)

        if not os.path.exists(gene_list_file_path):
            logging.info(f'      Generating gene list: {gene_list_file_path}')
            quantile_gene_list = get_gene_list_for_filtering(
                bulk_exp_file=generated_bulk_gep_fp,
                filtering_type='quantile_range',
                tcga_file=config.tcga_merged_tpm_file_path,
                quantile_range=config.gene_quantile_range,
                result_file_path=gene_list_file_path,
                q_col_name=q_names
            )
        else:
            logging.info(f'      Loading existing gene list: {gene_list_file_path}')
            quantile_gene_list = pd.read_csv(gene_list_file_path)['gene_name'].to_list()

        # Take intersection
        gene_list = [g for g in gene_list if g in quantile_gene_list]

    # B. Prepare data for PCA
    bulk_exp_obj = ReadH5AD(generated_bulk_gep_fp)
    bulk_exp = bulk_exp_obj.get_df()
    bulk_exp_cell_frac = bulk_exp_obj.get_cell_fraction()

    # Note: Reading large TCGA files is time-consuming.
    # If calling in a loop, consider passing pre-loaded data or using system caching.
    tcga_exp = ReadExp(config.tcga_merged_tpm_file_path, exp_type='TPM').get_exp()

    pca_model_fp = os.path.join(current_result_dir,
                                f'both_TCGA_and_simu_{dataset_name}_PCA_{config.gene_list_type}.joblib')
    pca_data_fp = os.path.join(current_result_dir, f'{dataset_name}_PCA_with_TCGA_{config.gene_list_type}.csv')

    # C. Execute filtering and plotting
    filtering_by_gene_list_and_pca_plot(
        bulk_exp=bulk_exp,
        tcga_exp=tcga_exp,
        gene_list=gene_list,
        result_dir=current_result_dir,
        n_components=2,
        simu_dataset_name=dataset_name,
        pca_model_name_postfix=config.gene_list_type,
        pca_model_file_path=pca_model_fp,
        pca_data_file_path=pca_data_fp,
        h5ad_file_path=filtered_file_path,
        cell_frac_file=bulk_exp_cell_frac,
        figsize=(5, 5)
    )


def get_gene_list_for_filtering(bulk_exp_file: str | Path, tcga_file: str | Path, result_file_path: str | Path, q_col_name: list = None,
                                filtering_type: str = 'quantile_range',
                                corr_threshold: float = 0.3, n_gene_max: int = 1000,
                                corr_result_fp: str = None, quantile_range: list = None) -> list:
    """
    Perform gene-level filtering based on the specific filtering type.

    :param bulk_exp_file: str, Path to the simulated bulk expression file in log2(TPM + 1) format to be filtered.
    :param tcga_file: str, Path to the TCGA reference file in TPM format
    :param result_file_path: str, Path where the filtered gene list will be saved.
    :param q_col_name: list, Optional. Column names for quantile ranges.
    :param filtering_type: str, Specify the method of filtering. Options include: 'high_corr_gene', 'quantile_range', 'all_genes', 'high_corr_gene_and_quantile_range'
      - 'high_corr_gene': Select genes that their expression values have high correlation with the cell proportions of any cell type
      - 'quantile_range': Select genes with their median of expression values within the [q_5, q_95] quantile range of their counterparts in TCGA (default)
    :param corr_result_fp: str, path to the result file after 'high_corr_gene' filtering
    :param quantile_range: tuple,  (lower_quantile, median_quantile, upper_quantile). Genes are removed if their median expression in simulated bulk GEPs is outside this range in TCGA data
        of the corresponding gene in TCGA dataset will be removed


    :param corr_threshold: float, correlation threshold for gene filtering
    :param n_gene_max: int, maximum number of genes to select for each cell type during 'high_corr_gene' filtering
    :return: list, filtered gene list
    """

    assert filtering_type in ['high_corr_gene', 'quantile_range', 'all_genes', 'high_corr_gene_and_quantile_range'], \
        f'filtering_type: {filtering_type} is not supported, only support high_corr_gene, quantile_range, all_genes, ' \
        f'high_corr_gene_and_quantile_range'
    if not os.path.exists(result_file_path):
        bulk_exp = ReadH5AD(bulk_exp_file).get_df(convert_to_tpm=True)
        tcga_obj = ReadExp(tcga_file, exp_type='TPM')
        tcga_obj.align_with_gene_list(gene_list=bulk_exp.columns.to_list())  # common genes will be kept
        tcga_obj.to_log2cpm1p()
        tcga = tcga_obj.get_exp()
        bulk_exp = bulk_exp.loc[:, tcga.columns].copy()
        bulk_exp = non_log2cpm(bulk_exp)
        bulk_exp = log2_transform(bulk_exp)
        assert (len(tcga.columns) == len(bulk_exp.columns)) and np.all(tcga.columns == bulk_exp.columns), \
            f'Columns of bulk_exp and tcga are not the same'
        gene_list = []
        if 'high_corr_gene' in filtering_type:
            h5_obj = ReadH5AD(bulk_exp_file)
            gene_exp = h5_obj.get_df(convert_to_tpm=True)
            cell_frac = h5_obj.get_cell_fraction()
            corr_df = cal_corr_gene_exp_with_cell_frac(gene_exp=gene_exp, cell_frac=cell_frac,
                                                       result_file_path=corr_result_fp, filtered_by_corr=corr_threshold,
                                                       filter_by_num=n_gene_max)
            del gene_exp
            gene_list = corr_df.index.to_list()
            logging.info(f'{len(gene_list)} genes are selected by high correlation')
        if 'quantile_range' in filtering_type:
            # both bulk_exp and tcga are in log-space
            gene_list_qr = get_gene_list_filtered_by_quantile_range(bulk_exp=bulk_exp, tcga_exp=tcga,
                                                                    quantile_range=quantile_range,
                                                                    q_col_name=q_col_name)
            logging.info(f'{len(gene_list_qr)} genes are selected by quantile range')
            if len(gene_list) > 0:  # if high-correlation genes exist, then filter by high-correlation genes
                gene_list = [gene for gene in gene_list if gene in gene_list_qr]
                logging.info(f'{len(gene_list)} genes are selected by both high correlation and quantile range')
            else:
                gene_list = gene_list_qr
        if filtering_type == 'all_genes':
            gene_list = bulk_exp.columns.to_list()
            logging.info(f'All {len(gene_list)} genes will be used')

        gene_list_df = pd.DataFrame(columns=['gene_name'])
        gene_list_df['gene_name'] = gene_list
        gene_list_df.to_csv(result_file_path, index=False)
    else:
        logging.info(f'Loading the gene list from file: {result_file_path}')
        gene_list_df = pd.read_csv(result_file_path)
        gene_list = gene_list_df['gene_name'].to_list()
    return gene_list


def filtering_by_gene_list_and_pca_plot(bulk_exp: pd.DataFrame, tcga_exp: pd.DataFrame, gene_list: list,
                                        result_dir: str, simu_dataset_name: str,
                                        n_components=5, pca_model_name_postfix='', bulk_exp_type='log_space',
                                        tcga_exp_type='TPM', pca_model_file_path=None, pca_data_file_path=None,
                                        h5ad_file_path=None, cell_frac_file: pd.DataFrame = None, figsize=(5, 5),
                                        if_plot_pca: bool = False):
    """
    Applying gene-level filtering based on a specific gene list to simulated bulk GEPs.
    After filtering, the bulk GEPs are rescaled to log2(TPM+1).
    And perform PCA and plot the results for both filtered bulk GEPs and TCGA samples.

    :param bulk_exp: pd.DataFrame, Simulated bulk expression data in log2cpm1p format.
    :param tcga_exp: pd.DataFrame, TCGA expression data in TPM format.
    :param gene_list: list, List of gene names used to filter the bulk GEPs.
    :param result_dir: str, Directory where results will be saved.
    :param simu_dataset_name: str, Name of the simulated dataset
    :param n_components: int, Number of PCA components to compute. Must be >= 2.
    :param pca_model_name_postfix: str, Postfix for naming the PCA model file.
    :param bulk_exp_type: str, Type of bulk GEPs, either 'TPM' or 'log_space'.
    :param tcga_exp_type: str, Type of TCGA samples, either 'TPM' or 'log_space'.
    :param pca_model_file_path: str, Path to save the fitted PCA model.
    :param pca_data_file_path: str, Path to save the PCA-transformed data.
    :param h5ad_file_path: str, Path to save the filtered bulk GEPs as a .h5ad file, if not None.
    :param cell_frac_file: pd.DataFrame, Cell proportion matrix used to generate simulated bulk GEPs. Required if h5ad_file_path is specified.
    :param figsize: tuple, Size of the figure for PCA plotting (width, height).
    :param if_plot_pca: boolean, if True plot PCA components using both simulated bulk GEPs and TCGA samples.
    :return: None
    """
    bulk_obj = ReadExp(bulk_exp, exp_type=bulk_exp_type)
    bulk_obj.align_with_gene_list(gene_list=gene_list)
    bulk_exp = bulk_obj.get_exp()

    # PCA and plot
    if if_plot_pca:
        tcga_obj = ReadExp(tcga_exp, exp_type=tcga_exp_type)
        tcga_obj.align_with_gene_list(gene_list=gene_list)
        tcga_obj.to_log2cpm1p()
        tcga_exp = tcga_obj.get_exp()

        check_dir(result_dir)
        assert n_components >= 2, 'n_components must be >= 2'
        if not os.path.exists(pca_data_file_path):
            # combine both simulated bulk cell GEPs and TCGA dataset together
            simu_bulk_with_tcga = pd.concat([bulk_exp, tcga_exp])
            pca_model = do_pca_analysis(exp_df=simu_bulk_with_tcga, n_components=n_components,
                                        pca_result_fp=pca_model_file_path)
            pcs = pca_model.transform(simu_bulk_with_tcga)
            if n_components >= 3:
                pca_df = pd.DataFrame(pcs[:, range(3)], index=simu_bulk_with_tcga.index, columns=['PC1', 'PC2', 'PC3'])
            else:
                pca_df = pd.DataFrame(pcs[:, range(2)], index=simu_bulk_with_tcga.index, columns=['PC1', 'PC2'])
            pca_df.to_csv(pca_data_file_path)
        else:
            logging.info(f'{pca_data_file_path} already exists, skip PCA analysis')
            pca_df = pd.read_csv(pca_data_file_path, index_col=0)
            pca_model = load(pca_model_file_path)

        # plot
        title = f'{simu_dataset_name}_PCA_with_TCGA_{pca_model_name_postfix}'
        color_code = np.array([simu_dataset_name] * bulk_exp.shape[0] + ['TCGA'] * tcga_exp.shape[0])
        # cumsum = np.cumsum(pca_model.explained_variance_ratio_)
        # pca_df['class'] = color_code
        plot_pca(data=pca_df, figsize=figsize,
                 result_fp=os.path.join(result_dir, title + '.png'),
                 color_code=color_code, explained_variance_ratio=pca_model.explained_variance_ratio_)

    if h5ad_file_path is not None:
        assert cell_frac_file is not None, 'cell_frac_file should be provided if h5ad_file_path is not None'
        logging.info(f'Saving filtered bulk exp to file: {h5ad_file_path}')
        if not os.path.exists(h5ad_file_path):
            logging.info('Saving as .h5ad file...')
            dataset_info = f'Some genes were removed from this dataset for increasing the similarity ' \
                           f'between TCGA and simulated bulk cell GEPs. This similarity was evaluated by PCA analysis.'
            create_h5ad_dataset(simulated_bulk_exp_file_path=bulk_exp,
                                cell_fraction_file_path=cell_frac_file, dataset_info=dataset_info,
                                result_file_path=h5ad_file_path)
