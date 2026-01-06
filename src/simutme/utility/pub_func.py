import os
import csv
import time
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import matplotlib as mpl
import multiprocessing as mp
from joblib import dump, load
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from anndata import AnnData, read_h5ad
from sklearn.metrics import euclidean_distances
from typing import Union, List, Dict, Optional, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

from ..config import DEFAULT_CELL_TYPES, DEFAULT_CORE_MARKER_GENES


def set_fig_style(font_family: str = None, font_size: int = None):
    """
    Set matplotlib figure style to 'SciencePlots' or fallback to seaborn.
    """
    # fig, ax = plt.subplots()

    sns.set_style("white")

    try:
        # Try using SciencePlots
        plt.style.use(['science', 'no-latex'])
    except (OSError, ImportError):
        print('Warning: "SciencePlots" style not found. Falling back to seaborn default. '
              'See https://github.com/garrettj403/SciencePlots')
        sns.set_theme(palette='muted', font_scale=1.5)

    # Set global parameters
    params = {
        'figure.dpi': 300,
        'figure.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none'
    }
    mpl.rcParams.update(params)

    if font_family:
        mpl.rcParams['font.family'] = font_family
    if font_size:
        mpl.rcParams['font.size'] = font_size

    # plt.close('all')


def log_exp2cpm(exp_df: Union[pd.DataFrame, np.ndarray], log_base: int = 2, correct: int = 1) -> Union[
    pd.DataFrame, np.ndarray]:
    """
    Convert log-space expression (e.g., log2(CPM+1)) back to linear space (CPM/TPM).
    :param exp_df: Log-transformed expression data (DataFrame or ndarray). Assumes rows are samples, columns are genes.

    :param log_base: Base of the logarithm used in the transformation.

    :param correct: The constant added before log transformation (e.g., 1 in log2(CPM + 1)).

    :return: CPM/TPM normalized expression data.

    """
    # 1. Inverse Log
    exp = np.power(log_base, exp_df) - correct

    # 2. Normalize
    if isinstance(exp, pd.DataFrame):
        cpm = exp.div(exp.sum(axis=1), axis=0) * 1e6
    else:
        # Numpy array
        cpm = exp / exp.sum(axis=1, keepdims=True) * 1e6

    return cpm


def non_log2log_cpm(input_data: Union[str, pd.DataFrame], result_file_path: str = None,
                    transpose: bool = True, correct: int = 1) -> Optional[pd.DataFrame]:
    """
    Convert non-log expression data to log2(CPM + 1).
    :param input_data: Input expression data (file path or DataFrame). Rows are genes, columns are samples by default.
    :param result_file_path: If provided, save the result to this file path.
    :param transpose: Whether to transpose the input data (default True, to make samples by genes).
    :param correct: The constant to add before log transformation (default 1).
    """
    bulk_exp = pd.DataFrame()

    if isinstance(input_data, (str, Path)):
        bulk_exp = read_df(input_data)
    elif isinstance(input_data, pd.DataFrame):
        bulk_exp = input_data.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")

    if transpose:
        bulk_exp = bulk_exp.T  # transpose to samples by genes

    # Calculate CPM
    bulk_exp = non_log2cpm(bulk_exp)

    # Log transform
    bulk_exp = np.log2(bulk_exp + correct)

    # Output
    if result_file_path:
        bulk_exp.round(3).to_csv(result_file_path)
        return None
    else:
        return bulk_exp.round(3)


def non_log2cpm(exp_df: pd.DataFrame, sum_exp: float = 1e6) -> pd.DataFrame:
    """
    Normalize gene expression to CPM / TPM for non-log space.
    :param exp_df: Raw expression DataFrame. Assumes rows are samples, columns are genes.

    """
    return exp_df.div(exp_df.sum(axis=1), axis=0) * sum_exp


def log2_transform(df):
    """
    log2 transform expression values (plus 1)
    :param df:
    :return:
    """
    df = df.astype(np.float64)
    df = np.log2(df + 1)
    df = df.astype(np.float32)
    return df


def read_df(df_file: Union[str, Path, pd.DataFrame, np.ndarray], index_col: Union[str, int] = 0) -> Union[
    pd.DataFrame, np.ndarray]:
    """
    Robust reader function.
    """
    if isinstance(df_file, (str, Path)):
        sep = get_sep(str(df_file))
        return pd.read_csv(df_file, index_col=index_col, sep=sep)

    elif isinstance(df_file, pd.DataFrame):
        return df_file
    elif isinstance(df_file, np.ndarray):
        return df_file
    else:
        raise TypeError(f'Unsupported type: {type(df_file)}')


def get_sep(file_path: str, comment: str = None) -> str:
    """
    Detect separator using Python's built-in csv.Sniffer for robustness.
    """
    try:
        with open(file_path, 'r') as f:
            # Read 4096 bytes for guessing, avoiding large file read
            sample = f.read(4096)
            # Filter out comment lines if specified
            if comment:
                lines = [line for line in sample.splitlines() if not line.startswith(comment)]
                sample = '\n'.join(lines)

            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=[',', '\t'])
            return dialect.delimiter
    except Exception:
        # Back to simple logic if Sniffer fails (e.g., only one column of data)
        # Return tab by default, or comma if first line contains commas
        return ',' if file_path.endswith('.csv') else '\t'


def get_inx2cell_type(cell_type_list: List[str] = None) -> Dict[int, str]:
    if cell_type_list is None:
        cell_type_list = DEFAULT_CELL_TYPES

    inx2cell_type = {i: ct for i, ct in enumerate(cell_type_list)}
    inx2cell_type[-1] = 'Neg'
    return inx2cell_type


def map_cell_id2exp(sc_exp, selected_cell_id):
    """

    :param sc_exp: a AnnData object, log2(CPM + 1), samples by genes
    :param selected_cell_id: a dataFrame which contains cell_type, n_cell, selected_cell_id
    :return: a DataFrame, log2(CPM + 1), samples by genes
    """
    # sc_exp = an.read_h5ad(sc_exp)
    adata_df = pd.DataFrame(sc_exp.X.A, index=sc_exp.obs.index, columns=sc_exp.var.index)
    adata_df = np.power(2, adata_df) - 1  # convert to non-log values
    simulated_exp = {}

    for sample_id, group in selected_cell_id.groupby(by=selected_cell_id.index):
        cell_ids = []
        for i, row in group.iterrows():
            if row['n_cell'] > 0:
                cell_ids += row['selected_cell_id'].split(';')
        current_merged = adata_df.loc[cell_ids, :].copy()
        simulated_exp[sample_id] = current_merged.mean(axis=0)  # single simulated bulk expression profile

    simulated_exp_df = pd.DataFrame.from_dict(data=simulated_exp, orient='index')
    # simu_adata = an.AnnData(simulated_exp_df)
    # simu_adata.X = np.log2(simu_adata.X + 1)  # log2(CPM + 1)
    # simulated_exp_df.rename(index={i: j for i, j in enumerate(filtered_single_cell_exp.index)}, inplace=True)
    log2cpm = np.log2(simulated_exp_df.values + 1)
    return pd.DataFrame(log2cpm, index=simulated_exp_df.index, columns=sc_exp.var.index).round(2)


def aggregate_marker_gene_exp(exp_df: pd.DataFrame, marker_genes: dict = None,
                              agg_methods: dict = None, return_ratio: bool = False,
                              show_marker_gene: bool = True):
    """
    aggregate marker genes by the corresponding method in agg_methods (mean or max)
    :param exp_df: a DataFrame of gene expression profiles, TPM, samples by genes
    :param marker_genes:
    :param agg_methods: aggregate multiple marker genes of a cell type by `mean` or `max`
    :param return_ratio: return the ratio of aggregated marker gene exp for each cell type if True
    :param show_marker_gene: whether printing marker genes or not
    """
    if marker_genes is None:
        marker_genes = DEFAULT_CORE_MARKER_GENES.copy()
    if agg_methods is None:
        agg_methods = {}
    for k, _ in marker_genes.items():
        if k not in agg_methods:
            agg_methods[k] = 'mean'

    cell_type2marker_exp = {}
    for ct, marker in marker_genes.items():
        method = agg_methods[ct]
        current_tpm = exp_df.loc[:, exp_df.columns.isin(marker)].copy()  # sample by gene
        current_valid_marker = current_tpm.columns.to_list()
        if len(current_valid_marker) > 0:
            current_marker_str = ', '.join(current_valid_marker)
            if show_marker_gene:
                print(f'   Using {len(current_valid_marker)} marker genes for cell type {ct}: {current_marker_str}')
            # cell_type2mean_exp[ct + '_marker_mean'] = current_tpm.mean(axis=0)
            cell_type2marker_exp[ct + f'_marker_{method}'] = cal_exp_by_gene_list(exp_df=exp_df, method=method,
                                                                                  gene_list=current_valid_marker,
                                                                                  min_exp_value=1)
        else:
            Warning(f'No any marker genes in bulk cell TPM for cell type {ct}, '
                    f'this cell type will be ignored in later analysis.')
    marker_exp = pd.DataFrame.from_dict(cell_type2marker_exp, orient='columns')
    if return_ratio:
        return marker_exp / marker_exp.sum(axis=1).values.reshape(-1, 1)
    else:
        return marker_exp.round(3)


def cal_exp_by_gene_list(exp_df, gene_list, min_exp_value=0, method='mean'):
    """
    calculate mean expression (or max) of a gene list (for single cell type)
    :param exp_df: sample by gene, usually in CPM / TPM
    :param gene_list: a list of genes which all are included in exp_df
    :param min_exp_value: min mean expression of the gene list
    :param method: mean or max
    :return: mean or max expression value of marker genes for each sample
    """
    for gene in gene_list:
        if gene not in exp_df.columns:
            raise KeyError(f'Gene {gene} not include in exp_df')
    current_exp = exp_df.loc[:, gene_list].copy()
    if method == 'mean':
        current_value = current_exp.mean(axis=1)
    elif method == 'max':
        current_value = current_exp.max(axis=1)
    else:
        raise KeyError('Only "mean" or "max" allowed')
    current_value[current_value < min_exp_value] = min_exp_value
    if exp_df.shape[0] == 1:
        return round(float(current_value), 3)  # float
    return current_value.round(3)  # pd.Series


def get_cell_num(cell_type_frac, total_num=500):
    """
    calculate the number of cells to choose for each cell type based on the total number of cells
    and fraction of each cell type
    :param cell_type_frac: dataFrame
        the fraction of each cell type, sum to 1 for each row, samples by cell types
    :param total_num: the total number of cells contributing to one simulated RNA-seq sample
    :return:
    """
    assert type(cell_type_frac) is pd.DataFrame
    cell_num = cell_type_frac * total_num
    if total_num == 1:  # assign 1 for all cell types for matrix multiplication
        cell_num = cell_type_frac * 0 + 1
    elif total_num == 0:
        cell_num = np.ceil(cell_type_frac)  # assign 1 for the cell types with non-zero cell fractions
    return cell_num.round(0).astype(int)


def create_h5ad_dataset(simulated_bulk_exp_file_path, cell_fraction_file_path: str | Path | pd.DataFrame,
                        dataset_info, result_file_path: str | Path,
                        filtering: bool = False, merge_t_cell: bool = False, gep_type='bulk'):
    """
    create .h5ad file according to cell fraction and simulated bulk expression profiles
    https://anndata.readthedocs.io/en/latest/index.html
    :param simulated_bulk_exp_file_path: simulated bulk expression profile, samples by genes
        .csv file or .h5ad file
    :param cell_fraction_file_path: .csv file or a DataFrame, samples by cell types
    :param dataset_info: str
    :param result_file_path:
    :param filtering: if filtered by marker ratio
    :param merge_t_cell: whether merge the cell fraction of CD8 T and CD4 T cells
    :param gep_type: bulk (mixture contains >= 2 cell types) / sct (single cell type)
    :return:
    """
    try:
        if filtering:
            simulated_bulk_exp_file_path = simulated_bulk_exp_file_path.replace('_log2cpm1p.csv',
                                                                                '_log2cpm1p_filtered.csv')
            if cell_fraction_file_path is str or isinstance(cell_fraction_file_path, Path):
                cell_fraction_file_path = str(cell_fraction_file_path).replace('.csv', '_filtered.csv')
        simu_bulk_exp_raw = read_df(simulated_bulk_exp_file_path)
    except UnicodeDecodeError:
        simu_bulk_exp_raw = read_h5ad(simulated_bulk_exp_file_path)
    if cell_fraction_file_path is str or isinstance(cell_fraction_file_path, Path):
        cell_frac = read_df(cell_fraction_file_path)
    elif isinstance(cell_fraction_file_path, pd.DataFrame):
        cell_frac = cell_fraction_file_path
    else:
        raise TypeError(f'Unsupported type for cell_fraction_file_path: {type(cell_fraction_file_path)}')
    if merge_t_cell:
        if 'T Cells' not in cell_frac.columns:
            cell_frac['T Cells'] = cell_frac.loc[:, ['CD4 T', 'CD8 T']].sum(axis=1)
        cell_frac.drop(columns=['CD4 T', 'CD8 T'], inplace=True)
    sample_list = cell_frac.index.to_list()
    # cell_frac.index = cell_frac.index.astype(str)
    uns = {'cell_types': cell_frac.columns.to_list(),
           'dataset_info': dataset_info}
    simu_bulk_exp = pd.DataFrame()
    if isinstance(simu_bulk_exp_raw, pd.DataFrame):
        simu_bulk_exp = simu_bulk_exp_raw.loc[sample_list, :]
    elif isinstance(simu_bulk_exp_raw, AnnData):
        simu_bulk_exp_df = pd.DataFrame(simu_bulk_exp_raw.X, index=simu_bulk_exp_raw.obs.index,
                                        columns=simu_bulk_exp_raw.var.index)
        simu_bulk_exp_df = simu_bulk_exp_df.loc[sample_list, :]
        simu_bulk_exp = simu_bulk_exp_df.copy()
    if (cell_frac.shape[0] == simu_bulk_exp.shape[0]) and not np.all(simu_bulk_exp.index == cell_frac.index):
        cell_frac = cell_frac.loc[simu_bulk_exp.index, :].copy()
    elif cell_frac.shape[0] != simu_bulk_exp.shape[0]:
        print(f'   The shape of simu_bulk_exp: {simu_bulk_exp.shape}, the shape of cell_frac: {cell_frac.shape}')
        cell_frac = cell_frac[~cell_frac.index.duplicated(keep='first')].copy()
        simu_bulk_exp = simu_bulk_exp[~simu_bulk_exp.index.duplicated(keep='first')].copy()
        intersection_inx = [i for i in simu_bulk_exp.index if i in cell_frac.index]
        print(f'   The length of intersections in both cell_frac and simu_bulk_exp: {len(intersection_inx)}')
        cell_frac = cell_frac.loc[intersection_inx, :].copy()
        simu_bulk_exp = simu_bulk_exp.loc[intersection_inx, :].copy()
    if np.all(simu_bulk_exp.index == cell_frac.index):
        var = pd.DataFrame(index=simu_bulk_exp.columns, columns=[f'in_{gep_type}'])
        var[f'in_{gep_type}'] = 1
        adata = AnnData(X=simu_bulk_exp.values.astype('float32'), obs=cell_frac, uns=uns,
                        var=var, dtype=np.dtype('float32'))
        adata.write_h5ad(filename=Path(result_file_path), compression='gzip')
    else:
        raise KeyError('simu_bulk_exp and cell_frac file should have same sample order')


def print_msg(p_str, log_file_path=None):
    print()
    current_info = f'---->>> {p_str} <<<----'
    current_time = time.ctime()
    print(current_info)
    print(current_time)
    if log_file_path is not None:
        with open(log_file_path, 'a') as f_handle:
            f_handle.write(current_info + '\n')
            f_handle.write(current_time + '\n')
            f_handle.write('\n')


def check_dir(path):
    """
    check if a path exist, create if not exist
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_quantile(exp_df, quantile_range, col_name: list = None) -> pd.DataFrame:
    """
    Get quantile
    :param exp_df:
    :param quantile_range:
    :param col_name:
    :return:
    """
    quantile_df = pd.DataFrame(index=exp_df.columns, columns=col_name)
    quantile_df[col_name[0]] = exp_df.quantile(quantile_range[0], axis=0)
    quantile_df[col_name[1]] = exp_df.quantile(quantile_range[1], axis=0)
    quantile_df[col_name[2]] = exp_df.quantile(quantile_range[2], axis=0)
    return quantile_df


def get_gene_list_filtered_by_quantile_range(bulk_exp, tcga_exp, quantile_range: list = None,
                                             q_col_name: list = None, tcga_gene_info=None):
    """
    Get gene list filtered by quantile range
    :param bulk_exp: bulk expression, TPM
    :param tcga_exp: TCGA expression data, TPM
    :param quantile_range: lower boundary, median, upper boundary, such as [0.025, 0.5, 0.975]
    :param q_col_name: column names for quantile range
    :param tcga_gene_info: gene quantile values in TCGA
    """
    if quantile_range is None:
        quantile_range = [0.025, 0.5, 0.975]
        q_col_name = ['q_' + str(int(q * 1000) / 10) for q in quantile_range]
    if tcga_gene_info is None:
        tcga_gene_info = get_quantile(tcga_exp, quantile_range=quantile_range, col_name=q_col_name)
    if type(bulk_exp) is pd.DataFrame and bulk_exp.shape[0] > 1:
        bulk_gene_info = get_quantile(bulk_exp, quantile_range=quantile_range, col_name=q_col_name)
        gene_inx = (bulk_gene_info[q_col_name[1]] >= tcga_gene_info[q_col_name[0]]) & \
                   (bulk_gene_info[q_col_name[1]] <= tcga_gene_info[q_col_name[2]])
        gene_list_qr = bulk_exp.loc[:, gene_inx].columns.to_list()
    else:  # if bulk_exp is a series
        gene_inx = (bulk_exp >= tcga_gene_info[q_col_name[0]]) & (bulk_exp <= tcga_gene_info[q_col_name[2]])
        gene_list_qr = bulk_exp.loc[gene_inx].index.to_list()
    return gene_list_qr


def plot_pca(data: pd.DataFrame, result_fp=None, color_code=None, s=5, figsize=(8, 8),
             color_code2label: dict = None, explained_variance_ratio: np.array = None, label_name='PC',
             show_legend=True, show_xy_labels=True, anno=None, show_core_zone_of_tcga=False):
    """
    plot PCA result of simulated bulk cell dataset
    :param data: PCA table, samples by PCs
    :param result_fp:
    :param color_code: a "np.array" to mark the label of each sample
    :param color_code2label:
    :param explained_variance_ratio: pca_model.explained_variance_ratio_
    :param label_name: label name for x-axis
    :param show_legend:
    :param show_xy_labels:
    :param anno: annotation for x-axis, which layer was used to generate the data
    :param show_core_zone_of_tcga: whether to show the core zone of TCGA data
    :return:
    """
    # sns.set_style('white')
    # sns.set(font_scale=1.5)
    n_components = len([i for i in data.columns.to_list() if label_name in i])
    if n_components >= 3:
        pc_comb = [(0, 1), (1, 2), (0, 2)]
    elif n_components == 2:
        pc_comb = [(0, 1)]
    else:
        raise IndexError(f'data should have >= 2 columns, but {data.shape[1]} got')
    if color_code is not None:
        data['class'] = color_code
    for pc1, pc2 in pc_comb:
        # plt.figure(figsize=figsize)
        if 'class' in data.columns:
            col_x = f'{label_name}{pc1 + 1}'
            col_y = f'{label_name}{pc2 + 1}'
            g = sns.jointplot(x=col_x, y=col_y, data=data, kind='scatter', hue='class',
                              s=s, space=0, height=figsize[1], alpha=0.5)
            ax = g.ax_joint
            n_tcga, n_non_tcga = 0, 0
            q_lower, q_upper = 0.1, 0.9
            if show_core_zone_of_tcga and 'TCGA' in data['class'].unique():
                coord, n_tcga, n_non_tcga = get_core_zone_of_pca(pca_data=data, col_x=col_x, col_y=col_y,
                                                                 q_lower=q_lower, q_upper=q_upper)
                width = coord['x_upper'] - coord['x_lower']
                height = coord['y_upper'] - coord['y_lower']
                rect = patches.Rectangle((coord['x_lower'], coord['y_lower']), width, height,
                                         fill=False, color='red', linewidth=1,
                                         linestyle='dashed', label='Core Zone of TCGA')
                ax.add_patch(rect)
            if show_xy_labels:
                x_label = col_x
                y_label = col_y
                if (explained_variance_ratio is not None) and (anno is not None):
                    x_label = col_x + f' ({explained_variance_ratio[pc1] * 100:.1f}%, {anno})'
                    y_label = col_y + f' ({explained_variance_ratio[pc2] * 100:.1f}%)'
                elif explained_variance_ratio is not None:
                    x_label = col_x + f' ({explained_variance_ratio[pc1] * 100:.1f}%)'
                    y_label = col_y + f' ({explained_variance_ratio[pc2] * 100:.1f}%)'
                elif anno is not None:
                    x_label = col_x + f' ({anno})'
                if show_core_zone_of_tcga:
                    q_range = f'$q_{{{q_lower * 100:.0f}}}-q_{{{q_upper * 100:.0f}}}$'
                    x_label += f'\nCore Zone ({q_range}): TCGA ({n_tcga}), Non-TCGA ({n_non_tcga})'
                ax.set(xlabel=x_label, ylabel=y_label)
            else:
                ax.set(xlabel=None, ylabel=None)
            # Put the legend out of the figure
            if show_legend:
                # g_legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2 - 0.1 * n_class), ncol=2)
                g_legend = ax.legend(loc='best', ncol=2)
                for _ in g_legend.legendHandles:
                    _.set_linewidth(1)
            else:
                ax.legend([], [], frameon=False)
            # remove the top and right ticks
            g.ax_marg_x.tick_params(axis='x', which='both', top=False)
            g.ax_marg_x.grid(False)
            g.ax_marg_y.tick_params(axis='y', which='both', right=False)
            g.ax_marg_y.grid(False)
        else:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            for i in np.unique(color_code)[::-1]:
                current_part = data.loc[color_code == i, :].copy()
                if color_code2label is None:
                    ax.scatter(current_part.iloc[:, pc1], current_part.iloc[:, pc2], label=i, alpha=.3)
                else:
                    ax.scatter(current_part.iloc[:, pc1], current_part.iloc[:, pc2],
                               label=color_code2label[i], alpha=.3)

            # plt.title(title, fontsize=18)
            if explained_variance_ratio is None:
                plt.xlabel(f'{label_name}{pc1 + 1}')
                plt.ylabel(f'{label_name}{pc2 + 1}')
            else:
                plt.xlabel(f'{label_name}{pc1 + 1} ({(explained_variance_ratio[pc1] * 100): .1f}%)')
                plt.ylabel(f'{label_name}{pc2 + 1} ({(explained_variance_ratio[pc2] * 100): .1f}%)')

            plt.legend()
            plt.tight_layout()
        if result_fp is not None:
            if '.png' in result_fp:
                plt.savefig(result_fp.replace('.png', f'_{label_name}{pc1}_{label_name}{pc2}.png'),
                            bbox_inches='tight', dpi=300)
            if '.pdf' in result_fp:
                plt.savefig(result_fp.replace('.pdf', f'_{label_name}{pc1}_{label_name}{pc2}.pdf'),
                            bbox_inches='tight', dpi=300)


def get_core_zone_of_pca(pca_data: pd.DataFrame, col_x, col_y, q_lower, q_upper):
    """
    get core zone of TCGA in PCA plot
    :param pca_data: PCA data
    :param col_x: x axis column name
    :param col_y: y axis column name
    :param q_lower: lower quantile
    :param q_upper: upper quantile
    :return: coordinate_core_zone, n_tcga_in_core_zone, n_non_tcga_in_core_zone
    """
    assert 'TCGA' in pca_data['class'].unique()
    x_tcga = pca_data.loc[pca_data['class'] == 'TCGA', col_x]
    y_tcga = pca_data.loc[pca_data['class'] == 'TCGA', col_y]
    x_q_lower = np.quantile(x_tcga, q_lower)
    x_q_upper = np.quantile(x_tcga, q_upper)
    y_q_lower = np.quantile(y_tcga, q_lower)
    y_q_upper = np.quantile(y_tcga, q_upper)
    core_zone = pca_data.loc[(pca_data[col_x] >= x_q_lower) & (pca_data[col_x] <= x_q_upper) &
                             (pca_data[col_y] >= y_q_lower) & (pca_data[col_y] <= y_q_upper), :]
    coordinate_core_zone = {'x_lower': x_q_lower, 'x_upper': x_q_upper, 'y_lower': y_q_lower, 'y_upper': y_q_upper}
    n_tcga_in_core_zone = core_zone.loc[core_zone['class'] == 'TCGA', :].shape[0]
    n_non_tcga_in_core_zone = core_zone.loc[core_zone['class'] != 'TCGA', :].shape[0]
    return coordinate_core_zone, n_tcga_in_core_zone, n_non_tcga_in_core_zone


def cal_corr_gene_exp_with_cell_frac(gene_exp: pd.DataFrame, cell_frac: pd.DataFrame,
                                     result_file_path=None, filtered_by_corr: float = None,
                                     filter_by_num: int = None):
    """
    calculate correlation between gene expression values and cell fractions for each cell type
    :param gene_exp: non-log space, tpm/cpm, samples by genes
    :param cell_frac: samples by cell types
    :param result_file_path:
    :param filtered_by_corr: threshold of correlation, only keep genes with higher corr than this value
    :param filter_by_num: the max number of genes to keep for each cell type
    :return: correlation, genes by cell types, filtered or all gene list
    """
    if not os.path.exists(result_file_path):
        _, n_cell_type = cell_frac.shape
        _, n_gene = gene_exp.shape
        print(f'The shape of cell_frac: {cell_frac.shape}')
        print(f'The shape of gene_exp: {gene_exp.shape}')
        corr = np.corrcoef(cell_frac.values, gene_exp.values, rowvar=False)
        corr_df = pd.DataFrame(data=corr[:n_cell_type, n_cell_type:].T,
                               columns=cell_frac.columns, index=gene_exp.columns)
    else:
        print(f'Previous correlation file will be used: {result_file_path}')
        corr_df = pd.read_csv(result_file_path, index_col=0)
    if (filtered_by_corr is not None) and (filter_by_num is not None):
        for cell_type in corr_df.columns:
            if corr_df.loc[corr_df[cell_type] >= filtered_by_corr, cell_type].shape[0] > filter_by_num:
                corr_df2 = corr_df.sort_values(by=[cell_type], ascending=False)
                # set to 0 if there are too many high corr genes for the cell fraction of this cell type
                corr_df.loc[corr_df.index.isin(corr_df2.iloc[filter_by_num:, :].index), cell_type] = 0
        corr_df[f'n_at_least_one'] = np.sum(corr_df >= filtered_by_corr, axis=1)
    elif filtered_by_corr is not None:
        corr_df[f'n_at_least_one'] = np.sum(corr_df >= filtered_by_corr, axis=1)
    elif filter_by_num is not None:
        for cell_type in corr_df.columns:
            corr_df2 = corr_df.sort_values(by=[cell_type], ascending=False)
            # set to 0 if there are too many genes than filter_by_num of this cell type
            corr_df.loc[corr_df.index.isin(corr_df2.iloc[filter_by_num:, :].index), cell_type] = 0
        corr_df[f'n_at_least_one'] = np.sum(corr_df > 0, axis=1)
    if (filtered_by_corr is not None) or (filter_by_num is not None):
        corr_df_filtered = corr_df.loc[corr_df[f'n_at_least_one'] >= 1,
                                       [i for i in corr_df.columns if i != 'n_at_least_one']].copy()
        corr_df_filtered.to_csv(result_file_path, float_format='%g')
        return corr_df_filtered
    else:
        corr_df.to_csv(result_file_path, float_format='%g')
        return corr_df


def do_pca_analysis(exp_df, n_components=5, pca_result_fp=None, save_model: bool = False):
    """
    PCA analysis
    :param exp_df:
    :param n_components:
    :param pca_result_fp:
    :param save_model:
    :return: fitted PCA model
    """
    if os.path.exists(pca_result_fp):
        print(f'Loading PCA result from file: {pca_result_fp}')
        pca = load(pca_result_fp)
    else:
        pca = PCA(n_components=n_components)
        pca.fit(exp_df)
        if save_model:
            dump(pca, pca_result_fp)
    return pca


def get_distance_matrices_by_class(
        adata_X: np.ndarray,
        obs_df: pd.DataFrame,
        class_by: str = 'cell_type',
        n_jobs: int = 1,
        show_progress: bool = True
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """
    Compute pairwise distance matrices for each class separately.

    Parameters
    ----------
    adata_X : np.ndarray
        Gene expression matrix (n_cells × n_genes).
    obs_df : pd.DataFrame
        DataFrame corresponding to `adata.obs`. Index contains cell IDs.
    class_by : str, default='cell_type'
        Column name in `obs_df` that defines the classes.
    n_jobs : int, default=1
        Number of parallel jobs. Use -1 for all CPU cores.
    show_progress : bool, default=True
        Whether to show progress bar.

    Returns
    -------
    distance_dict : dict
        Dictionary mapping each class to a tuple of:
        - distance_matrix (np.ndarray): Pairwise distance matrix
        - cell_ids (list of str): Ordered list of cell IDs
    """
    # Validation
    if class_by not in obs_df.columns:
        raise KeyError(f'Column "{class_by}" not found in obs_df.')

    # Convert to float32
    adata_X = adata_X.astype(np.float32)

    # Group by class
    grouped = obs_df.groupby(class_by, observed=True)

    # Determine number of workers
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    # Sequential processing (default)
    if n_jobs == 1:
        return _compute_distances_sequential(
            adata_X, grouped, obs_df, show_progress
        )

    # Parallel processing
    return _compute_distances_parallel(
        adata_X, grouped, obs_df, n_jobs, show_progress
    )


def _compute_distances_sequential(
        adata_X: np.ndarray,
        grouped: pd.core.groupby.DataFrameGroupBy,
        obs_df: pd.DataFrame,
        show_progress: bool
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Sequential distance computation."""
    distance_dict = {}

    iterator = grouped
    if show_progress:
        iterator = tqdm(grouped, desc='Computing distances', unit='class')

    for cell_class, group in iterator:
        cell_ids = group.index.to_list()
        cell_indices = obs_df.index.get_indexer(cell_ids)
        X_subset = adata_X[cell_indices, :]

        if len(cell_ids) > 1:
            distance_matrix = euclidean_distances(X_subset).astype(np.float32)
        else:
            distance_matrix = np.array([[0.0]], dtype=np.float32)

        distance_dict[cell_class] = (distance_matrix, cell_ids)

        if show_progress:
            iterator.set_postfix({'cells': len(cell_ids)})

    return distance_dict


def _compute_distances_parallel(
        adata_X: np.ndarray,
        grouped: pd.core.groupby.DataFrameGroupBy,
        obs_df: pd.DataFrame,
        n_jobs: int,
        show_progress: bool
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Parallel distance computation."""
    # Prepare tasks
    tasks = []
    for cell_class, group in grouped:
        cell_ids = group.index.to_list()
        cell_indices = obs_df.index.get_indexer(cell_ids)
        X_subset = adata_X[cell_indices, :]
        tasks.append((cell_class, X_subset, cell_ids))

    distance_dict = {}

    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_class = {
            executor.submit(_compute_single_distance, X_subset, cell_ids): cell_class
            for cell_class, X_subset, cell_ids in tasks
        }

        # Collect results with progress bar
        iterator = as_completed(future_to_class)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(tasks),
                desc=f'Computing distances ({n_jobs} workers)',
                unit='class'
            )

        for future in iterator:
            cell_class = future_to_class[future]
            distance_matrix, cell_ids = future.result()
            distance_dict[cell_class] = (distance_matrix, cell_ids)

            if show_progress:
                iterator.set_postfix({'cells': len(cell_ids)})

    return distance_dict


def _compute_single_distance(
        X_subset: np.ndarray,
        cell_ids: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Worker function for parallel processing."""
    if len(cell_ids) > 1:
        distance_matrix = euclidean_distances(X_subset).astype(np.float32)
    else:
        distance_matrix = np.array([[0.0]], dtype=np.float32)

    return distance_matrix, cell_ids


def get_available_threads(requested: int) -> int:
    """Get available thread count."""
    return min(multiprocessing.cpu_count() - 2, requested)
