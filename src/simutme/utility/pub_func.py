import os
import sys
from typing import Union, List, Dict, Optional
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from ..config import DEFAULT_CELL_TYPES


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
        sns.set(palette='muted', font_scale=1.5)

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
