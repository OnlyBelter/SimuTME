import os
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import anndata as ad  # Convention is usually 'ad', not 'an'
from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import MinMaxScaler

# Assuming these are available in your local package
from .pub_func import (
    log_exp2cpm,
    read_df,
    non_log2log_cpm,
    non_log2cpm,
    get_inx2cell_type
)


class ReadH5AD(object):
    """
    Reads .h5ad files (AnnData format), typically handling log2 transformed data.

    Attributes:
        dataset (anndata.AnnData): The loaded AnnData object.
    """

    def __init__(self, file_path: str, show_info: bool = False, backed: Optional[str] = None):
        """
        Initialize the ReadH5AD object.

        :param file_path: Path to the .h5ad file (samples x genes, usually log2cpm1p).
        :param show_info: If True, prints the AnnData summary after loading.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.dataset = ad.read_h5ad(file_path, backed=backed)

        if show_info:
            print(f"   > Loaded H5AD file: {file_path}")
            print(self.dataset)

    def get_df(self, result_file_path: Optional[str] = None,
               convert_to_tpm: bool = False,
               scaling_by_sample: bool = False) -> pd.DataFrame:
        """
        Convert the AnnData object to a pandas DataFrame.

        :param result_file_path: If provided, saves the DataFrame to this CSV path.
        :param convert_to_tpm: If True, converts log2cpm1p data back to TPM.
        :param scaling_by_sample: If True, scales each sample's expression to [0, 1] using MinMax.
        :return: A DataFrame (samples x genes).
        """
        # Efficiently handle sparse vs dense matrices
        if issparse(self.dataset.X):
            x_data = self.dataset.X.toarray().astype(np.float32)
        else:
            x_data = np.asarray(self.dataset.X, dtype=np.float32)

        if convert_to_tpm:
            x_data = log_exp2cpm(x_data)

        if scaling_by_sample:
            # Scale per sample (row-wise). MinMaxScaler works on columns, so we transpose.
            scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
            x_data = scaler.fit_transform(x_data.T).T

        df = pd.DataFrame(data=x_data,
                          index=self.dataset.obs.index,
                          columns=self.dataset.var.index).round(3)

        if result_file_path is not None:
            df.to_csv(result_file_path, float_format='%.3f')

        return df

    def get_rows_df(self, row_ids: List[str], convert_to_tpm: bool = False) -> pd.DataFrame:
        """Materialize only selected rows as a DataFrame."""
        subset = self.dataset[row_ids, :]

        if issparse(subset.X):
            x_data = subset.X.toarray().astype(np.float32)
        else:
            x_data = np.asarray(subset.X, dtype=np.float32)

        if convert_to_tpm:
            x_data = log_exp2cpm(x_data)

        return pd.DataFrame(
            data=x_data,
            index=subset.obs.index,
            columns=subset.var.index,
        ).round(3)

    def get_cell_fraction(self) -> Union[pd.DataFrame, None]:
        """
        Extract cell fraction data from observations (obs).

        :return: DataFrame of cell fractions (cells x cell types) or None if empty.
        """
        if self.dataset.obs.shape[1] > 0:
            return self.dataset.obs.round(3)
        else:
            print('   > Warning: No cell fraction data found in .obs')
            return None

    def get_h5ad(self) -> ad.AnnData:
        """
        Return the underlying AnnData object.
        """
        return self.dataset


class ReadExp(object):
    """
    Reads and processes gene expression files, supporting format conversions.

    Supported formats:
    - TPM: Transcripts Per Million.
    - CPM: Counts Per Million (treated equivalently to TPM for bulk).
    - log_space: log2(TPM + 1) or log2(CPM + 1).
    - non_log: Raw counts or non-log normalized data.

    Attributes:
        exp (pd.DataFrame): The expression matrix (samples x genes).
        file_type (str): Current data format (TPM, log_space, etc.).
    """

    def __init__(self, exp_file: str | pd.DataFrame | Path, exp_type: str = 'TPM', transpose: bool = False):
        """
        Initialize the ReadExp object.

        :param exp_file: File path (str) or existing DataFrame.
        :param exp_type: One of ['TPM', 'CPM', 'log_space', 'non_log'].
        :param transpose: Set True if input is (Genes x Samples). Result will be (Samples x Genes).
        """
        valid_types = ['TPM', 'CPM', 'log_space', 'non_log']
        if exp_type not in valid_types:
            raise ValueError(f"Invalid exp_type: {exp_type}. Must be one of {valid_types}")

        self.file_type = exp_type
        self.exp = read_df(exp_file) if isinstance(exp_file, str) else exp_file.copy()

        if transpose:
            self.exp = self.exp.T

        self.scaled_by_sample = False

    def to_tpm(self):
        """
        Convert the current expression matrix to TPM.
        """
        if self.file_type == 'non_log':
            self.exp = non_log2cpm(self.exp)
        elif self.file_type == 'log_space':
            self.exp = log_exp2cpm(self.exp)

        # If already TPM or CPM, do nothing
        self.file_type = 'TPM'

    def to_log2cpm1p(self):
        """
        Convert the current expression matrix to log2(TPM + 1).
        """
        if self.file_type != 'log_space':
            self.exp = non_log2log_cpm(self.exp, transpose=False)
        else:
            print('   > Info: Data is already in log space.')

        self.file_type = 'log_space'

    def get_file_type(self) -> str:
        """Return the current file type."""
        return self.file_type

    def get_exp(self) -> pd.DataFrame:
        """Return the expression matrix rounded to 3 decimal places."""
        return self.exp.round(3)

    def save(self, file_path: str, sep: str = ',', transpose: bool = False):
        """
        Save the expression matrix to a file.

        :param file_path: Destination path.
        :param sep: Delimiter (default: ',').
        :param transpose: If True, saves as (Genes x Samples).
        """
        df_to_save = self.exp.T if transpose else self.exp
        df_to_save.to_csv(file_path, sep=sep, float_format='%.3f')

    def do_scaling(self):
        """
        Scale GEPs sample-wise to [0, 1] (MinMax scaling).
        """
        if not self.scaled_by_sample:
            scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
            # fit_transform expects (n_samples, n_features).
            # If we want to scale *within* a sample (row), we transpose, scale columns, then transpose back.
            x_scaled = scaler.fit_transform(self.exp.T).T

            self.exp = pd.DataFrame(data=x_scaled,
                                    index=self.exp.index,
                                    columns=self.exp.columns).round(3)
            self.scaled_by_sample = True

    def do_scaling_by_constant(self, divide_by: float = 20.0):
        """
        Scale GEPs by dividing by a constant (only for log space).
        Ensures values are roughly in [0, 1].
        """
        if self.file_type != 'log_space':
            raise ValueError('   > Error: do_scaling_by_constant requires log_space data.')

        if np.any(self.exp.values > 1.0):
            self.exp = self.exp / divide_by

    def align_with_gene_list(self, gene_list: List[str],
                             fill_not_exist: bool = False,
                             pathway_list: bool = False):
        """
        Align the expression matrix with a specific gene list.

        If genes are removed, the data is re-normalized to TPM (unless it's a pathway list).

        :param gene_list: Target list of genes.
        :param fill_not_exist: If True, missing genes are added with 0 expression.
        :param pathway_list: Set True if gene_list represents pathways (skips re-normalization).
        """
        original_gene_count = self.exp.shape[1]

        # Use Pandas reindex for efficient alignment (handles both intersection and filling)
        if fill_not_exist:
            # Keep all genes in gene_list, fill missing with 0
            self.exp = self.exp.reindex(columns=gene_list, fill_value=0.0)

            # Calculate stats for logging
            current_genes = self.exp.columns
            missing_count = len(set(gene_list) - set(current_genes))  # Should be 0 after reindex
            # Actually, reindex adds them, so we check how many were added (which were NaN/0)
            # But for logging "removed", we compare with original
            # This logic is simplified: reindex does the job of "common + fill"
        else:
            # Only keep intersection
            common_genes = self.exp.columns.intersection(gene_list)
            removed_count = original_gene_count - len(common_genes)
            print(f'   > Aligning: {len(common_genes)} common genes used. {removed_count} genes removed.')
            self.exp = self.exp[common_genes].copy()

        # Re-normalization Logic
        # If we subset genes, the relative abundance (TPM) changes, so we must re-calculate.
        if not pathway_list:
            if self.file_type == 'log_space':
                # Convert back to TPM -> Re-normalize -> Log transform
                self.to_tpm()
                self.to_log2cpm1p()
            else:
                # Assume raw/non-log needs TPM normalization
                self.file_type = 'non_log'
                self.to_tpm()


def read_single_cell_type_dataset(
    sct_dataset_file_path: str,
    latent_z_nn_info_file: Union[str, pd.DataFrame] = None,
    load_expression: bool = True,
    backed: Optional[str] = None,
):
    """
    positive samples of SCT (single cell type), generated by SingleCellTypeGEPGenerator
    :param sct_dataset_file_path: the file path of GEPs for single cell type (SCT), positive samples
    :param latent_z_nn_info_file: neighbor information of latent z for all samples, used for QC
    """
    sct_dataset_obj = ReadH5AD(sct_dataset_file_path, backed=backed)
    sct_dataset_df = sct_dataset_obj.get_df(convert_to_tpm=True) if load_expression else None
    cell_type_list = sct_dataset_obj.get_h5ad().obs.columns.tolist()
    inx2cell_type = get_inx2cell_type(cell_type_list=cell_type_list)
    if (latent_z_nn_info_file is not None) and os.path.exists(latent_z_nn_info_file):
        latent_z_nn_info = read_df(latent_z_nn_info_file)
        latent_z_nn_info = \
            latent_z_nn_info.loc[
                (latent_z_nn_info['class'] != -1) &  # only positive samples
                (latent_z_nn_info['n_neighbor_class'] == 1) &  # all neighbors belong to the same cell type
                (latent_z_nn_info['class'] == latent_z_nn_info['pred_class']),  # predicted class is same as true class
                ['class']].copy()
        latent_z_nn_info['cell_type'] = latent_z_nn_info['class'].map(lambda x: inx2cell_type[x])
        sct_dataset_obs = latent_z_nn_info.copy()
    else:
        sct_dataset_obs = sct_dataset_obj.get_cell_fraction()
        sct_dataset_obs['class'] = sct_dataset_obs.values.argmax(axis=1)
        sct_dataset_obs['cell_type'] = sct_dataset_obs['class'].map(lambda x: inx2cell_type[x])
    return sct_dataset_obs, sct_dataset_df, sct_dataset_obj
