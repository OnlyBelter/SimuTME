import os
import sys
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import anndata as ad  # Convention is usually 'ad', not 'an'
from scipy.sparse import csr_matrix, issparse
from sklearn import preprocessing as pp

# Assuming these are available in your local package
from .pub_func import (
    log_exp2cpm,
    read_df,
    non_log2log_cpm,
    non_log2cpm,
    get_inx2cell_type
)


class H5ADLoader:
    """
    A class to handle loading and basic access to .h5ad files.
    """

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self._adata: Optional[ad.AnnData] = None

        self._load_data()

    def _load_data(self):
        """Internal method to load data."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            self._adata = ad.read_h5ad(self.file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load h5ad file: {e}")

    @property
    def dataset(self) -> ad.AnnData:
        """Returns the loaded AnnData object."""
        return self._adata

    @property
    def shape(self):
        return self._adata.shape


class ExpressionLoader:
    """
    A class to load raw expression matrices (CSV/TXT) and perform normalization.
    """

    def __init__(self, file_path: Union[str, Path], file_format: str = 'csv'):
        self.file_path = Path(file_path)
        self.raw_df: Optional[pd.DataFrame] = None

        # Load data immediately upon instantiation
        self._load_data(file_format)

    def _load_data(self, fmt: str):
        """Wrapper around the custom read_df function."""
        # Assuming read_df is your custom function from pub_func
        self.raw_df = read_df(self.file_path, fmt=fmt)

    def to_tpm(self) -> pd.DataFrame:
        """
        Convert raw counts to TPM (Transcripts Per Million).
        Note: This assumes the input is raw counts.
        """
        if self.raw_df is None:
            raise ValueError("Data not loaded.")

        print("Converting to TPM...")

        # 1. Copy data to avoid modifying original
        df = self.raw_df.copy()

        # 2. Handle sparse matrices if necessary
        # Logic depends on if your read_df returns pandas or sparse matrix
        # Here is a standard pandas implementation:

        # Calculate total counts per cell (column sum)
        total_counts = df.sum(axis=0)

        # Avoid division by zero
        total_counts[total_counts == 0] = 1

        # TPM calculation: (Count / Total) * 10^6
        tpm_df = (df / total_counts) * 1e6

        return tpm_df

    def to_cpm(self, log: bool = False) -> pd.DataFrame:
        """Example of using your imported pub_func"""
        if log:
            return non_log2log_cpm(self.raw_df)
        else:
            return non_log2cpm(self.raw_df)
