import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .simulator import BulkGEPGenerator
from ..utility import ReadExp, log2_transform, create_h5ad_dataset, get_distance_matrices_by_class
from .simu_helper import validate_generation_params_sct, SimuMethod, SamplingMethodGEP, SampleType
import logging
logger = logging.getLogger(__name__)


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
                         sample_type: str = SampleType.POSITIVE,
                         sep_by_patient: bool = False,
                         simu_method: str = SimuMethod.AVERAGE,
                         test_set: bool = False,
                         minimum_n_base: int = 1,
                         ref_gene_list_file_path: Optional[str] = None,
                         sampling_method: str = SamplingMethodGEP.RANDOM,
                         ) -> None:
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
        validate_generation_params_sct(
            n_sample_each_cell_type=n_sample_each_cell_type,
            n_base_for_positive_samples=n_base_for_positive_samples,
            sample_type=sample_type,
            simu_method=simu_method,
            sampling_method=sampling_method,
            minimum_n_base=minimum_n_base,
        )

        if self.generated_bulk_gep_fp.exists():
            logger.info(f'Previous result exists: {self.generated_bulk_gep_fp}')
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
        chunk_size = max(100, int(n_sample_each_cell_type / 5)) if n_sample_each_cell_type > 1000 else n_sample_each_cell_type
        total_rows = self.n_samples
        n_chunks = (total_rows + chunk_size - 1) // chunk_size

        # Check for existing partial results
        existed_ids = pd.Index([])
        if self.generated_bulk_gep_csv_fp.exists():
            existed_ids = pd.read_csv(self.generated_bulk_gep_csv_fp, index_col=0, usecols=[0]).index

        logger.info(f"Starting generation: {self.n_samples} samples in ~{n_chunks} chunks of size {chunk_size}.")

        processed_samples = 0
        with pd.read_csv(self.generated_cell_fraction_fp, chunksize=chunk_size, index_col=0) as reader:
            pbar = tqdm(reader, desc="Processing Chunks", total=n_chunks)
            for chunk_rows in pbar:
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
                    processed_samples += chunk_rows.shape[0]
                    pbar.set_postfix({'processed': processed_samples, 'total': self.n_samples})
        logger.info(f"Generation completed: {self.generated_bulk_gep_counter} samples generated.")
        # 6. Finalize
        self._finalize_dataset()

    def _prepare_fractions(self, sample_type: str, n_sample_each_cell_type: int):
        """Helper to generate or verify cell fraction file."""
        if not self.generated_cell_fraction_fp.exists():
            logging.info(f'Generating cell proportions for {self.bulk_dataset_name} ({sample_type})...')
            generated_cell_frac = self.generate_frac_sc(
                sample_type=sample_type,
                sample_prefix=f'{self.bulk_dataset_name}_{sample_type[:3]}'
            )
            generated_cell_frac.to_csv(self.generated_cell_fraction_fp, float_format='%g', index_label='sample_id')
        else:
            logging.info(f'Using existing cell fractions: {self.generated_cell_fraction_fp}')

    def _prepare_gene_expression(self, ref_gene_list_fp: Optional[str]):
        """Helper to align expression data with reference gene list."""
        if ref_gene_list_fp:
            logging.info(f'Aligning with reference genes from: {ref_gene_list_fp}')
            ref_gene_list = pd.read_csv(ref_gene_list_fp, index_col=0).index.to_list()

            # Use ReadExp for safe handling
            sc_df_obj = ReadExp(self.merged_sc_dataset_df, exp_type='TPM')
            sc_df_obj.align_with_gene_list(ref_gene_list)
            # Update the main dataframe in memory
            self.merged_sc_dataset_df = sc_df_obj.get_exp()
            logging.info(f'Dataset aligned. Genes: {self.merged_sc_dataset_df.shape[1]}')

    def _get_cell_distance_dict(self):
        """Calculate cell distance dictionary for n_neighbors sampling for all cell types in "output_types"."""
        logging.info('Calculating cell distance dictionary for n_neighbors sampling...')
        self.cell_distance_dict = get_distance_matrices_by_class(
            adata_X=self.merged_sc_dataset.X.toarray(),
            obs_df=self.merged_sc_dataset_obs,
            class_by='output_types',
            metric='euclidean'
        )

    def _process_chunk(
            self,
            rows: pd.DataFrame,
            sample_type: str,
            n_base: int,
            min_n_base: int,
            sep_by_patient: bool,
            simu_method: str = 'ave',
            sampling_method_gep: str = 'random'
    ) -> None:
        """Process a single chunk of samples: select cells -> aggregate -> save."""
        # Determine total cells to aggregate
        total_cell_number = n_base if (sample_type == SampleType.POSITIVE and n_base > 1) else 0

        # Load base data
        current_obs_df, gene_list_in_sc_ds, sc_dataset_key = self._load_sc_data_for_generation(simu_method)

        # For positive samples, optimize by filtering obs if chunk is homogeneous
        if sample_type == SampleType.POSITIVE:
            current_obs_df = self._get_filtered_obs_for_chunk(rows, min_n_base, n_base)

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
        self._save_chunk_results(simulated_gep, selected_cell_ids)

    def _get_filtered_obs_for_chunk(
            self,
            rows: pd.DataFrame,
            min_n_base: int,
            n_base: int
    ) -> pd.DataFrame:
        """Get filtered observation DataFrame for a chunk."""
        unique_targets = rows.idxmax(axis=1).unique()

        if len(unique_targets) == 1:
            # Homogeneous chunk - filter to single cell type
            return self._get_obs_for_type(unique_targets[0], min_n_base, n_base)
        else:
            # Mixed chunk - return full obs, let _sc_sampling handle filtering
            logger.debug(f"Mixed chunk with {len(unique_targets)} cell types, using full obs")
            return self.merged_sc_dataset_obs

    def _save_chunk_results(
            self,
            simulated_gep: Optional[pd.DataFrame],
            selected_cell_ids: pd.DataFrame
    ) -> None:
        """Save chunk results if valid."""
        if simulated_gep is None or simulated_gep.empty:
            return

        simulated_gep = log2_transform(simulated_gep)
        self.generated_bulk_gep_counter += simulated_gep.shape[0]

        # Ensure alignment
        valid_indices = simulated_gep.index
        selected_cell_ids = selected_cell_ids.loc[valid_indices].copy()

        self._save_simulated_bulk_gep(gep=simulated_gep, cell_id=selected_cell_ids)

    def _get_obs_for_type(
            self,
            cell_type: str,
            min_n: int,
            warn_n: int
    ) -> pd.DataFrame:
        """
        Filter observation DataFrame for a specific cell type.

        Parameters
        ----------
        cell_type : str
            Target cell type to filter.
        min_n : int
            Minimum required cells (raises error if not met).
        warn_n : int
            Warning threshold for low cell count.

        Returns
        -------
        pd.DataFrame
            Filtered observation DataFrame.

        Raises
        ------
        ValueError
            If column not found or insufficient cells.
        """
        # Determine which column to use
        if cell_type in self.cell_subtype_used:
            col_name = self.subtype_col_name
        else:
            col_name = self.cell_type_col_name

        if col_name not in self.merged_sc_dataset_obs.columns:
            available_cols = list(self.merged_sc_dataset_obs.columns)
            raise ValueError(
                f"Column '{col_name}' not found in obs. "
                f"Available columns: {available_cols}"
            )

        filtered_obs = self.merged_sc_dataset_obs[
            self.merged_sc_dataset_obs[col_name] == cell_type
            ].copy()

        n_cells = filtered_obs.shape[0]

        if n_cells < min_n:
            raise ValueError(
                f"Insufficient cells for '{cell_type}': found {n_cells}, "
                f"minimum required is {min_n}. "
                f"Consider reducing 'minimum_n_base' or 'n_base_for_positive_samples'."
            )

        if n_cells <= warn_n:
            logger.warning(
                f"Cell type '{cell_type}' has only {n_cells} cells "
                f"(requested {warn_n}). Results may have limited diversity."
            )

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

    def generate_frac_sc(
            self,
            sample_prefix: Optional[str] = None,
            sample_type: str = 'positive'
    ) -> pd.DataFrame:
        """
        Generate cell fractions for single cell samples.

        Parameters
        ----------
        sample_prefix : str, optional
            Prefix for sample names.
        sample_type : str
            'positive' (100% one cell type) or 'negative' (mixture).

        Returns
        -------
        pd.DataFrame
            DataFrame (samples x cell_types) with fractions.
        """
        if sample_prefix is None:
            sample_prefix = f's_sc_{sample_type}'

        cols = self.cell_type_used + self.cell_subtype_used
        n_cell_types = len(cols)

        sample_names = [f'{sample_prefix}_{i}' for i in range(self.n_samples)]

        if sample_type == SampleType.POSITIVE:
            return self._generate_positive_fractions(sample_names, cols, n_cell_types)
        else:
            return self._generate_negative_fractions(sample_names, cols, n_cell_types)

    def _generate_positive_fractions(
            self,
            sample_names: List[str],
            cols: List[str],
            n_cell_types: int
    ) -> pd.DataFrame:
        """Generate fractions for positive (pure) samples."""
        # Create sparse representation first, then convert to DataFrame
        samples_per_type = self.n_samples // n_cell_types

        # Pre-allocate array
        data = np.zeros((self.n_samples, n_cell_types), dtype=np.float32)

        for i in range(n_cell_types):
            start_idx = i * samples_per_type
            end_idx = self.n_samples if i == n_cell_types - 1 else (i + 1) * samples_per_type
            data[start_idx:end_idx, i] = 1.0

        return pd.DataFrame(data, index=sample_names, columns=cols).round(4)

    def _generate_negative_fractions(
            self,
            sample_names: List[str],
            cols: List[str],
            n_cell_types: int
    ) -> pd.DataFrame:
        """Generate fractions for negative (mixed) samples - vectorized."""
        # Pre-allocate array
        data = np.zeros((self.n_samples, n_cell_types), dtype=np.float32)

        # Complexity levels: 2 to n_cell_types
        complexity_levels = list(range(2, n_cell_types + 1))
        n_levels = len(complexity_levels)
        samples_per_level = self.n_samples // n_levels

        current_idx = 0

        for level_i, n_ct in enumerate(complexity_levels):
            # Determine range for this complexity level
            is_last = level_i == n_levels - 1
            end_idx = self.n_samples if is_last else current_idx + samples_per_level
            n_rows = end_idx - current_idx

            if n_rows <= 0:
                break

            frac_val = 1.0 / n_ct

            if n_ct == n_cell_types:
                # All cell types - simple assignment
                data[current_idx:end_idx, :] = frac_val
            else:
                # Vectorized random selection for multiple rows
                # Generate all random choices at once
                for row_offset in range(n_rows):
                    chosen_indices = np.random.choice(n_cell_types, size=n_ct, replace=False)
                    data[current_idx + row_offset, chosen_indices] = frac_val

            current_idx = end_idx

        return pd.DataFrame(data, index=sample_names, columns=cols).round(4)
