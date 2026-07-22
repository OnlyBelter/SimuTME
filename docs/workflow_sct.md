# SCT Workflow

## Goal

The SCT workflow generates single-cell-type gene expression profiles from a
merged scRNA-seq dataset. Each simulated sample represents one cell type or
subtype and is stored in `log2(TPM+1)` format.

This workflow is used to:

- create SCT reference datasets
- provide reference profiles for later Mixed bulk simulation
- optionally produce an imputed SCT reference using a sorted bulk dataset

## High-Level Steps

1. Load the merged scRNA-seq dataset.
2. Restrict to the selected datasets and configured cell types/subtypes.
3. Optionally remove cells with very high zero ratio.
4. Sample cells for each target cell type or subtype.
5. Average the sampled cells into SCT profiles.
6. Save CSV and `.h5ad` outputs.
7. Optionally apply sorted-bulk-guided SCT post-imputation.

The visual summary is shown below.

![SCT workflow](figures/workflow_sct.svg)

## Required Inputs

### Global YAML fields

- `paths.merged_sc_dataset_file_path`
- `paths.simu_bulk_dir`
- `constants.sc_dataset_ids`
- `constants.cell_type2subtypes`
- `constants.output_types`
- `parameters.n_base`

### Dataset-level fields

- `dataset_type: "SCT"`
- `n_samples`

### Common optional dataset-level fields

- `sampling_method_gep`
- `zero_ratio_threshold`
- `cell_type_col`
- `cell_subtype_col`
- `max_cells_per_cell_type`

## Sampling Behavior

### `sampling_method_gep: "random"`

Randomly sample cells from the target class and average them.

### `sampling_method_gep: "n_neighbors"`

Select a seed cell and then use its nearest neighbors within the same class to
construct the SCT profile. This mode is more expensive and may require limiting
the number of cells per class via `max_cells_per_cell_type`.

## Main Outputs

For a dataset named `<dataset_name>` and sampling method `<sampling_method_gep>`,
the SCT generator writes files under:

`paths.simu_bulk_dir/generated_sct_dataset_<n_sc_datasets>ds_n_base<n_base>_all_subtypes/`

Main files:

- `cell_fractions_<dataset_name>_gepsamp-<sampling_method_gep>.csv`
- `simu_gep_<dataset_name>_gepsamp-<sampling_method_gep>_sampled_sc_cell_id.csv`
- `simu_gep_<dataset_name>_gepsamp-<sampling_method_gep>_log2cpm1p.csv`
- `simu_gep_<dataset_name>_gepsamp-<sampling_method_gep>_log2cpm1p.h5ad`

## Optional Sorted-Bulk-Guided SCT Post-Imputation

This workflow is enabled at the dataset level with:

```yaml
apply_sorted_bulk_imputation: true
```

It only applies to datasets with:

```yaml
dataset_type: "SCT"
```

### Additional Required Inputs

- `paths.sorted_bulk_expr_file_path`
- `paths.sorted_bulk_annot_file_path`
- `paths.sorted_bulk_cell_type_mapping_file_path`

### Additional Optional Global Parameters

- `parameters.sorted_bulk_cell_type_col`
- `parameters.sorted_bulk_imputation_eps`
- `parameters.sorted_bulk_imputation_seed`
- `parameters.sorted_bulk_imputation_batch_size`
- `parameters.sorted_bulk_imputation_boxplot_sample_size`
- `parameters.sorted_bulk_imputation_y_frac_threshold`
- `parameters.sorted_bulk_imputation_output_suffix`
- `paths.sorted_bulk_imputation_out_dir`

### What The Post-Imputation Step Does

1. Load the newly generated SCT `.h5ad`.
2. Load and align the sorted bulk expression matrix to the SCT gene list.
3. Map each SCT cell type to a sorted bulk reference cell type.
4. For each SCT sample, find the Euclidean-nearest sorted bulk sample in TPM
   space within the mapped reference class.
5. Branch by the `included` flag in the mapping file:
   - `included == 1`: replace the SCT profile with the TPM-space average
     `0.5 * SCT + 0.5 * bulk`
   - `included == 0`: impute genes that are near zero in the SCT sample using
     the chosen sorted bulk sample
6. For `included == 0`, limit the newly added expression by
   `sorted_bulk_imputation_y_frac_threshold`.
7. Save an imputed SCT `.h5ad`.
8. Save diagnostics such as pre/post sparsity summaries and `y_fraction`
   summaries.

### Outputs From The Post-Imputation Step

The imputed SCT `.h5ad` is saved beside the original SCT `.h5ad` using:

`<original_stem><sorted_bulk_imputation_output_suffix>.h5ad`

Diagnostic outputs are saved under:

`paths.sorted_bulk_imputation_out_dir/<dataset_name>/`

Examples of diagnostic files:

- `<dataset_name>_pre_imputation_sct_sparsity_per_cell.csv`
- `<dataset_name>_pre_imputation_sct_sparsity_summary.csv`
- `<dataset_name>_pre_imputation_sct_sparsity_boxplot.png`
- `<dataset_name>_post_imputation_sct_sparsity_per_cell.csv`
- `<dataset_name>_post_imputation_sct_sparsity_summary.csv`
- `<dataset_name>_post_imputation_sct_sparsity_boxplot.png`
- `<dataset_name>_sct_added_expression_y_per_cell.csv`
- `<dataset_name>_sct_added_expression_y_summary.csv`
- `<dataset_name>_sct_added_expression_y_fraction_before_adjustment_boxplot.png`
- `<dataset_name>_sct_added_expression_y_fraction_after_adjustment_boxplot.png`

The `y` diagnostics remain meaningful for rows processed with `included == 0`.
Rows processed with `included == 1` use direct TPM averaging and therefore
contribute zero in the `y` output tables.

## Handoff To Mixed Simulation

If the sorted-bulk-guided post-imputation step is enabled, SimuTME updates the
in-memory `config.sct_dataset_file_path` to the imputed SCT `.h5ad`. That means
later Mixed simulations in the same run will use the imputed SCT dataset as
their reference input.

Because datasets are executed in YAML order, the SCT dataset should appear
earlier than any Mixed dataset that should consume its imputed output.
