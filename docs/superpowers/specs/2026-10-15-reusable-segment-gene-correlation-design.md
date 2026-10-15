# Reusable Segment-Sampled Gene Correlation Design

## Problem

High-correlation gene selection currently calculates gene-expression/cell-fraction correlations from the target Mixed dataset and applies `gene_filtering_corr_threshold` and `gene_filtering_n_gene_max` before saving the result. That makes the saved correlation table dependent on the first filtering settings and unsuitable for reuse with different thresholds.

For broader cell-fraction coverage, gene-filtering targets should be able to use correlations calculated from a dedicated segment-sampled Mixed dataset generated without GEP-level or gene-level filtering.

## Goals

- Allow a Mixed dataset to name a separate Mixed dataset as its correlation source.
- Calculate and retain a complete, unfiltered gene-by-cell-type correlation table for that source.
- Reuse the raw table for gene filtering with each target's own correlation threshold and maximum genes per cell type.
- Preserve the existing high-correlation gene-list file precedence and current behavior when no source dataset is configured.
- Retain the source Mixed GEP and cell-fraction files for reproducibility; do not delete them after calculating correlations.

## Configuration

Add an optional global switch to calculate and persist complete correlations,
defaulting to off and overridable per Mixed dataset. Add an optional reference
dataset setting, also overridable per target:

```yaml
parameters:
  calculate_gene_cell_prop_correlation: false
  gene_filtering_correlation_reference_dataset: null

datasets:
  Mixed_segment_correlation_source:
    dataset_type: "Mixed"
    sampling_method_prop: "segment"
    gep_filtering: false
    gene_filtering: false
    plot_pca_for_comparison: false
    calculate_gene_cell_prop_correlation: true

  Mixed_gene_filtered_target:
    dataset_type: "Mixed"
    gene_filtering: true
    gene_filtering_correlation_reference_dataset: "Mixed_segment_correlation_source"
```

The source dataset must be declared before its target, use `sampling_method_prop: "segment"`, and have both `gep_filtering` and `gene_filtering` disabled. It must use compatible output cell types and gene identifiers.

## Correlation Data Flow

1. A Mixed dataset with `calculate_gene_cell_prop_correlation: true` computes Pearson correlations from all its genes and cell-fraction columns after generating its bulk data. This runs independently of `gene_filtering` and PCA plotting.
2. Save the raw matrix as `full_gene_corr_with_cell_fraction.csv` in that dataset's result directory. Its rows are genes, its columns are cell types, and no threshold or top-N filtering is applied. If the file already exists, reuse it without recomputation.
3. When a target names this dataset with `gene_filtering_correlation_reference_dataset`, load the raw matrix and apply that target's `gene_filtering_corr_threshold` and `gene_filtering_n_gene_max` in memory. Write the target-specific selected correlations and gene list under the target result directory. Never overwrite the raw matrix with a filtered table.
4. Generate the referenced source before its target. If a reference is configured but its raw matrix does not exist and correlation calculation is disabled for that source, fail with a message identifying the source and required setting. An existing raw matrix remains reusable when calculation is disabled.
5. Retain the current target-self correlation behavior when no reference dataset is configured, but save its raw matrix separately before deriving a filtered result.

An explicit dataset-level `gene_filtering_high_corr_gene_file` or shared
`paths.high_corr_gene_file_path` remains higher priority than correlation
calculation. If a precomputed gene list is configured, use it as today and do
not require a correlation source for that target.

## Validation And Errors

- Fail with a clear error if the referenced dataset is missing, appears after its target, is not Mixed, does not use segment sampling, or enables GEP filtering.
- Require a dedicated correlation source to have `gene_filtering: false`; retain its raw GEP and cell-fraction files for reproducibility.
- Fail if source correlation columns do not cover the target's requested cell types or no genes overlap with the target data.

## Testing

- Verify the raw correlation file includes every source gene and cell type without an `n_at_least_one` column or threshold-based row removal.
- Verify `calculate_gene_cell_prop_correlation` writes the raw file even when gene filtering and PCA are disabled.
- Verify two targets with different thresholds/top-N limits derive different selections from the same unchanged raw matrix.
- Verify precomputed high-correlation gene-list precedence and the no-reference fallback.
- Verify source dataset validation, execution order, and generated file paths.
