# Mixed Bulk RNA-seq Workflow

## Goal

The Mixed workflow generates synthetic bulk RNA-seq gene expression profiles by
combining cell-type proportions with sampled expression profiles from
single-cell-derived references.

Each generated mixed sample is stored in `log2(TPM+1)` format together with its
ground-truth cell fractions.

## High-Level Steps

1. Load the YAML configuration and dataset-specific parameters.
2. Determine which SCT reference dataset should be used.
3. Initialize `BulkGEPGenerator`.
4. Generate cell-type proportions using `sampling_method_prop`.
5. Sample expression profiles from the reference source.
6. Compose mixed bulk GEPs.
7. Save cell-fraction, CSV, and `.h5ad` outputs.
8. Optionally apply filtering and PCA-based validation.

The visual summary is shown below.

![Mixed workflow](figures/workflow_bulk.svg)

## Required Inputs

### Global YAML fields

- `paths.simu_bulk_dir`
- `paths.merged_sc_dataset_file_path`
- `paths.sct_dataset_file_path`
- `paths.tcga2cancer_type_file_path` for some filtering workflows
- `paths.tcga_merged_tpm_file_path` for filtering and PCA reference
- `constants.sc_dataset_ids`
- `constants.cell_type2subtypes`
- `constants.total_rna_coefficient`
- `parameters.n_base`

### Dataset-level fields

- `dataset_type: "Mixed"`
- `n_samples`
- `sampling_method_prop`
- `filtering`

### Common optional dataset-level fields

- `sampling_method_gep`
- `filtering_ref_types`
- `cell_prop_prior`
- `filtering_in_pca_space`

## Reference Source Used By Mixed Simulation

Mixed simulation uses `config.sct_dataset_file_path` as the SCT reference input.

This path can come from two cases:

1. a static path from `paths.sct_dataset_file_path` in the YAML file
2. an SCT dataset generated earlier in the same run

If a preceding SCT dataset also enables sorted-bulk-guided post-imputation, then
Mixed simulation will use the **imputed SCT dataset**, because SimuTME updates
`config.sct_dataset_file_path` in memory after the SCT post-step finishes.

## Proportion Sampling Modes

Supported values for `sampling_method_prop`:

- `segment`
- `seg_random`
- `sparse`
- `random`
- `dirichlet`
- `mixed`

These control how cell-type proportions are generated before composing the mixed
bulk samples.

## Main Outputs

For a dataset named `<dataset_name>` and SCT sampling method
`<sampling_method_gep>`, the Mixed workflow writes files under:

`paths.simu_bulk_dir/<sampling_method_prop>_<n_sc_datasets>ds_n_base<n_base>_<filtering_tag>_median_gep/`

Main files:

- `cell_fractions_<dataset_name>_gepsamp-<sampling_method_gep>.csv`
- `ref2n_neighbors_<dataset_name>_gepsamp-<sampling_method_gep>.csv`
- `simu_gep_<dataset_name>_gepsamp-<sampling_method_gep>_sampled_sc_cell_id.csv`
- `simu_gep_<dataset_name>_gepsamp-<sampling_method_gep>_log2cpm1p.csv`
- `simu_gep_<dataset_name>_gepsamp-<sampling_method_gep>_log2cpm1p.h5ad`

## Optional Filtering And PCA Outputs

If `filtering: true`, the workflow may additionally generate:

- filtered `.h5ad` files
- filtered gene lists
- PCA model files
- PCA-transformed data tables
- PCA plots comparing simulated bulk data with TCGA reference samples

These outputs are created by the post-processing stage in the workflow.

## When To Use This Workflow

Use Mixed simulation when you need:

- training data with known cell fractions
- benchmarking data for deconvolution methods
- mixed bulk data with different composition priors
- optional reference refinement using an SCT dataset generated or updated in the
  same workflow run
