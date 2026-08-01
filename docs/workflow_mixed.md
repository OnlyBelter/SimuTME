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
8. Save cell-fraction distribution outputs for the dataset.
9. Optionally apply filtering and PCA-based validation.

The visual summary is shown below.

![Mixed workflow](figures/workflow_bulk.svg)

## Required Inputs

### Global YAML fields

- `paths.simu_bulk_dir`
- `paths.merged_sc_dataset_file_path`
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

- `sct_reference`
- `sampling_method_gep`
- `filtering_ref_types`
- `cell_prop_prior`
- `filtering_in_pca_space`
- `dirichlet_alpha` for `sampling_method_prop: "dirichlet"`
- `pred_cell_prop_file_path` for `sampling_method_prop: "dirichlet"`

## Reference Source Used By Mixed Simulation

Mixed simulation now requires an explicit `sct_reference` block in each Mixed
dataset entry.

Supported modes:

1. external path

```yaml
sct_reference:
  source_type: "path"
  file_path: "./datasets/.../toy_sct.h5ad"
```

2. earlier SCT dataset in the same YAML file

```yaml
sct_reference:
  source_type: "generated_dataset"
  dataset_name: "SCT_N6K_31Jul_07"
  version: "sorted_bulk_imputed"
```

For `source_type: "generated_dataset"`, `version` must be one of:

- `"original"`
- `"sorted_bulk_imputed"`

For `source_type: "path"`, SimuTME uses `file_path` directly and does not infer
another SCT path from it.

This removes the previous implicit behavior where Mixed simulation depended on
the last in-memory SCT path update.

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

For `sampling_method_prop: "dirichlet"`, the workflow also supports:

- `dirichlet_alpha`
  A dictionary of concentration parameters keyed by output cell type. Missing
  keys default to `1.0` at the sampling layer.
- `pred_cell_prop_file_path`
  An optional CSV containing predicted cell proportions across samples. When
  provided, SimuTME estimates `dirichlet_alpha` from the matched cell-type
  columns and uses the estimated values instead of the manual dictionary.

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

Dataset-local diagnostic outputs under:

`paths.simu_bulk_dir/<dataset_name>/`

- `cancer_cell_fraction_distribution.png`
- `cancer_cell_fraction_distribution.pdf`
- `cancer_cell_fraction_summary.csv`
- `cell_fraction_summary_all_cell_types.csv` when `sampling_method_prop: "dirichlet"`
- `<cell_type>_fraction_distribution.png` when `sampling_method_prop: "dirichlet"`
- `<cell_type>_fraction_distribution.pdf` when `sampling_method_prop: "dirichlet"`
- `dirichlet_alpha_estimated_from_pred_cell_props.csv` when `sampling_method_prop: "dirichlet"` and `pred_cell_prop_file_path` is provided

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
- explicit reuse of either an external SCT reference or an SCT dataset
  generated earlier in the same workflow run
