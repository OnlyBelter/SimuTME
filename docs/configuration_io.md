# Configuration and I/O Reference

## YAML Structure

SimuTME expects a YAML file with four top-level sections:

- `paths`
- `constants`
- `parameters`
- `datasets`

The package loads `paths`, `constants`, and `parameters` into `SimuConfig`, and
then reads `datasets` separately when executing the workflow.

## Paths

### Core paths

- `simu_bulk_dir`
  Root output directory for generated datasets.

- `sct_dataset_file_path`
  Path to an SCT `.h5ad` reference file. This is used directly by Mixed
  simulations unless it is replaced in memory by an earlier SCT run in the same
  workflow.

- `merged_sc_dataset_file_path`
  Merged single-cell input `.h5ad`.

- `tcga2cancer_type_file_path`
  Sample-to-cancer-type annotation for TCGA reference data.

- `tcga_merged_tpm_file_path`
  TCGA bulk expression matrix used by filtering and PCA-based validation.

- `high_corr_gene_file_path`
  Gene list used by filtering and post-processing.

### Optional sorted-bulk-guided SCT imputation paths

- `sorted_bulk_expr_file_path`
  Sorted bulk reference expression matrix.

- `sorted_bulk_annot_file_path`
  Annotation table for the sorted bulk samples.

- `sorted_bulk_cell_type_mapping_file_path`
  Cell-type mapping file from SCT labels to sorted bulk reference labels. This
  file must also include an `included` column:
  - `included == 1`: average the SCT TPM profile with the Euclidean-nearest
    matched sorted-bulk TPM profile using `0.5 / 0.5`
  - `included == 0`: keep the zero-guided replacement workflow

- `sorted_bulk_imputation_out_dir`
  Parent directory for SCT post-imputation diagnostics. The actual outputs are
  written under:

  `sorted_bulk_imputation_out_dir/<dataset_name>/`

## Constants

### `sc_dataset_ids`

List of single-cell dataset identifiers to retain from the merged scRNA-seq
dataset.

### `cell_type2subtypes`

Maps major cell types to the subtypes used in simulation.

### `output_types`

Defines which cell types or subtypes should appear in generated outputs.

### `total_rna_coefficient`

Per-cell-type scaling coefficients used during bulk simulation.

## Parameters

### Core simulation parameters

- `gene_quantile_range`
- `gep_filtering_quantile`
- `pca_n_components`
- `gene_list_type`
- `n_base`

### Optional sorted-bulk-guided SCT imputation parameters

- `sorted_bulk_cell_type_col`
  Cell-type column in the sorted bulk annotation file.

- `sorted_bulk_imputation_eps`
  Zero threshold used by sparsity calculations and imputation logic.

- `sorted_bulk_imputation_seed`
  Random seed used for diagnostic sampling such as boxplot subsampling.

- `sorted_bulk_imputation_batch_size`
  Batch size used during SCT post-imputation.

- `sorted_bulk_imputation_boxplot_sample_size`
  Number of SCT samples used when drawing diagnostic boxplots.

- `sorted_bulk_imputation_bulk_weight`
  Sorted-bulk contribution used by the `included == 1` averaging branch during
  SCT post-imputation. A value of `0.5` keeps the previous 50/50 blend, `0.0`
  keeps the SCT profile, and `1.0` uses only the matched sorted-bulk profile.

- `sorted_bulk_imputation_y_frac_threshold`
  Maximum fraction of newly added expression allowed in each SCT sample for the
  `included == 0` zero-guided branch. It is not used for rows processed by the
  `included == 1` averaging branch.

- `sorted_bulk_imputation_output_suffix`
  Suffix appended to the imputed SCT `.h5ad` filename.

## Dataset Definitions

Each entry under `datasets:` defines one simulation task.

### Common dataset fields

- `dataset_type`
- `n_samples`
- `filtering`
- `filtering_ref_types`
- `cell_prop_prior`
- `zero_ratio_threshold`
- `cell_type_col`
- `cell_subtype_col`

### SCT-only fields

- `sampling_method_gep`
- `max_cells_per_cell_type`
- `apply_sorted_bulk_imputation`

### Mixed-only fields

- `sampling_method_prop`
- `sampling_method_gep` (optional but recommended)
- `filtering_in_pca_space`
- `dirichlet_alpha` (only when `sampling_method_prop: "dirichlet"`)
- `pred_cell_prop_file_path` (only when `sampling_method_prop: "dirichlet"`)

## Input Files By Workflow

### SCT simulation inputs

Required:

- merged scRNA-seq `.h5ad`
- YAML configuration file

Optional:

- sorted bulk expression matrix
- sorted bulk annotation table
- SCT-to-sorted-bulk cell-type mapping file

### Mixed simulation inputs

Required:

- YAML configuration file
- merged scRNA-seq `.h5ad`
- SCT reference `.h5ad` via `sct_dataset_file_path`

Optional but commonly used:

- TCGA TPM matrix
- TCGA sample annotation
- high-correlation gene list
- predicted cell-proportion CSV for Dirichlet alpha estimation when using
  `sampling_method_prop: "dirichlet"`

## Output Files By Workflow

### SCT simulation outputs

Typical outputs:

- cell fraction table
- sampled single-cell ID table
- simulated SCT CSV in `log2(TPM+1)` format
- simulated SCT `.h5ad`
- optional `sample_diversity/` result folder with:
  - `config_used.yaml`
  - `per_cell_type_distance.csv`
  - `per_cell_type_summary.csv`
  - `per_cell_type_mGEP_log.csv`
  - cell-type boxplots and distance-distribution figures

If sorted-bulk-guided post-imputation is enabled:

- imputed SCT `.h5ad`
- pre/post SCT sparsity tables
- pre/post SCT sparsity boxplots
- `y_fraction` per-cell tables and summaries
- `y_fraction` before/after-adjustment boxplots
- sorted bulk sparsity summaries

Rows processed through the `included == 1` averaging branch still appear in the
same output files, but their `y` values are recorded as zero because no
zero-guided addition or cap step is applied there.

Sample-diversity outputs are saved under:

- `paths.sorted_bulk_imputation_out_dir/<dataset_name>/sample_diversity/` when
  `paths.sorted_bulk_imputation_out_dir` is configured
- otherwise `paths.simu_bulk_dir/<dataset_name>/sample_diversity/`

### Mixed simulation outputs

Typical outputs:

- cell fraction table
- sampled single-cell ID table
- neighbor usage table
- simulated mixed bulk CSV in `log2(TPM+1)` format
- simulated mixed bulk `.h5ad`
- dataset-local `Cancer Cells` fraction diagnostics under
  `paths.simu_bulk_dir/<dataset_name>/`:
  - `cancer_cell_fraction_distribution.png`
  - `cancer_cell_fraction_distribution.pdf`
  - `cancer_cell_fraction_summary.csv`

Optional post-processing outputs:

- filtered `.h5ad`
- gene list CSVs
- PCA model files
- PCA score tables
- PCA figures

## Usage Pattern

Minimal execution pattern:

```python
import yaml
from simutme import SimuConfig, run_bulk_simulation

config_path = "config.yaml"
config = SimuConfig.from_yaml(config_path)

with open(config_path, "r", encoding="utf-8") as f:
    datasets_to_run = yaml.safe_load(f).get("datasets", {})

for dataset_name, params in datasets_to_run.items():
    run_bulk_simulation(dataset_name, params, config)
```

## Ordering Recommendation

If a Mixed dataset should consume an SCT dataset generated and optionally
imputed in the same run, define the SCT dataset before the Mixed dataset in the
YAML file.
