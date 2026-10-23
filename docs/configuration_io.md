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

- `merged_sc_dataset_file_path`
  Merged single-cell input `.h5ad`.

- `tcga2cancer_type_file_path`
  Sample-to-cancer-type annotation for TCGA reference data.

- `tcga_merged_tpm_file_path`
  TCGA bulk expression matrix used by filtering and PCA-based validation.

- `high_corr_gene_file_path`
  Optional shared fallback high-correlation gene list used by Mixed
  filtering and post-processing. Set it to `null` for a clean first run.

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

These parameters affect shared simulation behavior across workflows.

- `n_base`
  Number of cells combined into each simulated profile.

- `gene_quantile_range`
  Legacy global quantile triplet still reused by some quantile-based
  filtering paths. Defaults to `[0.005, 0.5, 0.995]` when omitted.

- `gep_filtering_quantile`
  Legacy global quantile pair still reused by some GEP-level filtering paths.

- `pca_n_components`
  Legacy global PCA default still reused by some filtering paths. Defaults to
  `0.9` when omitted.

- `gene_list_type`
  Legacy global gene-filtering mode. It maps to `gene_filtering_type`
  and defaults to `high_corr_gene`.

### Mixed filtering parameters

These parameters define the global defaults for Mixed dataset filtering.
Dataset-level Mixed fields can override them.

#### GEP-level filtering defaults

- `gep_filtering_enabled_default`
- `gep_filtering_reference_file`
- `gep_filtering_ref_exp_type`
- `gep_filtering_method`
- `gep_filtering_quantile`
- `gep_n_top`
- `gep_show_filtering_info`
- `gep_filtering_by_gene_range`
- `gep_min_percentage_within_gene_range`
- `gep_gene_quantile_range`
- `gep_filtering_in_pca_space`
- `gep_pca_n_components`
- `gep_norm_ord`
- `gep_reference_holdout_fraction`
- `gep_reference_holdout_seed`
- `gep_knn_n_neighbors`
- `gep_knn_distance_quantile`
- `gep_pca_residual_quantile`
- `gep_max_generation_batches`

#### Gene-level filtering and PCA comparison defaults

- `gene_filtering_enabled_default`
- `gene_filtering_reference_file`
- `gene_filtering_type`
- `calculate_gene_cell_prop_correlation`
- `gene_filtering_correlation_reference_dataset`
- `gene_filtering_corr_threshold`
- `gene_filtering_reference_variance_threshold`
- `gene_filtering_n_gene_max`
- `gene_filtering_save_filtered_h5ad`
- `gene_filtering_filtered_dataset_postfix`
- `plot_pca_for_comparison`
- `gene_filtering_pca_n_components`
- `gene_filtering_pca_figsize`

When you use `pred_cell_prop_file_path` with segment sampling, SimuTME normalizes
each row over the output cell types and caps each generated fraction at that
cell type's global observed maximum. This requires GEP filtering. Segment
sampling without this file and Dirichlet alpha estimation retain their
existing behavior.

`gene_filtering_reference_variance_threshold` accepts a finite, nonnegative
number in TPM squared or `null` to disable selection (the code default).
The example sets `1.0`. Genes must have population variance (`ddof=0`)
strictly above the threshold in linear reference expression before subsetting.
Local KNN methods use fitting samples only. This criterion applies when
either gene or GEP filtering is enabled, before correlation/supplied-list
selection, and both reference and simulated profiles are renormalized on the
final genes to 1,000,000. Dataset overrides take precedence, including `null`.
Unfiltered correlation sources remain unchanged.

`plot_pca_for_comparison` is independent of `gene_filtering`: when enabled
without gene filtering, the PCA comparison uses all genes shared by the Mixed
dataset and TCGA. The legacy `gene_filtering_plot_pca` key remains supported;
the new key takes precedence if both are set.

`calculate_gene_cell_prop_correlation` writes the complete, unfiltered
gene-by-cell-type correlation table for a Mixed dataset. A target can name an
earlier segment-sampled, unfiltered Mixed dataset with
`gene_filtering_correlation_reference_dataset`; its own correlation threshold
and maximum genes per cell type are applied without changing the source table.
An explicit `gene_filtering_high_corr_gene_file` or
`paths.high_corr_gene_file_path` gene list takes precedence.

Gene selection runs before target generation, using only output cell types.
The selected list G is intersected with SCT and reference genes, persisted
in a deterministic order, and reused for normalization, profile checks,
PCA, and export. Each profile is normalized over G to 1,000,000 before
`log2(TPM + 1)` transformation. When GEP filtering is enabled, its reference
file and `filtering_ref_types` define the reference space; otherwise,
`gene_filtering_reference_file` is used.

Use `gene_filtering_type: high_corr_gene`. Without a supplied gene list,
`gene_filtering_correlation_reference_dataset` is required. Explicit legacy
range-selection modes fail with a migration message. Remove
`gene_filtering_quantile_range` and `gene_filtering_q_col_name`; the separate
`gep_gene_quantile_range` controls profile acceptance on G.

`gep_pca_n_components` is the only GEP-filtering PCA configuration parameter.
`gep_filtering_n_pcs` is read-only summary metadata obtained from the model.
The supported methods are `median_gep`, `mgep`, `mean_gep`, `linear_mmd`,
`knn`, and `knn_residual`. Existing methods retain their behavior.

`mgep` computes a coordinate-wise median from all resolved reference samples
using linear TPM values normalized over the selected genes to 1,000,000. It
keeps simulated profiles whose L1 distance from this mGEP is at or below the
upper quantile in `gep_filtering_quantile`. It does not apply a lower-distance
cutoff, so set the lower quantile to `null` or `0`. It requires
`filtering_in_pca_space: false`. Unlike `median_gep`, `mgep` does not transform
expression to `log2(TPM + 1)` or project it into PCA space.

For either local method, provide `paths.tcga2cancer_type_file_path` with unique
sample IDs and a `cancer_type` column. After exact sample alignment and
`filtering_ref_types` selection, hold out `ceil(fraction * n)` individual
samples per cancer type. Every type must retain fitting and held-out
samples. The default fraction is 0.2 and the dedicated split seed is 42.
Repeated patient samples are not grouped; patient overlap is reported.

Only fitting TCGA samples estimate PCA, neighbors, and optional gene-range
bounds. All held-out samples calibrate the two upper score percentiles,
including samples that fail the gene-range check. In `knn`, only neighbor
distance affects local acceptance; residual scores and their percentile
remain diagnostic. In `knn_residual`, both score thresholds must pass.

| Parameter | Default | Meaning in local methods |
| --- | --- | --- |
| `gep_reference_holdout_fraction` | `0.2` | Held-out fraction within each cancer type |
| `gep_reference_holdout_seed` | `42` | Reproducible individual-sample split seed |
| `gep_knn_n_neighbors` | `10` | Mean distance to k fitting-reference neighbors |
| `gep_knn_distance_quantile` | `0.95` | Held-out neighbor-distance upper percentile |
| `gep_pca_residual_quantile` | `0.95` | Held-out residual upper percentile; diagnostic only in `knn` |
| `gep_max_generation_batches` | `100` | Cumulative candidate-batch bound |

Both local methods require `gep_filtering_in_pca_space: true`. Use
`gep_pca_n_components: 0.9` to retain 90% of fitting-reference variance, or
an integer for an explicit component count. Raw PCs are not whitened.
`gep_norm_ord` selects Manhattan (1) or Euclidean (2) neighbor distance;
reconstruction residuals always use L2 in normalized log-expression space.
`gep_filtering_quantile` applies only to centroid methods and does not
impose a lower distance bound in either local method. KNN-only proximity
constrains the retained reference-PC space, not omitted expression directions.

In KNN-only candidate tables, `joint_pass` equals `knn_pass`; held-out
`joint_pass` also includes the optional gene-range check. An exported
candidate can have `residual_pass: false`. KNN summaries record
`residual_filter_enabled: false`, `local_filter_kept_n_samples`, and separate
diagnostic residual and neighbor/residual conjunction counts. Residual
plots label the percentile marker as diagnostic only.

With `gep_filtering: false`, no local split or calibration runs. Changing
methods, labels, split membership, or calibration settings requires
compatible metadata or a new output location; old artifacts are not
reused as local-filter runs.

### SCT sample-diversity parameters

These parameters control the optional SCT sample-diversity analysis step.

- `run_sample_diversity_analysis`
- `sample_diversity_gene_mode`
- `sample_diversity_n_top_genes`
- `sample_diversity_gene_list_path`
- `sample_diversity_distance_metric`
- `sample_diversity_min_samples_per_cell_type`

### Dirichlet sampling

- `dirichlet_sampling_seed`
  Optional seed for Dirichlet cell-fraction draws. A Mixed dataset can override
  the global value with its own `dirichlet_sampling_seed`. If neither value is
  set, SimuTME keeps using the global NumPy random state.

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

`filtering` is no longer a Mixed workflow field. Mixed datasets now use
`gep_filtering` and `gene_filtering` explicitly.

### SCT-only fields

- `sampling_method_gep`
- `max_cells_per_cell_type`
- `apply_sorted_bulk_imputation`
- `neighbor_sampling_scheme`
- `run_sample_diversity_analysis`
- `stratify_by_dataset`
- `cancer_cell_dataset_ids`
- `non_cancer_cell_dataset_ids`

For SCT datasets:

- `max_cells_per_cell_type` accepts a positive integer or `"all"`; omitted
  values default to 15,000. `"all"` retains all eligible cells and uses
  sampled-anchor exact queries for `n_neighbors` with `single_anchor` and no
  patient constraints. Other neighbor schemes reject `"all"` before loading.
  Existing QC, dataset selection, and balanced-quota downsampling still apply.
- `sampling_method_gep` is required and must be set explicitly to `random` or `n_neighbors`

### Mixed-only fields

- `sct_reference`
- `sampling_method_prop`
- `gep_filtering`
- `gene_filtering`
- `filtering_in_pca_space`
- `gep_filtering_method`
- `gep_filtering_quantile`
- `gep_filtering_reference_file`
- `gep_filtering_by_gene_range`
- `gep_min_percentage_within_gene_range`
- `gep_gene_quantile_range`
- `gep_pca_n_components`
- `gep_norm_ord`
- `gep_n_top`
- `gep_show_filtering_info`
- `gene_filtering_type`
- `gene_filtering_reference_file`
- `gene_filtering_high_corr_gene_file`
- `calculate_gene_cell_prop_correlation`
- `gene_filtering_correlation_reference_dataset`
- `gene_filtering_corr_threshold`
- `gene_filtering_n_gene_max`
- `plot_pca_for_comparison`
- `gene_filtering_pca_n_components`
- `gene_filtering_pca_figsize`
- `dirichlet_alpha` (only when `sampling_method_prop: "dirichlet"`)
- `pred_cell_prop_file_path` for Dirichlet alpha estimation, or for empirical
  per-type caps when `sampling_method_prop: "segment"` and `gep_filtering: true`
- `dirichlet_sampling_seed` (only when `sampling_method_prop: "dirichlet"`)

For Mixed datasets:

- `sampling_method_gep` is ignored if provided; Mixed generation always uses `random`
  when drawing SCT profiles from the referenced SCT dataset
- `gep_filtering` controls GEP-level filtering during Mixed generation
- `gene_filtering` selects the fixed gene list before generation
- `parameters.mixed_sct_loading_mode` controls whether the upstream SCT
  reference is loaded eagerly in memory, lazily in backed mode, or selected
  automatically based on matrix size
- `paths.high_corr_gene_file_path` is the shared fallback when a dataset does
  not provide a filtering-specific high-correlation gene file
- the old Mixed key `filtering` is no longer supported

### Global Mixed SCT loading mode

Set `parameters.mixed_sct_loading_mode` to control how Mixed simulation loads
its SCT reference `.h5ad`:

```yaml
parameters:
  mixed_sct_loading_mode: "auto"
  mixed_sct_auto_eager_max_gb: 4.0
```

Supported values:

- `auto`
  Use eager in-memory loading for smaller SCT references and lazy backed
  loading for larger ones. `mixed_sct_auto_eager_max_gb` sets the cutoff in
  GiB. SimuTME estimates the dense `float32` matrix size as
  `n_obs * n_vars * 4`.
- `eager`
  Load the full SCT matrix into memory before Mixed generation. This is usually
  the fastest option when RAM is sufficient.
- `lazy`
  Keep the SCT reference backed on disk and materialize rows on demand during
  Mixed aggregation. This uses less memory but can be slower for large runs.

### Mixed `sct_reference` schema

Each Mixed dataset must define:

```yaml
sct_reference:
  source_type: "path" | "generated_dataset"
  file_path: "..."      # required when source_type == "path"
  dataset_name: "..."   # required when source_type == "generated_dataset"
  version: "original" | "sorted_bulk_imputed"  # required only when source_type == "generated_dataset"
```

Notes:

- `source_type: "path"` uses the exact external SCT `.h5ad` file in `file_path`.
  No path inference is applied, and `version` must not be provided.
- `source_type: "generated_dataset"` references an SCT dataset defined earlier
  in the same YAML file.
- `version: "sorted_bulk_imputed"` applies only to `source_type: "generated_dataset"`
  and requires an imputed SCT output to exist.

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
- SCT reference `.h5ad` resolved from each Mixed dataset's `sct_reference`
- explicit `sampling_method_prop`

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

- `<generated_sct_parent>/<dataset_name>/sample_diversity/`

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
- gene and cell-fraction correlation tables
- PCA model files
- PCA score tables
- PCA figures
- dataset-level summary JSON

For targets with gene or GEP filtering enabled, the primary CSV and `.h5ad`
contain exactly the fixed list G in normalized `log2(TPM + 1)` space.
`gene_filtering_save_filtered_h5ad` controls an additional named copy, not
whether the primary output is restricted to G.

Dataset-local filtering artifacts include `fixed_filtering_gene_list.csv`,
`filtering_metadata.json`, and, when PCA filtering is used,
`gep_filtering_pca.joblib` with `gep_filtering_pca.metadata.json`.
The summary separates `pre_selection_n_genes`, `shared_n_genes`, and
`exported_n_genes`; `generated_n_genes` reports the actual artifact dimension.

Missing or incompatible filtering metadata prevents reuse of target outputs,
models, and intermediate batches. Use a new dataset name/output location.
Existing unfiltered correlation-source artifacts remain reusable.

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

If a Mixed dataset uses:

```yaml
sct_reference:
  source_type: "generated_dataset"
```

then define the referenced SCT dataset earlier than the Mixed dataset in the
YAML file.
