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
4. Resolve and persist the fixed gene list G before target generation.
5. Generate cell-type proportions and sample SCT profiles.
6. Compose bulk GEPs in the original linear expression space.
7. Subset reference and simulated profiles to G, normalize each row to
   1,000,000, and transform to `log2(TPM + 1)`.
8. Optionally reject profiles by gene ranges and reference-PCA distances.
9. Save accepted expression on G with matching cell-fraction rows.
10. Save summaries, fraction distributions, and optional comparison PCA.

The independent unfiltered correlation source retains its full gene universe
and existing generation behavior.

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
- `sct_reference`
- `n_samples`
- `sampling_method_prop`
- `gep_filtering`
- `gene_filtering`

### Common optional dataset-level fields

- `filtering_ref_types`
- `cell_prop_prior`
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
- `dirichlet_alpha` for `sampling_method_prop: "dirichlet"`
- `pred_cell_prop_file_path` for `sampling_method_prop: "dirichlet"`
- `dirichlet_sampling_seed` for `sampling_method_prop: "dirichlet"`

The old Mixed-dataset key `filtering` is no longer supported. Use
`gep_filtering` for GEP-level filtering instead.

## Reference Source Used By Mixed Simulation

Mixed simulation now requires an explicit `sct_reference` block in each Mixed
dataset entry.

### `sct_reference` fields

`source_type` is required and determines which remaining fields are valid:

| `source_type` | Required fields | Purpose and constraints |
| --- | --- | --- |
| `"path"` | `file_path` | Uses the specified external SCT `.h5ad` file directly. Do not set `dataset_name` or `version`. |
| `"generated_dataset"` | `dataset_name`, `version` | References an SCT dataset defined earlier in the same YAML file and generated during the current workflow run. `version` must be `"original"` or `"sorted_bulk_imputed"`. The latter requires the referenced SCT dataset to have produced its imputed output. |

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

Mixed generation always uses `random` when drawing SCT profiles from the
referenced SCT dataset. If `sampling_method_gep` is present in a Mixed dataset
entry, it is ignored and the workflow logs that `random` is being used.

This removes the previous implicit behavior where Mixed simulation depended on
the last in-memory SCT path update.

## Mixed SCT loading mode

Mixed simulation uses the global parameter
`parameters.mixed_sct_loading_mode` to choose how the referenced SCT `.h5ad`
is loaded before aggregation.

Supported values:

- `auto`
  Use eager in-memory loading for smaller SCT references and lazy backed
  loading for larger ones. The global parameter
  `parameters.mixed_sct_auto_eager_max_gb` sets the cutoff in GiB. SimuTME
  estimates the dense `float32` SCT matrix size as `n_obs * n_vars * 4`.
- `eager`
  Load the full SCT matrix into memory up front. This is usually the fastest
  option when the SCT reference fits comfortably in RAM.
- `lazy`
  Keep the SCT `.h5ad` backed on disk and materialize rows on demand. This
  reduces memory pressure for very large SCT references but can slow down large
  Mixed runs because it performs repeated row reads.

We recommend `auto` unless you already know the memory limits of the machine
running the workflow.

## Proportion Sampling Modes

Supported values for `sampling_method_prop`:

- `segment`: iteratively allocates fractions from the remaining total, producing
  uneven compositions; optional `cell_prop_prior` ranges are enforced by
  rejection sampling.
- `seg_random`: assigns one randomly sized fraction, then distributes the
  remaining fraction across the other cell types using normalized random draws.
- `sparse`: randomly selects the cell types present in each sample, sets all
  other cell types to zero, and normalizes random fractions among the present
  types.
- `random`: draws an independent uniform value for every cell type and
  normalizes the values to sum to one.
- `dirichlet`: draws a composition from a Dirichlet distribution; the
  `dirichlet_alpha` vector controls each cell type's expected abundance and the
  overall concentration of compositions.
- `mixed` (a composite mode that distributes samples across all five methods above)

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
- `dirichlet_sampling_seed`
  An optional seed for reproducible Dirichlet cell-fraction draws. A dataset
  value overrides `parameters.dirichlet_sampling_seed`; the seed does not
  control other stochastic steps in Mixed generation.

## Main Outputs

For a dataset named `<dataset_name>`, the Mixed workflow writes files under:

`paths.simu_bulk_dir/<sampling_method_prop>_<n_sc_datasets>ds_n_base<n_base>_<filtering_tag>_median_gep/`

The `knn_residual` method uses
`<sampling_method_prop>_<n_sc_datasets>ds_n_base<n_base>_with_filtering_knn_residual/`
instead, so new local-filter outputs are not labeled as centroid-filtered
datasets. Existing centroid and unfiltered-source locations are unchanged.

Main files:

- `cell_fractions_<dataset_name>_gepsamp-random.csv`
- `ref2n_neighbors_<dataset_name>_gepsamp-random.csv`
- `simu_gep_<dataset_name>_gepsamp-random_sampled_sc_cell_id.csv`
- `simu_gep_<dataset_name>_gepsamp-random_log2cpm1p.csv`
- `simu_gep_<dataset_name>_gepsamp-random_log2cpm1p.h5ad`

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

Gene selection runs before GEP-level profile filtering. With
`gene_filtering: true`, use `gene_filtering_type: high_corr_gene`. A supplied
`gene_filtering_high_corr_gene_file`, falling back to
`paths.high_corr_gene_file_path`, takes precedence over a correlation source.
Otherwise, `gene_filtering_correlation_reference_dataset` is required.

The fixed list G is shared by normalization, optional per-gene range checks,
reference-only GEP-filtering PCA, and accepted-profile export. With
`gene_filtering: false` and `gep_filtering: true`, G is the common
reference/SCT gene universe. With GEP filtering disabled and gene filtering
enabled, all generated profiles are normalized and exported on G without
profile rejection.

`gep_filtering` gates profile rejection. `gep_filtering_by_gene_range`
enables the range acceptance check; it does not remove genes. The supported
reference-distance methods are `median_gep`, `mean_gep`, and `linear_mmd`;
`knn_residual` adds held-out calibration for local TCGA proximity.
PCA fit and transform both use `log2(TPM + 1)` normalized over G.

`plot_pca_for_comparison` independently enables a diagnostic joint PCA on
the same normalized gene space. Its component count is separate from the
reference-only GEP-filtering PCA target.

These artifacts are written under the dataset-local directory:

`<mixed_output_dir>/<dataset_name>/`

Typical files include:

- `fixed_filtering_gene_list.csv` for targets with either filter enabled
- `filtering_metadata.json` for output and intermediate compatibility
- `gep_filtering_pca.joblib` and `gep_filtering_pca.metadata.json`
- `comparison_pca_<n_components>.joblib`, `.csv`, and `.metadata.json`
- `<dataset_name>_PCA_with_TCGA_correlation-first-v1_PC0_PC1.png`
- `full_gene_corr_with_cell_fraction.csv` for reusable, unfiltered correlations
- `mixed_dataset_summary.json`

The summary records the workflow version, cache identity, pre-selection,
shared, and exported gene counts, and acceptance totals. It records
`gep_pca_n_components` as the configured target and `gep_filtering_n_pcs`
as the actual fitted count, not a second configuration setting. Both PCA
fields are null when GEP filtering does not use PCA space.

For correlation-based gene selection, `calculate_gene_cell_prop_correlation`
creates `full_gene_corr_with_cell_fraction.csv` without applying a threshold
or top-N limit. A target can reuse that matrix by naming an earlier Mixed
dataset with `gene_filtering_correlation_reference_dataset`. The source must
use segment sampling with GEP- and gene-level filtering disabled. Each target
applies its own `gene_filtering_corr_threshold` and
`gene_filtering_n_gene_max`; those selections do not alter the reusable
matrix. No correlations are calculated from target samples, and no implicit
pilot dataset is created. A supplied high-correlation gene list takes
precedence and is not modified by correlation thresholds.

The primary target `.h5ad` contains exactly G as normalized
`log2(TPM + 1)`. `gene_filtering_save_filtered_h5ad` optionally creates an
additional named copy beside it, using `gene_filtering_filtered_dataset_postfix`
or the default `filtered_by_high_corr_gene` suffix.

Existing targets or resumable batches with missing or incompatible filtering
metadata fail before reuse. Use a new dataset name/output location; SimuTME
does not delete old artifacts or relabel historical summaries. The independent
unfiltered correlation source remains reusable.

### Cancer-stratified local filtering

Opt into `gep_filtering_method: knn_residual` when you need candidates
near actual TCGA profiles rather than only near a global reference center.

1. Match TCGA expression sample IDs to `paths.tcga2cancer_type_file_path`.
2. Reserve `ceil(0.2 * n)` individual samples within each cancer type,
   using dedicated split seed 42 by default.
3. Fit PCA, the neighbor database, and gene-range bounds on fitting samples
   only. Transform held-out samples using the same PCA and normalization.
4. Set upper thresholds from held-out neighbor distances and L2
   reconstruction residuals, using the 95th percentile of each by default.
5. Retain candidates that pass the optional range check and both score
   thresholds. There is no minimum distance cutoff.

Use `gep_pca_n_components: 0.9` for a 90% variance target and
`gep_norm_ord: 2` for Euclidean neighbor distance. The default neighbor count
is 10. All held-out samples contribute to calibration; they are not added
to the fitting PCA or neighbor database.

The method saves the split, per-cancer counts, held-out scores, neighbor
database, thresholds, and per-batch candidate scores. The summary records
patient overlap, marginal and joint pass counts, and exported sample counts.
Individual-sample stratification does not guarantee patient-independent
calibration.

The default `gep_max_generation_batches: 100` stops a run that cannot
collect the requested sample count. It counts candidate batches, not
individual profiles, and persists across resumptions. Thresholds are not
automatically loosened and intermediate artifacts are preserved on failure.

When comparison plotting is enabled, local-filter runs additionally save:

- `reference_only_PCA_coordinates.csv`
- `reference_only_PCA_PC1_PC2.png` and `.pdf`, using the filtering model
- `local_filter_score_distributions.png` and `.pdf`, with threshold markers

The joint comparison plot still uses all eligible TCGA samples. Plotting
does not fit or recalibrate the acceptance model. Two-dimensional visual
overlap is not a guarantee of high-dimensional similarity. Use a fresh
target name when migrating an existing centroid-filtered run.

## Global Defaults And Dataset Overrides

Global filtering defaults live under `parameters`:

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
- `gene_filtering_enabled_default`
- `gene_filtering_reference_file`
- `gene_filtering_type`
- `calculate_gene_cell_prop_correlation`
- `gene_filtering_correlation_reference_dataset`
- `gene_filtering_corr_threshold`
- `gene_filtering_n_gene_max`
- `gene_filtering_save_filtered_h5ad`
- `gene_filtering_filtered_dataset_postfix`
- `plot_pca_for_comparison`
- `gene_filtering_pca_n_components`
- `gene_filtering_pca_figsize`

Mixed datasets may override these defaults with the flat dataset-level fields
listed above.

The legacy `gene_filtering_plot_pca` key remains supported as an alias for
`plot_pca_for_comparison`; the new key takes precedence when both are set.

For backward compatibility, older global fields still work:

- `gene_list_type` maps to `gene_filtering_type`
- `gene_quantile_range` still supplies the default profile-range quantiles
- `pca_n_components` still feeds GEP-level PCA filtering defaults
- `gene_filtering` is independent from `gep_filtering`

Explicit legacy range-selection modes require migration to `high_corr_gene`
unless a supplied gene list takes precedence. Remove
`gene_filtering_quantile_range` and `gene_filtering_q_col_name`; retain
`gep_gene_quantile_range` for profile acceptance.

## When To Use This Workflow

Use Mixed simulation when you need:

- training data with known cell fractions
- benchmarking data for deconvolution methods
- mixed bulk data with different composition priors
- explicit reuse of either an external SCT reference or an SCT dataset
  generated earlier in the same workflow run
