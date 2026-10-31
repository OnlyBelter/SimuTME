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
4. When gene or GEP filtering is enabled, apply reference-variance screening
   using `gene_filtering_reference_variance_threshold` when it is not `null`,
   then resolve and persist the fixed gene list G before target generation.
5. Generate cell-type proportions and sample SCT profiles.
6. Compose bulk GEPs in the original linear expression space.
7. Subset reference and simulated profiles to G and normalize each row to
   1,000,000 in linear TPM.
8. Optionally reject profiles by gene ranges and the configured
   reference-based GEP filtering method.
9. Save output expression on G as `log2(TPM + 1)` with matching cell-fraction
   rows.
10. Save summaries, fraction distributions, and optional comparison PCA.

The independent unfiltered correlation source retains its full gene universe
and existing generation behavior.

## CPU candidate generation

Random sampling without patient constraints builds eligible ID pools once per
batch and avoids spawning worker processes for individual selection tasks.
Eager mixed-profile aggregation gathers bounded expression chunks and uses
float64 arithmetic. These paths activate automatically; no YAML change is
required. Patient-constrained sampling, neighbor generation, and lazy
aggregation retain their existing implementations.

Sampling probabilities and constraints are preserved, but random draws can
differ from the old multiprocessing implementation. RNA weighting,
normalization, noise, and filtering rules are unchanged. Each completed batch
logs time spent on fractions, sampling, aggregation, filtering, and saving.
Restart the simulation process to load updated source code.

Use `scripts/benchmark_cpu_candidates.py` to compare against Git revision
`801f248`. It supports synthetic workloads and `--reference <SCT_H5AD>` for
actual SCT expression. Filtering and file export are excluded from benchmark
timings.

## Required Inputs

### Global YAML fields

- `paths.simu_bulk_dir`
- `paths.merged_sc_dataset_file_path`
- `paths.tcga2cancer_type_file_path` when selecting reference cancer types with
  `filtering_ref_types`
- `paths.tcga_merged_tpm_file_path` for filtering and PCA reference
- `paths.gep_filtering_reference_file` as an optional GEP reference override
- `paths.gene_filtering_reference_file` as an optional gene-selection
  reference override when GEP filtering is disabled
- `paths.pred_cell_prop_file_path` (optional) for Dirichlet inference or
  segment caps when GEP filtering is enabled
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
- `gep_filtering_reference_file` as a dataset-level override for
  `paths.gep_filtering_reference_file`
- `gep_filtering_by_gene_range`
- `gep_min_percentage_within_gene_range`
- `gep_gene_quantile_range`
- `gep_pca_n_components`
- `gep_norm_ord`
- `gep_n_top`
- `gep_show_filtering_info`
- `gene_filtering_type`
- `gene_filtering_reference_file` as a dataset-level override for
  `paths.gene_filtering_reference_file`
- `gene_filtering_high_corr_gene_file`
- `calculate_gene_cell_prop_correlation`
- `gene_filtering_correlation_reference_dataset`
- `gene_filtering_corr_threshold`
- `gene_filtering_reference_variance_threshold`
- `gene_filtering_n_gene_max`
- `plot_pca_for_comparison`
- `gene_filtering_pca_n_components`
- `gene_filtering_pca_figsize`
- `dirichlet_alpha` for `sampling_method_prop: "dirichlet"`
- `pred_cell_prop_file_path` as an optional dataset-level override of
  `paths.pred_cell_prop_file_path`
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
  version: "sorted_bulk_imputed"  # Requires imputation on the referenced SCT dataset
```

For `source_type: "generated_dataset"`, `version` must be one of:

- `"original"`: use the generated SCT output before sorted-bulk imputation.
  This output remains available even when imputation is enabled.
- `"sorted_bulk_imputed"`: use the SCT output after sorted-bulk-guided
  imputation. Set `apply_sorted_bulk_imputation: true` on the referenced SCT
  dataset, not on the Mixed dataset, and run that SCT dataset first so its
  imputed output is recorded and available.

`version` selects an output; it does not enable or perform imputation.
If the referenced SCT dataset has imputation disabled or no recorded imputed
output, requesting `"sorted_bulk_imputed"` raises an error. There is no
automatic fallback to `"original"`. Use `version: "original"` when you want
the pre-imputation reference.

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
- `segment` with GEP filtering and `pred_cell_prop_file_path`: rejects segment
  compositions that exceed any output cell type's global maximum in the
  normalized prediction table.
- `mixed` (a composite mode that distributes samples across all five methods above)

These control how cell-type proportions are generated before composing the mixed
bulk samples.

Set `paths.pred_cell_prop_file_path` once to share a predicted proportion CSV
across Mixed datasets. A dataset-level `pred_cell_prop_file_path` overrides
the shared path. Dirichlet sampling uses the file to estimate alpha values;
segment sampling uses it for maximum caps only when `gep_filtering: true`
and no `cell_prop_prior` is supplied.
If a dataset is neither Dirichlet nor segment with GEP filtering enabled, the
shared path is not used.

For `sampling_method_prop: "dirichlet"`, the workflow also supports:

- `dirichlet_alpha`
  A dictionary of concentration parameters keyed by output cell type. Missing
  keys default to `1.0` at the sampling layer.
- `pred_cell_prop_file_path`
  An optional dataset-level override of `paths.pred_cell_prop_file_path`.
  When a file is configured, SimuTME estimates `dirichlet_alpha` from the
  matched cell-type columns and uses the estimated values instead of the
  manual dictionary.
- `dirichlet_sampling_seed`
  An optional seed for reproducible Dirichlet cell-fraction draws. A dataset
  value overrides `parameters.dirichlet_sampling_seed`; the seed does not
  control other stochastic steps in Mixed generation.

For GEP-filtered segment sampling, set `cell_prop_prior` to a mapping from
output cell types to `[minimum, maximum]` ranges, or to the
`"DeSide_predicted_TCGA_range"` preset. The preset uses the raw ranges in
`simutme.config.DESIDE_TCGA_CELL_PROPORTION_RANGES`. Set the optional
`enlarge_cell_prop_prior` to a nonnegative number, default `0`, to subtract it
from each minimum and add it to each maximum, clipped to `[0, 1]`.
For example, `0.15` expands `[0.2, 0.9]` to `[0.05, 1.0]`.

When a prior is supplied, it takes precedence and prediction-file maximum
caps are not applied. Missing labels in an explicit mapping remain
unconstrained; the preset must cover every output label. The workflow checks
invalid or infeasible bounds before candidate expression generation.

Without a prior, `paths.pred_cell_prop_file_path` or its dataset-level override
provides caps. SimuTME requires a numeric, nonnegative column for every
output cell type, normalizes each row to sum to one, and uses the maximum
normalized value in each column as that cell type's cap. Extra metadata
columns are ignored.

These caps constrain marginal maxima across the complete file; they do not
condition on its `cancer_type` column or reproduce the joint distribution of
observed cell fractions. Segment proposals are rejection-sampled before
expression aggregation, and constrained runs log proposal acceptance. If proposal
acceptance is too low, fraction generation stops at a bounded attempt count
with an error rather than looping indefinitely. The DeSide file contains
predicted proportions, so these maxima define a sampling constraint rather
than biological ground truth.

## Main Outputs

For a dataset named `<dataset_name>`, the Mixed workflow writes files under:

`paths.simu_bulk_dir/<sampling_method_prop>_<n_sc_datasets>ds_n_base<n_base>_<output_tag>/`

When `gep_filtering: false`, `<output_tag>` is `no_filtering`, regardless of
the configured `gep_filtering_method`. When filtering is enabled, methods other
than `mgep` retain the historical `<filtering_tag>_median_gep` suffix.

The `mgep` method uses a distinct
`<sampling_method_prop>_<n_sc_datasets>ds_n_base<n_base>_<filtering_tag>_mgep/`
suffix.
Because the non-`mgep` methods share an output suffix, use a new target name or
output location when switching between them; filtering metadata rejects
incompatible reuse.

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
- `cell_fraction_summary_all_cell_types.csv` when `sampling_method_prop: "dirichlet"` or GEP filtering is enabled; contains non-cancer cell types
- `<cell_type>_fraction_distribution.png` when `sampling_method_prop: "dirichlet"` or GEP filtering is enabled
- `<cell_type>_fraction_distribution.pdf` when `sampling_method_prop: "dirichlet"` or GEP filtering is enabled
- `dirichlet_alpha_estimated_from_pred_cell_props.csv` when `sampling_method_prop: "dirichlet"` and `pred_cell_prop_file_path` is provided

The plots use the fraction table of accepted samples after GEP filtering.
Gene filtering or PCA visualization alone does not enable all-cell-type plots.

## Optional Filtering And PCA Outputs

Gene selection runs before GEP-level profile filtering. With
`gene_filtering: true`, use `gene_filtering_type: high_corr_gene`. A supplied
`gene_filtering_high_corr_gene_file`, falling back to
`paths.high_corr_gene_file_path`, takes precedence over a correlation source.
Otherwise, `gene_filtering_correlation_reference_dataset` is required.
The reference-variance screen described below runs before correlation-based or
supplied-list selection when either gene or GEP filtering is enabled.

The fixed list G is shared by normalization, optional per-gene range checks,
reference-only GEP-filtering PCA, and accepted-profile export. With
`gene_filtering: false` and `gep_filtering: true`, G starts from the common
reference/SCT gene universe and is narrowed by reference-variance screening
when that threshold is configured. With GEP filtering disabled and gene
filtering enabled, all generated profiles are normalized and exported on G
without profile rejection.

`gep_filtering` gates profile rejection. `gep_filtering_by_gene_range`
enables the range acceptance check; it does not remove genes. The supported
reference-distance methods are `median_gep`, `mean_gep`, `mgep`, and
`linear_mmd`. Removed or unknown method names fail during configuration
resolution and list the supported methods.

Set `gep_filtering_method: "mgep"` with `filtering_in_pca_space: false` to
filter in linear TPM space. SimuTME normalizes reference and simulated
profiles over G to 1,000,000, calculates the per-gene median across the
resolved reference samples, and computes each reference profile's L1 distance
to that mGEP. It keeps simulated profiles at or below the configured upper
reference-distance quantile. Set the lower value in `gep_filtering_quantile`
to `null` or `0`; this method applies no lower-distance cutoff.

The other centroid methods use `log2(TPM + 1)` values, either in G or in
reference-fitted PCA space according to `filtering_in_pca_space`.

Every distance-filtered Mixed dataset saves
`gep_filtering_distance_distribution.png` and `.svg` in its result directory.
The figure compares the filtering-reference distances with distances of the
final saved simulated profiles. Dotted lines mark the applicable reference
quantile cutoffs. The labels identify the distance space, metric, and sample
counts. The reference is the cohort used for filtering, including any
`filtering_ref_types` selection. Rejected candidates are not shown.

The figure is regenerated for compatible cached outputs. Saved log-expression
values are converted back to linear TPM and normalized on G before calculating
distances, so rounding can cause small differences near a cutoff. The
diagnostic does not change the filter or its accepted dataset.

Set `gene_filtering_reference_variance_threshold: 1.0` to retain only reference
genes with population variance strictly greater than 1 TPM squared
(`ddof=0`, calculated in linear expression before subsetting or
renormalization). This criterion applies before correlation-based or supplied
gene-list selection and cannot be bypassed by a supplied gene list. The
selected reference cohort is used for variance screening.

The example YAML uses 1.0; the code default is `null` (disabled) for existing
configurations. Dataset-level values override the global setting. Zero removes
constant genes only. Negative or non-finite thresholds are invalid. The
criterion runs when either gene or GEP filtering is enabled; fully unfiltered
datasets, including reusable correlation sources, retain their gene universe.

After selection, both reference and simulated profiles are subset to final G
and renormalized to 1,000,000 before the log transform. The input reference CSV
is not modified. Changing the threshold requires a fresh target name/output
location because existing filtering caches are incompatible.

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
- `reference_gene_variance.csv` when variance selection is enabled, including
  `variance_TPM2`, `variance_pass`, and `selected_in_G` for each reference gene
- `comparison_pca_<n_components>_gene_loadings_sorted_PC1.csv`
- `<dataset_name>_PCA_with_TCGA_correlation-first-v1_PC0_PC1.png`
- `full_gene_corr_with_cell_fraction.csv` for reusable, unfiltered correlations
- `mixed_dataset_summary.json`

The summary records the workflow version, cache identity, pre-selection,
shared, and exported gene counts, and acceptance totals. It records
`gep_pca_n_components` as the configured target and `gep_filtering_n_pcs`
as the actual fitted count, not a second configuration setting. Both PCA
fields are null when GEP filtering does not use PCA space.

The single loading table includes all final genes with signed `PC1_loading`
and `PC2_loading`. It is sorted descending by `PC1_loading`, with
gene names breaking ties; sorting is by signed value, not absolute magnitude.
It describes the joint comparison PCA, not the TCGA-only acceptance model,
and is saved beside the comparison plot for fresh and reused models.
The summary records its path and the variance threshold, scope, input gene
count, removed gene count, and number of reference samples used for selection.

For correlation-based gene selection, `calculate_gene_cell_prop_correlation`
creates `full_gene_corr_with_cell_fraction.csv` without applying a threshold
or top-N limit. Each value is the correlation between a gene's expression and
one cell type's fraction across source samples. A target can reuse that matrix
by naming an earlier Mixed dataset with
`gene_filtering_correlation_reference_dataset`. The source must use segment
sampling with GEP- and gene-level filtering disabled. Each target applies its
own `gene_filtering_corr_threshold` and selects up to
`gene_filtering_n_gene_max` genes per output cell type. The final list is the
union across cell types, so a gene is included when it passes the threshold
and top-N selection for at least one cell type. These selections do not alter
the reusable matrix. No correlations are calculated from target samples, and
no implicit pilot dataset is created. A supplied high-correlation gene list
takes precedence and is not modified by correlation thresholds.

The primary target `.h5ad` contains exactly G as normalized
`log2(TPM + 1)`. `gene_filtering_save_filtered_h5ad` optionally creates an
additional named copy beside it, using `gene_filtering_filtered_dataset_postfix`
or the default `filtered_by_high_corr_gene` suffix.

Existing targets or resumable batches with missing or incompatible filtering
metadata fail before reuse. Use a new dataset name/output location; SimuTME
does not delete old artifacts or relabel historical summaries. The independent
unfiltered correlation source remains reusable. Existing KNN-generated files
are left untouched and are never automatically reused.

## Global Defaults And Dataset Overrides

Global filtering options live under `parameters`; shared reference-file paths
live under `paths`. Dataset-level values for `gep_filtering_reference_file`
and `gene_filtering_reference_file` override their corresponding shared paths.

- `gep_filtering_enabled_default`
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
