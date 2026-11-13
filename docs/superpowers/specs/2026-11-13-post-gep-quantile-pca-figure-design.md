# Post-GEP quantile PCA figure

This design adds a second PCA comparison figure for the accepted Mixed
dataset after the post-GEP gene-quantile filter. The existing pre-filter
figure remains unchanged. The new figure is generated only for the combined
gene-filtering mode when PCA comparison is enabled.

## Scope and activation

Generate the additional figure when `gene_filtering_type` is
`high_corr_gene_and_quantile_range` and the resolved
`plot_pca_for_comparison` setting is true. Reuse the existing plotting flag;
do not add another configuration option.

Keep the existing pre-filter PCA model, coordinates, figure, and summary
fields unchanged. The new figure uses its own model, coordinate file, cache
identity, and summary fields. `high_corr_gene` mode and runs with PCA
comparison disabled do not create the new artifacts.

## PCA inputs

Use the separate post-GEP filtered H5AD produced by the combined
gene-filtering mode. Its ordered variables are the final gene list G2. Use
the resolved TCGA comparison reference restricted to G2 and normalized over
G2. Both inputs are in normalized `log2(TPM + 1)` space.

Fit a new joint PCA over the post-filter Mixed profiles and TCGA profiles.
Reuse `_load_full_comparison_reference_log` and
`filtering_by_gene_list_and_pca_plot` rather than adding another PCA
implementation. Fit the PCA independently from the pre-filter model because
the two figures use different gene sets. The figures show alignment with
TCGA within each PCA space; PC coordinates are not directly comparable
between the figures. Run this step after writing the post-filter H5AD and
before optional sample-count PCA branches so it is not skipped by an early
return.

## Outputs and cache identity

Save the post-filter model, coordinates, and PC1-PC2 PNG beside the existing
comparison artifacts, using names that identify the post-GEP quantile
filter. For example:

- `comparison_pca_2_post_gep_quantile_range.joblib`
- `comparison_pca_2_post_gep_quantile_range.csv`
- `<dataset>_PCA_after_gene_quantile_filter_PC0_PC1.png`

Include the ordered G2 hash, post-filter H5AD identity, TCGA reference
identity, comparison component count, and normalization definition in the
new PCA cache metadata. Never reuse the pre-filter PCA cache for the
post-filtered inputs.

Preserve `pca_plot_file`, `pca_model_file`, and `pca_data_file` for the
pre-filter figure. Add separate `post_gep_quantile_pca_plot_file`,
`post_gep_quantile_pca_model_file`, and
`post_gep_quantile_pca_data_file` fields to `mixed_dataset_summary.json`.

## Failure handling

If the combined mode is active but the filtered H5AD or G2 cannot be loaded
or aligned with the TCGA reference, stop with an actionable error. If a
post-filter PCA artifact already exists, reuse it only when its metadata
matches the current G2, filtered input, reference, and PCA settings.

## Validation

Tests must verify:

- The combined mode with PCA enabled writes a second PNG, model, coordinates,
  and metadata without replacing the pre-filter artifacts.
- The post-filter PCA uses G2 and a TCGA reference normalized over G2.
- The new cache identity changes when the filtered input, G2, reference, or
  PCA component count changes.
- The summary retains the original pre-filter paths and reports distinct
  post-filter PCA paths.
- `high_corr_gene` mode and disabled PCA runs do not create post-filter PCA
  artifacts.
