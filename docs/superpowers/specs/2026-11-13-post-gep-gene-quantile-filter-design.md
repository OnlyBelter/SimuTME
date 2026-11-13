# Post-GEP gene quantile filtering

This design restores the combined correlation-and-quantile gene-filtering
mode as an explicit opt-in. It retains the correlation-selected gene
universe for generation and GEP filtering, then applies a second gene-level
quantile check to the final accepted Mixed samples and writes a separate,
renormalized `.h5ad`.

## Scope and activation

Enable the second stage only when `gene_filtering` is true and
`gene_filtering_type` is `high_corr_gene_and_quantile_range`. Keep the
`high_corr_gene` mode unchanged.

The combined mode requires a configured correlation source or a supplied
high-correlation gene list, as the current correlation-first workflow does.
It also requires `gene_filtering_save_filtered_h5ad: true`, because the
post-GEP filtered `.h5ad` is a required output in this mode. Reject
incompatible settings with an actionable configuration error.

Use `gene_filtering_quantile_range` as `[q_low, q_mid, q_high]`. To match
the q5-q95 rule, set `[0.05, 0.5, 0.95]` explicitly for the target dataset.
The middle quantile is the D1 statistic; it is the median when `q_mid` is
`0.5`. Do not use `gep_gene_quantile_range` for this gene-level filter;
that parameter controls the separate per-profile GEP acceptance check.

## Stage order

Keep the existing fixed gene list G and GEP sample acceptance behavior.
Apply the additional gene-level screen only after GEP filtering has produced
the final accepted Mixed profiles.

1. Select the fixed correlation-based gene list G before Mixed generation.
2. Generate Mixed samples and apply the configured GEP-level filters on G.
3. Calculate the gene quantile-range mask from all final accepted D1 samples
   and the resolved TCGA reference.
4. Intersect the passing genes with G to form the final gene list G2.
5. Subset each accepted D1 profile to G2, renormalize it, and write a
   separate filtered `.h5ad`.

The primary accepted Mixed `.h5ad`, sample set, and cell fractions remain
unchanged. The quantile stage must not reject or add samples.

## Quantile calculation and helper reuse

Use the existing `get_gene_list_filtered_by_quantile_range` helper for the
gene mask. It already compares the configured middle D1 quantile with the
inclusive lower and upper reference quantiles in log space.

Use the final accepted Mixed `.h5ad` as D1. Compare only genes in G and use
the same resolved TCGA reference cohort and normalization basis as the
fixed filtering space. Honor `filtering_ref_types` when it restricts that
reference cohort. For each gene g, retain it exactly when

```text
Q_q_low(TCGA_g) <= Q_q_mid(D1_g) <= Q_q_high(TCGA_g)
```

Both expression matrices use selected-gene row normalization to 1,000,000
in linear expression followed by `log2(TPM + 1)`. This constrains the
selected D1 quantile; it does not claim that the complete D1 and TCGA
expression distributions overlap.

Reuse the existing H5AD reader and normalization functions. Convert
accepted D1 `log2(TPM + 1)` values back to linear expression, subset to G2,
normalize each row over G2 to 1,000,000 with `normalize_selected`, and
transform the result with `log2_transform` before export. Process
normalization in bounded row batches so the extra linear-expression matrix
does not need to be materialized for all accepted samples at once.

## Outputs and compatibility

Write the new artifact beside the primary accepted Mixed dataset using the
combined-mode filtered filename, for example
`<primary_stem>_filtered_by_high_corr_gene_and_quantile_range.h5ad`.
Preserve accepted sample IDs and observation metadata. The output contains
only G2 in a deterministic gene order, with each sample renormalized over
G2.

Keep `fixed_filtering_gene_list.csv` as the pre-generation list G. Save G2
as a separate post-GEP gene-list artifact and expose both paths in
`mixed_dataset_summary.json`. Point `filtered_bulk_gep_file` to the new
artifact for the combined mode; preserve its current meaning for
`high_corr_gene`.

Record the quantile triplet, reference identity, accepted-input identity,
final gene-list identity, and normalization definition in filtered-artifact
metadata. Reuse an existing output only when this metadata matches. Fail
without overwriting when the existing artifact is incompatible.

## Error handling

Fail with an actionable error when the combined mode lacks a correlation
source or supplied list, the quantile triplet is invalid, no genes pass, or
the accepted D1 and TCGA data cannot be aligned. If any D1 sample has a
nonpositive or non-finite total expression over G2, identify the affected
sample IDs and do not silently drop samples or write invalid values.

## Validation

Tests must verify:

- `high_corr_gene` runs keep their current behavior.
- The combined mode runs correlation selection before GEP generation and
  quantile filtering after final GEP acceptance.
- D1 uses the configured middle quantile and TCGA supplies the lower and
  upper bounds.
- Values equal to either reference bound pass.
- The q5-q95 configuration uses `[0.05, 0.5, 0.95]`.
- The post-filtered `.h5ad` contains only G2, retains sample and observation
  metadata, and has per-row linear TPM totals of 1,000,000 before log
  transformation.
- The primary accepted `.h5ad` and cell-fraction file remain unchanged.
- Empty G2, zero-sum rows, invalid ranges, missing genes, and incompatible
  cached outputs fail without silently changing the sample set.
