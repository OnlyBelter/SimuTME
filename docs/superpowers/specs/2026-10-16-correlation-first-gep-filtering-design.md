# Correlation-first GEP filtering design

This specification changes Mixed simulation so correlation-based gene
selection precedes normalization, optional gene-range profile screening,
and PCA-space GEP screening. It describes proposed behavior, not an
implemented feature.

## Scope and scientific semantics

The target uses one fixed ordered gene list for reference normalization,
simulated-profile normalization, per-gene range checks, PCA filtering,
and accepted-profile export. Denote this fixed list by G. Accepted target
profiles contain only G, not the full gene universe used for generation.
No post-generation expression-range gene selection takes place.

`gep_filtering_by_gene_range` remains a profile acceptance check. It does
not remove genes. The correlation thresholds determine which genes enter
the filtering space.

SCT generation, cell-fraction sampling, and sorted-bulk imputation are
unchanged. The unfiltered segment-sampled correlation source retains its
full gene universe and its existing generation behavior.

## Stage order

Apply the following steps to a target with gene selection enabled.

1. Resolve a supplied gene list or an earlier unfiltered segment-sampled
   correlation source before target GEP generation.
2. Intersect the candidate gene universe with the target SCT genes and
   reference genes. For a correlation source, select eligible genes with
   `gene_filtering_corr_threshold` and `gene_filtering_n_gene_max`, then
   take the union across the target output cell types.
3. Persist the resulting unique ordered gene list G. Resolve ties
   deterministically, and use identical gene order for both datasets.
4. Generate simulated bulk profiles in the original linear expression
   space. Subset both reference and simulated profiles to G.
5. Renormalize each reference and simulated sample over G to a total of
   1,000,000. Transform the normalized values to `log2(TPM + 1)`.
6. If enabled, reject simulated profiles that fail the per-gene range
   acceptance fraction on the fixed gene list.
7. Fit or load a compatible PCA model fitted on reference log-expression
   only. Transform reference and surviving simulated profiles with that
   same model.
8. Apply the configured reference-distance quantiles and norm in PC
   space. Use surviving sample IDs to retrieve the normalized
   selected-gene expression from step 5, not the PC coordinates or
   inverse-PCA reconstructions.
9. Export accepted profiles as `log2(TPM + 1)` on G in the fixed gene
   order. Do not restore excluded genes or select genes again.

Run the range and PC-distance acceptance checks only when their respective
GEP-filtering options are enabled. Gene selection, normalization, and
export do not require profile rejection to be enabled.

## Normalization and PCA

For the fixed gene list G and a nonnegative linear profile x, use:

```text
x_normalized[g] = 1,000,000 * x[g] / sum(x[h] for h in G)
x_log[g] = log2(x_normalized[g] + 1)
```

Normalize after subsetting, separately for each sample. Do not normalize
each SCT profile over G before mixing; that would change the relative
contributions of cell types to the simulated bulk.

Fail with a descriptive error on an empty gene list, duplicate gene
labels, negative or non-finite expression, or a nonpositive selected-gene
row sum. Do not silently fill missing selected genes with zeros: choose
the shared gene universe before accepting profiles.

PCA fit and transform inputs must both be `x_log`. The reference used to
estimate PCA and distance thresholds respects `filtering_ref_types`.
Use `gep_pca_n_components` as the only GEP-filtering PCA configuration
parameter. It specifies an integer component count or an explained-variance
target. Record the actual fitted component count from
`pca_model.n_components_` in `gep_filtering_n_pcs` as read-only summary
metadata, not as a separate configuration setting.

The optional range check compares reference and simulated expression on
the same normalization basis. It may compare normalized linear values:
because log2(x + 1) is monotonic, the range acceptance decision is
equivalent when the same linear quantile boundaries are transformed.
Do not assume separately interpolated quantiles in log space are exactly
equal to log-transformed linear quantiles.

## Correlation source and configuration

Reuse the existing source and threshold configuration names.

- An explicit `gene_filtering_high_corr_gene_file`, falling back to
  `paths.high_corr_gene_file_path`, takes precedence over a correlation
  reference. A supplied gene list is already selected; correlation
  thresholds do not modify it.
- Otherwise, require `gene_filtering_correlation_reference_dataset` when
  correlation-based gene selection is enabled.
- The source must precede the target in YAML order and must use segment
  sampling with GEP and gene filtering disabled. Its full correlation
  matrix must be available before target generation.
- Require all target output cell types in the correlation matrix.
  Exclude non-finite correlations from qualifying gene/cell-type pairs.
- Do not calculate correlations from already-filtered target samples,
  and do not create an implicit pilot dataset.
- With `gene_filtering: false`, skip correlation selection. GEP filtering
  uses the common reference/target gene universe as G, normalized
  consistently. This does not change the independent unfiltered
  correlation-source workflow.
- With `gep_filtering: false` and `gene_filtering: true`, resolve G and
  export all generated profiles as normalized `log2(TPM + 1)` on G,
  without range or PC-distance rejection.

Use `gene_filtering_type: high_corr_gene` for correlation selection.
Migrate the selected example config from
`high_corr_gene_and_quantile_range` to `high_corr_gene`. Remove its
`gene_filtering_quantile_range` and `gene_filtering_q_col_name` settings;
retain the separate `gep_gene_quantile_range` settings.

Existing public gene-selection helper functions can remain available for
older callers, but this workflow must not silently execute their
expression-range gene-selection branch. Reject explicitly configured
legacy range-selection modes for an enabled correlation-first target
with a migration message. Update the default mode for omitted keys to
`high_corr_gene`, preserving explicit supplied-list precedence.

## Caches and existing artifacts

A PCA target alone is not a sufficient cache identity. Record a
versioned identity including the ordered gene-list hash, normalization
definition, reference identity, reference subset, and PCA target.
Validate that the stored model gene order agrees with the fixed list G.

Generated target outputs and intermediate batches require compatible
filtering metadata too. Their identity must include the gene-list hash,
normalization version, selected-gene output definition, range-check
settings, and PC-distance settings.
Preserve existing generation parameters and seed handling.

If an existing output or resumable intermediate artifact has missing or
incompatible metadata, fail before reuse with an instruction to use a
new dataset name/output location. Do not delete old artifacts, overwrite
them, or relabel historical summaries as runs of the new workflow.

Compatibility checks apply to outputs generated under this filtering
workflow; the independent unfiltered correlation-source output remains
reusable.

## Outputs and observability

Retain existing sample IDs in the generated accepted-profile artifact.
Its expression matrix must contain exactly G in the persisted gene order,
with normalized `log2(TPM + 1)` values. Selected-gene expression is the
primary output, not an optional additional export. Do not restore the
full-gene profiles after filtering. Keep cell-fraction rows synchronized
with accepted sample IDs.

The summary must distinguish the pre-selection generation gene count,
the shared reference/target gene count, and the exported gene count
`len(G)`. Record the full generation gene count as provenance, not as the
accepted artifact's dimension. Retain the correlation-source path,
selected gene-list path, removed gene count, PCA target, actual PC count,
and filter totals. Define the removed gene count as the pre-selection
generation gene count minus `len(G)`. Record the filtering workflow
version and cache identity.

Comparison PCA remains a diagnostic controlled by
`plot_pca_for_comparison`. It must use the same selected gene list and
normalization, but its plotting component count is separate from the
GEP-filtering PCA target. Do not substitute a joint comparison PCA model
for the reference-only GEP-filtering model.

Log selected and shared gene counts, optional gene-range acceptance
counts, PCA component counts, and PC-distance acceptance counts.
Retain the empty-batch early return and survivor-index handling.

## Verification

The implementation must cover these behaviors with focused tests.

- Correlation selection precedes the first target generation batch.
- Source matrices remain unchanged across target thresholds.
- Correlation selection uses only the target output cell types.
- Different input gene orders yield identical aligned normalized inputs.
- Each selected-gene linear row sums to 1,000,000.
- Accepted-profile exports contain exactly G in the persisted order,
  with values equal to the surviving rows of `x_log`.
- Cell-fraction rows retain the same sample IDs and order as the
  accepted-profile exports.
- Captured PCA fit and transform inputs equal `log2(TPM + 1)`.
- Fresh and cached PCA paths use identical normalization.
- `gep_filtering_n_pcs` equals `pca_model.n_components_` on fresh and
  cached paths and remains read-only summary metadata.
- Gene-range acceptance divides by the selected gene count.
- No post-generation expression-range gene selection occurs.
- Empty and partially accepted batches retain existing correct behavior.
- Changed gene lists or filtering settings invalidate incompatible
  models, generated outputs, and intermediate artifacts.
- Supplied gene lists, disabled gene filtering, and disabled GEP
  filtering follow their documented branches.
- With GEP filtering disabled and gene filtering enabled, every generated
  profile is exported on G without range or PC-distance rejection.
- Summary fields distinguish pre-selection counts from exported counts
  and agree with the fixed list G.
