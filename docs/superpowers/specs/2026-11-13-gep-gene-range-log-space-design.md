# Log-space per-gene GEP range filtering

## Goal

When `gep_filtering_by_gene_range` is enabled in the active fixed-gene
filtering workflow, evaluate the per-gene range acceptance check in normalized
`log2(TPM + 1)` space. Keep the existing minimum fraction of genes and
quantile settings. When the flag is false, filtering behavior remains
unchanged.

This range check is separate from reference-distance filtering. In
particular, the configured `median_gep` PCA-distance filter retains its
existing behavior.

## Range calculation

Use the fixed selected gene list G shared by reference and simulated profiles.
For each reference and candidate profile:

1. Subset to G and normalize its selected-gene values to a row sum of
   1,000,000 in linear TPM.
2. Transform the normalized values with `log2(TPM + 1)`.
3. Compute per-gene lower and upper quantile bounds from the transformed
   reference profiles.
4. Accept a candidate when the fraction of its genes within the inclusive
   per-gene bounds is at least `gep_min_percentage_within_gene_range`.

Use the first and third elements of `gep_gene_quantile_range` as the lower
and upper quantiles; the middle element does not participate in this
acceptance check. Calculate quantiles directly from transformed reference
values rather than transforming bounds computed in linear space, because
quantile interpolation can make those results differ.

Apply the range check before the existing reference-distance filter. Do not
change the `median_gep` center, PCA transform, distance norm, or
`gep_filtering_quantile` behavior.

## Scope and compatibility

Apply the change in `FixedGeneFilter`, which is the active path used by the
configuration workflow, including serial and parallel candidate generation.
The parallel path must precompute range bounds from the same transformed
reference values as the serial path.

Leave the older direct-generator filtering fallback unchanged. It is not used
by the configuration path covered by this change.

Add a range-space marker to the filtering identity when the range check is
enabled. This prevents metadata and resumable parallel checkpoints created
with linear-space range checks from being reused under the new behavior.
Unchanged configurations with the range check disabled retain their existing
identity.

The related design note's option to compare normalized linear-space values
is superseded for this range check. Other filtering methods, gene selection,
and accepted-profile export remain unchanged.

## Validation

Tests must verify:

- Quantile bounds are computed from normalized reference
  `log2(TPM + 1)` values.
- Candidate profiles are transformed the same way before range comparison.
- Inclusive bounds and the minimum in-range gene fraction are preserved.
- A fixture whose linear- and log-space quantile decisions differ follows the
  log-space result.
- The disabled range check does not change existing behavior.
- Parallel precomputation uses the same log-space bounds.
- The filtering identity changes only when the enabled range space changes.
