# GEP filtering distance distributions

## Goal

For each Mixed dataset with distance-based GEP filtering enabled, compare the
filtering-reference distance distribution with the distribution of the final
saved simulated profiles. Display the filtering cutoffs used for acceptance.
The diagnostic is automatic and does not change filtering or add configuration
parameters.

## Distance calculation

Reuse the filtering calculation for the resolved reference, selected genes,
normalization, center, distance metric, quantiles, and PCA model. The reference
distribution contains every sample in the reference cohort actually used by
the filter, including any `filtering_ref_types` restriction.

Support every distance method accepted by `FixedGeneFilter`: `median_gep`,
`mean_gep`, `mgep`, and `linear_mmd`. For `mgep`, use L1 distance from the
coordinate-wise median of normalized linear TPM. For the other methods, use
log2(TPM + 1), either directly or in reference-fitted PCA space. Preserve their
existing center, norm, and squared-distance rules.

Calculate simulated distances from the final generated `.h5ad`, in bounded
batches. This includes exactly the saved samples, excluding any excess
accepted candidates from the last generation batch. Convert stored
log2(TPM + 1) values back to linear TPM as needed and use the same selected
gene normalization and transformation as filtering. Account for numerical
rounding when checking samples near the cutoff.

## Figure and lifecycle

Save a density-normalized distance histogram comparing the reference cohort and
the final simulated samples in the Mixed dataset result directory, in PNG and
SVG formats. Label each group with its sample count and identify the method,
space, and distance metric. Draw a dotted vertical line for the upper
reference-quantile cutoff and, when applied, the lower cutoff. Label each line
with its quantile and cutoff value. The plot does not show rejected candidates.

Generate the diagnostic after output finalization for both newly generated and
compatible cached datasets. Do not run it when GEP filtering is disabled.
Existing generation identity and cache validation remain unchanged. Fail with
an informative error if the accepted dataset or its gene order is incompatible
with the filtering space.

## Verification

Test numeric distances and thresholds against the filtering implementation for
PCA-space `median_gep`, linear-TPM `mgep`, `mean_gep`, and `linear_mmd`.
Cover lower and upper quantiles, reference subset selection, final-sample
truncation, disabled filtering, and regeneration from a compatible cache.
Check that both output formats exist and that the plotted sample counts and
cutoff values match the underlying calculation. Update the Mixed workflow
documentation and changelog.
