# KNN-only GEP filtering

Add `gep_filtering_method: knn` alongside the existing `knn_residual`
method. This design implements the approved KNN-only approach without
changing the stricter method or historical outputs.

## Acceptance and calibration

Reuse fixed gene list G, selected-gene normalization, individual-sample
cancer-stratified holdout, fitting-only PCA and neighbor database, the
optional gene-range check, and the cumulative generation-batch bound.

| Method | Acceptance after gene-range screening |
| --- | --- |
| `knn` | Neighbor distance is at most the held-out cutoff |
| `knn_residual` | Both neighbor distance and residual are at most their cutoffs |

For `knn`, continue computing residual scores and the held-out residual
percentile for diagnostics. They must never affect acceptance, truncation,
export, or the batch limit. There is no minimum distance requirement.
KNN-only filtering constrains proximity in the retained reference-PC space,
not similarity in omitted directions or overlap in a joint comparison plot.

Use the existing parameter names and defaults. In the new example, use
PCA variance target 0.9, Euclidean distance, 10 neighbors, 20% held out per
cancer type, split seed 42, and neighbor quantile 0.99. Keep residual
quantile 0.95 as diagnostic context, not an acceptance setting.

## Scores, counters, and diagnostics

Preserve `knn_pass` and `residual_pass` as raw comparisons with their
respective calibrated cutoffs. In `knn`, `residual_pass` is diagnostic only
and may be false for an exported sample.

`joint_pass` denotes acceptance by the active local method: it equals
`knn_pass` in KNN-only candidate tables, and the conjunction of both
comparisons in strict tables. Held-out `joint_pass` also includes the
optional gene-range check. `exported` remains the post-truncation decision.

Keep diagnostic residual and neighbor/residual conjunction counters
separate from actual local-filter acceptance totals. Add explicit
`residual_filter_enabled` and diagnostic residual-count fields for `knn`
summaries. Do not change v2 strict-mode summary values or score semantics.

Update checkpoint flag validation to use the active method. KNN-only
exports with `residual_pass: false` are valid; contradictory neighbor or
export flags must still fail before reuse.

Residual plots in KNN-only runs must say "diagnostic only" and label their
percentile marker as not applied to acceptance. Keep the reference-only
and joint comparison plots independent of filtering decisions.

## Compatibility and example

Use a distinct KNN-only workflow identity, including the method and
disabled residual gate. Preserve the existing v2 identity for
`knn_residual`. Reject cross-method outputs, models, calibration tables,
or resumable batches with incompatible metadata; do not relabel them.

Both local methods require a prepared `LocalTCGAFilter` in the low-level
generator API. Disabled GEP filtering must not split, calibrate, or reject
profiles. Existing centroid methods and the unfiltered correlation source
retain their existing behavior and output paths.

Create a separate example file ending in `_knn.yaml`, preserving the
existing `_knn_residual.yaml` file. Use fresh target name
`Mixed_N5K_pan_cancer_Oct16_8_0p9_knn` under the
`dirichlet_12ds_n_base30_with_filtering_knn` output directory. Reuse the
earlier SCT and unfiltered correlation-source entries.

## Verification

Test acceptance of close-neighbor profiles with failing residual scores,
strict rejection of the same profiles, calibration and candidate flag
semantics, marginal versus actual acceptance counters, valid cached and
resumed KNN-only exports, cross-method cache rejection, disabled filtering,
the generation bound, low-level API validation, and diagnostic labeling.
Run the existing strict-mode and centroid regression tests unchanged.
