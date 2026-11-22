# Batch-Specific GEP PCA Reference Subgroups

## Goal

Allow PCA-space `median_gep` filtering to fit its reference PCA and distance
distribution on a configurable number of reference samples per candidate
generation batch. A reduced subgroup gives each batch its own PCA space and
median-GEP acceptance thresholds, while gene-level filtering continues to
use the full resolved reference.

## Configuration

Add `n_subgroup` as a global GEP-filtering setting, with dataset-level
overrides using the same name:

```yaml
parameters:
  n_subgroup: null  # null uses every eligible GEP-reference sample

datasets:
  Mixed_example:
    n_subgroup: 200
```

`null` or an omitted value resolves to the number of eligible samples in the
GEP reference. An explicit integer sets the per-batch subgroup size.

## Reference Sample Universe

Let `N_ref` be the number of reference samples remaining after the existing
reference-file, expression-type, and `filtering_ref_types` selection. The
subgroup is sampled from these `N_ref` rows without stratification or
replacement within a batch. Different batches draw independently and may
overlap; disjoint subsets are not required.

The selected fixed gene list and current reference normalization remain
unchanged. PCA input continues to be normalized over the selected genes to
1,000,000 and transformed to `log2(TPM + 1)`.

## Batch Filtering

Apply this behavior to both serial and parallel Mixed candidate generation.
The existing candidate generation batch (`n_round`/`batch_id`, using the
configured or resolved candidate batch size) is the unit of subgroup
sampling.

When `n_subgroup` is less than `N_ref`, each batch:

1. Draws a fresh subgroup of `n_subgroup` reference rows.
2. Fits a transient PCA model using that subgroup's normalized
   `log2(TPM + 1)` profiles and the configured `gep_pca_n_components`.
3. Transforms the subgroup into PCA coordinates and calculates its
   coordinate-wise median.
4. Calculates each subgroup reference sample's configured distance from that
   median and derives the existing lower and upper quantile cutoffs.
5. Transforms every candidate in the batch with this same PCA, calculates its
   distance to the subgroup median, and applies the batch's inclusive
   quantile cutoffs.

Discard the batch PCA and its subgroup-specific reference state after the
batch completes. Do not save a PCA model per batch. When `n_subgroup` resolves
to `N_ref`, preserve current behavior: fit or load the full-reference PCA
once and reuse it across batches.

## Other Filtering Stages

`n_subgroup` changes only the PCA reference used for GEP-level
`median_gep` distances and their reference-distance quantiles. All reference
samples continue to be used for fixed-gene selection and any gene-level
filtering, including configured gene-range bounds and post-GEP gene-quantile
filtering. Other GEP filtering methods and non-PCA filtering behavior remain
unchanged.

Skip the existing pooled final GEP-distance distribution plot when reduced
subgroups are used. Its raw distances come from different batch-specific PCA
spaces and are not comparable in one histogram. Preserve the existing plot
for full-reference mode and log why it is skipped for reduced-subgroup runs.

## Reproducibility and Metadata

In parallel mode, derive a dedicated subgroup-sampling seed from
`mixed_candidate_seed` and the candidate `batch_id`. This keeps a batch's
subgroup stable across worker counts and checkpoint resume without consuming
the candidate-generation random stream. In serial mode, use the existing
NumPy RNG stream; do not add another required seed setting.

Include the configured/resolved subgroup size and `N_ref` in the filtering
identity so incompatible outputs cannot be silently reused. Record the
subgroup size, reference count, and actual fitted PCA component counts in
filtering metadata/logs. Preserve the existing single-model metadata for
full-reference mode. In reduced-subgroup mode, report component-count
variation across batches in summary metadata when the PCA variance target
causes the fitted count to vary.

## Validation

- Accept `null` or a positive integer `n_subgroup`; reject booleans, zero,
  negative values, non-integers, and values greater than `N_ref`.
- Reject a subgroup too small for the configured PCA component target with a
  clear error before candidate filtering begins.
- Permit `n_subgroup == N_ref` and treat it as full-reference mode.
- Restrict subgroup mode to enabled PCA-space `median_gep`; reject
  incompatible filtering configurations rather than silently ignoring
  `n_subgroup`.
- Verify every candidate in one batch uses the same PCA and cutoffs, while
  reduced-subgroup batches get independently derived subgroup state.
- Test the full-reference path, subgroup draws without replacement, batch
  refresh, inclusive distance boundaries, lower and upper quantiles, and
  preservation of full-reference gene-level filtering.
- Test deterministic parallel subgroup draws across worker counts and resume,
  plus dataset-level override precedence and invalid subgroup sizes.
- Retain regression coverage for existing serial and parallel filtering
  behavior when `n_subgroup` is unset.

## Non-Goals

This change does not alter the gene list, GEP distance norm, quantile
definitions, simulated batch size, or filtering behavior for `mgep`,
`mean_gep`, or `linear_mmd`. It does not persist one PCA artifact per batch or
subsample reference rows for gene-level filtering.
