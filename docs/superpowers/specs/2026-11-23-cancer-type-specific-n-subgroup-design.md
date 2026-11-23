# Cancer-Type-Specific GEP PCA Subgroups

## Goal

When `n_subgroup` is configured for PCA-space `median_gep` filtering, make each
candidate batch use reference samples from exactly one cancer type. This
prevents a batch's PCA reference from mixing cancer types.

## Sampling behavior

The eligible reference is the expression reference after the existing
`filtering_ref_types` selection. Every eligible reference sample has a
cancer-type annotation in the configured TCGA mapping table.

For each candidate batch:

1. Choose one cancer type uniformly from the types represented in the eligible
   reference.
2. Sample without replacement up to `n_subgroup` reference samples from that
   cancer type. If that type has fewer than `n_subgroup` samples, use all of
   its samples.
3. Fit the batch PCA and calculate the batch-specific distance reference from
   only those selected samples.
4. Filter every candidate in the batch against that same type-specific PCA and
   distance reference.

All samples in one subgroup must have the selected cancer type. Different
batches may select the same type and may reuse reference samples.

When `n_subgroup` is unset, preserve the existing full-reference behavior.
When only one cancer type is represented and its entire cohort is selected,
the existing full-reference PCA path may be reused. With multiple cancer
types, a configured `n_subgroup` always uses a batch-specific PCA, even when
the cap is larger than every individual cancer-type cohort, because a batch
still selects only one cancer type.

## Randomness and validation

Serial candidate generation uses the existing NumPy random stream. Parallel
candidate generation derives the type choice and sample draw from the existing
per-batch subgroup seed, preserving results across worker counts and resumes.

Treat `n_subgroup` as a per-cancer-type cap rather than a total reference size:
do not reject it merely because it exceeds the total or per-type sample count.
Validate before candidate generation that every cancer type eligible for
selection has enough samples for the configured PCA. Fail with a clear error
if a type cannot support that PCA, rather than failing after that type is
randomly selected.

Include the aligned cancer-type assignments and the one-type-per-batch
sampling rule in filtering identity metadata so cached outputs cannot be
reused when the subgroup definition changes. Log the selected type and actual
number of reference samples for each batch.

## Kept-profile counts

Attribute every candidate batch's input and accepted candidates to the cancer
type selected for its reference PCA. Accumulate both counts alongside the
existing totals, persist them in filtering metadata and parallel checkpoints,
and write them to `mixed_dataset_summary.json` as
`gep_filtering_input_n_samples_by_cancer_type` and
`gep_filtering_kept_n_samples_by_cancer_type`. Report
`gep_filtering_acceptance_rate_by_cancer_type` as accepted divided by input,
from 0 to 1, or `null` when a type has no input candidates. Initialize every
eligible type to zero. For a single cancer type that uses the full-reference
PCA path, attribute candidates to that type. When `n_subgroup` is unset, report
no per-type breakdown. Version the subgroup filtering identity so outputs
created before these metrics existed cannot be reported with incorrect zeros.

## Scope and tests

Only the PCA reference used for `n_subgroup`-enabled `median_gep` filtering
changes. Keep gene selection, gene-range filtering, candidate generation, and
filtering behavior when `n_subgroup` is unset unchanged.

Tests cover:

- Every batch subgroup contains samples from exactly one cancer type.
- Cancer types are selected uniformly and deterministically for a given
  parallel batch seed.
- Sampling is without replacement and selects `min(n_subgroup, type size)`
  samples.
- A cap larger than the full reference is accepted and still selects one
  cancer type per batch.
- PCA-ineligible cancer types fail validation before sampling.
- A single cancer type with its full cohort selected can reuse the full
  reference PCA.
- Per-type input counts, accepted counts, and acceptance rates use the batch
  reference type and survive parallel checkpoint resume.
- Existing serial, parallel, filtering, and `n_subgroup: null` behavior remains
  unchanged.
