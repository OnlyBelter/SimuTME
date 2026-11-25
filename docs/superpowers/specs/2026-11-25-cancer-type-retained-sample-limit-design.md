# Cancer-type retained-sample limit

Add an optional limit on the number of Mixed samples retained after GEP-level
filtering for each cancer type used as a batch reference. The total
`n_samples` target remains in effect. This feature applies only to
`full_simulation`; it doesn't change `simulation_for_stat` or later
gene-level filtering.

## Configuration

Add the dataset-level `max_samples_per_cancer_type`, with a default of
`null`, which preserves the current workflow. A non-null value must be a
positive integer, excluding Boolean values.

```yaml
datasets:
  Mixed_Balanced:
    dataset_type: Mixed
    simu_mode: full_simulation
    n_samples: 3000
    gep_filtering: true
    gep_filtering_method: median_gep
    gep_filtering_in_pca_space: true
    filtering_ref_types: [ACC, BLCA, LIHC]
    n_subgroup: 200
    max_samples_per_cancer_type: 1000
```

With three eligible types, this example retains exactly 1,000 samples per
type at the GEP stage. With `n_samples: 2000`, each type remains limited to
1,000, but the counts can differ when the total target is reached.

The cancer type refers to the reference cancer type selected for the batch.
It is not an inferred biological label for the simulated sample.

In full simulation, a non-null limit requires enabled PCA-space `median_gep`
filtering, a positive `n_subgroup`, and the existing cancer-type reference
mapping. Use the same eligible reference types resolved by
`filtering_ref_types` and the GEP reference cohort. Reject an invalid
configuration before candidate generation, including a target greater than
the limit multiplied by the number of eligible types.

In `simulation_for_stat`, ignore this setting and retain the existing
statistics-only behavior, including stopping, sampling, reporting, and cache
behavior. This permits a shared dataset configuration to be used in either
mode without affecting the statistics-only result.

## GEP filtering and retention

Keep candidate generation, cell-fraction sampling, SCT sampling, filtering
criteria, PCA settings, and the subsequent gene-level filtering unchanged.
Maintain a separate count of retained samples for each eligible reference
type, initialized to zero.

In serial execution, choose each new batch's reference type uniformly among
types whose retained count is below the limit. After GEP filtering, retain at
most the smaller of the type's remaining capacity and the remaining total
target, using the existing deterministic accepted-row order. Update the
per-type retained count only for rows actually retained. Stop when the total
retained count reaches `n_samples`.

When the total target equals the number of eligible types multiplied by the
limit, reaching the target necessarily gives equal GEP-stage retained counts.
Otherwise the limit alone doesn't guarantee equal counts. Any later
gene-level filter may remove samples independently and change those counts;
don't change that filter or regenerate samples to compensate.

The existing full-simulation workflow has no batch limit. If an eligible type
never accepts enough samples, a capped run can continue indefinitely,
especially when the total target requires all types to reach their limits.
This feature doesn't add a stopping limit or change that existing behavior.

## Parallel execution and resume

Preserve deterministic results across supported parallel worker counts and
checkpoint resume. Use fixed-size scheduling waves independent of the number
of workers. At the beginning of a wave, snapshot the types with remaining
capacity and deterministically assign a type from that snapshot to each
batch, using the configured candidate seed and batch ID. Consume batch
results in batch-ID order.

A type can reach its limit while other batches assigned to it are already in
flight. For such batches, record the raw GEP-filtering results when their
turn is committed, but retain no rows once the type has no remaining
capacity. At the next wave, exclude that type from selection. Cancel or
ignore work beyond the batch that reaches the total target. A wave's fixed
size and assignment procedure are part of the generation identity and must
not depend on worker count or completion timing.

Persist enough scheduling and retained-count state to resume an interrupted
wave with the same assignments and results as an uninterrupted run. Continue
to use the existing per-batch candidate and subgroup random streams, with a
separate deterministic type-selection stream where needed. Enabling the
limit may produce a different sequence from an uncapped run. Uncapped runs
retain their existing scheduling, identities, and results.

## Counts, metadata, and reuse

Keep the existing GEP-filtering input counts, accepted counts, and acceptance
rates as raw filtering measurements, before either the per-type limit or
total-target truncation. When the limit is enabled, add separate
full-simulation summary fields for:

- `max_samples_per_cancer_type`: the configured limit.
- `gep_retained_n_samples_by_cancer_type`: the number actually retained for
  each eligible reference type, including zero-count types.
- `gep_discarded_by_cancer_type_limit`: the total number of GEP-accepted
  candidates discarded because their reference type reached its limit.
- `gep_discarded_by_total_target`: the total number of GEP-accepted
  candidates discarded only because the overall target was reached.

Apply the type limit first, then apply the total target to the remaining
rows, so each discarded row has exactly one reason. The sum of retained
per-type counts must equal the number of samples saved at the GEP stage.
Keep expression, cell fractions, and selected cell IDs aligned after
truncation. When the limit is absent or null, preserve existing summary
fields and semantics.

Persist the per-type retained counts and any wave state in the existing
filtering metadata and parallel checkpoint as appropriate. Include the limit
and capped scheduling version in the generation identity, so enabling,
disabling, or changing the limit cannot reuse incompatible generated outputs
or checkpoints. Require the existing fresh-output compatibility workflow
rather than silently overwriting incompatible artifacts. Don't invalidate a
reusable PCA model solely because the output limit changed.

## Verification and documentation

Test serial and supported parallel full simulation with uneven acceptance
rates, including a final batch that crosses a type limit, a total target
smaller than total capacity, and a target equal to total capacity. Verify
saved output alignment, exact retained counts, uncapped raw acceptance
rates, and exclusion of capped types from future scheduling waves.

Test invalid and impossible configurations, a single eligible type,
zero-acceptance batches, unchanged behavior when the setting is absent or
null, and unchanged statistics-only and post-GEP gene-filtering behavior.
Test deterministic output across worker counts, interrupted-wave resume, and
rejection of incompatible cached outputs or checkpoints.

Document the setting, GEP-stage attribution, distinction between acceptance
and retained counts, total-target interaction, possible prolonged runs, and
the scope of later gene filtering. Add a commented example to the package
configuration without changing user configurations or existing outputs.
