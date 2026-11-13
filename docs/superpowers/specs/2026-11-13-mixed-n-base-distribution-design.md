# Mixed SCT `n_base` usage distribution

## Goal

For a Mixed dataset that uses an SCT reference generated with at least two
`n_base_values`, report how often each configured `n_base` is selected for
each cell type. Split the counts by the corresponding Mixed sample's cell
fraction: `>= 0.01` and `< 0.01`.

This is a reporting-only change. It does not alter SCT selection, cell
fractions, expression aggregation, filtering, or generated expression.

## Sampling and counting semantics

Mixed generation uses matrix multiplication. It selects one SCT GEP for every
cell type in each candidate sample, including cell types whose fraction is
zero, then weights each selected GEP by that sample's cell fraction. The
report uses the selected SCT GEP's saved per-profile `n_base` assignment.

Count exactly one selection for each accepted Mixed sample and cell type.
Repeated selection of the same SCT GEP in different Mixed samples counts
again. Assign each selection to one fraction group using the fraction for
that cell type in that same Mixed sample:

- `>= 0.01` includes exactly `0.01`.
- `< 0.01` includes zero.

Use only final accepted Mixed samples, including the final output-size
truncation. For each cell type, the counts across both fraction groups and all
configured `n_base` values must sum to the number of final Mixed samples.
Within each cell-type and fraction group, the counts across `n_base` values
must sum to the number of samples in that group.

## Data sources and output

Build the report from the existing persisted artifacts:

- The accepted Mixed samples' cell-type-specific selected SCT GEP IDs.
- The accepted samples' cell fractions.
- The source SCT dataset's per-profile `n_base` assignment.

For each selected-source row, use its Mixed sample ID and cell type to look up
the matching cell fraction. Use its selected SCT GEP ID to look up that
profile's `n_base` assignment. Each selected-source row must resolve to
exactly one fraction and one assignment. Fail with an actionable error if
required assignment or provenance data are missing, duplicated, or cannot be
aligned. Do not silently omit selections.

Write `n_base_distribution_by_cell_type.csv` beside
`mixed_dataset_summary.json`. Include these columns:

- `cell_type`
- `cell_fraction_group` (`>=0.01` or `<0.01`)
- `n_base`
- `count`
- `proportion`
- `group_sample_count`

`proportion` is `count / group_sample_count`. Emit rows for every cell type,
both fraction groups, and every configured `n_base` value, including zero
counts. When a cell-type/fraction group has no samples,
`group_sample_count` and `count` are zero and `proportion` is null.

Add the distribution file path and counting metadata to
`mixed_dataset_summary.json`, including the configured `n_base_values`, the
fraction threshold and inclusivity, and the counting unit. When the source
uses a scalar `n_base` or a list with fewer than two values, do not create the
distribution CSV or add distribution metadata.

## Workflow integration

Calculate the distribution once the final accepted Mixed output and its
sample metadata are known. Use the same post-generation calculation for
serial and parallel candidate generation. This avoids separate counting
implementations and ensures rejected candidates and speculative parallel
batches do not contribute.

The SCT assignment remains the source of truth for each selected GEP's
`n_base`. Do not infer `n_base` from the number of selected cells, expression
values, or the Mixed cell fraction.

## Compatibility and error handling

- Preserve existing simulation and filtering behavior.
- Preserve existing Mixed output data and sample IDs.
- Add the CSV and summary fields only for SCT references with at least two
  configured `n_base_values`.
- If assignment, selected-source, or fraction metadata do not align, stop
  report generation with an error naming the missing or inconsistent
  artifact.

## Validation

Tests must verify:

- Each accepted Mixed sample contributes one observation for every cell type.
- A selected GEP reused across Mixed samples is counted once per selection.
- Fractions of exactly `0.01` enter the `>= 0.01` group; zero enters `< 0.01`.
- For each cell type, all `n_base` counts across both fraction groups sum to
  the final accepted sample count.
- Counts within each fraction group sum to that group's sample count.
- Empty strata and unobserved configured `n_base` values appear as zero-count
  rows with a null proportion only when the stratum denominator is zero.
- Rejected candidates and samples trimmed beyond the requested output size do
  not contribute.
- Serial and parallel generation produce equivalent summaries from equivalent
  persisted artifacts.
- Scalar `n_base` and one-value lists do not produce the report.
- Missing assignments, selected IDs, or fractions fail with actionable
  messages.
