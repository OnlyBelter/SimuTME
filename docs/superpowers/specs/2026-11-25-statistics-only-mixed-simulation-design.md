# Statistics-only Mixed simulation

## Goal

Provide a statistics-only Mixed simulation mode for comparing GEP-filtering
parameter combinations using overall and per-cancer-type candidate acceptance rates.
The existing full-simulation workflow remains the default and keeps its current
behavior.

## Configuration

Add `simu_mode` with values `full_simulation` and `simulation_for_stat`.
The global default is `full_simulation`; a Mixed dataset can override it.
Reject unknown values. The new mode applies to Mixed datasets and requires:

- `gep_filtering: true`, `median_gep`, PCA-space filtering, and a positive
  `n_subgroup`, so each candidate batch has an attributable cancer type.
- An optional dataset-level `max_batches`, defaulting to `null` for no batch
  limit. When set, it must be a positive integer; reject nonpositive,
  noninteger, and Boolean values.
- The existing positive `n_samples` target and other filtering, gene-selection,
  fraction-sampling, SCT-reference, and worker settings.
- Already available SCT and gene-selection inputs. Do not generate or modify
  upstream datasets as part of a statistics-only Mixed run. Fail clearly when
  a needed input is unavailable.

For example:

```yaml
parameters:
  simu_mode: "full_simulation"

datasets:
  Mixed_Filter_Pilot:
    dataset_type: "Mixed"
    simu_mode: "simulation_for_stat"
    n_samples: 100000
    max_batches: null  # Optional: use a positive integer to limit batches.
    gep_filtering: true
    gep_filtering_method: "median_gep"
    gep_filtering_in_pca_space: true
    n_subgroup: 100
```

The example shows only the additional or relevant fields; the existing Mixed
dataset reference and sampling settings are still required.

## Candidate processing

Use the existing scientific generation and GEP-filtering calculations,
including fraction sampling, SCT-profile sampling, aggregation, gene selection,
normalization, cancer-type subgroup selection, PCA fitting, distance cutoffs,
and optional gene-range screening. The filtering PCA must still be fitted:
skipping PCA analysis means skipping diagnostic comparisons and plots, not the
PCA required for acceptance.

Use the current candidate-batch size rule based on `n_samples`. Process whole
batches and stop after the first completed batch for which either:

1. A non-null `max_batches` is reached.
2. The total number of accepted candidates reaches or exceeds `n_samples`.

When `max_batches` is omitted or `null`, continue until the accepted candidate
target is reached.

Count every candidate in the final completed batch, even when its accepted
count exceeds the target. If both conditions occur on the same batch, report
`n_samples_reached` as the stop reason. If no candidates are accepted, keep
processing until the batch limit when one is configured; without a limit, the
run cannot finish until enough candidates are accepted. Do not dispatch batches
with IDs beyond a configured batch limit.

Each batch uses the existing uniform selection of one eligible cancer type,
followed by subgroup sampling from that type. Candidate counts are attributed
to that selected batch reference type, not inferred biological labels for the
simulated profiles. A type not selected during the run has zero input and kept
counts and a `null` acceptance rate.

Support the existing serial and supported parallel candidate modes. Preserve
the existing per-batch seed roles and ordered consumption in parallel mode.
Parallel workers may use temporary shared input storage, but must not persist
simulated expression, fractions, or sampled cell IDs.

## Output and isolation

Write only a statistics summary JSON for the Mixed target under a dedicated
`simulation_for_stat` subdirectory of its usual dataset result directory:

```text
<dataset-result-directory>/simulation_for_stat/mixed_dataset_summary.json
```

Do not write simulated CSV or H5AD data, sampled-cell records, diagnostic
plots, PCA analysis artifacts, filtering model artifacts, or checkpoints.
Temporary shared input storage must be cleaned up after completion or failure.
Do not modify existing inputs or full-simulation outputs. A pre-existing
statistics summary must not be overwritten silently; use a fresh target
dataset name or output location.

The summary records the mode, resolved scientific and sampling settings,
input and kept candidate totals, per-cancer-type input and kept counts,
per-cancer-type acceptance rates, actual batch count, candidate batch size,
requested limits, stop reason, and relevant seed and PCA settings. Rates are
kept divided by input, and `null` where no input was evaluated. The recorded
kept counts are full-batch filtering counts, not the number of saved profiles.
Use the existing field names for the three per-type count and rate mappings.

This first version does not support checkpoint resume for statistics-only
jobs. A failed or interrupted job must not claim to be a completed run.

## Validation

Test that:

- Omitting `simu_mode` preserves existing full-simulation behavior and output.
- Invalid modes, invalid non-null `max_batches`, and missing required GEP
  filtering or `n_subgroup` settings fail before simulation.
- An omitted or `null` `max_batches` imposes no batch limit and stops when the
  accepted candidate target is reached.
- Runs stop at the first applicable limit, include all candidates in the final
  batch, and report the stop reason and correct per-type rates.
- Zero-acceptance batches and cancer types that receive no batches are reported
  correctly.
- The same bounded, seeded parallel run produces the same committed statistics
  across worker counts.
- The statistics run writes only its summary, never touches existing inputs or
  full-simulation artifacts, and rejects an existing summary rather than
  overwriting it.
- Diagnostic PCA, plots, accepted-profile exports, and downstream operations
  requiring generated profiles are skipped.
- Existing segment and Dirichlet sampling settings continue to work with their
  existing seed and prior semantics.

Document the new mode, required inputs, stopping rules, output location, and
limitations in the configuration and Mixed workflow guides.
