# Mixed GEP Filter Counts Design

## Goal

Record how many Mixed candidates are produced before GEP-level filtering and how many remain after GEP-level filtering, then save those totals in `mixed_dataset_summary.json`.

## Scope

This change applies only to Mixed dataset generation.

It does not change:

- the filtering algorithm
- per-round output files
- gene-level filtering behavior

## Design

### Source of Truth

The source of truth will live inside `BulkGEPGenerator`, because GEP-level filtering happens there during the generation loop.

New cumulative counters will be tracked across the full Mixed run:

- `gep_filtering_input_n_samples`
- `gep_filtering_kept_n_samples`
- `gep_filtering_removed_n_samples`

These totals represent:

- input: number of simulated Mixed candidates before GEP-level filtering
- kept: number of candidates that survived GEP-level filtering
- removed: number of candidates rejected by GEP-level filtering

When GEP filtering is disabled, these counters will stay `null` in the summary to avoid implying that a filtering stage ran.

### Runtime Updates

The generator loop already computes `simulated_gep` before filtering and again after filtering.

The implementation will:

1. Count candidate rows immediately after expression aggregation.
2. Run `_filter_simulated_geps(...)` when enabled.
3. Count surviving rows after filtering.
4. Update cumulative totals.

The counters will be cumulative across all rounds, not just the final saved batch.

### Summary JSON

`_build_mixed_dataset_summary(...)` in `src/simutme/workflow.py` will be extended with three new fields:

- `gep_filtering_input_n_samples`
- `gep_filtering_kept_n_samples`
- `gep_filtering_removed_n_samples`

Existing summary fields keep their current meaning:

- `generated_n_samples` remains the final saved Mixed dataset size
- `filtered_n_samples` remains the final size after optional gene-level filtering

This keeps the GEP-level filtering totals distinct from the final output dataset shapes.

## Data Flow

1. `BulkGEPGenerator.generate_gep(...)` aggregates candidate Mixed samples.
2. If GEP-level filtering is enabled, the generator updates cumulative pre/post counts.
3. `run_bulk_simulation(...)` passes those totals into `_build_mixed_dataset_summary(...)`.
4. `mixed_dataset_summary.json` stores the totals alongside the existing dataset metadata.

## Error Handling

- If GEP-level filtering is disabled, all three new summary fields will be `null`.
- If a filtering pass returns no samples, the kept count for that batch is `0`, and removed count increases accordingly.
- No backward-incompatible changes are introduced to existing summary fields.

## Testing

Add or extend tests to verify:

- counters accumulate correctly when GEP-level filtering is enabled
- removed count equals input minus kept
- summary JSON contains the new fields
- disabled GEP filtering writes `null` for the new totals
