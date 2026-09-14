# Mixed SCT Loading Mode Design

## Goal

Add one global configuration parameter under `parameters:` to control how Mixed
dataset generation loads its SCT reference `.h5ad`.

This is needed because:

- eager in-memory loading is much faster for smaller runs
- lazy backed loading prevents memory blow-ups on very large runs

## Scope

This change applies only to Mixed dataset generation that uses an SCT reference.

It includes:

- config parsing
- Mixed runtime loading behavior
- documentation updates in the example config, README, and docs

It does not change:

- SCT dataset generation
- filtering algorithms
- output file formats

## Proposed Parameter

Add one global parameter:

```yaml
parameters:
  mixed_sct_loading_mode: "auto"
```

Allowed values:

- `auto`
- `eager`
- `lazy`

## Behavior

### `eager`

Load the entire SCT reference expression matrix into memory before Mixed
generation.

Use this when:

- the SCT reference fits comfortably in RAM
- throughput matters more than memory usage

Tradeoff:

- fastest steady-state generation
- can trigger OOM or `SIGKILL` on very large SCT references

### `lazy`

Open the SCT reference in backed mode and materialize rows on demand during
Mixed aggregation.

Use this when:

- the SCT reference is too large to hold in memory safely

Tradeoff:

- safest memory behavior
- slower because generation repeatedly reads SCT rows from disk

### `auto`

Choose between eager and lazy based on SCT matrix size.

Proposed rule:

- use eager loading when the SCT matrix size is below a fixed threshold
- use lazy loading when it exceeds that threshold

Implementation detail:

- estimate matrix size from `n_obs * n_vars * 4` bytes (`float32`)
- keep the threshold as an internal constant for now

This keeps the config simple while restoring fast behavior for smaller runs.

## Runtime Design

Mixed generation currently enters the SCT path in `BulkGEPGenerator` and then
either:

- materializes the full TPM dataframe, or
- uses the backed reader plus row-wise access

The new parameter will control that choice centrally.

Implementation plan:

1. Add `mixed_sct_loading_mode` to config loading with default `auto`.
2. Pass the resolved value into `BulkGEPGenerator`.
3. During Mixed SCT setup:
   - `eager`: call `_read_sct_dataset(load_expression=True, backed=None)`
   - `lazy`: call `_read_sct_dataset(load_expression=False, backed='r')`
   - `auto`: inspect SCT shape and choose eager or lazy
4. Log which loading mode was selected for the run.

## Validation

Reject invalid values with a clear error:

- allowed values are `auto`, `eager`, and `lazy`

## Documentation Changes

Update:

- `src/config_example.yaml`
- `README.md`
- `docs/configuration_io.md`
- `docs/workflow_mixed.md`

Documentation should explain:

- what the parameter does
- default behavior
- when to prefer `eager`
- when to prefer `lazy`
- that `auto` is the recommended default

## Testing

Add focused coverage to verify:

- config accepts the new parameter
- invalid values are rejected
- Mixed generation selects eager vs lazy as requested
- `auto` selects a mode based on matrix size logic

## Notes

This design intentionally keeps the parameter global-only, because the user
requested it under the global `parameters:` section and that is enough for the
current workflow.
