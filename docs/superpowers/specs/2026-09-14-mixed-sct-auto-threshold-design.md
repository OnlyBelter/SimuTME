# Mixed SCT auto threshold design

## Goal

Add one global configuration parameter to control the eager-versus-lazy
threshold used when `parameters.mixed_sct_loading_mode: "auto"`.

## Scope

This change applies only to Mixed SCT loading mode selection.

It includes:

- config parsing and validation
- Mixed auto-mode threshold selection
- documentation updates in the example config, README, and docs

It does not change:

- explicit `eager` mode behavior
- explicit `lazy` mode behavior
- SCT generation logic

## Proposed parameter

Add one global parameter:

```yaml
parameters:
  mixed_sct_loading_mode: "auto"
  mixed_sct_auto_eager_max_gb: 4.0
```

## Behavior

- `mixed_sct_auto_eager_max_gb` is used only when
  `mixed_sct_loading_mode == "auto"`.
- The value is interpreted as a dense `float32` matrix-size threshold in GiB.
- SimuTME estimates SCT matrix bytes as:

```text
n_obs * n_vars * 4
```

- If the estimate is less than or equal to the configured threshold, use
  `eager`.
- Otherwise, use `lazy`.

## Validation

- The parameter must be a positive number.
- Default value: `4.0`

## Runtime design

Implementation steps:

1. Add `mixed_sct_auto_eager_max_gb` to `SimuConfig`.
2. Validate and normalize it in `__post_init__`.
3. Pass the resolved value into `BulkGEPGenerator`.
4. Replace the hard-coded `4 GiB` constant in `auto` mode selection with the
   configured threshold.
5. Keep logging the estimated SCT matrix size and the chosen loading mode.

## Documentation changes

Update:

- `src/config_example.yaml`
- `README.md`
- `docs/configuration_io.md`
- `docs/workflow_mixed.md`

Documentation should explain that:

- `mixed_sct_loading_mode` selects `auto`, `eager`, or `lazy`
- `mixed_sct_auto_eager_max_gb` tunes the cutoff for `auto`
- the default is `4.0`

## Testing

Add focused coverage to verify:

- config accepts the new parameter
- invalid non-positive values are rejected
- `auto` honors the configured threshold instead of the old fixed constant
