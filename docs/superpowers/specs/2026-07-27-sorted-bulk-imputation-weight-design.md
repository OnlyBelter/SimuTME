# Sorted-bulk imputation bulk-weight parameter

## Goal

Add a configuration parameter to control how much the sorted-bulk reference GEP
contributes when SimuTME applies sorted-bulk-guided SCT imputation.

The target use case is:

- the current `included == 1` branch averages SCT and sorted-bulk GEPs with a
  fixed `0.5 / 0.5` split
- the user wants to control that contribution explicitly
- the default behavior should remain the current `50%` sorted-bulk weight

## Current behavior

Today, the sorted-bulk-guided SCT imputation path uses a hard-coded average in
`src/simutme/sct_imputation.py`:

```python
averaged = 0.5 * tpm_batch[include_rows, :] + 0.5 * bulk_vec[None, :]
```

This behavior applies to rows where the mapping file marks the matched sorted
bulk donor as:

- `included == 1`

There is currently no config parameter that changes this weight.

## Requested behavior

Add a new parameter that directly represents the sorted-bulk contribution:

- `sorted_bulk_imputation_bulk_weight`

The imputation formula for the `included == 1` averaging branch becomes:

```python
imputed = (1 - bulk_weight) * sct + bulk_weight * sorted_bulk
```

## Config design

### New global parameter

Add the parameter under the existing global `parameters:` block:

```yaml
parameters:
  sorted_bulk_imputation_bulk_weight: 0.5
```

### Default

Default value:

- `0.5`

This preserves the current behavior exactly for existing configs.

## Validation rules

The new parameter should be validated as:

- numeric
- within `[0, 1]`

Clear errors should be raised for invalid values such as:

- negative weights
- values greater than `1`

## Workflow behavior

The parameter should flow through the existing sorted-bulk imputation path:

1. read `sorted_bulk_imputation_bulk_weight` from the global config
2. allow dataset-level `params` to override it, matching the existing pattern
   already used for other sorted-bulk imputation parameters
3. pass the resolved weight into `run_sorted_bulk_guided_sct_imputation()`
4. pass it further into `_impute_batch()`
5. use it only for the `included == 1` averaging branch

No behavior change is needed for the `included != 1` zero-guided replacement
branch.

## Code changes

### Primary files

- `src/simutme/workflow.py`
- `src/simutme/sct_imputation.py`
- `src/config_example.yaml`
- `README.md`
- `docs/changelog.md`

### Implementation shape

1. add `sorted_bulk_imputation_bulk_weight` to the global config model in
   `workflow.py`
2. thread the parameter through `_run_sorted_bulk_imputation_for_sct_dataset()`
3. extend `run_sorted_bulk_guided_sct_imputation()` and `_impute_batch()` to
   accept the resolved bulk weight
4. replace the hard-coded `0.5 / 0.5` average with:
   ```python
   sct_weight = 1.0 - bulk_weight
   averaged = sct_weight * sct + bulk_weight * sorted_bulk
   ```
5. document the parameter in the example config, README, and changelog

## Documentation updates

Update the user-facing docs to explain:

- the new parameter name
- that it only affects the `included == 1` averaging branch
- that `0.5` keeps the current behavior
- that `0.0` means SCT-only for the averaging branch
- that `1.0` means sorted-bulk-only for the averaging branch

## Verification

After implementation, verify:

1. omitted parameter preserves the previous `0.5 / 0.5` behavior
2. `bulk_weight = 0.0` produces SCT-only output in the averaging branch
3. `bulk_weight = 1.0` produces sorted-bulk-only output in the averaging branch
4. `bulk_weight = 0.25` and `0.75` are threaded correctly through the workflow
5. invalid values outside `[0, 1]` raise clear errors

## Out of scope

The following are not part of this change:

- changing the `included != 1` zero-guided replacement branch
- adding two independent weights that must later be normalized
- redesigning the mapping-file semantics
- changing how nearest sorted-bulk donors are selected
