# SCT Dataset-Specific `n_base` Override

## Goal

Allow an SCT dataset definition to override the global `parameters.n_base`
value used when constructing positive SCT samples. Existing configurations
without a dataset-level override must retain their current behavior.

Example:

```yaml
parameters:
  n_base: 30

datasets:
  SCT_N1K_GBM_nbase50_reference:
    dataset_type: "SCT"
    n_base: 50
```

## Scope

- Dataset-level `n_base` is supported only for `dataset_type: "SCT"`.
- Mixed datasets must not define their own `n_base`.
- Mixed datasets referencing a generated SCT dataset inherit that SCT
  dataset's effective `n_base` for output-directory naming and metadata.
- The global `parameters.n_base` remains the fallback for SCT datasets that do
  not define an override.

## Design

Add a workflow resolver that:

1. Reads `params["n_base"]` for SCT datasets when present.
2. Falls back to the global YAML value `parameters.n_base` when the
   dataset-level key is absent. Internally, this value is available as
   `config.n_base` after configuration loading.
3. Converts the value to an integer and rejects non-positive values with a
   clear dataset-specific `ValueError`.

The resolved value is used consistently for:

- `generated_sct_dataset_*_n_base{n_base}_all_subtypes` directory naming.
- `SingleCellTypeGEPGenerator.generate_samples(...,
  n_base_for_positive_samples=...)`.
- Mixed output-directory naming when its `sct_reference` points to a generated
  SCT dataset.

Mixed datasets do not generate positive SCT samples, so their own `n_base`
field is rejected rather than silently ignored. When a Mixed dataset
references a generated SCT dataset, the workflow resolves the referenced
dataset's effective value using the same global fallback and override logic.

## Configuration Contract

SCT dataset definitions may optionally provide a positive integer `n_base`.
When omitted, the global `parameters.n_base` value is used.

## Validation and Tests

Add or update tests to verify:

- An SCT dataset without `n_base` uses the global value.
- An SCT dataset with `n_base: 50` resolves to `50`.
- Zero, negative, non-numeric, and null overrides fail clearly.
- Mixed datasets reject a local `n_base` field.
- Mixed naming inherits the effective `n_base` from its referenced SCT
  definition.
- Existing dataset-subset naming behavior remains unchanged.

No changes are required in the low-level SCT simulator because it already
accepts `n_base_for_positive_samples`.
