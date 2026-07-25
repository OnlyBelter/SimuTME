# Mixed dataset selection of original vs sorted-bulk-imputed SCT input

## Goal

Add a Mixed-only configuration parameter that explicitly selects which SCT
dataset version should be used as the input for Mixed bulk RNA-seq simulation.

The target use case is:

- an SCT dataset is generated earlier in the same workflow
- sorted-bulk-guided imputation may optionally produce a second SCT `.h5ad`
- a later Mixed dataset should explicitly choose either:
  - the original SCT dataset
  - the sorted-bulk-imputed SCT dataset

This removes the current ambiguity where Mixed simulation follows whichever SCT
path was last written into workflow state.

## Current behavior

Today, Mixed simulation uses:

- `config.sct_dataset_file_path`

When an earlier SCT dataset runs with:

- `apply_sorted_bulk_imputation: true`

the workflow updates `config.sct_dataset_file_path` to the imputed SCT file.
That means later Mixed datasets inherit the imputed SCT dataset implicitly via
side effect.

This creates two problems:

- users cannot explicitly request the original SCT dataset once imputation has
  run
- the chosen SCT source is not visible in the Mixed dataset config block

## Requested behavior

Each Mixed dataset entry under `datasets:` should support a new optional
parameter:

- `sct_dataset_version`

Allowed values:

- `original`
- `sorted_bulk_imputed`

This parameter decides which SCT dataset file is used by the Mixed dataset.

## Config design

### New Mixed-only parameter

Example:

```yaml
paths:
  sct_dataset_file_path: "path/to/sct_dataset_gepsamp-random.h5ad"

datasets:
  Mixed_N100_segment:
    dataset_type: "Mixed"
    sct_dataset_version: "sorted_bulk_imputed"
    n_samples: 100
    sampling_method_prop: "segment"
```

### Default

If `sct_dataset_version` is omitted, default to:

- `sorted_bulk_imputed`

This preserves the project-level rule already used in this repository: Mixed
simulation should use the imputed SCT dataset when available.

## Path resolution rule

### Original SCT path

If `sct_dataset_version == "original"`:

- use `paths.sct_dataset_file_path` directly

### Imputed SCT path

If `sct_dataset_version == "sorted_bulk_imputed"`:

1. start from `paths.sct_dataset_file_path`
2. read `sorted_bulk_imputation_output_suffix`
3. infer the imputed SCT path by applying the suffix to the original SCT file
   name

This keeps the design deterministic and avoids storing extra workflow state
just to remember both paths.

## Inference details

The imputed path inference should follow the same naming rule already used by
the sorted-bulk imputation workflow so that the Mixed dataset resolves the same
file that the imputation step writes.

The resolution should be centralized in one helper function so the naming logic
is defined once and reused by:

- the optional imputation step
- the Mixed dataset SCT input selection step

## Validation rules

The implementation should validate:

- `sct_dataset_version` is only used for Mixed datasets
- valid values are only `original` or `sorted_bulk_imputed`
- `paths.sct_dataset_file_path` is present before Mixed simulation starts
- if `sorted_bulk_imputed` is requested, the inferred imputed SCT file exists

If the imputed file is missing, raise a clear error that explains:

- which Mixed dataset requested it
- which path was inferred
- that the user must either run sorted-bulk imputation first or switch
  `sct_dataset_version` to `original`

## Workflow behavior

### SCT datasets

No behavior change is required for SCT generation itself.

The existing `apply_sorted_bulk_imputation` flow remains unchanged. It may
still generate the imputed SCT file in the same way as today.

### Mixed datasets

Before Mixed simulation starts:

1. read `sct_dataset_version`
2. resolve the target SCT path using the rule above
3. pass that resolved file path into the Mixed generator
4. log which version and which concrete file path are being used

This makes Mixed dataset provenance explicit in both config and logs.

## Code changes

### Primary files

- `src/config_example.yaml`
- `src/simutme/workflow.py`

### Implementation shape

1. add `sct_dataset_version` to the Mixed example config comments
2. add a helper in `workflow.py` to resolve the SCT path for Mixed datasets
3. reuse or align with the existing sorted-bulk output naming logic
4. validate the parameter and resolved file existence
5. pass the resolved path into Mixed simulation instead of relying only on the
   mutable `config.sct_dataset_file_path`

## Documentation updates

Update the example config comments to explain:

- `sct_dataset_version` is Mixed-only
- `original` uses `paths.sct_dataset_file_path`
- `sorted_bulk_imputed` infers the imputed path from
  `paths.sct_dataset_file_path` and `sorted_bulk_imputation_output_suffix`
- the imputed file must already exist when requested

Update the changelog to record the new explicit Mixed SCT-version selector.

## Verification

After implementation, verify:

1. existing Mixed configs without `sct_dataset_version` still resolve to the
   expected default behavior
2. `sct_dataset_version: original` uses the original SCT dataset path
3. `sct_dataset_version: sorted_bulk_imputed` uses the inferred imputed path
4. a missing imputed file raises a clear error
5. invalid values raise a clear config error

## Out of scope

The following are not part of this change:

- redesigning sorted-bulk imputation outputs
- adding direct per-Mixed-dataset custom SCT file paths
- allowing Mixed datasets to reference arbitrary earlier SCT blocks by name
- changing how SCT datasets themselves are generated
