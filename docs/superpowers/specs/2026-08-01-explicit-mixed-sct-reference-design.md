# Explicit SCT reference selection for Mixed dataset simulation

## Goal

Make Mixed bulk RNA-seq simulation use an explicit SCT reference declaration
instead of relying on the mutable global path
`paths.sct_dataset_file_path`.

The current workflow is confusing because:

- `paths.sct_dataset_file_path` looks like a fixed global reference
- SCT generation mutates `config.sct_dataset_file_path` in memory
- later Mixed datasets implicitly consume that updated value
- dataset behavior depends on execution order in a non-obvious way

The new design should make the data source for every Mixed dataset explicit and
auditable.

## Scope

This design changes the **YAML configuration contract** for Mixed datasets and
the corresponding workflow resolution logic.

It does **not** require renaming the internal runtime argument
`sct_dataset_file_path` inside simulation classes such as `BulkGEPGenerator`.
That internal name is still reasonable because the generator still needs a
resolved SCT `.h5ad` file path at runtime.

## Summary of the new behavior

1. Remove `paths.sct_dataset_file_path` from the public YAML schema.
2. Require every Mixed dataset to declare its SCT reference explicitly.
3. Stop mutating a global SCT reference path during SCT generation.
4. Allow Mixed datasets to reference either:
   - an external SCT `.h5ad` path
   - a generated SCT dataset defined earlier in the same YAML file

This removes implicit coupling between earlier SCT tasks and later Mixed tasks.

## New Mixed-only config schema

Each Mixed dataset must contain a new block:

```yaml
sct_reference:
  source_type: "path" | "generated_dataset"
  version: "original" | "sorted_bulk_imputed"
  file_path: "..."        # required when source_type == "path"
  dataset_name: "..."     # required when source_type == "generated_dataset"
```

### Semantics

#### `source_type: "path"`

Use a user-provided SCT `.h5ad` file directly.

- `file_path` is required
- `dataset_name` must not be provided
- `version: "original"` means use `file_path` as-is
- `version: "sorted_bulk_imputed"` means infer the sibling path using
  `sorted_bulk_imputation_output_suffix`

#### `source_type: "generated_dataset"`

Use the output of an SCT dataset defined in the same YAML file.

- `dataset_name` is required
- `file_path` must not be provided
- `dataset_name` must refer to an SCT dataset entry
- that SCT dataset must appear earlier than the current Mixed dataset
- `version` chooses the original or sorted-bulk-imputed SCT output

## Example configuration

```yaml
paths:
  simu_bulk_dir: "./datasets/simulated_bulk_cell_dataset"
  tcga2cancer_type_file_path: "./datasets/TCGA/tcga_sample_id2cancer_type.csv"
  tcga_merged_tpm_file_path: "./datasets/TCGA/merged_tpm.csv"
  high_corr_gene_file_path: "./datasets/high_corr_gene.csv"
  merged_sc_dataset_file_path: "./datasets/single_cell/merged_12_sc_datasets_231003.h5ad"

datasets:
  SCT_N6K_31Jul_07:
    dataset_type: "SCT"
    sampling_method_gep: "n_neighbors"
    n_samples: 6000
    apply_sorted_bulk_imputation: true

  Mixed_A:
    dataset_type: "Mixed"
    sampling_method_prop: "segment"
    n_samples: 20000
    sct_reference:
      source_type: "path"
      file_path: "./datasets/simulated_bulk_cell_dataset/generated_sct_dataset_12ds_n_base30_all_subtypes/simu_gep_SCT_N10K_31Jul_01_gepsamp-random_log2cpm1p.h5ad"
      version: "original"

  Mixed_B:
    dataset_type: "Mixed"
    sampling_method_prop: "dirichlet"
    n_samples: 20000
    sct_reference:
      source_type: "generated_dataset"
      dataset_name: "SCT_N6K_31Jul_07"
      version: "sorted_bulk_imputed"
```

## Resolution rules

Add a new resolver for Mixed datasets that operates only on the explicit
`sct_reference` block.

### Path source

For:

```yaml
sct_reference:
  source_type: "path"
  file_path: "/path/to/toy_sct.h5ad"
  version: "original"
```

Resolve to `/path/to/toy_sct.h5ad`.

For:

```yaml
sct_reference:
  source_type: "path"
  file_path: "/path/to/toy_sct.h5ad"
  version: "sorted_bulk_imputed"
```

Resolve by appending `sorted_bulk_imputation_output_suffix` before the file
extension.

### Generated-dataset source

For:

```yaml
sct_reference:
  source_type: "generated_dataset"
  dataset_name: "SCT_N6K_31Jul_07"
  version: "original"
```

Resolve to the generated SCT `.h5ad` path for `SCT_N6K_31Jul_07`.

For:

```yaml
sct_reference:
  source_type: "generated_dataset"
  dataset_name: "SCT_N6K_31Jul_07"
  version: "sorted_bulk_imputed"
```

Resolve to the corresponding imputed SCT `.h5ad` path for that dataset.

This requires the workflow to keep an explicit registry such as:

```python
generated_sct_outputs = {
    "SCT_N6K_31Jul_07": {
        "original": Path(...),
        "sorted_bulk_imputed": Path(...) | None,
    }
}
```

instead of mutating `config.sct_dataset_file_path`.

## Validation rules

### Mixed dataset validation

Every Mixed dataset must:

- provide `sct_reference`
- provide `sct_reference.source_type`
- provide `sct_reference.version`

#### For `source_type: "path"`

- require `file_path`
- reject `dataset_name`
- require the resolved file to exist

#### For `source_type: "generated_dataset"`

- require `dataset_name`
- reject `file_path`
- require that the target dataset exists in `datasets`
- require that the target dataset has `dataset_type: "SCT"`
- require that the target dataset appears earlier in the YAML order

#### For `version: "sorted_bulk_imputed"`

- when referencing a generated SCT dataset, require that the target SCT dataset
  has `apply_sorted_bulk_imputation: true`
- when referencing an external path, require the inferred imputed file to exist

### Deprecated field handling

Reject the old Mixed-only field:

- `sct_dataset_version`

and remove the top-level YAML field:

- `paths.sct_dataset_file_path`

If either appears in the new schema, raise a clear error telling the user to
use `sct_reference`.

## Workflow changes

### Configuration model

- remove `sct_dataset_file_path` from the public `paths` section
- update `SimuConfig` accordingly
- keep the internal runtime argument `sct_dataset_file_path` for lower-level
  classes that consume a fully resolved SCT file

### Execution flow

During dataset iteration:

1. when an SCT dataset runs, record its resolved outputs in a dedicated
   registry instead of mutating a global config field
2. when a Mixed dataset runs, resolve its explicit `sct_reference`
3. pass the resolved SCT path into `BulkGEPGenerator`

This preserves clear ownership:

- SCT tasks produce outputs
- Mixed tasks choose their inputs explicitly

## Documentation changes

Update:

- `src/config_example.yaml`
- `docs/configuration_io.md`
- `docs/workflow_mixed.md`
- `docs/workflow_sct.md`
- `docs/changelog.md`

The examples should show both supported Mixed reference modes:

- external path
- generated SCT dataset

## Testing

Add tests for:

1. Mixed dataset resolves `sct_reference.source_type: "path"` with
   `version: "original"`
2. Mixed dataset resolves `sct_reference.source_type: "path"` with
   `version: "sorted_bulk_imputed"`
3. Mixed dataset resolves `sct_reference.source_type: "generated_dataset"` to
   an earlier SCT task
4. Mixed dataset rejects forward references to a later SCT task
5. Mixed dataset rejects references to non-SCT datasets
6. Mixed dataset rejects `version: "sorted_bulk_imputed"` when the referenced
   generated SCT dataset does not enable imputation
7. old `sct_dataset_version` config is rejected with a migration hint
8. old `paths.sct_dataset_file_path` config is rejected with a migration hint

## Migration guidance

### Old style

```yaml
paths:
  sct_dataset_file_path: "./datasets/.../toy_sct.h5ad"

datasets:
  Mixed_A:
    dataset_type: "Mixed"
    sct_dataset_version: "original"
```

### New style

```yaml
datasets:
  Mixed_A:
    dataset_type: "Mixed"
    sct_reference:
      source_type: "path"
      file_path: "./datasets/.../toy_sct.h5ad"
      version: "original"
```

## Why this design is better

- no hidden mutation of global state
- no accidental dependency on dataset execution order
- supports generating extra SCT datasets without changing Mixed inputs
- each Mixed dataset documents its true SCT source directly in the YAML file
- easier to debug and safer to extend later
