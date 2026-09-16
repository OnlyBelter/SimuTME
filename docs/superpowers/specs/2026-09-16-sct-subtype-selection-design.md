# SCT subtype-restricted sampling for parent cell types

## Summary

Add an optional SCT-only dataset parameter that lets users restrict which
single-cell subtypes may be used when simulating a parent cell type, while
keeping the generated SCT sample labeled as the parent cell type.

Example target behavior:

- `DC` normally samples from all configured DC single cells
- with an override, `DC` can be simulated from only `pDC` cells or only `mDC`
- the generated SCT output remains labeled as `DC`, not `pDC` or `mDC`

If the new parameter is omitted, SimuTME must keep the current behavior and use
all single cells belonging to each configured cell type.

## Goal

Support dataset-level subtype restriction during SCT simulation without changing:

- the ontology in `constants.cell_type2subtypes`
- the generated output label names
- Mixed dataset behavior
- default SCT behavior for existing configs

## Current behavior

Today, SCT simulation uses:

- `constants.cell_type2subtypes` to describe which subtypes belong to each
  parent cell type
- `output_types` to decide which parent types and subtype outputs exist in the
  generated SCT dataset

In the SCT path, `SingleCellTypeGEPGenerator` eventually filters candidate cells
by either:

- `cell_type_col_name` for parent-only outputs such as `DC`
- `subtype_col_name` for subtype outputs such as `CAFs`

There is currently no dataset-level way to say:

- generate `DC`
- but only draw the underlying single cells from `pDC`

## Requested behavior

For SCT datasets only, users should be able to specify a mapping from parent
cell type to an allowed subset of its known subtypes.

Example:

```yaml
datasets:
  SCT_N10_hnscc_07Jan_01:
    dataset_type: "SCT"
    selected_subtypes_by_cell_type:
      DC: ["pDC"]
      Fibroblasts: ["CAFs"]
```

With this configuration:

- SCT samples labeled `DC` are built only from single cells whose subtype is
  `pDC`
- SCT samples labeled `Fibroblasts` are built only from single cells whose
  subtype is `CAFs`
- all other cell types continue to use all their available single cells

## Config design

### New optional SCT dataset parameter

Name:

- `selected_subtypes_by_cell_type`

Type:

- `Dict[str, List[str]]`

Scope:

- dataset-level
- SCT-only

Default:

- omitted or `null` means no subtype restriction
- SimuTME uses all single cells belonging to that configured cell type, which is
  the current behavior

### Example config block

```yaml
datasets:
  SCT_N10_hnscc_07Jan_01:
    dataset_type: "SCT"
    n_samples: 10
    sampling_method_gep: "n_neighbors"
    selected_subtypes_by_cell_type:
      DC: ["pDC"]
      Fibroblasts: ["CAFs", "Myofibroblasts"]
```

### Meaning

For each configured parent cell type:

- if the parent cell type is absent from the mapping, use all of its subtypes
- if it is present, only use the listed subtype names as the candidate pool for
  that parent cell type

This parameter does not redefine `cell_type2subtypes`. It is only a per-dataset
sampling restriction.

## Recommended semantics

### Labeling

The generated SCT output label stays as the parent cell type.

That means:

- `DC -> ["pDC"]` still produces SCT samples labeled `DC`
- the underlying candidate single cells come only from rows whose subtype is
  `pDC`

This matches the requested behavior and avoids downstream changes to output
columns, file naming, and Mixed dataset expectations.

### Coverage of cell types

The new parameter should apply only to parent cell types that have explicitly
configured subtype lists in `constants.cell_type2subtypes`.

For parent cell types with a single same-name subtype such as:

- `Endothelial Cells: ["Endothelial Cells"]`

the override is technically allowed but unnecessary.

## Filtering logic

### SCT path

The subtype restriction should be applied only when generating SCT datasets.

Recommended flow:

1. start from the current filtered `merged_sc_dataset_obs`
2. keep existing dataset-id filtering, balanced-quota stratification, and
   zero-ratio filtering behavior unchanged
3. before SCT sampling for a parent cell type, narrow the candidate rows using
   `selected_subtypes_by_cell_type`
4. continue with the existing SCT neighbor/random sampling logic

### Per-cell-type rule

When generating one SCT output type:

- if the output is already a subtype output, keep current behavior unchanged
- if the output is a parent cell type and that parent has a subtype override,
  filter candidate rows by:
  - `obs[cell_type_col_name] == parent_cell_type`
  - `obs[subtype_col_name] in selected_subtypes_by_cell_type[parent_cell_type]`
- if the parent has no subtype override, keep current behavior

This keeps the change local to SCT parent-type sampling and avoids altering the
existing subtype-output path.

## Validation rules

The implementation should validate:

1. `selected_subtypes_by_cell_type` is only accepted for SCT datasets
2. every key is a known parent cell type in `config.cell_type2subtypes`
3. every listed subtype belongs to that parent cell type in
   `config.cell_type2subtypes`
4. subtype lists must not be empty
5. duplicate subtype names within one list should be rejected or normalized
   deterministically
6. if filtering leaves too few cells for a parent type, raise the same kind of
   clear insufficiency error used elsewhere in SCT generation

## Code changes

### Primary files

- `src/simutme/workflow.py`
- `src/simutme/simulation/simulator.py`
- `src/simutme/simulation/simulator_sct.py`
- `src/config_example.yaml`

### Implementation shape

1. In `workflow.py`, read `selected_subtypes_by_cell_type` from SCT dataset
   params
2. reject this key for Mixed datasets during dataset-type validation
3. pass the resolved mapping into `SingleCellTypeGEPGenerator`
4. store the mapping in `BulkGEPGenerator` / `SingleCellTypeGEPGenerator`
5. update the SCT candidate-row filtering logic so parent cell types can be
   restricted by subtype before sample generation
6. document the new parameter in `config_example.yaml`

## Suggested implementation boundary

The cleanest place to apply the subtype restriction is inside the SCT
single-cell candidate filtering path rather than globally altering
`merged_sc_dataset_obs`.

Why:

- the restriction is specific to certain parent outputs, not to all rows in the
  merged dataset
- subtype outputs should keep their current behavior
- global row deletion would be harder to reason about and could accidentally
  affect unrelated outputs

So the preferred design is:

- keep the global merged single-cell pool intact
- apply subtype restriction only when fetching candidate rows for a specific
  parent cell type

## Documentation updates

Update `src/config_example.yaml` near the SCT dataset block with:

- the new key
- a short example for `DC`
- an explanation that the default is to use all single cells for that parent
  cell type
- an explanation that the generated SCT label remains the parent cell type

Suggested comment style:

```yaml
selected_subtypes_by_cell_type:
  DC: ["pDC"]  # Optional SCT-only override: when generating parent-cell-type SCT samples such as "DC", only use single cells from these subtype labels; omit to use all DC single cells. The generated SCT sample is still labeled as "DC".
```

## Verification

After implementation, verify:

1. an SCT config without `selected_subtypes_by_cell_type` behaves exactly as
   before
2. `DC: ["pDC"]` generates `DC` SCT samples using only `pDC` rows
3. `DC: ["mDC"]` generates `DC` SCT samples using only `mDC` rows
4. cell types without overrides still use all eligible single cells
5. invalid subtype names raise a clear error
6. using the new key on a Mixed dataset raises a clear error

## Out of scope

This change does not include:

- relabeling SCT outputs from parent type to subtype
- subtype restriction for Mixed datasets directly
- changing `constants.cell_type2subtypes` semantics
- subtype-specific dataset-id source lists
- redesigning SCT neighbor sampling itself
