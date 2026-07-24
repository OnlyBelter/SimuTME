# SCT dataset_id source selection for cancer vs non-cancer cells

## Goal

Add a configurable way to choose which `dataset_id` values contribute cancer
cells and which contribute non-cancer cells during SCT simulation in `SimuTME`.

The target use case is:

- cancer cells come from a limited subset of datasets, such as HNSCC only
- non-cancer cells come from all datasets, or from a separate configured subset

This selection is needed only for SCT dataset generation. Later Mixed
simulation should continue to use the SCT dataset that was generated earlier in
the workflow.

## Current behavior

Today, SimuTME uses a single dataset filter:

- `constants.sc_dataset_ids`

This list is passed into the SCT and Mixed generators and is applied uniformly
to all cells through:

- `BulkGEPGenerator.sc_dataset_used`
- `BulkGEPGenerator.get_info_in_merged_single_cell_dataset()`

That means cancer and non-cancer cells are currently drawn from the same set of
`dataset_id` values.

## Requested behavior

For SCT datasets only, users should be able to specify:

- `cancer_cell_dataset_ids`
- `non_cancer_cell_dataset_ids`

The filtering rule should use the major cell-type column:

- cancer rows are those where `obs[cell_type_col] == "Cancer Cells"`
- all other rows are treated as non-cancer rows

## Config design

### Scope

These parameters should live in each SCT dataset entry under `datasets:`.

They should not be global-only because the user explicitly wants this to be
configured for SCT generation, while later Mixed datasets inherit the produced
SCT dataset as a downstream dependency.

### New optional SCT dataset parameters

Example:

```yaml
datasets:
  SCT_N10_07Jan_01:
    dataset_type: "SCT"
    cancer_cell_dataset_ids:
      - "hnscc_puram_03"
    non_cancer_cell_dataset_ids:
      - "hnscc_cillo_01"
      - "pdac_peng_02"
      - "hnscc_puram_03"
      - "pdac_steele_04"
      - "luad_kim_05"
      - "nsclc_guo_06"
      - "pan_cancer_07"
      - "prad_cheng_08"
      - "prad_dong_09"
      - "hcc_sun_10"
      - "gbm_neftel_11"
      - "gbm_abdelfattah_12"
```

### Defaults

Both parameters should be optional.

If either parameter is omitted or `null`, SimuTME should fall back to:

- `constants.sc_dataset_ids`

This preserves current behavior for existing configs.

## Filtering logic

### SCT path

When `dataset_type == "SCT"`:

1. load merged single-cell metadata as before
2. identify cancer rows using `cell_type_col_name == "Cancer Cells"`
3. keep cancer rows only if `dataset_id` is in `cancer_cell_dataset_ids`
4. keep non-cancer rows only if `dataset_id` is in
   `non_cancer_cell_dataset_ids`
5. concatenate those retained rows and continue with the existing zero-ratio
   filtering, output-type assignment, distance-matrix generation, and SCT
   sample generation

### Mixed path

No separate dataset-id split is needed for Mixed simulation.

Mixed datasets already consume:

- `config.sct_dataset_file_path`

So if a preceding SCT dataset used this cancer/non-cancer source selection,
Mixed simulation naturally inherits that choice through the generated SCT
dataset.

## Code changes

### Primary files

- `src/simutme/workflow.py`
- `src/simutme/simulation/simulator.py`
- `src/simutme/simulation/simulator_sct.py`
- `src/config_example.yaml`

### Implementation shape

1. Extend the SCT workflow branch in `workflow.py` to read:
   - `cancer_cell_dataset_ids`
   - `non_cancer_cell_dataset_ids`
2. Pass those parameters into `SingleCellTypeGEPGenerator`
3. Extend the generator / base generator state to store both lists
4. Update `get_info_in_merged_single_cell_dataset()` to apply the split filter
   when those lists are present
5. Preserve the old behavior for datasets that do not set the new parameters

## Validation rules

The implementation should validate:

- configured dataset IDs exist in the merged single-cell dataset
- the two new parameters are only used for SCT datasets
- empty strings are rejected if passed as dataset IDs

It is acceptable for the two lists to overlap. For example, a cancer dataset
such as `hnscc_puram_03` may also appear in the non-cancer list.

## Documentation updates

Update the example config comments to explain:

- what `cancer_cell_dataset_ids` means
- what `non_cancer_cell_dataset_ids` means
- that the selection applies during SCT generation only
- that later Mixed datasets use the generated SCT dataset and therefore inherit
  the chosen source composition

## Verification

After implementation, verify:

1. existing configs without the new parameters behave exactly as before
2. SCT generation can restrict cancer cells to one dataset while leaving
   non-cancer cells drawn from all datasets
3. invalid `dataset_id` values raise a clear error
4. later Mixed simulation uses the SCT dataset produced by that SCT step,
   without adding a second independent dataset-id filter

## Out of scope

The following are not part of this change:

- changing how Mixed datasets directly filter merged scRNA-seq cells
- splitting non-cancer cells by subtype-specific dataset lists
- redesigning the SCT sampling algorithms
- changing the sorted-bulk imputation workflow
