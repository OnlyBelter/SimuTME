# Multi-dataset SCT sample-diversity comparison

## Goal

Add an independently callable function to compare sample-diversity results across
multiple simulated SCT datasets.

The target use case is:

- start from multiple existing `per_cell_type_distance.csv` files
- compare the same diversity metric across SCT datasets
- generate outputs similar to the current single-dataset `sample_diversity/`
  folder
- show all datasets together in one boxplot for each cell type comparison

This comparison step should not require reloading the original SCT `.h5ad`
files.

## Input definition

The new function should accept a list of paths to existing:

- `per_cell_type_distance.csv`

Example inputs include files such as:

- `.../SCT_N10K_30Jul_02_HVG_n-neighbor_sorted_bulk_imputation_5%/sample_diversity/per_cell_type_distance.csv`
- `.../SCT_N10K_29Jul_01_HVG_n-neighbor_no_imputation/sample_diversity/per_cell_type_distance.csv`
- `.../SCT_N10K_27Jul_01_HVG_n-neighbor_sorted_bulk_imputation_15%/sample_diversity/per_cell_type_distance.csv`

Each input CSV is expected to contain at least:

- `dataset_id`
- `cell_type`
- `sample_id`
- `distance`
- `distance_metric`
- `n_genes`

## Proposed public API

Add a reusable function under the sample-diversity module, for example:

```python
compare_sct_sample_diversity_across_datasets(
    distance_csv_paths: list[str | Path],
    output_dir: str | Path,
    dataset_name_map: dict[str, str] | None = None,
)
```

### Expected behavior

The function should:

1. read and validate all input `per_cell_type_distance.csv` files
2. merge them into one long table
3. verify that the distance metric is consistent across all inputs
4. summarize diversity per dataset and cell type
5. generate figures under a new output folder
6. return the output directory path

This function should be callable independently from the main simulation
workflow.

## Dataset label handling

### Default behavior

Use the `dataset_id` column from each input CSV as the dataset label in summary
tables and plots.

### Optional override

Support an optional `dataset_name_map` argument so users can replace long
dataset names with shorter display labels.

If a dataset label override is provided, it should affect only the comparison
outputs and should not modify the original input files.

## Validation rules

The implementation should validate:

- at least two input CSV files are provided
- every input file exists
- required columns are present
- `distance` is numeric
- all inputs use the same `distance_metric`

If the input files contain different distance metrics, raise a clear error
instead of silently merging incompatible results.

If some cell types are missing from some datasets, the function should still
run. The available rows should be compared for the overlapping cell types, and
missing combinations should simply be absent from the merged outputs.

## Output structure

The new comparison output folder should mirror the current single-dataset
structure where practical:

```text
<output_dir>/
  config_used.yaml
  per_cell_type_distance_all_datasets.csv
  per_dataset_per_cell_type_summary.csv
  figures/
    distance_boxplot_by_cell_type.png
    distance_boxplot_by_cell_type.pdf
    <cell_type>_distance_distribution.png
    <cell_type>_distance_distribution.pdf
```

## Saved result tables

### `per_cell_type_distance_all_datasets.csv`

This file should contain the merged long-format distance table across all input
SCT datasets.

It should preserve the original row-level information:

- `dataset_id`
- `cell_type`
- `sample_id`
- `distance`
- `distance_metric`
- `n_genes`

### `per_dataset_per_cell_type_summary.csv`

This file should summarize diversity for each dataset and cell type.

Recommended columns:

- `dataset_id`
- `cell_type`
- `n_samples`
- `n_genes`
- `median_distance`
- `mean_distance`
- `iqr_distance`
- `p90_distance`

## Figures

### Combined boxplot

Generate one figure named:

- `figures/distance_boxplot_by_cell_type.png`
- `figures/distance_boxplot_by_cell_type.pdf`

Design requirements:

- x-axis: `cell_type`
- y-axis: `distance`
- hue: `dataset_id`
- all datasets shown in the same figure
- y-axis label should include the metric name, such as:
  - `1-cosine distance to mGEP_log`
  - `l2 distance to mGEP_log`
- horizontal guide lines should be enabled, consistent with the current
  single-dataset style
- a dataset legend should be shown

### Per-cell-type distribution plots

For each cell type, generate one comparison figure across datasets:

- `figures/<cell_type>_distance_distribution.png`
- `figures/<cell_type>_distance_distribution.pdf`

These plots should show all input datasets together for that cell type, using a
shared style such as KDE overlays or histogram+KDE overlays with a legend.

## Config record

Write a `config_used.yaml` file capturing:

- the list of input CSV paths
- the resolved dataset labels
- the shared distance metric
- the output directory

This keeps the comparison run auditable and reproducible.

## Implementation scope

The implementation should stay focused on post hoc comparison of already saved
distance tables.

It should not:

- rerun SCT simulation
- recompute distances from `.h5ad`
- change the current single-dataset diversity workflow

## Testing

Add a focused test that:

1. creates multiple tiny `per_cell_type_distance.csv` files
2. runs the new comparison function
3. checks that:
   - the merged CSV exists
   - the summary CSV exists
   - the combined boxplot exists
   - at least one per-cell-type comparison plot exists

Add a validation test that mixing different `distance_metric` values raises a
clear error.
