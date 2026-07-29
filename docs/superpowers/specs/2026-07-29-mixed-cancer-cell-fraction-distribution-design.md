# Mixed Cancer Cells fraction distribution output

## Goal

Add a lightweight post-processing step for each simulated Mixed bulk RNA-seq
dataset to visualize the distribution of `Cancer Cells` proportions across all
samples.

## Requested behavior

For each `dataset_type: "Mixed"` dataset:

1. reuse the generated cell-fraction CSV
2. extract the `Cancer Cells` column
3. save a histogram with KDE showing the distribution across all samples
4. save a small summary CSV
5. write outputs under the dataset-local subfolder:
   - `<dataset_name>/`

## Output files

Save the following files under:

```text
paths.simu_bulk_dir/<dataset_name>/
```

Files:

- `cancer_cell_fraction_distribution.png`
- `cancer_cell_fraction_distribution.pdf`
- `cancer_cell_fraction_summary.csv`

## Data source

The new output should read the existing Mixed cell-fraction table:

- `cell_fractions_<dataset_name>_gepsamp-<sampling_method_gep>.csv`

No new simulation data should be generated.

## Figure choice

Use:

- histogram
- KDE overlay

This keeps the output focused on the actual distribution shape.

## Summary table

The summary CSV should contain one row with:

- `dataset_id`
- `cell_type`
- `n_samples`
- `mean`
- `median`
- `std`
- `min`
- `max`
- `p25`
- `p75`
- `p90`

## Error handling

The workflow should fail clearly when:

- the Mixed cell-fraction CSV does not exist
- the `Cancer Cells` column is missing

## Integration point

Run this as a Mixed-only post-step after the Mixed dataset has been generated
or detected as already existing.

This should happen before or independently of any optional filtering and PCA
post-processing so the distribution output is always available for Mixed
datasets.
