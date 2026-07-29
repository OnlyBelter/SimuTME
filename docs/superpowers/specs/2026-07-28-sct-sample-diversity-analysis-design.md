# SCT sample-diversity analysis after simulation

## Goal

Add a final post-simulation analysis step for each simulated SCT dataset to
measure how diverse the sample-level profiles are within each cell type.

The target use case is:

- after SimuTME generates an SCT dataset, run a lightweight diversity analysis
  directly in gene space
- avoid joint PCA or other cross-dataset embeddings
- use log-transformed TPM profiles with a robust centroid for each dataset and
  cell type
- save tables and figures next to the simulated SCT dataset in a dedicated
  subfolder

## Proposed analysis workflow

For each simulated SCT dataset and each cell type:

1. build a sample-by-gene matrix from the saved `log2(TPM + 1)` results
2. analyze each cell type separately, so diversity is measured only across
   samples with the same cell type
3. select a gene set for the diversity analysis
4. define the robust centroid `mGEP_log` as the gene-wise median of the
   log-transformed sample profiles
5. compute the distance from each sample to `mGEP_log`
6. summarize the distance distribution and generate figures

This analysis is a within-dataset dispersion metric:

- smaller distances mean samples of that cell type are more homogeneous
- larger distances mean samples of that cell type are more diverse

## Requested behavior

The diversity analysis should run automatically as the final step after an SCT
dataset is fully simulated and saved.

The default analysis should:

- use all genes within each cell type
- reuse the saved `log2(TPM + 1)` values directly
- use `cosine` distance as the default metric
- save both raw results and summary figures

The workflow should remain optional through config so users can disable it when
they only want dataset generation.

## Output structure

Each simulated SCT dataset should get a dedicated result subfolder:

```text
<sct_dataset_output_dir>/
  <sct_dataset_name>/sample_diversity/
    config_used.yaml
    per_cell_type_distance.csv
    per_cell_type_summary.csv
    per_cell_type_mGEP_log.csv
    figures/
      distance_boxplot_by_cell_type.png
      distance_boxplot_by_cell_type.pdf
      <cell_type>_distance_distribution.png
      <cell_type>_distance_distribution.pdf
```

This keeps the diversity outputs colocated with the corresponding simulated SCT
dataset.

## Input definition

The analysis input should be the sample-level, cell-type-specific saved
`log2(TPM + 1)` profiles already produced by the SCT simulation workflow.

For each cell type:

- rows are sample IDs
- columns are genes
- values are saved `log2(TPM + 1)` abundances

The implementation should make this input source explicit so the diversity
analysis always uses the same level of data and does not silently switch
between per-cell and per-sample matrices.

## Gene selection design

### Default mode

Use all genes for each cell type by default.

This keeps the initial workflow simple and avoids dropping genes before the
first diversity analysis pass.

### Configurable options

Support at least:

- `all`
- `hvg`
- `provided_gene_list`

If a provided gene list is used, only genes present in the simulated dataset
should be retained.

### Minimum requirements

The implementation should:

- filter to genes present in the saved expression matrix
- deduplicate gene names if needed
- raise a clear warning or error if too few genes remain after filtering

## Log-space centroid design

If the distance is computed in log-space, the centroid should also be defined in
log-space.

For a given cell type:

```python
log_profile = saved_log2_tpm_plus_1
mGEP_log = median(log_profile across samples, gene-wise)
```

The analysis should not use:

- `log2(median(TPM) + 1)`

because that quantity is not equivalent to the median profile in the same space
where distance is measured.

## Distance metric design

### Default metric

Default to:

- `cosine`

This is preferred because it focuses on pattern differences in high-dimensional
 gene space and is less sensitive than `l2` to absolute magnitude differences.

### Optional metric

Also support:

- `l2`

This is useful as a sensitivity analysis, but it should not be the default.

## Saved result tables

### `per_cell_type_distance.csv`

One row per sample and cell type.

Recommended columns:

- `dataset_id`
- `cell_type`
- `sample_id`
- `distance`
- `distance_metric`
- `n_genes`

### `per_cell_type_summary.csv`

One row per cell type.

Recommended columns:

- `dataset_id`
- `cell_type`
- `n_samples`
- `n_genes`
- `median_distance`
- `mean_distance`
- `iqr_distance`
- `p90_distance`

### `per_cell_type_mGEP_log.csv`

Store the gene-wise log-space median profile for reuse and auditing.

This can be stored in wide format:

- rows: cell types
- columns: genes

## Figures

The initial plotting scope should include:

1. a boxplot of sample-to-centroid distances across cell types for the simulated
   SCT dataset
2. a per-cell-type distance distribution figure, such as a histogram, density
   plot, or violin plot

These figures should be saved in both `png` and `pdf` formats.

## Config design

Add analysis controls to the SimuTME config, for example:

```yaml
parameters:
  run_sample_diversity_analysis: true
  sample_diversity_gene_mode: "all"
  sample_diversity_n_top_genes: 2000  # only used when HVG mode is used
  sample_diversity_gene_list_path: null
  sample_diversity_distance_metric: "cosine"
  sample_diversity_min_samples_per_cell_type: 5
```

### Validation rules

The implementation should validate:

- `sample_diversity_gene_mode` is one of the supported options
- `sample_diversity_n_top_genes` is positive when HVG mode is used
- `sample_diversity_distance_metric` is one of `cosine` or `l2`
- `sample_diversity_min_samples_per_cell_type` is at least `2`
- `sample_diversity_gene_list_path` exists when provided

## Workflow integration

The new analysis should be inserted after SCT dataset generation completes.

The intended flow is:

1. generate and save the SCT dataset as usual
2. collect the sample-level, cell-type-specific saved `log2(TPM + 1)` profiles
3. run the diversity analysis if enabled
4. write outputs into the dataset-local `<sct_dataset_name>/sample_diversity/`
   subfolder, creating `<sct_dataset_name>/` first if it does not already
   exist
5. continue without affecting downstream dataset generation logic

This analysis should be tied to `dataset_type == "SCT"` and should not run for
other dataset types unless explicitly extended later.

## Code changes

### Primary files

- `src/simutme/config.py`
- `src/config_example.yaml`
- `src/simutme/workflow.py`
- `src/simutme/simulation/simulator_sct.py`
- `src/simutme/simulation/sample_diversity.py`
- `docs/workflow_sct.md`
- `docs/configuration_io.md`
- `docs/changelog.md`

### Implementation shape

1. add config fields for the diversity analysis
2. create a dedicated module for gene selection, log-space centroid
   computation, distance calculation, table export, and plotting
3. expose a workflow entry point that runs the analysis for one SCT dataset
4. call that entry point at the end of the SCT simulation workflow
5. save all outputs under the simulated dataset's own `sample_diversity/`
   folder, reusing the existing dataset-local directory layout when it already
   exists

## Logging and robustness

The analysis should emit clear warnings when:

- a cell type has fewer than the minimum number of samples
- too few genes remain after filtering
- the input expression matrix has duplicated or missing gene identifiers

Cell types that fail the minimum-sample requirement may be skipped with a clear
message rather than failing the full workflow.

## Verification

After implementation, verify:

1. each simulated SCT dataset gets its own `sample_diversity/` subfolder
2. the analysis reads the saved `log2(TPM + 1)` values directly
3. `mGEP_log` is computed as the median of the saved `log2(TPM + 1)` profiles
   rather than the log of median TPM
4. all genes are used by default and can be switched to `hvg` or a provided
   gene list
5. cosine distance is used by default and can be switched to `l2`
6. CSV outputs contain the expected columns and aligned cell-type/sample labels
7. figures are created for datasets and cell types with sufficient samples
8. disabling the analysis through config skips the step cleanly
9. existing SCT simulation behavior is unchanged when the new analysis is not
   used

## Out of scope

The following are not part of this change:

- joint PCA, UMAP, or other cross-dataset embedding workflows
- pairwise all-vs-all sample similarity matrices as the primary diversity
  output
- adding cross-dataset statistical tests for diversity differences
- extending the analysis to non-SCT dataset types
- redesigning how SCT sample-level TPM profiles are generated
