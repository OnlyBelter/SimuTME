# Changelog

This page records notable documentation, workflow, and implementation changes in
`SimuTME`. Use it to track recent updates and to find the files and behaviors
that changed on a given date.

## October 20, 2026

### CPU candidate simulation

Mixed candidate generation now avoids repeated metadata scans and per-sample pandas expression lookups on the ordinary eager path.

- Build random-selection pools per classification column once per batch.
- Avoid a spawn pool for random sampling without patient constraints.
- Aggregate eager expression in bounded chunks with float64 arithmetic.
- Log per-batch stage timings and provide a baseline CPU benchmark script.
- Preserve filtering rules; random draws can differ from the legacy workers.

## August 1, 2026

### Explicit Mixed SCT reference selection

This update removes the old public YAML dependency on
`paths.sct_dataset_file_path` for Mixed bulk simulation and replaces it with an
explicit per-dataset `sct_reference` block.

- Mixed datasets must now declare their SCT source explicitly:
  - external path via `sct_reference.source_type: "path"`
  - earlier SCT dataset via `sct_reference.source_type: "generated_dataset"`
- Removed the previous implicit behavior where an SCT run mutated the in-memory
  Mixed reference path for later datasets.
- Added validation for forward references, non-SCT references, and unavailable
  `_sorted_bulk_imputed` outputs.
- Updated the example config and workflow/configuration docs to describe the
  new schema.

Relevant files:

- `src/simutme/workflow.py`
- `src/config_example.yaml`
- `tests/test_workflow_sct_version.py`
- `docs/configuration_io.md`
- `docs/workflow_mixed.md`
- `docs/workflow_sct.md`
- `docs/superpowers/specs/2026-08-01-explicit-mixed-sct-reference-design.md`

## July 29, 2026

### Mixed Dirichlet alpha estimation from predicted proportions

This update adds two Mixed-only dataset parameters for
`sampling_method_prop: "dirichlet"`:

- `dirichlet_alpha`
- `pred_cell_prop_file_path`

When `pred_cell_prop_file_path` is provided, SimuTME now estimates the
Dirichlet concentration parameters from the matched predicted cell-proportion
columns and uses the estimated values during mixed bulk simulation.

Relevant files:

- `src/simutme/workflow.py`
- `src/simutme/simulation/sampling.py`
- `src/simutme/simulation/simulator.py`
- `src/config_example.yaml`
- `docs/workflow_mixed.md`
- `docs/configuration_io.md`

### Mixed Cancer Cells fraction distribution output

This update adds a Mixed-only post-processing output that plots the
distribution of `Cancer Cells` proportions across all simulated samples.

- Reused the existing Mixed cell-fraction CSV rather than generating new data.
- Added a histogram with KDE for the `Cancer Cells` proportion distribution.
- Saved the figure in both `png` and `pdf` formats.
- Added a summary CSV with sample count and basic distribution statistics.
- Wrote all outputs under:
  - `paths.simu_bulk_dir/<dataset_name>/`

Relevant files:

- `src/simutme/workflow.py`
- `docs/workflow_mixed.md`
- `docs/configuration_io.md`

### Related spec

- `docs/superpowers/specs/2026-07-29-mixed-cancer-cell-fraction-distribution-design.md`

## July 28, 2026

### SCT sample-diversity analysis

This update adds an optional final post-simulation analysis step for each SCT
dataset to measure within-cell-type sample diversity directly from the saved
`log2(TPM+1)` profiles.

- Added optional SCT post-step controls:
  - `run_sample_diversity_analysis`
  - `sample_diversity_gene_mode`
  - `sample_diversity_n_top_genes`
  - `sample_diversity_gene_list_path`
  - `sample_diversity_distance_metric`
  - `sample_diversity_min_samples_per_cell_type`
- Defaulted the analysis to use all genes and cosine distance on the saved
  `log2(TPM+1)` SCT profiles.
- Computed one log-space median centroid (`mGEP_log`) per cell type and saved
  sample-to-centroid distances, summaries, and the centroid table.
- Saved per-dataset figures and tables under:
  - `<output_parent>/<dataset_name>/sample_diversity/`
- Kept the scope limited to SCT datasets only.

Relevant files:

- `src/simutme/workflow.py`
- `src/simutme/simulation/sample_diversity.py`
- `src/config_example.yaml`
- `docs/workflow_sct.md`
- `docs/configuration_io.md`

### Related spec

- `docs/superpowers/specs/2026-07-28-sct-sample-diversity-analysis-design.md`


## July 27, 2026

This release adds an explicit weight parameter for the sorted-bulk contribution
used by the SCT post-imputation averaging branch.

### Sorted-bulk imputation weight control

This update removes the hard-coded `0.5 * SCT + 0.5 * bulk` blend for rows with
`included == 1` and makes that contribution configurable.

- Added a new parameter:
  - `parameters.sorted_bulk_imputation_bulk_weight`
- Defaulted the new parameter to `0.5` so existing workflows keep the previous
  50/50 averaging behavior.
- Applied the weight only to the `included == 1` averaging branch using:
  - `(1 - bulk_weight) * SCT + bulk_weight * bulk`
- Kept the `included == 0` zero-guided replacement branch unchanged.
- Added validation so the weight must stay within `[0, 1]`.

Relevant files:

- `src/simutme/workflow.py`
- `src/simutme/sct_imputation.py`
- `src/config_example.yaml`
- `README.md`

### Related spec

- `docs/superpowers/specs/2026-07-27-sorted-bulk-imputation-weight-design.md`

## July 25, 2026

This release adds an explicit Mixed-dataset selector for choosing whether Mixed
simulation should use the original SCT dataset or the sorted-bulk-imputed SCT
dataset.

### Mixed SCT version selection

This update makes the Mixed dataset's SCT input explicit instead of relying
only on the last in-memory `config.sct_dataset_file_path` update.

- Added a new Mixed-only dataset parameter:
  - `sct_dataset_version`
- Supported values are:
  - `original`
  - `sorted_bulk_imputed`
- Defaulted omitted values to `sorted_bulk_imputed` so existing workflows keep
  preferring the imputed SCT dataset when it is available.
- Added reversible path resolution so Mixed simulation can:
  - use the original SCT path directly
  - infer the imputed SCT path from `paths.sct_dataset_file_path` and
    `sorted_bulk_imputation_output_suffix`
- Added validation so:
  - invalid `sct_dataset_version` values raise a clear error
  - `sct_dataset_version` cannot be used on SCT datasets
  - requesting the imputed SCT version fails clearly if the inferred file does
    not exist

Relevant files:

- `src/simutme/workflow.py`
- `src/config_example.yaml`

### Related spec

- `docs/superpowers/specs/2026-07-25-mixed-sct-version-selection-design.md`

## July 24, 2026

This release adds SCT-only dataset source selection by `dataset_id`, so you can
choose different source cohorts for cancer and non-cancer cells before SCT
simulation.

### SCT dataset_id source selection

This update lets you control where cancer and non-cancer cells come from during
SCT dataset generation.

- Added two optional SCT dataset parameters:
  - `cancer_cell_dataset_ids`
  - `non_cancer_cell_dataset_ids`
- Applied the split filtering rule on the merged scRNA-seq metadata using the
  major cell-type column:
  - rows with `cell_type == "Cancer Cells"` are filtered by
    `cancer_cell_dataset_ids`
  - all other rows are filtered by `non_cancer_cell_dataset_ids`
- Kept backward compatibility by falling back to `constants.sc_dataset_ids`
  when either new parameter is omitted.
- Kept the scope limited to SCT datasets only. Later Mixed simulations still
  use the SCT dataset produced earlier in the workflow, so they inherit this
  source selection automatically.
- Added validation so invalid `dataset_id` values raise a clear error, and so
  the new parameters cannot be used on Mixed datasets.

Relevant files:

- `src/simutme/workflow.py`
- `src/simutme/simulation/simulator.py`
- `src/simutme/simulation/simulator_sct.py`
- `src/config_example.yaml`

### Related spec

This spec note records the approved design for the new SCT source-selection
workflow.

- `docs/superpowers/specs/2026-07-24-sct-dataset-id-source-selection-design.md`

## July 19, 2026

This release updates the SCT post-imputation workflow and the nearest-neighbor
sampling implementation used during simulation.

### Sorted-bulk-guided SCT post-imputation

This update changes how `SimuTME` selects the sorted-bulk donor profile for SCT
post-imputation.

- Replaced random donor selection with Euclidean-nearest donor matching in TPM
  space within the mapped sorted-bulk cell type.
- Kept the existing `included` branch behavior:
  - `included == 1` still uses TPM-space averaging with `0.5 * SCT + 0.5 * bulk`.
  - `included == 0` still uses the zero-guided replacement workflow with
    `y_frac_threshold`.
- Clarified that after sorted-bulk-guided imputation, `SimuTME` updates the
  in-memory SCT reference path so later Mixed bulk RNA-seq simulation uses the
  newly imputed SCT dataset, not the original SCT dataset without imputation.
- Updated the SCT workflow documentation and configuration reference to describe
  nearest matched donor selection.

Relevant files:

- `src/simutme/sct_imputation.py`
- `README.md`
- `docs/workflow_sct.md`
- `docs/configuration_io.md`
- `src/config_example.yaml`

### Nearest-neighbor shared-memory sampling

This update makes the nearest-neighbor sampling path more efficient and removes
an import-time multiprocessing side effect.

- Removed the module-level `multiprocessing.set_start_method('spawn', force=True)`
  call from `nneighbors_shared_memory.py`.
- Moved the `spawn` start-method setup to the example entrypoint
  `src/main_example.py`.
- Reduced worker initialization overhead by unpickling `obs_df` directly from
  the shared-memory buffer.
- Sped up neighbor lookup by:
  - building a `cell_id -> matrix_index` map once per sampling call
  - using `np.argpartition(...)` for top-k selection
  - sorting only the selected neighbor subset by distance

Relevant files:

- `src/simutme/simulation/nneighbors_shared_memory.py`
- `src/main_example.py`

### Related specs

These spec notes record the approved design for today's changes.

- `docs/superpowers/specs/2026-07-19-sorted-bulk-nearest-match-design.md`
- `docs/superpowers/specs/2026-07-19-nneighbors-shared-memory-necessary-fixes.md`
