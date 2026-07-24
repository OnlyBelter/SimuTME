# Changelog

This page records notable documentation, workflow, and implementation changes in
`SimuTME`. Use it to track recent updates and to find the files and behaviors
that changed on a given date.

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
