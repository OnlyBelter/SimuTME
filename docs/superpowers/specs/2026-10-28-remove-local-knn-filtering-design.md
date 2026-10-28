# Remove local KNN GEP filtering

## Goal

Remove active support for the `knn` and `knn_residual` GEP-filtering methods
without changing the remaining reference-based filtering workflows or
deleting existing generated outputs.

## Design

Retain these `gep_filtering_method` values:

- `median_gep`
- `mean_gep`
- `mgep`
- `linear_mmd`

Reject `knn` and `knn_residual` during configuration resolution, before
loading reference data, with an error that names the remaining supported
methods. Do not silently map old values to another method.

Remove the active local-filter implementation and its exclusive components:
the TCGA holdout split and calibration, local neighbor/residual scoring,
local score diagnostics, local-only configuration parameters, local-only
output metadata, and KNN-only tests. Remove these methods and parameters from
the example config and current workflow/configuration docs. Preserve
historical changelog entries and design records.

Keep `paths.tcga2cancer_type_file_path` because other workflows use it for
`filtering_ref_types`. Keep shared gene selection, variance screening, and
centroid-based filtering unchanged. Without local KNN modes, variance
screening uses the selected reference cohort rather than a fitting-only
subset.

Do not delete, rewrite, or automatically reuse existing KNN-generated files.
No migration or output cleanup runs as part of this change.

## Verification

- Confirm KNN methods fail during filtering-parameter validation with an
  actionable supported-method list.
- Confirm all four remaining filtering methods continue to pass their
  existing tests.
- Remove or update tests that only exercise local KNN behavior; retain tests
  for shared variance filtering and centroid modes.
- Search current source and user-facing docs for active KNN configuration and
  behavior references, excluding historical changelog and design records.
- Run the test suite and `git diff --check`.
