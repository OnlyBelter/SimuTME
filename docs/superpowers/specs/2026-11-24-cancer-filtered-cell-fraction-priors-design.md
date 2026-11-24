# Cancer-Filtered Cell-Fraction Priors

## Goal

When a Mixed dataset uses Dirichlet sampling and sets `filtering_ref_types`,
estimate its Dirichlet alpha from prediction rows for those cancer types only.
This keeps the Dirichlet sampling prior aligned with the cohort selected for
GEP filtering.

## Current behavior

`filtering_ref_types` currently selects the expression samples used for
GEP-level filtering. Dirichlet alpha estimation currently uses every row in
the configured prediction file. Segment sampling uses its configured prior,
static DeSide range preset, and prediction-derived caps independently of
`filtering_ref_types`.

## Configuration behavior

Use the resolved `filtering_ref_types` values to restrict prediction rows for
Dirichlet alpha estimation only when GEP filtering is enabled and a cancer-type
subset is configured. `filtering_ref_types: null` preserves current all-row
behavior.

The prediction CSV must contain a `cancer_type` column when a subset is
configured and Dirichlet alpha is estimated from the file. Filter by exact
cancer-type labels before estimating alpha. Fail before candidate generation
if the column is missing or no rows match the requested types; do not silently
use all cancer types.

### Dirichlet sampling

When `sampling_method_prop: "dirichlet"` and
`pred_cell_prop_file_path` is configured, estimate `dirichlet_alpha` using only
the rows matching `filtering_ref_types`. Retain the existing behavior when the
subset is unset or the prediction file is not configured. A configured manual
`dirichlet_alpha` continues to be overridden by prediction-file estimation
when that file is configured.

### Segment sampling

Leave segment sampling unchanged. `filtering_ref_types` does not restrict
segment prediction-derived caps or the static
`"DeSide_predicted_TCGA_range"` preset. Explicit dictionary `cell_prop_prior`
bounds remain authoritative and retain their existing precedence over
prediction-file caps.

## Implementation boundaries

- Centralize prediction-row filtering for Dirichlet alpha estimation.
- Keep the source CSV unchanged.
- Include the resolved cancer subset and prediction-file content fingerprint
  in the generation identity so cached Dirichlet outputs cannot be reused
  across different subsets or prediction data.
- Preserve all segment sampling behavior.

## Validation and tests

Tests cover:

- Dirichlet alpha estimation uses only rows from selected cancer types.
- Missing `cancer_type` columns and subsets with no matching rows fail clearly.
- Manual or default `dirichlet_alpha` is unchanged when no prediction file is
  configured.
- Segment caps, static DeSide prior ranges, manual segment priors, and
  enlargement behavior remain unchanged when `filtering_ref_types` is set.
- Unrestricted Dirichlet runs retain current all-row alpha estimation.
- Generation identity differs when the selected cancer types or prediction
  source changes.
