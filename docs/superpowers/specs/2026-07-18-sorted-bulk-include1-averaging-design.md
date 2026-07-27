# Sorted-Bulk SCT Post-Imputation: `included == 1` Averaging

## Goal

Revise the optional sorted-bulk-guided post-imputation workflow for SCT datasets
in `SimuTME` so that matched cell types use full-profile averaging instead of
zero-only replacement.

The change applies to rows in the mapping file
`sorted_bulk_cell_type_mapping_file_path` where:

- `included == 1`

For those SCT cell types, the workflow should:

1. randomly sample one sorted-bulk reference GEP from the mapped bulk cell type
2. convert the SCT sample to TPM space
3. compute the simple average in TPM space
4. convert the averaged profile back to `log2(TPM+1)`

The averaging weight is fixed at:

- `0.5 * SCT_TPM + 0.5 * BULK_TPM`

## Existing Behavior

The current workflow samples one sorted-bulk reference profile for each SCT
cell, then performs zero-guided replacement:

- identify near-zero SCT genes using `eps`
- copy only nonzero bulk values into the SCT zero positions
- compute added expression `y`
- cap `y` using `y_frac_threshold`
- rescale the original SCT values so the final row total stays at `1e6`

This logic currently applies to all mapped SCT cell types, regardless of the
`included` column in the mapping file.

## Approved Change

### Branch on `included`

The workflow should branch by the mapping row for each SCT cell type.

### Case 1: `included == 1`

Use full-profile averaging in TPM space:

```text
averaged_tpm = 0.5 * sct_tpm + 0.5 * bulk_tpm
```

Then convert:

```text
log2(TPM+1) = log2(averaged_tpm + 1)
```

For this branch:

- do not use zero-only replacement
- do not calculate `y`
- do not apply `y_frac_threshold`
- do not renormalize after averaging

Rationale:

- both `sct_tpm` and `bulk_tpm` are already normalized to sum to `1e6`
- therefore their `0.5 / 0.5` average also sums to `1e6`, aside from tiny
  floating-point error

### Case 2: `included == 0`

Keep the current workflow unchanged:

- zero-only replacement using nonzero bulk genes
- `y` calculation before and after the cap
- `y_frac_threshold` enforcement
- rescaling to keep the row sum at `1e6`

## Implementation Scope

The change should be surgical and localized.

### Code changes

Primary logic change:

- `SimuTME/src/simutme/sct_imputation.py`

Supporting workflow integration is already present and should remain unchanged:

- `SimuTME/src/simutme/workflow.py`

### Data flow update

The implementation should derive, for each SCT cell:

- the mapped bulk cell type
- the `included` flag for that SCT cell type
- the randomly assigned bulk sample index

During batch processing:

- rows with `included == 1` follow the averaging branch
- rows with `included == 0` follow the existing zero-guided branch

## Diagnostics

### `y` diagnostics

The existing `y` diagnostics are only meaningful for the `included == 0`
branch, because they quantify newly added expression under the zero-guided
replacement workflow.

Approved behavior for `included == 1`:

- set `y_added_expression_before_adjustment = 0`
- set `y_added_expression_after_adjustment = 0`

This keeps the existing output format unchanged while making it explicit that no
`y`-based adjustment was applied to those matched cell types.

### Sparsity diagnostics

Keep all existing pre/post SCT sparsity outputs unchanged:

- per-cell CSV
- summary CSV
- boxplots

These diagnostics should reflect the new averaging behavior naturally through
the output matrix and do not need a structural redesign.

## Documentation Updates

Update the package documentation so it clearly states:

- `included == 1` means full averaging with a randomly selected matched
  sorted-bulk GEP in TPM space
- `included == 0` means the original zero-guided replacement workflow remains
  in effect
- `y` diagnostics only correspond to the `included == 0` branch; rows processed
  by the averaging branch contribute zero in those `y` output tables

Relevant docs likely include:

- `SimuTME/README.md`
- `SimuTME/docs/configuration_io.md`
- any workflow documentation that describes the sorted-bulk-guided SCT
  post-imputation behavior

## Verification

After implementation, verify:

1. matched cell types with `included == 1` no longer use zero-only replacement
2. rows in the averaging branch preserve a TPM row sum of approximately `1e6`
   before conversion back to `log2(TPM+1)`
3. rows in the `included == 0` branch behave exactly as before
4. output files are still written to the same locations
5. `y` diagnostic tables remain structurally unchanged
6. documentation reflects the new branch behavior correctly

## Out of Scope

The following are not part of this change:

- changing the averaging weights away from `0.5 / 0.5`
- redesigning the `y` diagnostics into a new metric for averaged rows
- changing how bulk samples are randomly assigned
- altering the downstream handoff so Mixed simulations use the imputed SCT file
- modifying unrelated ALRA or standalone single-cell imputation workflows
