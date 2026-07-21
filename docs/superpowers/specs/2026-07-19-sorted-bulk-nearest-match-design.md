# Sorted-Bulk SCT Post-Imputation: Nearest Matched Bulk Selection

## Goal

Revise the optional sorted-bulk-guided SCT post-imputation workflow in
`SimuTME` so that the sorted-bulk reference profile is no longer chosen at
random.

Instead, for each SCT sample, the workflow should find the most similar
sorted-bulk profile within the mapped bulk cell type and use that matched bulk
profile in the existing post-imputation branch logic.

This is a follow-up change to the already approved `included`-aware behavior:

- `included == 1`: TPM-space averaging
- `included == 0`: zero-guided replacement with `y` diagnostics

## Existing Behavior

The current workflow:

1. maps each SCT cell type to a sorted-bulk reference cell type
2. randomly selects one sorted-bulk sample from that mapped bulk class
3. branches on the `included` flag

Current branch behavior:

- `included == 1`: use `0.5 * SCT_TPM + 0.5 * BULK_TPM`
- `included == 0`: use the existing zero-guided replacement workflow

The requested change only affects step 2.

## Approved Change

### Bulk Reference Selection Rule

For each SCT sample:

1. identify its mapped sorted-bulk cell type from the mapping table
2. collect all sorted-bulk reference profiles belonging to that mapped bulk
   cell type
3. compute Euclidean distance in aligned TPM space between the SCT sample and
   each candidate bulk profile
4. choose the single nearest bulk profile

The selected nearest bulk profile then feeds into the existing `included`
branch logic.

### Distance Definition

Use:

- Euclidean distance on aligned TPM vectors

This matches the user's approved design direction and is intentionally simple.

## Branch Logic After Matching

### Case 1: `included == 1`

Keep the existing averaging behavior, but replace the random bulk profile with
the nearest matched bulk profile:

```text
averaged_tpm = 0.5 * sct_tpm + 0.5 * nearest_bulk_tpm
```

Then convert back to:

```text
log2(TPM+1)
```

Keep the previously approved rules unchanged:

- do not calculate `y`
- do not apply `y_frac_threshold`
- do not renormalize after averaging

### Case 2: `included == 0`

Keep the existing zero-guided replacement workflow unchanged, except that the
bulk donor profile is now the nearest matched bulk profile instead of a random
bulk profile.

Keep unchanged:

- zero-only replacement logic
- `y` calculation before and after adjustment
- `y_frac_threshold`
- renormalization to preserve the TPM total

## Implementation Scope

This change should remain surgical.

### Primary code path

- `SimuTME/src/simutme/sct_imputation.py`

### Likely code updates

- replace the current random-assignment helper with nearest-match assignment
- preserve the existing output structure returned to the batch imputation step
- keep the downstream `_impute_batch()` logic as stable as possible

### Data flow

The implementation should still produce, for each SCT cell:

- one bulk sample index
- one `included` flag

The difference is that the bulk sample index should now come from nearest-match
search within the mapped bulk cell type instead of random sampling.

## Performance And Memory

The nearest-match search should be implemented in batches or by bulk cell-type
groups, rather than by constructing unnecessarily large all-by-all matrices
across unrelated cell types.

Practical guidance:

- restrict candidates to the mapped bulk cell type first
- compute distances only within that candidate set
- prefer batched computation if a mapped bulk class is large

The goal is exact nearest selection, not approximate nearest neighbor search.

## Documentation Updates

Update package documentation so it consistently describes:

- "find the most similar sorted-bulk sample" instead of "randomly sample one"
- Euclidean-nearest matching in TPM space within the mapped bulk cell type
- unchanged downstream behavior for `included == 1` and `included == 0`

Relevant documentation files include:

- `SimuTME/README.md`
- `SimuTME/docs/configuration_io.md`
- `SimuTME/docs/workflow_sct.md`

## Verification

After implementation, verify:

1. the sorted-bulk donor is no longer selected randomly
2. donor selection is restricted to the mapped bulk cell type
3. nearest-match selection uses Euclidean distance on aligned TPM vectors
4. `included == 1` still performs TPM-space averaging with the selected donor
5. `included == 0` still performs the existing zero-guided workflow
6. the docs no longer describe random bulk selection

## Out Of Scope

The following are not part of this change:

- changing the `0.5 / 0.5` averaging rule
- changing the `included` branching semantics
- introducing correlation, cosine, or approximate-NN matching
- redesigning the SCT simulation-side `n_neighbors` sampling workflow
- changing file outputs or Mixed-workflow handoff behavior
