# Two-half nearest-neighbor SCT sampling for rare cell types

Last updated: 2026-08-07

## Status

Draft for review.

## Goal

Improve SCT sample diversity for rare cell types when `sampling_method_gep: "n_neighbors"` is used, especially when:

- the candidate cell pool for a cell type is much smaller than the requested
  number of SCT samples, and
- the current one-anchor nearest-neighbor workflow repeatedly regenerates the
  same or nearly identical cell sets.

The motivating example is `Plasma B cells`, where the available pool can be
small enough that generating `10,000` SCT samples with `n_base=30` leads to
many repeated positive samples.

## Problem summary

Current `n_neighbors` SCT sampling behaves like:

1. randomly choose one anchor cell inside the target cell type,
2. deterministically take the nearest `n_base - 1` cells around that anchor,
3. average those selected cells into one SCT sample.

This is implemented in
`src/simutme/simulation/nneighbors_shared_memory.py:get_sample_id_with_neighbors`.

For rare cell types, many SCT samples inevitably reuse the same anchors. Because
the neighbor selection around each anchor is effectively deterministic, the same
anchor often produces the same selected-cell tuple. In practice this causes:

- repeated identical SCT samples,
- reduced within-cell-type diversity,
- a more centroid-like SCT target distribution for downstream training.

## Design constraints

The fix should:

- preserve the local-manifold intuition of the existing `n_neighbors` method,
- avoid a large rewrite of the shared-memory nearest-neighbor engine,
- be backward compatible unless explicitly enabled,
- degrade gracefully when only one dataset exists for a cell type,
- remain applicable to rare cell types without failing too aggressively.

## Options considered

### Option A: average two random cells first, then query neighbors around the averaged pseudo-cell

Pros:

- introduces more anchor diversity than the current one-anchor design,
- could reduce exact sample repetition.

Cons:

- requires querying neighbors around a virtual averaged point rather than a real
  cell, which does not fit the current precomputed cell-to-cell distance matrix
  implementation,
- is a larger algorithmic change,
- may bias samples further toward local centroids and reduce diversity if the
  pseudo-anchor becomes overly averaged.

### Option B: split `n_base` into two half-neighbor samples and then average them

Example for `n_base=30`:

1. run the current `n_neighbors` workflow once with `n_base=15`,
2. run it a second time with `n_base=15`,
3. concatenate the selected cells,
4. average all `30` cells into one SCT sample.

Pros:

- preserves the current real-cell-anchor nearest-neighbor logic,
- increases combinatorial diversity substantially,
- requires only moderate changes around the existing sampling path,
- remains interpretable and close to the original method.

Cons:

- if the two half-samples overlap heavily, diversity gain is smaller,
- some repeated structures can still occur for extremely small pools.

## Recommended approach

Use **Option B** as the new optional SCT-only sampling mode:

- generate each positive SCT sample from **two nearest-neighbor half-groups**
  rather than one full-group anchor,
- add a **soft preference** that the two halves come from different datasets
  when more than one dataset exists for that cell type,
- if that preference cannot be satisfied, fall back to the best available
  same-dataset solution rather than failing.

This keeps the current algorithm recognizable while directly addressing the
duplication problem.

## Proposed interface

Add one new optional SCT-only parameter:

```yaml
datasets:
  SCT_example:
    dataset_type: "SCT"
    sampling_method_gep: "n_neighbors"
    neighbor_sampling_scheme: "single_anchor"   # default
    # allowed: "single_anchor", "two_half_anchor"
```

Behavior:

- `"single_anchor"`: current behavior, unchanged.
- `"two_half_anchor"`: new behavior described in this spec.

Why a new parameter is preferred over silently changing `n_neighbors`:

- preserves backward compatibility,
- allows controlled A/B comparison,
- makes the change explicit in configs and downstream result naming.

No new Mixed-dataset behavior is needed. This change applies only during SCT
generation.

## Algorithm details

### 1. Half-size definition

For a requested `n_base`:

- `half_1_size = ceil(n_base / 2)`
- `half_2_size = n_base - half_1_size`

So:

- `30 -> 15 + 15`
- `31 -> 16 + 15`

This preserves the total number of cells contributing to each SCT sample.

### 2. Per-sample generation flow

For one positive SCT sample of a target cell type:

1. determine the candidate cell pool for that cell type using the already
   filtered `obs_df`,
2. identify the set of available `dataset_id` values for that cell type,
3. sample the first half-group with the current nearest-neighbor logic using
   `half_1_size`,
4. sample the second half-group with the same logic using `half_2_size`,
5. concatenate the two selected cell tuples,
6. average the expression of all selected cells into one SCT sample.

### 3. Soft preference for different datasets

If more than one dataset exists for the target cell type and `dataset_id` is
available in `obs_df`:

- first half: no dataset restriction,
- second half: prefer a different dataset from the first half's anchor dataset,
- if a valid second half cannot be formed from a different dataset, fall back to
  unrestricted sampling.

This is a **soft preference**, not a strict requirement.

### 4. Dataset choice definition

The dataset constraint should apply at the **half-sample anchor level**.

That means:

- each half-group is centered on one anchor cell,
- the "different dataset" preference refers to the dataset of the two anchor
  cells,
- neighbors are selected from the candidate pool restricted to the chosen anchor
  dataset when the preference is being applied.

This keeps the logic simple and interpretable:

- half-group 1 comes from anchor dataset A,
- half-group 2 preferably comes from anchor dataset B,
- final SCT sample is the average of these two locally coherent half-groups.

### 5. Fallback behavior

The new mode must degrade gracefully:

- if only one dataset exists for the target cell type, use two half-groups from
  that dataset,
- if multiple datasets exist but no alternate dataset has enough cells to form
  the second half-group, fall back to unrestricted second-half sampling,
- if a half-group has between `minimum_n_base` and its requested half-size,
  re-use the current "take all available cells" behavior,
- if the cell pool is too small even for one half-group, keep the existing error
  semantics.

### 6. Overlap policy

This first version will **not** add explicit overlap penalties between the two
halves.

Reason:

- it keeps the implementation smaller and easier to verify,
- the two-half split itself already provides a large diversity gain,
- overlap control can be added later only if sample-diversity metrics still show
  collapse.

### 7. Aggregation semantics

No change is needed in downstream aggregation:

- the selected cell IDs from both halves are concatenated,
- `_map_cell_id2exp(...)` continues to average the selected single-cell
  expression profiles into one SCT GEP.

This keeps file formats and downstream interfaces unchanged.

## Implementation placement

### `src/simutme/simulation/nneighbors_shared_memory.py`

Add a helper that can sample one nearest-neighbor group with optional dataset
restriction, reusing the current anchor-and-neighbor logic.

Then add a wrapper for the new two-half strategy that:

- computes `half_1_size` and `half_2_size`,
- samples two half-groups,
- applies the soft different-dataset preference for half-group 2,
- returns one concatenated selected-cell tuple.

### `src/simutme/simulation/simu_helper.py`

Extend the SCT neighbor sampling wrapper so the new scheme can be chosen without
affecting existing random sampling or Mixed-dataset paths.

### `src/simutme/simulation/simulator_sct.py`

Add the new SCT-only config plumbing and pass the selected neighbor sampling
scheme into the neighbor-sampling path.

### `src/simutme/workflow.py`

Add config validation and YAML plumbing for the new parameter, following the
same pattern used for other SCT-only knobs.

### `src/config_example.yaml`

Add a commented example for the new parameter near `sampling_method_gep`.

## Logging / diagnostics

Add lightweight INFO logging for the new mode:

- current neighbor sampling scheme,
- when the second half successfully uses a different dataset,
- when the second half falls back to the same dataset because no alternative is
  feasible.

This makes rare-cell-type behavior inspectable without changing output file
schemas.

## Testing plan

Add focused tests covering:

1. **Backward compatibility**
   - `neighbor_sampling_scheme="single_anchor"` reproduces current behavior.

2. **Half-size correctness**
   - `n_base=30` yields exactly `30` selected cells total from `15 + 15`,
   - odd `n_base` values split correctly.

3. **Different-dataset soft preference**
   - when two datasets both have enough cells for a target cell type, the two
     half-group anchors come from different datasets.

4. **Fallback behavior**
   - when only one dataset exists, the new scheme still returns a valid sample,
   - when multiple datasets exist but the alternate dataset cannot support the
     second half, fallback succeeds rather than failing.

5. **Rare-cell-type feasibility**
   - small synthetic pools still produce valid outputs when feasible,
   - impossible cases raise the same class of error as the current code.

6. **Diversity sanity check**
   - on a synthetic repeated-run setting, the two-half scheme produces more
     unique selected-cell tuples than the current single-anchor scheme.

## Out of scope for this version

- explicit overlap minimization between the two halves,
- pseudo-cell query anchors based on averaged expression,
- changing Mixed bulk generation behavior,
- changing output file formats for generated SCT `.h5ad`,
- adding a strict different-dataset requirement.

## Expected outcome

Compared with the current `n_neighbors` SCT sampling, the new mode should:

- reduce exact duplicate SCT samples for rare cell types,
- increase within-cell-type SCT diversity,
- better preserve multi-dataset structure for scarce cell types such as
  `Plasma B cells`,
- remain close enough to the existing method that behavior is still easy to
  reason about and compare.
