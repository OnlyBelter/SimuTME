# Per-profile randomized SCT `n_base`

## Goal

Allow each generated single-cell-type (SCT) GEP to average a different number
of source single cells. An SCT dataset can draw each profile's `n_base`
independently from a configured list, such as
`[30, 40, 50, 60, 70, 80, 90, 100]`. Existing scalar `n_base` configurations
retain their current fixed-count behavior.

This setting varies the averaging depth used to construct pure SCT GEPs. It
does not claim to reproduce the actual number of cells contributing to TCGA
bulk RNA-seq samples.

## Configuration

Keep `parameters.n_base` as the fixed global fallback for existing
configurations. Add a global seed with a stable default:

```yaml
parameters:
  n_base: 30
  n_base_seed: 42

datasets:
  SCT_N10K_pan_cancer_Oct23:
    dataset_type: "SCT"
    n_base_values: [30, 40, 50, 60, 70, 80, 90, 100]
```

An SCT dataset may define either the existing scalar `n_base` override or the
new `n_base_values` list, but not both. If neither is present, the global
scalar `parameters.n_base` applies. A dataset-level `n_base_seed` may override
the global seed only when `n_base_values` is used. Seeds must be nonnegative
integers.

`n_base_values` must be a nonempty list of distinct positive integers. Boolean,
fractional, zero, negative, and duplicate entries fail validation. For this
first version, every configured value has equal probability.

## Sampling behavior

Resolve the SCT dataset's fixed or variable `n_base` policy before generation.
For a variable policy, use a dedicated NumPy random generator and the resolved
seed to assign one value independently to every stable SCT sample ID. Generate
the complete assignment before processing chunks so chunk size and resume
order do not change the selected values.

Pass each sample's assigned value into the existing per-sample `n_cell`
sampling task. Both random and nearest-neighbor sampling must select up to that
number of cells before averaging expression. Preserve current behavior when
the eligible pool is smaller than the requested value: use all eligible cells
if the existing minimum-cell requirement is met. Record the requested
`n_base` and the actual selected-cell count for each profile.

Mixed-dataset generation does not draw another `n_base`. It consumes the SCT
GEP reference as it does today; the varying averaging depths are represented
by the individual reference profiles.

## Reproducibility and outputs

Keep scalar output directory names unchanged. Variable-policy output names
must identify the full configured value list and seed, so different
distributions or seeds cannot reuse the same SCT or Mixed output directory.
Mixed output naming must inherit the effective policy from its generated SCT
reference.

Save the per-sample `n_base` assignment with the SCT artifacts, keyed by
sample ID. Store the value list, seed, uniform-probability policy, and sampled
counts in the run metadata. When resuming partial generation, validate that
the saved assignment matches the current sample IDs, values, and seed; fail
with an actionable error if it does not.

The dedicated seed makes the `n_base` assignment reproducible. It does not
change or guarantee reproducibility of the existing random anchor and
single-cell selection draws, which remain outside this change.

## Compatibility and scope

- Existing global and dataset-level scalar `n_base` configurations keep their
  current sampling behavior and output names.
- `n_base_values` is valid only for SCT dataset definitions. Mixed datasets
  continue to inherit their SCT reference's effective policy.
- Nearest-neighbor preparation must support the maximum configured
  `n_base_values` value. Per-profile sampling uses each assigned value.
- No probability weights or direct per-Mixed-sample cell-count model are
  included in this change.

## Validation

Tests must verify:

- Scalar configurations retain their current resolved value and output names.
- Valid lists draw only configured values, independently by SCT sample ID.
- The same sample IDs and seed produce the same assignment across chunking
  and resume.
- Both random and nearest-neighbor samplers honor each row's assigned count.
- Smaller eligible pools preserve current minimum-cell behavior and report the
  actual selected count.
- Invalid values, duplicate entries, and conflicting local scalar/list
  settings fail with clear errors.
- Mixed datasets inherit the source SCT policy for output naming.
- A changed list or seed cannot silently reuse prior generated artifacts.

The SCT run summary must report the configured distribution and the observed
`n_base` counts per cell type.
