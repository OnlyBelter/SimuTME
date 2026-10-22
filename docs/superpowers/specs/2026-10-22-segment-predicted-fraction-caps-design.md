# Predicted-fraction caps for segment sampling

When a Mixed dataset uses `sampling_method_prop: "segment"`,
`gep_filtering: true`, and provides `pred_cell_prop_file_path`, use the
prediction file to cap each output cell type's generated fraction at its
maximum observed predicted fraction. Apply these bounds during fraction
sampling, before constructing expression profiles.

The user selected global empirical maxima across all prediction-file rows.
The user approved this design before implementation. The implementation,
regression tests, documentation, and pilot configuration now follow this
contract.

## Existing behavior and input evidence

`pred_cell_prop_file_path` currently estimates a Dirichlet alpha vector and is
accepted only with `sampling_method_prop: "dirichlet"`. The `segment` sampler
already supports per-cell-type minimum and maximum ranges through
`cell_prop_prior`, using rejection before expression aggregation. The
requested behavior reuses that segment constraint mechanism, but derives its
upper bounds from the prediction file.

The configured DeSide prediction table contains 7,699 rows and the 16 target
cell-type columns. Values are finite, nonnegative fractions, and row totals
range from 0.996 to 1.004 due to rounding. Normalize each row before
calculating caps. The observed per-type maxima sum to approximately 5.216,
so the upper-bound simplex is feasible.

A vectorized simulation of 200,000 segment proposals found that about 4.95%
pass all 16 empirical-maximum caps. This is a rough proposal-acceptance
estimate, not a production timing result. Rejected fraction proposals do not
load expression or run GEP filters, but can increase composition-generation
time.

## Configuration contract

Apply caps only when all of these conditions hold:

- `dataset_type: "Mixed"`
- `sampling_method_prop: "segment"`
- dataset-level resolved `gep_filtering` is `true`
- `pred_cell_prop_file_path` is provided

For this combination, resolve and validate the file before bulk generation.
Keep the existing Dirichlet-alpha estimation behavior unchanged for
`sampling_method_prop: "dirichlet"`.

If a prediction file is supplied with segment sampling while GEP filtering is
disabled, fail fast with an error explaining that these caps are enabled only
for filtered segment generation. For other non-Dirichlet sampling methods,
retain the existing validation that rejects Dirichlet-only prediction inputs.
If no prediction file is supplied, segment behavior remains unchanged.

## Cap derivation and prior combination

Derive the required cell-type columns from the output types used by
`generate_cell_fraction`. Ignore metadata and extra prediction columns.
Require one column for every output cell type, with numeric, finite,
nonnegative entries and strictly positive row totals. Normalize each
prediction row to sum to one, then set each cap to the maximum normalized
value in that cell-type column across all rows.

Build a range `(0.0, empirical_max)` for every output type. If the dataset
also provides `cell_prop_prior`, intersect its range with the empirical cap:

- combined minimum is the larger of the manual and derived minima
- combined maximum is the smaller of the manual maximum and empirical cap

Reject malformed ranges, a combined minimum above its maximum, or caps whose
sum is below one. This first version uses global maxima across the entire
prediction file; it does not condition on the `cancer_type` metadata column.

Pass the combined ranges to the existing segment fraction sampler. It keeps
drawing segment proposals until it obtains the requested number of valid
fractions. Validate caps before drawing; add a clear bounded-attempt failure
for the cap-constrained path so unexpectedly low feasibility cannot loop
indefinitely. Log requested, attempted, accepted, and rejected fraction
counts. Do not run expression aggregation or GEP filtering on rejected
fraction proposals.

## Provenance and resume compatibility

Record the prediction-file path, hash of the normalized per-cell maxima,
normalized caps, cap scope (`global`), and row-normalization policy in the
generation identity. This makes resumed intermediates incompatible when the
file contents or derived caps change, even if the path stays the same.

Record the caps and source path in generation logs and the Mixed dataset
summary. Do not persist or alter the input prediction CSV. Do not estimate a
Dirichlet alpha or write the Dirichlet-alpha artifact for segment sampling.

## Tests

Add tests that verify:

- prediction rows are normalized before taking per-type maxima
- output-type columns are required and extra metadata columns are ignored
- nonnumeric, missing, negative, nonfinite, and zero-sum inputs fail clearly
- the generated segment fractions satisfy every empirical maximum
- manual `cell_prop_prior` bounds are intersected and conflicts fail
- infeasible caps and bounded-attempt exhaustion report clear errors
- segment plus prediction path is allowed only with GEP filtering enabled
- Dirichlet alpha behavior and segment behavior without this option are unchanged
- cap values enter checkpoint identity and summary provenance
- invalid cap proposals are rejected before expression aggregation

Use a fixed synthetic prediction table to test deterministic maxima and
constraint behavior. Run a bounded benchmark using the provided DeSide table
and current 1,000-profile variance-1 segment pilot settings. Report fraction
proposal acceptance and generation-stage timings; do not claim improved final
GEP acceptance or biological similarity until the filtered pilot finishes.

## Alternatives considered

Using the 99th percentile instead of the observed maximum would reduce
sensitivity to isolated prediction outliers, but would impose a stricter
ceiling than the user requested. Clipping and redistributing excess fractions
would avoid proposal rejection but would alter the segment distribution.
The selected design uses the literal normalized empirical maximum and
rejection sampling, preserving the existing segment proposal distribution
conditional on passing the caps.

The configured pilot points to the existing DeSide CSV; the prediction table
itself is read-only and is not modified.
