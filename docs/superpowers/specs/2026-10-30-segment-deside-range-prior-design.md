# Segment cell-proportion range priors

## Goal

Let GEP-filtered Mixed segment simulations use the saved DeSide TCGA ranges or
an explicit cell-type range mapping. Explicit priors take precedence over
prediction-file maximum caps.

## Configuration and precedence

- `cell_prop_prior: "DeSide_predicted_TCGA_range"` selects the raw
  `DESIDE_TCGA_CELL_PROPORTION_RANGES` snapshot. Restrict it to the actual
  segment output labels; fail with the missing labels if any are unavailable.
- A mapping supplies explicit `[minimum, maximum]` bounds. Missing output
  labels remain unconstrained, and unknown labels fail validation.
- `enlarge_cell_prop_prior` is a finite nonnegative number, defaulting to
  zero. For each supplied range, compute `(max(minimum - value, 0),
  min(maximum + value, 1))`. Do not modify the package constant or the input
  mapping.
- Any supplied prior, including an empty mapping, disables segment caps from
  `pred_cell_prop_file_path`; the unused file is not opened. When no prior is
  supplied, keep the existing normalized prediction-file cap behavior.
- The preset and enlargement require Mixed segment sampling with GEP
  filtering enabled. Invalid or unsupported combinations fail explicitly
  rather than being silently ignored. Dirichlet prediction-file alpha
  estimation remains unchanged.

## Validation and provenance

Validate range values, output labels, finite nonnegative enlargement, and
simplex feasibility before sampling. Validate these settings at CLI preflight
before running earlier datasets and again for direct workflow calls.

Pass resolved bounds to the existing rejection sampler. Apply a bounded
attempt limit and log proposal counts for explicit priors as well as file
caps. Record the effective ranges in generation identity so cached datasets
cannot silently reuse results generated with different ranges. When a prior
overrides the configured prediction file, record that decision in logs and do
not report prediction-file caps in metadata.

## Verification

Test preset resolution, enlargement and clipping, partial explicit mappings,
invalid values and output labels, simplex feasibility, prior-over-cap
precedence, global and dataset prediction paths, cache identity, CLI
preflight, and unchanged Dirichlet and cap-only behavior. Update the
configuration reference and example without overwriting the user's example
configuration.
