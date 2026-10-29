# GEP-filtered cell-fraction distribution diagnostics

## Goal

Generate cell-type-specific fraction diagnostics for Mixed datasets whose
GEPs undergo filtering, while preserving the existing Dirichlet diagnostics.

## Behavior

The Mixed workflow enables all-cell-type fraction plots when either the
sampling method is `dirichlet` or the dataset's resolved `gep_filtering` is
enabled. The latter uses the effective dataset-level override and global
default, not merely the presence of gene filtering or PCA visualization.

The existing plotting helper reads the generated fraction CSV, whose rows are
aligned with the accepted simulated GEPs. It continues to emit the existing
Cancer Cells PNG, PDF, and summary for every Mixed dataset. When all-cell
plots are enabled, it also emits the existing per-cell-type PNG and PDF files
and the non-cancer cell-type summary CSV. Unfiltered non-Dirichlet datasets
retain their current behavior.

Cached datasets use the same diagnostics step if their fraction CSV is
available. This change does not regenerate missing fraction data or alter the
sampling, GEP filtering, plots, or existing file naming conventions.

## Validation

Exercise the Mixed workflow with Dirichlet, segment, and sparse sampling,
both with and without GEP filtering, including dataset overrides. Verify the
helper receives the intended flag and that filtered fractions generate the
expected per-cell-type figures and summary. Update the workflow documentation
to describe the new condition.
