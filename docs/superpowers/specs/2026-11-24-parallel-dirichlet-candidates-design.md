# Parallel Dirichlet Candidate Generation

## Goal

Allow Mixed datasets using `sampling_method_prop: "dirichlet"` to use the
existing parallel candidate engine with GEP filtering. Preserve the configured
worker count, Dirichlet alpha, and seed behavior without changing serial
generation or the existing segment-sampling path.

## Current behavior

Parallel candidate generation currently accepts only segment sampling. The
workflow preflight and `GEPGenerationConfig` both reject Dirichlet sampling,
and `_run_batch` hard-codes segment fractions. The target configuration uses
Dirichlet sampling with 64 workers, seed 42 for candidate generation, seed 42
for Dirichlet draws, and PCA-space L1 `median_gep` filtering with
`n_subgroup: 100`.

## Design

Cell-proportion generation and SCT-profile generation are separate stages in
the candidate pipeline. `sampling_method_prop` selects only how cell fractions
are generated. Those fractions then determine per-cell-type sampling counts;
the worker samples SCT profiles, aggregates them using those fractions, and
applies the same GEP filter regardless of the fraction-sampling method. The
stages therefore remain modular, while SCT sampling is conditional on the
generated fractions.

Extend the supported parallel sampling methods to `segment` and `dirichlet`.
Keep the existing requirements for enabled GEP filtering with `mgep` or
`median_gep`; parallel `median_gep` still requires PCA-space L1 filtering.
The current `n_subgroup` behavior and reference handling remain unchanged.

Pass the resolved `sampling_method_prop` and `dirichlet_alpha` to workers.
Each worker generates fractions using the configured method. Only the
fraction-generation call receives method-specific inputs: Dirichlet alpha and
its local NumPy generator for Dirichlet sampling, or the existing prior and
attempt limit for segment sampling. For Dirichlet sampling, derive the local
generator deterministically from `dirichlet_sampling_seed` and the batch ID.
If the Dirichlet seed is unset, derive that stream from `mixed_candidate_seed`
and the batch ID instead. Continue deriving SCT-selection and other worker
randomness from `mixed_candidate_seed` and the batch ID. Both streams remain
deterministic across worker counts and completion order. Parallel results need
not match historical serial draws sample-for-sample.

The serial generation path and segment sampling behavior are unchanged.
Dirichlet alpha estimation remains in the coordinator, so workers receive the
already-resolved alpha values. No new command-line or YAML settings are
required.

## Cache identity and recovery

Increment the parallel candidate engine identity version so checkpoints created
by the segment-only implementation cannot be reused by the updated engine.
Keep worker count as execution metadata rather than part of identity, preserving
worker-count-independent results. The configured Dirichlet alpha and seed remain
part of the generation identity.

## Validation

- Accept parallel Dirichlet with enabled `mgep` or valid PCA-space L1
  `median_gep`; continue rejecting disabled or unsupported GEP filters.
- Verify configured alpha reaches workers and Dirichlet seed changes produce
  distinct fraction draws.
- Verify repeated runs with the same seeds produce identical ordered outputs
  with one and multiple workers.
- Verify an unset Dirichlet seed falls back deterministically to the candidate
  seed.
- Retain tests for segment sampling, subgroup filtering, checkpoint recovery,
  and invalid parallel settings.
- Run the focused parallel-candidate and Dirichlet-seed tests, then the full
  SimuTME test suite.

## Documentation

Update the Mixed workflow and configuration documentation to list Dirichlet
sampling as supported by parallel candidate generation and explain the separate
seed roles and fallback behavior.
