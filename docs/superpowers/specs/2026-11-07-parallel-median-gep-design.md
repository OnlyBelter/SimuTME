# Parallel median-GEP candidate generation

Extend the existing Mixed candidate worker engine to support `median_gep`
filtering in reference PCA space with L1 distance. Preserve the serial filter
and the existing `mgep` parallel path.

## Scope and configuration

Parallel `median_gep` requires segment sampling, enabled GEP filtering, an
eager SCT reference, `gep_filtering_in_pca_space: true`, and
`gep_norm_ord: 1`. Positive `mixed_candidate_workers` requires a nonnegative
`mixed_candidate_seed`. Reject other parallel `median_gep` configurations
with a clear error rather than changing their filtering semantics. Dataset
overrides retain their existing precedence.

The current user's target configuration already selects PCA-space L1 filtering.
Keep its worker settings opt-in, and use a fresh dataset name and output
location when starting a parallel run. Never silently reuse serial
intermediates as a parallel checkpoint.

## Scientific behavior

Use the existing fixed-gene filter and selected-gene TPM normalization. Fit
or load the reference PCA once in the coordinator on `log2(TPM + 1)`. Use
the coordinate-wise median of reference PCA scores as the center. Workers
transform their candidates with this same PCA and calculate L1 distances to
the center. Apply both configured reference-distance quantile boundaries
inclusively, including an optional gene-range filter when configured.

Keep accepted sample order, gene order, expression format, fraction labels,
and selected-cell output consistent with the existing engine. Batch seeds and
ordered commits preserve reproducibility across worker counts. Parallel
samples need not match historical serial random draws.

## Execution and recovery

The coordinator prepares the PCA, center, and reference distances before
starting workers. Workers reuse that prepared filter without refitting PCA.
Reset all per-batch range and distance counters before each worker batch, and
merge returned counters exactly once in batch order. Persist PCA component
information and filter counters in metadata and committed checkpoints.

Keep the existing memory-mapped SCT matrix, output writer, target lock, and
resume consistency checks. Spawned processes receive their own PCA/filter
state, so benchmark their memory use on the target server. Existing filter
identity already records the method, PCA settings, norm, and quantiles; keep
the `mgep` engine identity compatible.

## Validation

Replay fixed candidate batches through serial and worker filters and compare
PCA scores, L1 distances, acceptance masks, and gene order. Test lower and
upper boundaries, gene-range filtering, empty accepted batches, accumulated
filter counters, and the PCA metadata. Compare one and multiple workers with
the same seed, including checkpoint interruption and resume.

Test configuration precedence and clear rejection of unsupported modes.
Retain the `mgep` regression tests. Extend the benchmark to accept the
PCA-space L1 `median_gep` target and report filtering acceptance without
changing the existing `mgep` measurement. Update configuration and workflow
documentation. Report local test results separately from any server
performance measurement.
