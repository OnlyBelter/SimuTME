# CPU-first candidate simulation optimization

Optimize generation of 5,000 mixed bulk candidates without changing GEP
filtering, the source reference, or the candidate batch-size policy. This
design selects the CPU option discussed on October 20, 2026.

## Scope and alternatives

The recommended implementation combines precomputed CPU sampling pools with
bounded-memory eager aggregation. A persistent multiprocessing pool avoids
process startup but retains serialization and metadata filtering overhead
unless its task interface also changes. GPU aggregation adds dependencies,
transfer costs, and precision trade-offs; it is deferred.

The first implementation targets random selection of SCT reference profiles
and eager MCT aggregation. Neighbor-based SCT generation, patient-constrained
sampling, and the existing lazy aggregation path remain supported through
their existing implementations.

## Random sampling

For calls without patient constraints, construct eligible source-ID arrays
once per classification column and class rather than filtering the metadata
for every task. Execute selections in the parent process, avoiding a new
spawn pool and repeated metadata serialization for each candidate batch.
Keep the public `sample_random` signature unchanged.

Preserve task order, subtype classification, zero-sized selections, minimum
pool validation, without-replacement selection, and sorted returned ID
tuples. When the requested selection exceeds the eligible pool, return all
eligible IDs as the existing implementation does.

For SCT cancer tasks, preserve the existing 0.1 glioma probability when both
glioma and epithelial labels exist in the task's classification column.
Choose the branch independently per task. Preserve the existing sampler for
patient-constrained calls rather than broadening this optimization.

Do not promise identical draws to the legacy spawn-based sampler. Its worker
random states are not controlled by the Dirichlet seed. Preserve sampling
constraints and probabilities, and test seeded draws against the existing
single-task selection helper where equivalent.

## Eager bulk aggregation

Resolve source-row positions, fraction-row positions, and fraction-column
positions in batches instead of repeatedly calling pandas `.loc` inside the
sample loop. Read selected expression into bounded chunks and aggregate in
float64, retaining source row order within each sample.

Use a fast path only for MCT tasks whose selected IDs each identify one source
row and whose source index is unique. Retain the legacy path for other task
representations. Unknown source IDs or fraction labels must fail clearly,
not accidentally index the last array row.

Bound temporary expression arrays by a memory budget rather than allocating
a full candidates-by-cell-types-by-genes tensor. Resolve chunks on complete
sample boundaries. Do not make a full float64 copy of the source reference.

Preserve the existing multiplication order: expression times RNA coefficient,
then times cell fraction, followed by summation over selected rows. Preserve
the optional RNA-weighted normalization before noise, sample-wise noise draw
order, final normalization to one million, rounding to three decimals, and
sample and gene labels. Preserve the legacy eager sample order.

Keep lazy SCT aggregation unchanged in this first implementation. Do not
subset to filtering genes before aggregation because normalization and noise
currently depend on the full source gene universe.

## Correctness and regression tests

Test ordinary and subtype pools, cancer branching, zero selections, small
pools, missing labels, patient fallback, and sorted IDs. Ensure the optimized
random path does not create a multiprocessing pool.

Compare eager fast-path aggregation with the legacy calculation on identical
fractions and selected rows. Cover repeated source IDs, unsorted sample IDs,
unequal task counts, chunk boundaries, extreme float32 input values, RNA
coefficient fallback to parent types, enabled and disabled noise, and missing
labels. Verify source expression is unchanged after aggregation.

Run existing numeric stability, mixed fraction, subtype/version, candidate
batch-size, and filtering regression tests. Filtering acceptance thresholds
and calibration artifacts are not modified.

## Benchmark protocol

Provide a reproducible script for 5,000 candidates that measures metadata
sampling separately from expression aggregation. Replay identical selected
IDs and fractions for legacy and optimized aggregation.

Report initialization separately from repeated execution. Use five measured
repetitions by default and report the median, range, speedup, workload shape,
thread count, software versions, and peak process memory where available.
Run variants in separate processes for meaningful peak-memory comparisons.
Compare full candidate generation without file export or filtering, and state
those exclusions in results.

Use a deterministic synthetic workload for portable comparisons and support
the actual SCT reference for realistic measurements. Synthetic timings must
not be presented as measured speedups for the user's running simulation.
Benchmark the legacy sampling helper both directly and through the existing
spawn pool to separate metadata scanning from process overhead.

No speedup target is asserted before measurement. A fast path becomes the
default only after correctness checks pass and representative benchmarks
show an improvement. No GPU dependency or new YAML setting is required.

## Delivery

Deliver focused source changes, regression tests, a benchmark script, and a
concise measured-results report. Keep temporary benchmark outputs and test
caches outside the repository. Do not commit or push without an explicit
user request.
