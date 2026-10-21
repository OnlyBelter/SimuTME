# All-cell SCT neighbor sampling

Keep every eligible cell available for SCT generation with
`max_cells_per_cell_type: "all"`. Use batched exact neighbor queries for sampled
anchors instead of precomputing neighbors for every cell. The conceptual
design was approved on October 21, 2026; this specification incorporates the
user's observed cell counts.

## Motivation and measured pool sizes

The following counts come from the user's pre-downsampling log, not from a
new inspection of the source dataset. Actual counts depend on configured QC,
dataset, output-type, and subtype restrictions.

| Cell type | Eligible cells before cap | Current cap | Retained |
| --- | ---: | ---: | ---: |
| CD4 T | 17,909 | 15,000 | 83.8% |
| CD8 T (GZMK high) | 19,429 | 15,000 | 77.2% |
| CD8 T effector | 21,616 | 15,000 | 69.4% |
| Cancer Cells | 119,970 | 15,000 | 12.5% |
| Double-neg-like T | 25,077 | 15,000 | 59.8% |
| Macrophages | 37,573 | 15,000 | 39.9% |

Removing the cap makes the full eligible population available, including rare
states that a capped random subset may omit. It does not guarantee every cell
appears in a finite generated dataset or that every biological state has equal
sampling probability.

## Configuration and compatibility

Accept a positive integer or the literal string `"all"` for the SCT dataset's
`max_cells_per_cell_type`. Reject booleans, zero, negative integers, other
strings, and fractional values with a clear error before expensive setup.
Keep the existing integer default of 15,000 for configurations that omit it.
Update the example SCT dataset to demonstrate `"all"` with a descriptive
comment.

An integer continues to use the existing capped neighbor workflow. `"all"`
selects the new exact sampled-anchor workflow for `single_anchor` with
`sep_by_patient=False`. Reject `"all"` with patient-constrained or
`two_half_anchor` neighbor sampling in this first implementation; do not
silently allocate an unlimited dense distance matrix. Random SCT sampling
needs no neighbor index and uses its existing eligible pools.

All means all cells remaining after existing QC, dataset, output-type, and
subtype selection. `stratify_by_dataset: "balanced_quota"` intentionally
removes cells earlier; use `"none"` when retaining all eligible dataset rows
is desired. Do not silently override this independent parameter.

Use a new SCT dataset name for an all-cell run. Existing completed outputs
are reused by the current workflow and are not regenerated merely because
this setting changes. Do not mix capped partial outputs with a new all-cell
generation run.

## Exact sampled-anchor architecture

Build eligible source-row arrays and ID mappings once for each requested
classification column and label. Honor task-level subtype overrides and
explicit parent subtype restrictions before selecting anchors or neighbors.
Do not prepare unrelated subtype indices when a task never uses them.

For each generation chunk:

1. Resolve each task's eligible pool and validate its minimum cell count.
2. If the pool is no larger than the requested group, return the complete
   pool as in the current sampler. For a one-cell group, select the anchor
   without running a distance query.
3. Draw an anchor uniformly for each remaining task from its full pool.
4. Deduplicate query anchors by eligible-pool identity and source row.
5. Retrieve cached neighbor rows where available and batch uncached queries.
6. Search each query against every cell in its eligible pool, retain the
   requested nearest indices, and restore task order and sorted selected IDs.
7. Average selected linear-expression profiles using the existing SCT
   aggregation and output transformations.

Run numerical queries in the manager process with compiled NumPy/scikit-learn
operations, avoiding a new sampling process pool and metadata copies per
chunk. Do not pass the full expression matrix through multiprocessing queues.

Use the same source expression space, float32 Euclidean distance calculation,
zero self-distance, and inclusion of the anchor as the existing neighbor
workflow. Do not introduce PCA, approximate neighbors, or reduced precision.
Preserve the legacy `argpartition` then distance-sort selection for a complete
distance row; nearest-neighbor ties must be tested explicitly. Numerical
equivalence means no intentional metric or candidate change beyond removing
the cap, not a universal bitwise guarantee across BLAS implementations.

## Bounded memory and cache

Use an internal working-memory budget of 256 MiB for explicit distance
temporaries and an internal 64 MiB LRU neighbor-index cache. These are code
defaults, not new user-facing YAML parameters in the first implementation.
Use query blocks of at most 128 anchors and choose a smaller block from the
candidate count and conservative temporary-array estimates.

Precompute reference row norms once per active pool when the distance API
supports compatible reuse. Retain sparse class expression when supported.
Do not densify the complete source matrix or create copies of every parent
and subtype expression pool at the same time. Pool-expression residency must
also be bounded, processing one active pool at a time.

If even one full distance row exceeds the working-memory budget, fail with a
resource-specific error in this first implementation rather than exceeding
the budget silently. A later reference-block top-k merge can extend that
case, but must define and verify boundary-tie behavior first.

Cached entries contain integer neighbor indices and are keyed by the exact
eligible pool, source anchor, and retained neighbor count. Cache eviction
must not change results. Do not persist cache files or reuse them across
changed source references in the first implementation.

These bounds apply to neighbor-search temporaries and cached indices.
Existing dense source loading, QC masks, SCT aggregation, and output storage
still contribute to total process memory. This change must not claim to
solve those independent allocations.

## Cost estimates

The current compact builder stores O(N * n_base) indices but still performs
O(N * N * G) distance work. Querying U unique uncached anchors performs
O(U * N * G) work, where G is the source feature count.

A 119,970-by-119,970 float32 distance matrix alone occupies approximately
53.6 GiB. For 10,000 independent uniform anchor draws from that pool, the
expected unique-anchor count is about 9,595:
`N * (1 - (1 - 1/N)^10000)`.

This yields approximately 12.5 times fewer query-reference comparisons than
precomputing all 119,970 anchors. It still performs approximately 5.1 times
as many comparisons as precomputing all anchors of a 15,000-cell pool.
These are operation-count estimates, not measured runtime speedups.
Batching, cached norms, reuse, and removal of task overhead require benchmarks.

The method is most advantageous when generated samples per class are much
fewer than eligible cells. Large sample counts approaching or exceeding the
pool size reduce that advantage; cache eviction can require repeated queries.
Exact full-gene search is not guaranteed to match capped-run runtime.

## Tests and benchmark protocol

Compare neighbor sets and generated SCT profiles against the existing exact
reference using identical eligible pools and fixed anchors. Cover unique
distances, duplicate-distance ties, repeated anchors, cache eviction, small
pools, one-cell groups, subtype restrictions, auto subtype selection, missing
labels, and anchor/self inclusion.

Verify all cells remain eligible in a pool larger than 15,000, including cells
outside the old capped subset. Check anchor frequencies separately from
neighbor participation; high-density cells can legitimately occur as
neighbors more often.

Verify no N-by-N distance allocation, no full-source densification for
neighbor setup, no per-task full-metadata scan, and no neighbor-sampling
process-pool creation on the all-cell path. Test early errors for unsupported
schemes and invalid cap settings, including completed-output paths.

Use the six logged class sizes as synthetic scaling targets, with
`n_base=30` and `n_base=100`. Test 1,000 and 10,000 generated groups per class.
Start with reduced feature counts for portable scaling tests, and label these
clearly; use actual source features for representative production benchmarks
when resources permit. Log actual pool sizes when using source data rather
than assuming they reproduce the supplied log.

Compare three workloads: capped 15,000-cell precomputation, all-cell
precomputation on feasible pools, and all-cell sampled-anchor queries.
Do not run infeasible all-cell precomputation solely to obtain a comparison;
report skipped cases and operation-count estimates separately.

Report eligible-pool preparation, uncached queries, cache hits and evictions,
aggregation, export, total runtime, peak memory, query count, and selected
cell count. Measure cold and repeated-query workloads separately and record
hardware, software, feature count, and thread settings. For long benchmarks,
use bounded measured runs and report extrapolations as estimates.

Run the existing bounded-neighbor, two-half, subtype, workflow, and numeric
stability regression suites. Existing capped behavior must remain unchanged.

## Delivery

Deliver focused source changes, fail-fast validation, tests, updated example
configuration and SCT workflow documentation, and a reproducible benchmark
script. The existing SCT averaging bottleneck can be measured, but unrelated
aggregation refactoring is outside this specification.

No code is implemented by this document. Do not commit or push without an
explicit user request.
