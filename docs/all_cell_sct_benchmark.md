# All-cell SCT scaling benchmark

The all-cell sampler keeps complete eligible pools and computes exact
float32 Euclidean neighbors only for sampled anchors. These measurements
exercise the six user-reported pool sizes, not the original gene matrix.

## Setup

Measurements use an Apple M4 Max, Python 3.12.13, NumPy 2.2.6, and pandas
2.3.3. OpenMP and OpenBLAS thread settings are 1. Synthetic expression has
16 features. Each pool runs in a separate process; uncached queries use three
repetitions, while compact-precomputation baselines use one. Several
benchmark jobs ran concurrently, so timing differences include contention.

The query medians below measure all unique anchors drawn for each workload,
without extrapolation. Cache clearing precedes every measured query
repetition; reference expression and norms remain active after the first
repetition. This is uncached-query timing, not completely cold process timing.
Setup and cache-hit replay are recorded separately by the script.

## Measured query times

The table reports seconds for exact all-cell queries. Every candidate remains
eligible; repeated anchors are queried once.

| Cell type | Pool cells | 1,000 samples, n_base=30 | 1,000 samples, n_base=100 | 10,000 samples, n_base=30 | 10,000 samples, n_base=100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CD4 T | 17,909 | 0.173 | 0.162 | 1.245 | 1.336 |
| CD8 T (GZMK high) | 19,429 | 0.182 | 0.172 | 1.375 | 1.465 |
| CD8 T effector | 21,616 | 0.207 | 0.210 | 1.574 | 1.604 |
| Cancer Cells | 119,970 | 0.904 | 0.898 | 8.877 | 8.535 |
| Double-neg-like T | 25,077 | 0.218 | 0.218 | 1.873 | 1.775 |
| Macrophages | 37,573 | 0.284 | 0.301 | 2.666 | 2.439 |

The cancer run draws 9,587 unique anchors from 10,000 samples, close to the
expected count of about 9,595. Replaying those cached queries takes about
0.0024 seconds for n_base=30. Actual production cache reuse depends on repeated
anchors, class changes, and cache eviction.

For 10,000 samples and n_base=30, comparison preparation times are:

| Cell type | Capped 15,000-cell precomputation | All-cell precomputation | All-cell sampled queries |
| --- | ---: | ---: | ---: |
| CD4 T | 1.958 | 2.862 | 1.245 |
| CD8 T (GZMK high) | 1.981 | 3.364 | 1.375 |
| CD8 T effector | 2.020 | 4.299 | 1.574 |
| Cancer Cells | 2.027 | Not run | 8.877 |
| Double-neg-like T | 2.222 | Not run | 1.873 |
| Macrophages | 2.139 | Not run | 2.666 |

All-cell precomputation is skipped above 25,000 cells by the benchmark's
default safety setting. These baseline values measure index preparation
only, not the legacy per-sample multiprocessing and metadata overhead.

All-cell query work is faster than full-pool precomputation in the feasible
comparisons. It is not uniformly faster than capped precomputation: the
cancer pool is almost eight times larger than the old cap. Retaining all
cells has a real computation cost.

## Memory and scope

Peak process RSS ranges from roughly 0.38 to 0.51 GiB in these reduced-feature
runs. It is a lifetime high-water mark including baselines, not isolated
query memory. Production full-gene expression loading can require much more
RAM.

The implementation uses a conservative 256 MiB explicit distance-working
budget, query blocks of at most 128 anchors, and a 64 MiB neighbor-index cache.
Only one class expression pool and its norms are active at a time. These
bounds do not cover existing source loading, QC masks, or output matrices.

The script times averaging and temporary CSV export for the measured unique
anchors. Those operations use synthetic 16-feature data and are not full
production SCT output timings. Source loading, fraction generation, final
H5AD export, imputation, and diagnostics are outside this benchmark.
Production chunk logs provide sampling, aggregation, and export durations.

No full-gene source benchmark was run here. The original source's feature
count, sparsity, and target-server hardware must be measured before making a
production runtime estimate.

## Verification and use

The focused all-cell, bounded-neighbor, two-half, and numeric stability suite
passes 59 tests. The full suite passes 235 tests and has the same seven known
failures previously reproduced on the untouched CPU-optimization baseline.
Tests cover exact neighbor sets including ties, dense and sparse inputs,
all-row eligibility beyond 15,000, subtype pools, cache eviction, chunk
integration, invalid modes, and incompatible partial outputs.

Use a new SCT dataset name and:

```yaml
sampling_method_gep: "n_neighbors"
neighbor_sampling_scheme: "single_anchor"
stratify_by_dataset: "none"
max_cells_per_cell_type: "all"
```

Positive integer caps and the omitted default of 15,000 retain existing
behavior. `"all"` rejects patient-constrained or two-half neighbor sampling.
QC, source dataset selection, and subtype restrictions remain active.

Reproduce the full synthetic query workloads from the SimuTME root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_all_cell_neighbors.py --features 16 --samples 10000 --n-base 30 --query-limit 0
```

Repeat with `--samples 1000` and `--n-base 100` for the other table entries.
Omitting `--query-limit 0` measures at most 256 unique anchors and marks
full-workload query estimates as extrapolations.
