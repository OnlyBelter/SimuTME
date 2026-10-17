# Bounded-memory exact SCT neighbor sampling

Replace full pairwise distance storage in SCT `single_anchor` sampling with
bounded-memory exact neighbor selection. Use a 10,000-cell candidate pool for
the affected SCT run while retaining the current exact Euclidean neighbor
calculation within that pool.

## Goal

Reduce peak memory during `sampling_method_gep: n_neighbors`, especially while
building per-class distances, without changing how `single_anchor` selects
neighbors from its seeded candidate pool.

The SCT run may set `max_cells_per_cell_type: 10000`. This intentionally
reduces the candidate pool from the current default of 15,000 cells per class.
Neighbor selection remains exact within the resulting 10,000-cell pool.

## Current memory use

`SingleCellTypeGEPGenerator._get_cell_distance_dict` densifies the full AnnData
matrix for the `output_types` distance pass and densifies it again for the
subtype pass. `get_distance_matrices_by_class` then extracts up to 15,000 rows
per class and computes a full pairwise distance matrix.

A 15,000-by-15,000 float32 matrix uses about 0.84 GiB. The implementation can
compute multiple classes in parallel, submits dense class matrices to worker
processes, and later copies the retained distance matrices into float64 shared
memory. The distance matrices are already computed and stored as float32;
converting them to float64 only widens the values and does not improve neighbor
ranking. The compact representation therefore shares integer neighbor indices
instead of distance matrices. These allocations compound.

`ReadH5AD.get_df` also materializes a dense expression DataFrame for existing
SCT filtering and aggregation paths; the neighbor setup must not create
further full-matrix copies.

## Design

Keep the existing seeded per-class candidate selection and exact float32
Euclidean distance semantics for `single_anchor` with
`sep_by_patient: false`, which is the workflow's current SCT setting. For a
fixed candidate pool, the memory optimization must preserve the current
neighbor-selection results. The affected run's 10,000-cell pool is a separate,
intentional configuration change and is not expected to produce the same
samples as a 15,000-cell pool.

Process one class at a time. Within a class, compute float32 distances in
query-row blocks of at most 128 rows against the complete candidate pool. For
each query row, use the existing `argpartition`-then-sort selection to retain
only the nearest `min(n_base, n_candidates)` cell indices in the same order
used by current sampling. Release each distance block before processing the
next block.

Store compact per-anchor neighbor indices instead of an N-by-N distance
matrix. Share those integer arrays with workers without converting them to
float64. Do not submit dense class matrices for every class to a process pool.
Sequential class processing bounds peak neighbor-preparation memory at the
cost of increased computation time.

Pass the existing AnnData matrix to neighbor preparation and slice class rows
as needed. Do not call `.toarray()` on the complete matrix separately for the
parent and subtype passes. Keep existing expression loading and aggregation
behavior outside the neighbor-preparation path unchanged in this change.

Set `max_cells_per_cell_type: 10000` explicitly on the SCT dataset definition
used by the affected run. Keep the package fallback value at 15,000 so
unrelated configurations retain their current candidate-pool behavior.

For `two_half_anchor` or `sep_by_patient: true`, retain the current distance
matrix path in this change. Those modes can restrict a candidate pool after
choosing an anchor, so a global per-anchor top-n list is not sufficient to
preserve their exact behavior.

## Precision

Use float32 as the production calculation and reference for exact neighbor
identity. Evaluate float16 only in an isolated benchmark that compares
per-anchor top-n identity, distance error, runtime, and peak memory against
float32. Do not enable reduced precision in the sampling workflow unless it
preserves every tested neighbor set and provides a meaningful resource benefit.


## Compatibility

Keep the default sampling scheme, sample-ID generation procedure, seeded
candidate-selection procedure, class-label resolution, and generated
expression format unchanged. For the active `single_anchor` path, given the
same candidate pool and random state, the compact neighbor representation
must return the same nearest-neighbor selections as the full-matrix reference,
including existing tie handling. The 10,000-cell configuration changes the
candidate pool selected by the existing seeded subsampling procedure, so
sampled cells and generated profiles may differ from a run using 15,000 cells.

The candidate-pool change is limited to the affected SCT dataset definition.
Keep the package fallback at 15,000 so unrelated configurations retain their
current candidate-pool behavior.

## Verification

Add tests that:

- Compare compact neighbor selections with the current full-matrix reference
  for multiple `n_base` values and fixed candidate pools.
- Cover classes smaller than `n_base`, duplicate-distance ties, and seeded
  candidate subsampling.
- Confirm `single_anchor` generation uses compact neighbor indices and does
  not construct or share an N-by-N matrix.
- Verify stored neighbor data scales as O(N × n_base), while each temporary
  distance block stays within the configured working-memory budget.
- Benchmark float16 and, when supported, float8 separately against float32.
- Run the existing SCT neighbor-sampling and workflow tests.

The implementation is successful when exact float32 `single_anchor` sampling
passes regression tests and peak neighbor-preparation memory depends on the
class input, a maximum 128-by-N distance block, and O(N × n_base) retained
neighbors, rather than the sum of full per-class N-by-N matrices.
