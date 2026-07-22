# Necessary Fixes For `nneighbors_shared_memory.py`

## Goal

Apply only necessary improvements to
`SimuTME/src/simutme/simulation/nneighbors_shared_memory.py` while preserving
the current sampling behavior.

## Scope

This change is limited to three issues identified during review:

1. avoid forcing the multiprocessing start method at module import time
2. avoid an unnecessary copy when reconstructing `obs_df` in worker processes
3. improve the nearest-neighbor hot path without changing output semantics

## Approved Changes

### 1. Remove import-time multiprocessing side effect

Current code calls:

```python
multiprocessing.set_start_method('spawn', force=True)
```

at module import time.

This should be removed because importing the module should not globally change
the multiprocessing configuration of the host process.

The file should rely on the caller's multiprocessing context instead.

### 2. Reduce worker initialization copying

Current worker initialization reconstructs `obs_df` by copying bytes out of
shared memory before unpickling.

This should be changed to load directly from the shared memory buffer so worker
startup uses less transient memory.

### 3. Speed up neighbor selection

The current neighbor-selection path:

- repeatedly maps cell IDs to matrix indices using `list.index(...)`
- fully sorts all distances using `np.argsort(...)`

This should be improved by:

- building a `cell_id -> matrix_index` mapping once per sampling call
- using `np.argpartition(...)` to select the top `n_base` nearest cells
- sorting only the selected subset by distance before converting to cell IDs

These changes should preserve the same effective nearest-neighbor semantics.

## Out Of Scope

The following are not part of this change:

- redesigning the shared-memory API
- changing the sampling contract or return types
- changing cancer-cell or patient-specific sampling rules
- changing how distance matrices are computed upstream

## Verification

After implementation, verify:

1. the file passes diagnostics
2. the Python module compiles
3. a small synthetic sampling example still returns valid nearest-neighbor
   groups
4. no new behavior is introduced beyond the reviewed fixes
