"""Bounded exact neighbor queries for uncapped SCT candidate pools."""

from collections import OrderedDict
import logging
from time import perf_counter

import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import euclidean_distances
from sklearn.utils.extmath import row_norms

logger = logging.getLogger(__name__)


def validate_cell_cap(value, sampling_method="n_neighbors",
                      neighbor_sampling_scheme="single_anchor", sep_by_patient=False):
    if value == "all":
        if sampling_method == "n_neighbors" and (
            sep_by_patient or neighbor_sampling_scheme != "single_anchor"
        ):
            raise ValueError(
                'max_cells_per_cell_type: "all" requires single_anchor '
                'neighbor sampling with sep_by_patient=False.'
            )
        return value
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError('max_cells_per_cell_type must be a positive integer or "all".')
    return int(value)


class AllCellNeighborSampler:
    """Keep all eligible IDs; query only sampled anchors in complete distance rows."""

    def __init__(self, expression, obs, parent_column, subtype_column,
                 selected_subtypes=None, working_memory_bytes=256 * 1024 ** 2,
                 cache_bytes=64 * 1024 ** 2):
        if expression.shape[0] != len(obs) or not obs.index.is_unique:
            raise ValueError("Neighbor expression and unique observation rows must align.")
        if working_memory_bytes < 1 or cache_bytes < 0:
            raise ValueError("Working memory must be positive and cache memory nonnegative.")
        self.expression = expression
        self.obs = obs
        self.parent_column = parent_column
        self.subtype_column = subtype_column
        self.selected_subtypes = selected_subtypes or {}
        self.working_memory_bytes = working_memory_bytes
        self.cache_bytes = cache_bytes
        self.cache = OrderedDict()
        self.column_groups = {}
        self.cache_size = 0
        self.pools = {}
        self.active_key = None
        self.active_expression = self.active_norms = None
        self.stats = dict(queries=0, cache_hits=0, evictions=0, query_seconds=0.0,
                          pool_seconds=0.0, activation_seconds=0.0)

    def pool(self, column, label):
        key = (column, label)
        if key not in self.pools:
            started = perf_counter()
            if column not in self.obs:
                raise ValueError(f"Neighbor classification column {column!r} is absent.")
            if column not in self.column_groups:
                self.column_groups[column] = self.obs.groupby(column, observed=True).indices
            rows = self.column_groups[column].get(label, np.array([], dtype=np.int64))
            allowed = self.selected_subtypes.get(label)
            if column == self.parent_column and allowed is not None and allowed != "auto":
                if self.subtype_column not in self.obs:
                    raise ValueError("Subtype restriction requires the subtype column.")
                rows = rows[self.obs[self.subtype_column].iloc[rows].isin(allowed).to_numpy()]
            self.pools[key] = (rows, self.obs.index.to_numpy()[rows])
            self.stats["pool_seconds"] += perf_counter() - started
            logger.info("All-cell neighbor pool %s: %d eligible cells (no cap).", label, len(rows))
        return self.pools[key]

    def _activate(self, key):
        if self.active_key != key:
            started = perf_counter()
            self.active_expression = self.active_norms = None
            rows, _ = self.pool(*key)
            values = self.expression[rows, :]
            if issparse(values):
                values = values.astype(np.float32, copy=False)
                finite = np.isfinite(values.data).all()
            else:
                values = np.asarray(values, dtype=np.float32)
                finite = np.isfinite(values).all()
            if not finite:
                raise ValueError("Neighbor expression contains non-finite values.")
            self.active_expression = values
            # Float64 norms avoid sklearn discarding float32 precomputed norms.
            self.active_norms = np.empty(len(rows), dtype=np.float64)
            norm_block = max(1, self.working_memory_bytes // max(1, values.shape[1] * 16))
            for start in range(0, len(rows), norm_block):
                stop = min(start + norm_block, len(rows))
                self.active_norms[start:stop] = row_norms(
                    values[start:stop].astype(np.float64), squared=True)
            self.active_key = key
            self.stats["activation_seconds"] += perf_counter() - started

    def neighbors(self, key, anchors, n_base):
        rows, _ = self.pool(*key)
        count = len(rows)
        if not count:
            raise ValueError(f"No eligible cells for neighbor pool {key!r}.")
        anchors = np.asarray(anchors, dtype=np.int64)
        if (anchors < 0).any() or (anchors >= count).any():
            raise ValueError("Anchor index is outside its eligible pool.")
        keep = min(n_base, count)
        if keep < 1:
            raise ValueError("Neighbor count must be positive.")
        block_size = min(128, self.working_memory_bytes // (count * 32))
        if block_size < 1:
            raise ValueError("One full neighbor distance row exceeds the working-memory budget.")
        result = {}
        missing = []
        for anchor in dict.fromkeys(anchors.tolist()):
            cache_key = (*key, anchor, keep)
            if cache_key in self.cache:
                self.cache.move_to_end(cache_key)
                result[anchor] = self.cache[cache_key]
                self.stats["cache_hits"] += 1
            else:
                missing.append(anchor)
        if missing:
            self._activate(key)
        started = perf_counter()
        for start in range(0, len(missing), block_size):
            block = np.asarray(missing[start:start + block_size])
            distances = euclidean_distances(
                self.active_expression[block], self.active_expression,
                Y_norm_squared=self.active_norms,
            ).astype(np.float32, copy=False)
            distances[np.arange(len(block)), block] = 0.0
            if not np.isfinite(distances).all():
                raise ValueError("Neighbor distances contain non-finite values.")
            for anchor, distance in zip(block, distances):
                nearest = np.argpartition(distance, keep - 1)[:keep]
                nearest = nearest[np.argsort(distance[nearest])].astype(np.int32)
                result[int(anchor)] = nearest
                # Account for key/container overhead as well as the numeric row.
                size = nearest.nbytes + 256
                if size <= self.cache_bytes:
                    while self.cache and self.cache_size + size > self.cache_bytes:
                        _, old = self.cache.popitem(last=False)
                        self.cache_size -= old.nbytes + 256
                        self.stats["evictions"] += 1
                    self.cache[(*key, int(anchor), keep)] = nearest
                    self.cache_size += size
            self.stats["queries"] += len(block)
        self.stats["query_seconds"] += perf_counter() - started
        return [result[int(anchor)] for anchor in anchors]

    def sample(self, tasks, minimum_n_base=1):
        results = [None] * len(tasks)
        queries = {}
        for i, task in enumerate(tasks.itertuples(index=False)):
            if task.n_cell == 0:
                results[i] = ()
                continue
            key = (task.class_by, task.cell_type)
            _, ids = self.pool(*key)
            if len(ids) < minimum_n_base:
                raise ValueError(f"Insufficient eligible cells for {task.cell_type!r}.")
            if len(ids) <= task.n_cell:
                results[i] = tuple(sorted(ids))
            else:
                anchor = int(np.random.randint(len(ids)))
                if task.n_cell == 1:
                    results[i] = (ids[anchor],)
                else:
                    queries.setdefault((key, task.n_cell), []).append((i, anchor))
        for (key, keep), requests in queries.items():
            _, ids = self.pool(*key)
            nearest = self.neighbors(key, [anchor for _, anchor in requests], keep)
            for (i, _), indices in zip(requests, nearest):
                results[i] = tuple(sorted(ids[indices]))
        return results
