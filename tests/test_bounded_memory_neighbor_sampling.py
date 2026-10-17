from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simutme.simulation import nneighbors_shared_memory as neighbor_module
from simutme.simulation.nneighbors_shared_memory import (
    CompactNeighborData,
    SharedDataManager,
    get_sample_id_with_neighbors,
    worker_cleanup,
    worker_init,
)
from simutme.simulation.simulator_sct import SingleCellTypeGEPGenerator
from simutme.utility import get_distance_matrices_by_class, get_neighbor_indices_by_class
from simutme.utility import pub_func


def _reference_top_indices(distance_matrix: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_keep = min(n_neighbors, distance_matrix.shape[0])
    result = np.empty((distance_matrix.shape[0], n_keep), dtype=np.int32)
    for anchor_idx, distances in enumerate(distance_matrix):
        indices = np.argpartition(distances, n_keep - 1)[:n_keep]
        result[anchor_idx] = indices[np.argsort(distances[indices])]
    return result


def _make_grouped_expression(n_cells: int = 24) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(31)
    expression = rng.normal(size=(n_cells, 7)).astype(np.float32)
    cell_types = np.where(np.arange(n_cells) % 2, "B", "A")
    obs = pd.DataFrame(
        {"cell_type": cell_types},
        index=[f"cell_{idx}" for idx in range(n_cells)],
    )
    return expression, obs


def test_compact_neighbors_match_full_matrix_for_multiple_n_base_values():
    expression, obs = _make_grouped_expression()
    for n_neighbors in (1, 3, 8, 20):
        reference = get_distance_matrices_by_class(
            expression,
            obs,
            class_by="cell_type",
            n_jobs=1,
            max_cells_per_class=None,
            show_progress=False,
        )
        compact = get_neighbor_indices_by_class(
            expression,
            obs,
            class_by="cell_type",
            n_neighbors=n_neighbors,
            max_cells_per_class=None,
            show_progress=False,
        )
        for cell_type, (distance_matrix, cell_ids) in reference.items():
            observed_indices, observed_ids = compact[cell_type]
            assert observed_ids == cell_ids
            np.testing.assert_array_equal(
                observed_indices,
                _reference_top_indices(distance_matrix, n_neighbors),
            )


def test_compact_neighbors_match_full_matrix_for_duplicate_distance_ties():
    expression = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {"cell_type": ["A"] * len(expression)},
        index=[f"cell_{idx}" for idx in range(len(expression))],
    )
    reference = get_distance_matrices_by_class(
        expression,
        obs,
        max_cells_per_class=None,
        show_progress=False,
    )["A"][0]

    for n_neighbors in (2, 4):
        compact = get_neighbor_indices_by_class(
            expression,
            obs,
            n_neighbors=n_neighbors,
            max_cells_per_class=None,
            show_progress=False,
        )["A"][0]
        np.testing.assert_array_equal(
            compact,
            _reference_top_indices(reference, n_neighbors),
        )


def test_class_smaller_than_n_base_returns_all_candidates():
    expression, obs = _make_grouped_expression(n_cells=12)
    obs = obs.loc[obs["cell_type"] == "A"].copy()
    expression_rows = [int(cell_id.split("_")[-1]) for cell_id in obs.index]
    expression = expression[expression_rows]
    distance_matrix, cell_ids = get_distance_matrices_by_class(
        expression,
        obs,
        max_cells_per_class=None,
        show_progress=False,
    )["A"]
    indices, compact_ids = get_neighbor_indices_by_class(
        expression,
        obs,
        n_neighbors=20,
        max_cells_per_class=None,
        show_progress=False,
    )["A"]

    np.testing.assert_array_equal(indices, _reference_top_indices(distance_matrix, 20))
    assert get_sample_id_with_neighbors(
        obs,
        n_samples=1,
        cell_type="A",
        n_base=20,
        distance_dict={"A": (distance_matrix, cell_ids)},
    ) == tuple(sorted(cell_ids))
    assert get_sample_id_with_neighbors(
        obs,
        n_samples=1,
        cell_type="A",
        n_base=20,
        distance_dict={"A": CompactNeighborData(indices, compact_ids)},
    ) == tuple(sorted(cell_ids))


def test_seeded_candidate_subsampling_matches_full_matrix_reference():
    expression, obs = _make_grouped_expression(n_cells=50)
    reference = get_distance_matrices_by_class(
        expression,
        obs,
        n_jobs=1,
        max_cells_per_class=9,
        random_seed=19,
        show_progress=False,
    )
    compact = get_neighbor_indices_by_class(
        expression,
        obs,
        n_neighbors=4,
        max_cells_per_class=9,
        random_seed=19,
        show_progress=False,
    )
    candidate_rng = np.random.RandomState(19)
    expected_ids = {}
    for cell_type, group in obs.groupby("cell_type", observed=True):
        selected = candidate_rng.choice(len(group), size=9, replace=False)
        selected.sort()
        candidate_ids = group.index.to_list()
        expected_ids[cell_type] = [candidate_ids[idx] for idx in selected]

    for cell_type, (distance_matrix, cell_ids) in reference.items():
        observed_indices, observed_ids = compact[cell_type]
        assert observed_ids == cell_ids
        assert observed_ids == expected_ids[cell_type]
        np.testing.assert_array_equal(
            observed_indices,
            _reference_top_indices(distance_matrix, 4),
        )


def test_distance_working_set_is_limited_to_query_blocks(monkeypatch):
    rng = np.random.default_rng(8)
    expression = rng.normal(size=(260, 6)).astype(np.float32)
    obs = pd.DataFrame(
        {"cell_type": ["A"] * len(expression)},
        index=[f"cell_{idx}" for idx in range(len(expression))],
    )
    original = pub_func.euclidean_distances
    observed_shapes = []

    def record_shapes(query, candidates):
        observed_shapes.append((query.shape, candidates.shape))
        return original(query, candidates)

    monkeypatch.setattr(pub_func, "euclidean_distances", record_shapes)
    compact = get_neighbor_indices_by_class(
        expression,
        obs,
        n_neighbors=12,
        max_cells_per_class=None,
        show_progress=False,
    )["A"][0]

    assert [shape[0][0] for shape in observed_shapes] == [128, 128, 4]
    assert all(query[0] <= 128 for query, _ in observed_shapes)
    assert all(candidates[0] == len(expression) for _, candidates in observed_shapes)
    assert compact.shape == (len(expression), 12)
    assert compact.dtype == np.int32


def test_query_block_size_cannot_exceed_memory_bound():
    expression, obs = _make_grouped_expression()
    with pytest.raises(ValueError, match="must be between 1 and 128"):
        get_neighbor_indices_by_class(
            expression,
            obs,
            n_neighbors=3,
            query_block_size=129,
            show_progress=False,
        )


def test_sct_compact_preparation_slices_sparse_matrix_by_class(monkeypatch):
    rng = np.random.default_rng(11)
    expression = rng.normal(size=(40, 5)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "output_types": ["A"] * 20 + ["B"] * 20,
            "cell_subtype": ["A1"] * 10 + ["A2"] * 10 + ["B1"] * 10 + ["B2"] * 10,
        },
        index=[f"cell_{idx}" for idx in range(len(expression))],
    )
    sparse_expression = csr_matrix(expression)
    dense_shapes = []
    original_toarray = csr_matrix.toarray

    def record_toarray(matrix, *args, **kwargs):
        dense_shapes.append(matrix.shape)
        return original_toarray(matrix, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", record_toarray)
    generator = object.__new__(SingleCellTypeGEPGenerator)
    generator.merged_sc_dataset = SimpleNamespace(X=sparse_expression)
    generator.merged_sc_dataset_obs = obs
    generator.subtype_col_name = "cell_subtype"
    generator.cell_distance_dict = {}

    generator._get_cell_distance_dict(
        max_cells_per_cell_type=15,
        n_neighbors=5,
        use_compact_neighbors=True,
    )

    assert generator.cell_distance_dict
    assert all(
        isinstance(data, CompactNeighborData)
        for data in generator.cell_distance_dict.values()
    )
    assert all(matrix_shape != sparse_expression.shape for matrix_shape in dense_shapes)
    assert all(
        data.neighbor_indices.shape[1] == 5
        for data in generator.cell_distance_dict.values()
    )


def test_restricted_sct_modes_keep_the_distance_matrix_path(monkeypatch):
    dense_expression = np.arange(24, dtype=np.float32).reshape(8, 3)
    obs = pd.DataFrame(
        {
            "output_types": ["A"] * 4 + ["B"] * 4,
            "cell_subtype": ["A1", "A1", "A2", "A2", "B1", "B1", "B2", "B2"],
        },
        index=[f"cell_{idx}" for idx in range(len(dense_expression))],
    )
    generator = object.__new__(SingleCellTypeGEPGenerator)
    generator.merged_sc_dataset = SimpleNamespace(X=dense_expression)
    generator.merged_sc_dataset_obs = obs
    generator.subtype_col_name = "cell_subtype"
    generator.cell_distance_dict = {}
    distance_calls = []

    def record_distance_call(**kwargs):
        distance_calls.append(kwargs)
        return {kwargs["class_by"]: (np.zeros((1, 1), dtype=np.float32), ["cell_0"])}

    def reject_compact_call(**kwargs):
        raise AssertionError("Restricted sampling must retain the distance path.")

    monkeypatch.setattr(
        "simutme.simulation.simulator_sct.get_distance_matrices_by_class",
        record_distance_call,
    )
    monkeypatch.setattr(
        "simutme.simulation.simulator_sct.get_neighbor_indices_by_class",
        reject_compact_call,
    )

    generator._get_cell_distance_dict(
        max_cells_per_cell_type=10,
        n_neighbors=3,
        use_compact_neighbors=False,
    )

    assert len(distance_calls) == 2
    assert all(call["adata_X"] is dense_expression for call in distance_calls)


def test_compact_shared_memory_preserves_integer_neighbor_arrays():
    obs = pd.DataFrame(
        {"cell_type": ["A"] * 5},
        index=[f"cell_{idx}" for idx in range(5)],
    )
    compact = CompactNeighborData(
        np.tile(np.arange(3, dtype=np.int32), (5, 1)),
        obs.index.tolist(),
    )
    with SharedDataManager(obs, {"A": compact}) as manager:
        metadata = manager.get_metadata()
        assert metadata["distance_shapes"]["A"] == (5, 3)
        assert metadata["distance_dtypes"]["A"] == "int32"
        assert metadata["distance_kinds"]["A"] == "neighbor_indices"
        worker_init(metadata)
        try:
            worker_data = neighbor_module._worker_data.distance_dict["A"]
            assert isinstance(worker_data, CompactNeighborData)
            np.testing.assert_array_equal(
                worker_data.neighbor_indices,
                compact.neighbor_indices,
            )
        finally:
            worker_cleanup()


def test_single_anchor_sampling_matches_full_distance_reference():
    expression, obs = _make_grouped_expression(n_cells=20)
    obs = obs.loc[obs["cell_type"] == "A"].copy()
    expression_rows = [int(cell_id.split("_")[-1]) for cell_id in obs.index]
    expression = expression[expression_rows]
    distance_matrix, cell_ids = get_distance_matrices_by_class(
        expression,
        obs,
        max_cells_per_class=None,
        show_progress=False,
    )["A"]
    indices, compact_ids = get_neighbor_indices_by_class(
        expression,
        obs,
        n_neighbors=4,
        max_cells_per_class=None,
        show_progress=False,
    )["A"]

    for seed in range(8):
        np.random.seed(seed)
        expected = get_sample_id_with_neighbors(
            obs,
            n_samples=1,
            cell_type="A",
            n_base=4,
            distance_dict={"A": (distance_matrix, cell_ids)},
        )
        np.random.seed(seed)
        observed = get_sample_id_with_neighbors(
            obs,
            n_samples=1,
            cell_type="A",
            n_base=4,
            distance_dict={"A": CompactNeighborData(indices, compact_ids)},
        )
        assert observed == expected
