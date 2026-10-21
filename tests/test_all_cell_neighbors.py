from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics import euclidean_distances

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simutme.simulation.all_cell_neighbors import AllCellNeighborSampler, validate_cell_cap


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, None, "ALL", "100"])
def test_invalid_caps(value):
    with pytest.raises(ValueError, match="positive integer"):
        validate_cell_cap(value)


def test_supported_caps_and_modes():
    assert validate_cell_cap("all") == "all"
    assert validate_cell_cap(15000) == 15000
    assert validate_cell_cap("all", "random", "two_half_anchor") == "all"
    for scheme, patient in [("two_half_anchor", False), ("single_anchor", True)]:
        with pytest.raises(ValueError, match="single_anchor"):
            validate_cell_cap("all", "n_neighbors", scheme, patient)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("ties", [False, True])
def test_exact_queries_and_cache(sparse, ties):
    values = np.random.default_rng(42).normal(size=(50, 7)).astype(np.float32)
    if ties:
        values[:] = 0
        values[10:] = 1
    obs = pd.DataFrame({"ct": ["A"] * 50}, index=[f"r{i}" for i in range(50)])
    sampler = AllCellNeighborSampler(csr_matrix(values) if sparse else values,
                                    obs, "ct", "sub", cache_bytes=600)
    distance = euclidean_distances(values, values).astype(np.float32)
    np.fill_diagonal(distance, 0)
    anchors = [2, 45, 2, 9]
    expected = []
    for anchor in anchors:
        nearest = np.argpartition(distance[anchor], 4)[:5]
        expected.append(nearest[np.argsort(distance[anchor, nearest])])
    actual = sampler.neighbors(("ct", "A"), anchors, 5)
    np.testing.assert_array_equal(actual, expected)
    assert sampler.stats["queries"] == 3
    assert sampler.stats["evictions"] > 0
    again = sampler.neighbors(("ct", "A"), anchors, 5)
    np.testing.assert_array_equal(again, expected)
    assert sampler.cache_size <= 600


def test_all_rows_eligible_and_sparse_retained(monkeypatch):
    n = 16001
    values = csr_matrix(np.arange(n, dtype=np.float32)[:, None])
    obs = pd.DataFrame({"ct": ["A"] * n}, index=[f"r{i}" for i in range(n)])
    sampler = AllCellNeighborSampler(values, obs, "ct", "sub")
    tasks = pd.DataFrame({"cell_type": ["A"], "class_by": ["ct"], "n_cell": [2]})
    monkeypatch.setattr(np.random, "randint", lambda count: count - 1)
    chosen = sampler.sample(tasks)[0]
    assert "r16000" in chosen
    assert len(sampler.pool("ct", "A")[0]) == n
    assert isinstance(sampler.active_expression, csr_matrix)
    assert sampler.stats["queries"] == 1


def test_subtype_pool_and_small_groups():
    obs = pd.DataFrame({"ct": ["A"] * 4, "sub": ["a", "a", "b", "b"]},
                       index=["r0", "r1", "r2", "r3"])
    sampler = AllCellNeighborSampler(np.ones((4, 3), np.float32), obs,
                                    "ct", "sub", {"A": ["a"]})
    tasks = pd.DataFrame({"cell_type": ["A", "b", "missing"],
                          "class_by": ["ct", "sub", "absent"], "n_cell": [5, 1, 0]})
    result = sampler.sample(tasks)
    assert result[0] == ("r0", "r1")
    assert result[1][0] in ("r2", "r3")
    assert result[2] == ()
    assert sampler.stats["queries"] == 0
    with pytest.raises(ValueError, match="Insufficient"):
        sampler.sample(pd.DataFrame({"cell_type": ["unknown"], "class_by": ["ct"], "n_cell": [1]}))


def test_distance_budget_fails_before_query():
    obs = pd.DataFrame({"ct": ["A"] * 10}, index=[f"r{i}" for i in range(10)])
    sampler = AllCellNeighborSampler(np.ones((10, 2), np.float32), obs, "ct", "sub",
                                    working_memory_bytes=100)
    with pytest.raises(ValueError, match="budget"):
        sampler.neighbors(("ct", "A"), [0], 2)


def test_generator_validation_before_completed_output(tmp_path):
    from simutme.simulation.simulator_sct import SingleCellTypeGEPGenerator

    generator = object.__new__(SingleCellTypeGEPGenerator)
    generator.generated_bulk_gep_fp = tmp_path / "complete.h5ad"
    generator.generated_bulk_gep_fp.touch()
    with pytest.raises(ValueError, match="single_anchor"):
        generator.generate_samples(max_cells_per_cell_type="all",
                                   sampling_method="n_neighbors",
                                   neighbor_sampling_scheme="two_half_anchor")


def test_sct_chunk_bypasses_legacy_sampling_and_averages(monkeypatch):
    from simutme.simulation.simulator_sct import SingleCellTypeGEPGenerator

    obs = pd.DataFrame({"cell_type": ["A"] * 4}, index=[f"r{i}" for i in range(4)])
    values = np.array([[1, 9], [2, 8], [8, 2], [9, 1]], dtype=np.float32)
    generator = object.__new__(SingleCellTypeGEPGenerator)
    generator.all_cell_sampler = AllCellNeighborSampler(values, obs, "cell_type", "sub")
    generator.merged_sc_dataset_df = pd.DataFrame(values, index=obs.index)
    generator.merged_sc_dataset_obs = obs
    generator.cell_type2subtype = {"A": ["A"]}
    generator.selected_subtypes_by_cell_type = {}
    generator.cell_subtype_used = []
    generator.cell_type_col_name = "cell_type"
    generator.subtype_col_name = "sub"
    monkeypatch.setattr(generator, "_load_sc_data_for_generation",
                        lambda method: (obs, [0, 1], "merged_sc_dataset"))
    monkeypatch.setattr(generator, "_sc_sampling",
                        lambda **kwargs: pytest.fail("Legacy sampler called"))
    captured = {}
    monkeypatch.setattr(generator, "_save_chunk_results",
                        lambda expression, tasks: captured.update(expression=expression, tasks=tasks))
    rows = pd.DataFrame({"A": [1, 1]}, index=["s1", "s2"])
    np.random.seed(42)
    generator._process_chunk(rows, "positive", 2, 1, False, sampling_method_gep="n_neighbors")
    for sample, task in captured["tasks"].iterrows():
        ids = task.selected_cell_id.split(";")
        expected = generator.merged_sc_dataset_df.loc[ids].mean(axis=0)
        expected = (expected / expected.sum() * 1e6).round(3)
        np.testing.assert_allclose(captured["expression"].loc[sample], expected, atol=0.1)


def test_block_budget_and_query_deduplication(monkeypatch):
    import simutme.simulation.all_cell_neighbors as module

    values = np.random.default_rng(1).normal(size=(100, 3)).astype(np.float32)
    obs = pd.DataFrame({"ct": ["A"] * 100}, index=[f"r{i}" for i in range(100)])
    sampler = AllCellNeighborSampler(values, obs, "ct", "sub", working_memory_bytes=6400)
    shapes = []
    original = module.euclidean_distances

    def record(query, reference, **kwargs):
        shapes.append((len(query), len(reference)))
        return original(query, reference, **kwargs)

    monkeypatch.setattr(module, "euclidean_distances", record)
    sampler.neighbors(("ct", "A"), [0, 1, 2, 3, 0], 3)
    assert shapes == [(2, 100), (2, 100)]
    assert sampler.stats["queries"] == 4


def test_auto_subtypes_and_uniform_anchor_frequency():
    obs = pd.DataFrame({"ct": ["A"] * 5, "sub": ["a"] * 2 + ["b"] * 3},
                       index=[f"r{i}" for i in range(5)])
    sampler = AllCellNeighborSampler(np.ones((5, 2), np.float32), obs,
                                    "ct", "sub", {"A": "auto"})
    assert len(sampler.pool("ct", "A")[0]) == 5
    assert len(sampler.pool("sub", "a")[0]) == 2
    assert len(sampler.pool("sub", "b")[0]) == 3
    tasks = pd.DataFrame({"cell_type": ["A"] * 5000, "class_by": ["ct"] * 5000, "n_cell": 1})
    np.random.seed(42)
    counts = pd.Series([ids[0] for ids in sampler.sample(tasks)]).value_counts()
    assert counts.between(850, 1150).all()


def test_unmarked_capped_partial_output_cannot_resume_as_all(tmp_path):
    from simutme.simulation.simulator_sct import SingleCellTypeGEPGenerator

    generator = object.__new__(SingleCellTypeGEPGenerator)
    generator.generated_bulk_gep_fp = tmp_path / "absent.h5ad"
    generator.generated_bulk_gep_csv_fp = tmp_path / "partial.csv"
    generator.generated_bulk_gep_csv_fp.touch()
    with pytest.raises(ValueError, match="Unmarked partial"):
        generator.generate_samples(max_cells_per_cell_type="all", sampling_method="n_neighbors")


@pytest.mark.parametrize("value,scheme", [(0, "single_anchor"), ("all", "two_half_anchor")])
def test_workflow_rejects_invalid_all_mode_before_loading(tmp_path, value, scheme):
    from types import SimpleNamespace
    from simutme.workflow import run_bulk_simulation

    config = SimpleNamespace(simu_bulk_dir=tmp_path, sc_dataset_ids=["ds"], n_base=30,
                             stratify_by_dataset="none")
    with pytest.raises(ValueError, match="max_cells_per_cell_type"):
        run_bulk_simulation("new", {
            "dataset_type": "SCT", "sampling_method_gep": "n_neighbors",
            "max_cells_per_cell_type": value, "neighbor_sampling_scheme": scheme,
        }, config)
