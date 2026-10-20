import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simutme.simulation.simu_helper import DatasetType, sample_random
from simutme.simulation.simulator import BulkGEPGenerator
from simutme.utility import non_log2cpm


def test_cached_sampling_constraints(monkeypatch):
    import simutme.simulation.simu_helper as helper

    monkeypatch.setattr(helper.multiprocessing, "get_context",
                        lambda *args: pytest.fail("Unexpected process pool"))
    obs = pd.DataFrame({"ct": ["A", "A", "B"], "sub": ["a", "b", "c"]},
                       index=["z", "x", "y"])
    tasks = pd.DataFrame({"cell_type": ["A", "a", "B", "missing"],
                          "class_by": ["ct", "sub", "ct", "ct"],
                          "n_cell": [10, 1, 1, 0]})
    assert sample_random(tasks, obs, False, DatasetType.SCT, 1, 10) == [
        ("x", "z"), ("z",), ("y",), (),
    ]
    tasks.loc[3, "n_cell"] = 1
    with pytest.raises(ValueError, match="minimum_n_base"):
        sample_random(tasks, obs, False, DatasetType.SCT, 1, 10)
    tasks = pd.DataFrame({"cell_type": ["missing"], "class_by": ["absent"], "n_cell": [0]})
    assert sample_random(tasks, obs, False, DatasetType.SCT, 1, 10) == [()]


def test_cancer_branching():
    obs = pd.DataFrame({"ct": ["Glioma Cells", "Epithelial Cells"]},
                       index=["glioma", "epi"])
    tasks = pd.DataFrame({"cell_type": ["Cancer Cells"] * 10000,
                          "class_by": ["ct"] * 10000, "n_cell": 1})
    np.random.seed(42)
    result = sample_random(tasks, obs, False, DatasetType.SCT, 1, 10)
    fraction = sum(ids == ("glioma",) for ids in result) / len(result)
    assert 0.08 < fraction < 0.12


@pytest.mark.parametrize("rna_enabled", [False, True])
@pytest.mark.parametrize("noise_enabled", [False, True])
def test_eager_aggregation_matches_legacy(rna_enabled, noise_enabled):
    rng = np.random.default_rng(42)
    # Several chunks, nonuniform task counts, repeated source rows, unsorted IDs.
    source = pd.DataFrame(rng.uniform(1, 1000, (4, 4096)).astype(np.float32),
                          index=["r0", "r1", "r2", "r3"])
    original = source.copy()
    ids = [f"s{i:03}" for i in range(260)][::-1]
    tasks = pd.DataFrame([
        (sample, ct, f"r{j % 4}")
        for i, sample in enumerate(ids)
        for j, ct in enumerate(["A", "B"] if i % 2 else ["A"])
    ], columns=["sample_id", "cell_type", "selected_cell_id"]).set_index("sample_id")
    fractions = pd.DataFrame({"A": 0.4, "B": 0.6}, index=ids)
    generator = object.__new__(BulkGEPGenerator)
    generator.total_rna_coefficient = {"A": 1e12, "B": 2e12} if rna_enabled else {}
    coefficients = {"A": 1e12, "B": 2e12} if rna_enabled else {"A": 1, "B": 1}
    np.random.seed(123)
    expected = {}
    for sample, group in tasks.groupby(level=0):
        expression = source.loc[group.selected_cell_id].to_numpy(dtype=np.float64, copy=True)
        if rna_enabled:
            expression *= np.array([coefficients[ct] for ct in group.cell_type])[:, None]
        values = (expression * fractions.loc[sample, group.cell_type].to_numpy()[:, None]).sum(0)
        if rna_enabled:
            values = values / values.sum() * 1e6
        if noise_enabled:
            noise = generator._sample_noise(n_samples=len(values), f=0.25)
            if noise.sum() > 100:
                noise = noise / noise.sum() * 100
            values += noise
        expected[sample] = values
    expected = non_log2cpm(pd.DataFrame.from_dict(expected, orient="index"), sum_exp=1e6).round(3)
    np.random.seed(123)
    actual = generator._aggregate_eager_bulk(
        tasks, fractions, source, coefficients, noise_enabled, (0.25, 100))
    actual = non_log2cpm(actual, sum_exp=1e6).round(3)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(source, original)
    tasks.iloc[0, tasks.columns.get_loc("selected_cell_id")] = "absent"
    with pytest.raises(ValueError, match="absent"):
        generator._aggregate_eager_bulk(tasks, fractions, source, coefficients)
    tasks.iloc[0, tasks.columns.get_loc("selected_cell_id")] = "r0"
    tasks.iloc[0, tasks.columns.get_loc("cell_type")] = "absent"
    with pytest.raises(ValueError, match="absent"):
        generator._aggregate_eager_bulk(tasks, fractions, source, coefficients)


def test_sampling_without_replacement_and_uniform_single_draws():
    obs = pd.DataFrame({"ct": ["A"] * 5}, index=[f"r{i}" for i in range(5)])
    tasks = pd.DataFrame({"cell_type": ["A"] * 5000,
                          "class_by": ["ct"] * 5000, "n_cell": 1})
    np.random.seed(42)
    counts = pd.Series([ids[0] for ids in sample_random(
        tasks, obs, False, DatasetType.SCT, 1, 10)]).value_counts()
    assert counts.between(850, 1150).all()
    tasks["n_cell"] = 3
    assert all(len(set(ids)) == 3 and list(ids) == sorted(ids)
               for ids in sample_random(tasks, obs, False, DatasetType.SCT, 1, 10))


def test_patient_sampling_retains_legacy_pool(monkeypatch):
    import simutme.simulation.simu_helper as helper

    class Pool:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def starmap(self, function, tasks):
            return [function(*task) for task in tasks]

    monkeypatch.setattr(helper.multiprocessing, "get_context",
                        lambda *args: type("Context", (), {"Pool": lambda self, n: Pool()})())
    obs = pd.DataFrame({"ct": ["A"] * 20, "sample_id": ["patient"] * 20},
                       index=[f"r{i}" for i in range(20)])
    tasks = pd.DataFrame({"cell_type": ["A"], "class_by": ["ct"], "n_cell": [2]})
    assert len(sample_random(tasks, obs, True, DatasetType.SCT, 1, 2)[0]) == 2


def test_eager_dispatch_uses_parent_rna_coefficient():
    generator = object.__new__(BulkGEPGenerator)
    generator.sct_dataset_df = pd.DataFrame(
        {"g1": [100, 400], "g2": [900, 600]}, index=["r1", "r2"])
    generator.sct_dataset_reader = None
    generator.cell_type2subtype = {"A": ["a"], "B": ["B"]}
    generator.total_rna_coefficient = {"A": 2, "B": 1}
    tasks = pd.DataFrame({"cell_type": ["a", "B"], "selected_cell_id": ["r1", "r2"]},
                         index=["sample", "sample"])
    fractions = pd.DataFrame({"a": [0.5], "B": [0.5]}, index=["sample"])
    result = generator._map_cell_id2exp(
        tasks, DatasetType.SCT, "mul", fractions)
    np.testing.assert_array_equal(result.to_numpy(), [[200000, 800000]])
