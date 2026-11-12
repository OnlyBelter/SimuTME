import json

import numpy as np
import pandas as pd
import pytest

from simutme.simulation.all_cell_neighbors import AllCellNeighborSampler
from simutme.simulation.simulator_sct import SingleCellTypeGEPGenerator
from simutme.simulation.simu_helper import (
    draw_n_base_assignments,
    sample_random,
)
from simutme.utility.pub_func import get_cell_num


def test_n_base_assignments_are_seeded_per_sample():
    sample_ids = [f"sample_{index}" for index in range(512)]
    values = [30, 40, 50, 60, 70, 80, 90, 100]

    first = draw_n_base_assignments(sample_ids, values, seed=42)
    repeated = draw_n_base_assignments(sample_ids, values, seed=42)
    different_seed = draw_n_base_assignments(sample_ids, values, seed=43)

    pd.testing.assert_series_equal(first, repeated)
    assert set(first.unique()) == set(values)
    assert not first.equals(different_seed)


@pytest.mark.parametrize(
    "values",
    [[], [0, 30], [-1, 30], [30, 30], [30, 40.5], [True, 30], "30,40"],
)
def test_n_base_assignments_reject_invalid_values(values):
    with pytest.raises(ValueError, match="n_base_values"):
        draw_n_base_assignments(["sample_1"], values, seed=42)


def test_n_base_assignments_reject_invalid_seed_and_sample_ids():
    with pytest.raises(ValueError, match="n_base_seed"):
        draw_n_base_assignments(["sample_1"], [30, 50], seed=-1)
    with pytest.raises(ValueError, match="unique and nonmissing"):
        draw_n_base_assignments(["sample_1", "sample_1"], [30, 50], seed=42)


def test_get_cell_num_applies_a_distinct_total_to_each_sample():
    fractions = pd.DataFrame(
        {"A": [1.0, 1.0, 0.0], "B": [0.0, 0.0, 1.0]},
        index=["sample_1", "sample_2", "sample_3"],
    )
    totals = pd.Series(
        [30, 50, 70],
        index=fractions.index,
    )

    counts = get_cell_num(fractions, total_num=totals)

    assert counts.to_dict(orient="list") == {
        "A": [30, 50, 0],
        "B": [0, 0, 70],
    }


def test_get_cell_num_keeps_single_cell_sct_counts_pure():
    fractions = pd.DataFrame(
        {"A": [1.0], "B": [0.0]},
        index=["sample_1"],
    )

    counts = get_cell_num(fractions, total_num=pd.Series([1], index=fractions.index))

    assert counts.to_dict(orient="list") == {"A": [1], "B": [0]}


def test_random_sampler_honors_each_sample_count():
    cell_ids = [f"cell_{index}" for index in range(12)]
    obs = pd.DataFrame({"cell_type": ["T"] * len(cell_ids)}, index=cell_ids)
    tasks = pd.DataFrame(
        {
            "n_cell": [3, 5],
            "cell_type": ["T", "T"],
            "class_by": ["cell_type", "cell_type"],
        },
        index=["sample_1", "sample_2"],
    )

    sampled = sample_random(
        tasks,
        obs,
        sep_by_patient=False,
        sc_dataset="merged_sc_dataset",
        minimum_n_base=1,
        n_threads=1,
    )

    assert [len(cell_ids) for cell_ids in sampled] == [3, 5]


def test_variable_n_base_keeps_single_cell_sct_profiles():
    generator = object.__new__(SingleCellTypeGEPGenerator)
    generator.merged_sc_dataset_df = pd.DataFrame(
        [[1.0, 3.0]],
        index=["cell_1"],
        columns=["gene_1", "gene_2"],
    )
    sampled_cells = pd.DataFrame(
        {
            "n_cell": [1],
            "n_base_requested": [1],
            "cell_type": ["A"],
            "selected_cell_id": ["cell_1"],
        },
        index=["sample_1"],
    )

    expression = generator._map_cell_id2exp(
        selected_cell_id=sampled_cells,
        sc_dataset="merged_sc_dataset",
        gep_type="SCT",
    )

    assert expression.index.tolist() == ["sample_1"]
    np.testing.assert_allclose(expression.loc["sample_1"], [250000.0, 750000.0])


def test_neighbor_sampler_honors_each_sample_count():
    values = np.random.default_rng(7).normal(size=(12, 4)).astype(np.float32)
    obs = pd.DataFrame(
        {"cell_type": ["T"] * len(values)},
        index=[f"cell_{index}" for index in range(len(values))],
    )
    tasks = pd.DataFrame(
        {
            "n_cell": [3, 5],
            "cell_type": ["T", "T"],
            "class_by": ["cell_type", "cell_type"],
        },
        index=["sample_1", "sample_2"],
    )
    sampler = AllCellNeighborSampler(values, obs, "cell_type", "subtype")

    sampled = sampler.sample(tasks, minimum_n_base=1)

    assert [len(cell_ids) for cell_ids in sampled] == [3, 5]


def test_sct_chunk_passes_sample_specific_counts_to_neighbor_sampler():
    values = np.random.default_rng(8).normal(size=(8, 3)).astype(np.float32)
    obs = pd.DataFrame(
        {"cell_type": ["A"] * len(values)},
        index=[f"cell_{index}" for index in range(len(values))],
    )
    generator = object.__new__(SingleCellTypeGEPGenerator)
    generator.all_cell_sampler = AllCellNeighborSampler(values, obs, "cell_type", "subtype")
    generator.merged_sc_dataset_obs = obs
    generator.cell_subtype_used = []
    generator.cell_type_col_name = "cell_type"
    generator.subtype_col_name = "subtype"
    generator.selected_subtypes_by_cell_type = {}
    rows = pd.DataFrame({"A": [1.0, 1.0]}, index=["sample_1", "sample_2"])
    assignments = pd.Series([2, 5], index=rows.index, name="n_base")
    captured = {}
    generator._load_sc_data_for_generation = lambda _method: (obs, ["g1", "g2", "g3"], "merged_sc_dataset")
    generator._map_cell_id2exp = lambda **kwargs: pd.DataFrame(
        {"g1": [1.0, 1.0]},
        index=kwargs["selected_cell_id"].index.unique(),
    )
    generator._save_chunk_results = lambda expression, tasks: captured.update(
        expression=expression,
        tasks=tasks,
    )

    generator._process_chunk(
        rows=rows,
        sample_type="positive",
        n_base=5,
        min_n_base=1,
        sep_by_patient=False,
        sampling_method_gep="n_neighbors",
        n_base_by_sample=assignments,
    )

    tasks = captured["tasks"]
    assert tasks["n_base_requested"].to_dict() == {"sample_1": 2, "sample_2": 5}
    assert tasks["n_cell_actual"].to_dict() == {"sample_1": 2, "sample_2": 5}


def test_sct_generator_requires_values_to_meet_minimum_n_base():
    generator = object.__new__(SingleCellTypeGEPGenerator)

    with pytest.raises(ValueError, match="Every n_base_values entry"):
        generator.generate_samples(
            n_base_for_positive_samples=[30, 50],
            minimum_n_base=40,
        )


def test_sct_generator_persists_assignments_and_dispatches_per_chunk(tmp_path):
    generator = object.__new__(SingleCellTypeGEPGenerator)
    generator.generated_bulk_gep_fp = tmp_path / "generated.h5ad"
    generator.generated_bulk_gep_csv_fp = tmp_path / "generated.csv"
    generator.generated_cell_fraction_fp = tmp_path / "fractions.csv"
    generator.cell_type_used = ["A"]
    generator.cell_subtype_used = []
    generator.all_cell_sampler = None
    generator.cell_distance_dict = {}
    generator.generated_bulk_gep_counter = 0
    captured = []

    def prepare_fractions(_sample_type, n_sample_each_cell_type):
        sample_ids = [f"SCT_positive_{index}" for index in range(n_sample_each_cell_type)]
        fractions = pd.DataFrame({"A": 1.0}, index=sample_ids)
        fractions.to_csv(generator.generated_cell_fraction_fp, index_label="sample_id")

    def process_chunk(**kwargs):
        captured.append(kwargs["n_base_by_sample"].copy())

    generator._prepare_fractions = prepare_fractions
    generator._prepare_gene_expression = lambda _path: None
    generator._process_chunk = process_chunk
    generator._finalize_dataset = lambda **_kwargs: None

    generator.generate_samples(
        n_sample_each_cell_type=1200,
        n_base_for_positive_samples=[30, 50, 100],
        n_base_seed=42,
        sampling_method="random",
        max_cells_per_cell_type=100,
    )

    assignments = pd.concat(captured)
    assignment_path = generator.generated_bulk_gep_csv_fp.with_name(
        "generated_n_base_assignment.csv"
    )
    saved = pd.read_csv(assignment_path, index_col=0)["n_base"]
    metadata = json.loads(assignment_path.with_suffix(".json").read_text())

    pd.testing.assert_series_equal(assignments, saved.rename(assignments.name))
    assert len(saved) == 1200
    assert set(saved.unique()) <= {30, 50, 100}
    assert metadata["distribution"] == "uniform_independent"
    assert metadata["n_base_seed"] == 42
    assert sum(metadata["counts_by_cell_type"]["A"].values()) == 1200
