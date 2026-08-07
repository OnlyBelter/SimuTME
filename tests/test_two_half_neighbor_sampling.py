import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simutme.workflow import _resolve_neighbor_sampling_scheme
from simutme.simulation.nneighbors_shared_memory import get_sample_id_with_neighbors
from simutme.simulation.simu_helper import NeighborSamplingScheme, validate_generation_params_sct


def _make_obs_and_distance_dict(
    ds1_count: int = 3,
    ds2_count: int = 3,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, list[str]]]]:
    rows = []
    cell_ids = []
    for ds_id, n_cells in [("ds1", ds1_count), ("ds2", ds2_count)]:
        for idx in range(n_cells):
            cell_id = f"{ds_id}_cell_{idx}"
            rows.append(
                {
                    "cell_id": cell_id,
                    "cell_type": "CT",
                    "dataset_id": ds_id,
                }
            )
            cell_ids.append(cell_id)

    obs_df = pd.DataFrame(rows).set_index("cell_id")
    n_total = len(cell_ids)
    distance_matrix = np.full((n_total, n_total), 100.0, dtype=float)
    np.fill_diagonal(distance_matrix, 0.0)
    for i, cid_i in enumerate(cell_ids):
        for j, cid_j in enumerate(cell_ids):
            if i == j:
                continue
            ds_i = cid_i.split("_")[0]
            ds_j = cid_j.split("_")[0]
            if ds_i == ds_j:
                distance_matrix[i, j] = 1.0 + abs(i - j)
    return obs_df, {"CT": (distance_matrix, cell_ids)}


def test_resolve_neighbor_sampling_scheme_defaults_and_validates():
    assert _resolve_neighbor_sampling_scheme({}) == "single_anchor"
    assert _resolve_neighbor_sampling_scheme({"neighbor_sampling_scheme": "two_half_anchor"}) == "two_half_anchor"
    with pytest.raises(ValueError):
        _resolve_neighbor_sampling_scheme({"neighbor_sampling_scheme": "bad"})


def test_validate_generation_params_sct_rejects_two_half_for_random():
    with pytest.raises(ValueError):
        validate_generation_params_sct(
            n_sample_each_cell_type=10,
            n_base_for_positive_samples=30,
            sample_type="positive",
            simu_method="ave",
            sampling_method="random",
            minimum_n_base=1,
            neighbor_sampling_scheme=NeighborSamplingScheme.TWO_HALF_ANCHOR,
        )


def test_single_anchor_default_matches_explicit_single_anchor():
    obs_df, distance_dict = _make_obs_and_distance_dict()
    np.random.seed(7)
    default_result = get_sample_id_with_neighbors(
        obs_df=obs_df,
        n_samples=1,
        cell_type="CT",
        n_base=3,
        distance_dict=distance_dict,
        class_by="cell_type",
    )
    np.random.seed(7)
    explicit_result = get_sample_id_with_neighbors(
        obs_df=obs_df,
        n_samples=1,
        cell_type="CT",
        n_base=3,
        distance_dict=distance_dict,
        class_by="cell_type",
        neighbor_sampling_scheme=NeighborSamplingScheme.SINGLE_ANCHOR,
    )
    assert default_result == explicit_result
    assert len(default_result) == 3


def test_two_half_anchor_prefers_different_datasets_when_feasible():
    obs_df, distance_dict = _make_obs_and_distance_dict(ds1_count=3, ds2_count=3)
    np.random.seed(3)
    result = get_sample_id_with_neighbors(
        obs_df=obs_df,
        n_samples=1,
        cell_type="CT",
        n_base=4,
        distance_dict=distance_dict,
        class_by="cell_type",
        neighbor_sampling_scheme=NeighborSamplingScheme.TWO_HALF_ANCHOR,
    )
    dataset_counts = obs_df.loc[list(result), "dataset_id"].value_counts().to_dict()
    assert len(result) == 4
    assert dataset_counts == {"ds1": 2, "ds2": 2}


def test_two_half_anchor_falls_back_when_only_one_dataset_exists():
    obs_df, distance_dict = _make_obs_and_distance_dict(ds1_count=4, ds2_count=0)
    np.random.seed(5)
    result = get_sample_id_with_neighbors(
        obs_df=obs_df,
        n_samples=1,
        cell_type="CT",
        n_base=4,
        distance_dict=distance_dict,
        class_by="cell_type",
        neighbor_sampling_scheme=NeighborSamplingScheme.TWO_HALF_ANCHOR,
    )
    dataset_counts = obs_df.loc[list(result), "dataset_id"].value_counts().to_dict()
    assert len(result) == 4
    assert dataset_counts == {"ds1": 4}


def test_two_half_anchor_splits_odd_n_base():
    obs_df, distance_dict = _make_obs_and_distance_dict(ds1_count=4, ds2_count=4)
    np.random.seed(11)
    result = get_sample_id_with_neighbors(
        obs_df=obs_df,
        n_samples=1,
        cell_type="CT",
        n_base=5,
        distance_dict=distance_dict,
        class_by="cell_type",
        neighbor_sampling_scheme=NeighborSamplingScheme.TWO_HALF_ANCHOR,
    )
    assert len(result) == 5
