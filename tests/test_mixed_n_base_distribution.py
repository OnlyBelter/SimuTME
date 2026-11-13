import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import simutme.workflow as workflow


def _write_inputs(tmp_path):
    fractions = pd.DataFrame(
        {
            "A": [0.01, 0.0, 0.005],
            "B": [0.99, 1.0, 0.995],
        },
        index=["mixed_1", "mixed_2", "mixed_3"],
    )
    selected = pd.DataFrame(
        {
            "cell_type": ["A", "B", "A", "B", "A", "B"],
            "selected_cell_id": [
                "sct_30", "sct_50",
                "sct_30", "sct_30",
                "sct_50", "sct_30",
            ],
        },
        index=[
            "mixed_1", "mixed_1",
            "mixed_2", "mixed_2",
            "mixed_3", "mixed_3",
        ],
    )
    assignments = pd.DataFrame(
        {"n_base": [30, 50]},
        index=["sct_30", "sct_50"],
    )
    fraction_path = tmp_path / "cell_fractions.csv"
    selected_path = tmp_path / "sampled_sct_geps.csv"
    assignment_path = tmp_path / "sct_n_base_assignment.csv"
    fractions.to_csv(fraction_path, index_label="sample_id")
    selected.to_csv(selected_path, index_label="sample_id")
    assignments.to_csv(assignment_path, index_label="sample_id")
    return fraction_path, selected_path, assignment_path


def _run_summary(
    tmp_path,
    monkeypatch,
    fraction_path,
    selected_path=None,
    assignment_path=None,
    n_base_values=None,
    dataset_name="Mixed_Test",
):
    monkeypatch.setattr(
        workflow,
        "_get_h5ad_shape",
        lambda path: (3, 2) if path is not None else None,
    )
    return workflow._build_mixed_dataset_summary(
        dataset_name=dataset_name,
        generated_bulk_gep_fp=tmp_path / "generated.h5ad",
        generated_cell_fraction_fp=fraction_path,
        gep_filtering=SimpleNamespace(enabled=False),
        gene_filtering=SimpleNamespace(
            enabled=False,
            correlation_reference_dataset=None,
        ),
        postprocess_outputs={},
        n_base_values=n_base_values,
        n_base_assignment_file=assignment_path,
        sampled_sct_cell_id_file=selected_path,
    )


def test_summary_writes_fraction_stratified_n_base_counts(tmp_path, monkeypatch):
    fraction_path, selected_path, assignment_path = _write_inputs(tmp_path)

    summary_path = _run_summary(
        tmp_path,
        monkeypatch,
        fraction_path,
        selected_path,
        assignment_path,
        n_base_values=[30, 50],
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    distribution_path = summary_path.parent / "n_base_distribution_by_cell_type.csv"
    distribution = pd.read_csv(distribution_path).set_index(
        ["cell_type", "cell_fraction_group", "n_base"]
    )

    assert summary["n_base_distribution_file"] == str(distribution_path)
    assert summary["n_base_distribution"]["n_base_values"] == [30, 50]
    assert summary["n_base_distribution"]["cell_fraction_threshold"] == 0.01
    assert summary["n_base_distribution"]["cell_fraction_groups"] == {
        ">=0.01": {"operator": ">=", "threshold": 0.01},
        "<0.01": {"operator": "<", "threshold": 0.01},
    }
    assert summary["n_base_distribution"]["sample_count"] == 3
    assert len(distribution) == 8

    assert distribution.loc[("A", ">=0.01", 30), "count"] == 1
    assert distribution.loc[("A", ">=0.01", 50), "count"] == 0
    assert distribution.loc[("A", "<0.01", 30), "count"] == 1
    assert distribution.loc[("A", "<0.01", 50), "count"] == 1
    assert distribution.loc[("B", ">=0.01", 30), "count"] == 2
    assert distribution.loc[("B", ">=0.01", 50), "count"] == 1
    assert distribution.loc[("B", "<0.01", 30), "count"] == 0
    assert distribution.loc[("B", "<0.01", 50), "count"] == 0
    assert distribution.loc[("A", ">=0.01", 30), "group_sample_count"] == 1
    assert distribution.loc[("A", "<0.01", 30), "group_sample_count"] == 2
    assert distribution.loc[("B", "<0.01", 30), "group_sample_count"] == 0
    assert np.isnan(distribution.loc[("B", "<0.01", 30), "proportion"])
    assert distribution.loc[("A", ">=0.01", 30), "proportion"] == 1.0
    assert distribution.loc[("A", "<0.01", 50), "proportion"] == 0.5
    assert distribution.groupby(level="cell_type")["count"].sum().to_dict() == {
        "A": 3,
        "B": 3,
    }


@pytest.mark.parametrize("n_base_values", [None, [30]])
def test_scalar_or_single_value_policy_does_not_write_distribution(
    tmp_path, monkeypatch, n_base_values
):
    fraction_path, _, _ = _write_inputs(tmp_path)
    summary_path = _run_summary(
        tmp_path,
        monkeypatch,
        fraction_path,
        n_base_values=n_base_values,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "n_base_distribution_file" not in summary
    assert "n_base_distribution" not in summary
    assert not (summary_path.parent / "n_base_distribution_by_cell_type.csv").exists()


def test_missing_selected_assignment_fails_with_source_id(tmp_path, monkeypatch):
    fraction_path, selected_path, assignment_path = _write_inputs(tmp_path)
    selected = pd.read_csv(selected_path, index_col=0)
    selected.iloc[0, selected.columns.get_loc("selected_cell_id")] = "missing_sct"
    selected.to_csv(selected_path, index_label="sample_id")

    with pytest.raises(ValueError, match="missing_sct"):
        _run_summary(
            tmp_path,
            monkeypatch,
            fraction_path,
            selected_path,
            assignment_path,
            n_base_values=[30, 50],
        )


def test_missing_fraction_sample_fails_alignment(tmp_path, monkeypatch):
    fraction_path, selected_path, assignment_path = _write_inputs(tmp_path)
    fractions = pd.read_csv(fraction_path, index_col=0).iloc[:2]
    fractions.to_csv(fraction_path, index_label="sample_id")

    with pytest.raises(ValueError, match="does not align with Mixed sample IDs"):
        _run_summary(
            tmp_path,
            monkeypatch,
            fraction_path,
            selected_path,
            assignment_path,
            n_base_values=[30, 50],
        )


def test_missing_cell_type_selection_fails_pair_validation(tmp_path, monkeypatch):
    fraction_path, selected_path, assignment_path = _write_inputs(tmp_path)
    selected = pd.read_csv(selected_path, index_col=0).iloc[:-1]
    selected.to_csv(selected_path, index_label="sample_id")

    with pytest.raises(ValueError, match="expected one row for each"):
        _run_summary(
            tmp_path,
            monkeypatch,
            fraction_path,
            selected_path,
            assignment_path,
            n_base_values=[30, 50],
        )


def test_multiple_selected_geps_for_one_output_cell_type_fails(tmp_path, monkeypatch):
    fraction_path, selected_path, assignment_path = _write_inputs(tmp_path)
    selected = pd.read_csv(selected_path, index_col=0)
    selected.iloc[0, selected.columns.get_loc("selected_cell_id")] = "sct_30;sct_50"
    selected.to_csv(selected_path, index_label="sample_id")

    with pytest.raises(ValueError, match="exactly one"):
        _run_summary(
            tmp_path,
            monkeypatch,
            fraction_path,
            selected_path,
            assignment_path,
            n_base_values=[30, 50],
        )
