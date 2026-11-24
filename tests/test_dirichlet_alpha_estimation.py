import hashlib
import json
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import simutme.workflow as workflow
from simutme.simulation.sampling import estimate_dirichlet_alpha_from_pred_cell_props
from simutme.workflow import SimuConfig, _save_estimated_dirichlet_alpha_from_pred_cell_props


def test_estimate_dirichlet_alpha_from_pred_cell_props_uses_requested_columns(tmp_path: Path):
    csv_path = tmp_path / "pred_cell_props.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "Cancer Cells": [0.7, 0.4],
            "CD4 T": [0.2, 0.3],
            "NK": [0.1, 0.3],
            "unused_meta": ["a", "b"],
        }
    ).to_csv(csv_path, index=False)

    fake_dirichlet = types.SimpleNamespace(
        mle=lambda x, method="meanversion": np.array([2.0, 3.0, 4.0])
    )
    original_module = sys.modules.get("dirichlet")
    sys.modules["dirichlet"] = fake_dirichlet
    try:
        alpha = estimate_dirichlet_alpha_from_pred_cell_props(
            pred_cell_prop_file_path=csv_path,
            cell_types=["Cancer Cells", "CD4 T", "NK"],
        )
    finally:
        if original_module is None:
            del sys.modules["dirichlet"]
        else:
            sys.modules["dirichlet"] = original_module

    assert alpha == {"Cancer Cells": 2.0, "CD4 T": 3.0, "NK": 4.0}


def test_estimate_dirichlet_alpha_uses_only_selected_cancer_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    csv_path = tmp_path / "pred_cell_props.csv"
    pd.DataFrame(
        {
            "cancer_type": ["BRCA", "LIHC", "BRCA"],
            "A": [0.8, 0.1, 0.6],
            "B": [0.2, 0.9, 0.4],
        }
    ).to_csv(csv_path, index=False)
    captured = {}

    def estimate(values, method="meanversion"):
        captured["values"] = values.copy()
        return np.array([2.0, 3.0])

    monkeypatch.setitem(
        sys.modules,
        "dirichlet",
        types.SimpleNamespace(mle=estimate),
    )

    alpha = estimate_dirichlet_alpha_from_pred_cell_props(
        pred_cell_prop_file_path=csv_path,
        cell_types=["A", "B"],
        cancer_types=["BRCA"],
    )

    assert alpha == {"A": 2.0, "B": 3.0}
    assert captured["values"].shape == (2, 2)
    expected = np.array([[0.8, 0.2], [0.6, 0.4]]) + 1e-6
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        captured["values"],
        expected,
    )


@pytest.mark.parametrize(
    "rows, cancer_types, message",
    [
        (
            {"A": [0.8], "B": [0.2]},
            ["BRCA"],
            'must contain a "cancer_type" column',
        ),
        (
            {"cancer_type": ["LIHC"], "A": [0.8], "B": [0.2]},
            ["BRCA"],
            "contains no rows for filtering_ref_types",
        ),
    ],
)
def test_estimate_dirichlet_alpha_rejects_unavailable_cancer_subset(
    tmp_path: Path,
    rows: dict,
    cancer_types: list[str],
    message: str,
):
    csv_path = tmp_path / "pred_cell_props.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match=message):
        estimate_dirichlet_alpha_from_pred_cell_props(
            pred_cell_prop_file_path=csv_path,
            cell_types=["A", "B"],
            cancer_types=cancer_types,
        )


def test_save_estimated_dirichlet_alpha_from_pred_cell_props_writes_csv(tmp_path: Path):
    csv_path = tmp_path / "pred_cell_props.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "Cancer Cells": [0.7, 0.4],
            "CD4 T": [0.2, 0.3],
            "NK": [0.1, 0.3],
        }
    ).to_csv(csv_path, index=False)

    fake_dirichlet = types.SimpleNamespace(
        mle=lambda x, method="meanversion": np.array([2.0, 3.0, 4.0])
    )
    original_module = sys.modules.get("dirichlet")
    sys.modules["dirichlet"] = fake_dirichlet
    try:
        out_path = _save_estimated_dirichlet_alpha_from_pred_cell_props(
            pred_cell_prop_file_path=csv_path,
            cell_types=["Cancer Cells", "CD4 T", "NK"],
            output_dir=tmp_path,
        )
    finally:
        if original_module is None:
            del sys.modules["dirichlet"]
        else:
            sys.modules["dirichlet"] = original_module

    assert out_path.exists()
    alpha_df = pd.read_csv(out_path)
    assert list(alpha_df["cell_type"]) == ["Cancer Cells", "CD4 T", "NK"]


@pytest.mark.parametrize(
    "case, extra_params, expected_method, expected_source, expected_alpha, expected_seed",
    [
        (
            "default_symmetric",
            {"sampling_method_prop": "dirichlet", "dirichlet_sampling_seed": 42},
            "dirichlet",
            "default_symmetric",
            {"A": 1.0, "B": 1.0},
            42,
        ),
        (
            "configured_partial",
            {
                "sampling_method_prop": "dirichlet",
                "dirichlet_alpha": {"A": 2.5},
                "dirichlet_sampling_seed": 7,
            },
            "dirichlet",
            "configured",
            {"A": 2.5, "B": 1.0},
            7,
        ),
        (
            "pred_file",
            {"sampling_method_prop": "dirichlet", "dirichlet_sampling_seed": 11},
            "dirichlet",
            "pred_cell_prop_file",
            {"A": 2.0, "B": 5.0},
            11,
        ),
        (
            "segment",
            {"sampling_method_prop": "segment"},
            "segment",
            None,
            None,
            None,
        ),
    ],
)
def test_mixed_summary_reports_resolved_dirichlet_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    extra_params: dict,
    expected_method: str,
    expected_source: str | None,
    expected_alpha: dict | None,
    expected_seed: int | None,
):
    sct_path = tmp_path / "sct.h5ad"
    sct_matrix = np.log2(
        np.array(
            [
                [800000.0, 200000.0],
                [700000.0, 300000.0],
                [200000.0, 800000.0],
                [300000.0, 700000.0],
            ],
            dtype=np.float32,
        )
        + 1.0
    )
    ad.AnnData(
        X=sct_matrix,
        obs=pd.DataFrame(
            {"A": [1.0, 1.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0, 1.0]},
            index=["s1", "s2", "s3", "s4"],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(sct_path)

    reference_path = tmp_path / "reference.csv"
    pd.DataFrame(
        {"g1": [200000.0, 500000.0, 800000.0], "g2": [800000.0, 500000.0, 200000.0]},
        index=["r1", "r2", "r3"],
    ).to_csv(reference_path)

    pred_path = None
    filtered_alpha_calls = []
    if case == "pred_file":
        pred_path = tmp_path / "pred_cell_props.csv"
        pd.DataFrame(
            {
                "cancer_type": ["BRCA", "BRCA", "LIHC"],
                "A": [0.3, 0.25, 0.95],
                "B": [0.7, 0.75, 0.05],
            }
        ).to_csv(pred_path, index=False)

        def estimate(values, method="meanversion"):
            filtered_alpha_calls.append(values.copy())
            return np.array([2.0, 5.0])

        fake_dirichlet = types.SimpleNamespace(mle=estimate)
        monkeypatch.setitem(sys.modules, "dirichlet", fake_dirichlet)
        pd.DataFrame(
            {"cancer_type": ["BRCA", "BRCA", "LIHC"]},
            index=["r1", "r2", "r3"],
        ).to_csv(tmp_path / "mapping.csv")

    config = SimuConfig(
        simu_bulk_dir=tmp_path / f"out_{case}",
        tcga2cancer_type_file_path=tmp_path / "mapping.csv",
        tcga_merged_tpm_file_path=reference_path,
        high_corr_gene_file_path=None,
        merged_sc_dataset_file_path=tmp_path / "unused.h5ad",
        sc_dataset_ids=[],
        cell_type2subtypes={"A": ["A"], "B": ["B"]},
        total_rna_coefficient={"A": 1.0, "B": 1.0},
        n_base=1,
        output_types=["A", "B"],
        gep_filtering_quantile=(None, 1.0),
        plot_pca_for_comparison=False,
        mixed_sct_loading_mode="eager",
        pred_cell_prop_file_path=pred_path,
    )
    params = {
        "dataset_type": "Mixed",
        "n_samples": 3,
        "sct_reference": {"source_type": "path", "file_path": str(sct_path)},
        "gep_filtering": True,
        "gep_filtering_method": "median_gep",
        "gep_filtering_in_pca_space": True,
        "gep_pca_n_components": 1,
        "gep_norm_ord": 1,
        "gene_filtering": False,
        **extra_params,
    }
    if case == "pred_file":
        params["filtering_ref_types"] = ["BRCA"]
    monkeypatch.setattr(workflow, "_save_cancer_cell_fraction_distribution", lambda **kwargs: None)

    output = workflow.run_bulk_simulation(f"target_{case}", params, config)
    result_dir = workflow._resolve_mixed_dataset_result_dir(output, f"target_{case}")
    summary = json.loads((result_dir / "mixed_dataset_summary.json").read_text(encoding="utf-8"))

    assert summary["sampling_method_prop"] == expected_method
    assert summary["dirichlet_alpha_source"] == expected_source
    assert summary["dirichlet_alpha"] == expected_alpha
    assert summary["dirichlet_sampling_seed"] == expected_seed
    estimated_csv = result_dir / "dirichlet_alpha_estimated_from_pred_cell_props.csv"
    assert estimated_csv.exists() is (case == "pred_file")
    if case == "pred_file":
        assert filtered_alpha_calls
        assert all(values.shape[0] == 2 for values in filtered_alpha_calls)
        metadata = json.loads(
            (result_dir / "filtering_metadata.json").read_text(encoding="utf-8")
        )
        generation = metadata["identity"]["generation"]
        assert generation["dirichlet_alpha_filtering_ref_types"] == ["BRCA"]
        assert generation["dirichlet_prediction_file_sha256"] == hashlib.sha256(
            pred_path.read_bytes()
        ).hexdigest()
