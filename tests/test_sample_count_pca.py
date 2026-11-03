from pathlib import Path
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import simutme.workflow as workflow
from simutme.simulation.filtering import FixedGeneFilter, normalize_selected
from tests.test_correlation_first_filtering import gep_params, workflow_case


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (12, (12,)),
        ([5, 10, 20], (5, 10, 20)),
    ],
)
def test_resolve_pca_sample_counts_accepts_null_scalar_and_list(value, expected):
    assert workflow._resolve_pca_sample_counts(value, "target", 20) == expected


@pytest.mark.parametrize(
    "value,total,error",
    [
        (True, 10, "positive integer"),
        (0, 10, "positive integer"),
        ([], 10, "empty list"),
        ([2, 2], 10, "duplicate"),
        ([2, 11], 10, "exceed n_samples"),
        ([2, 3.5], 10, "positive integers"),
    ],
)
def test_resolve_pca_sample_counts_rejects_invalid_values(value, total, error):
    with pytest.raises(ValueError, match=error):
        workflow._resolve_pca_sample_counts(value, "target", total)


def test_gene_filter_resolver_validates_configured_sample_counts(workflow_case):
    config, params, _, _ = workflow_case
    params["n_samples"] = 3
    params["plot_pca_for_comparison"] = True
    params["n_sample_for_pca"] = [1, 3]
    resolved = workflow._resolve_gene_filtering_params("target", params, config)
    assert resolved.n_sample_for_pca == (1, 3)
    assert resolved.pca_sample_seed == 42
    assert resolved.incremental_pca_batch_size == 5000

    params["n_sample_for_pca"] = [4]
    with pytest.raises(ValueError, match="exceed n_samples"):
        workflow._resolve_gene_filtering_params("target", params, config)

    params["n_sample_for_pca"] = [1]
    params["incremental_pca_batch_size"] = 3
    with pytest.raises(ValueError, match="at least twice"):
        workflow._resolve_gene_filtering_params("target", params, config)

    params["plot_pca_for_comparison"] = False
    params["n_sample_for_pca"] = [4]
    assert workflow._resolve_gene_filtering_params(
        "target", params, config
    ).n_sample_for_pca is None


def test_pca_sample_indices_are_reproducible_and_nested():
    first = workflow._nested_pca_sample_indices(50, (10, 25, 40), 17)
    second = workflow._nested_pca_sample_indices(50, (10, 25, 40), 17)

    for count in first:
        np.testing.assert_array_equal(first[count], second[count])
    assert set(first[10]).issubset(first[25])
    assert set(first[25]).issubset(first[40])
    assert all(np.all(values[:-1] < values[1:]) for values in first.values())


def test_pca_batch_slices_respect_size_and_minimum_batch():
    for n_rows in (4, 6, 9, 17):
        slices = workflow._pca_batch_slices(n_rows, batch_size=8, min_batch_size=2)
        sizes = [stop - start for start, stop in slices]
        assert sum(sizes) == n_rows
        assert min(sizes) >= 2
        assert max(sizes) <= 8


def test_sample_count_pca_streams_rows_and_saves_separate_artifacts(
    tmp_path, monkeypatch
):
    genes = ["g1", "g2"]
    sample_ids = [f"sim_{i}" for i in range(6)]
    linear = np.array(
        [[i + 1, 10 - i] for i in range(6)],
        dtype=np.float32,
    )
    simulated_log = np.log2(linear / linear.sum(axis=1, keepdims=True) * 1e6 + 1)
    bulk = ad.AnnData(
        X=simulated_log,
        obs=pd.DataFrame(index=sample_ids),
        var=pd.DataFrame(index=genes),
    )
    generated_file = tmp_path / "generated.h5ad"
    bulk.write_h5ad(generated_file)

    reference_linear = pd.DataFrame(
        [[2.0, 8.0], [6.0, 4.0], [9.0, 1.0]],
        index=["tcga_1", "tcga_2", "tcga_3"],
        columns=genes,
    )
    reference_normalized = normalize_selected(reference_linear, genes)
    reference_log = pd.DataFrame(
        np.log2(reference_normalized.to_numpy(dtype=np.float32) + 1),
        index=reference_linear.index,
        columns=genes,
    )
    params = SimpleNamespace(
        n_sample_for_pca=(3, 6),
        pca_sample_seed=17,
        incremental_pca_batch_size=4,
        pca_n_components=2,
        pca_figsize=(5.0, 5.0),
    )
    plotted = []

    def fake_plot(data, result_fp, color_code, **kwargs):
        plotted.append((data.copy(), list(color_code)))
        output = Path(result_fp.replace(".png", "_PC0_PC1.png"))
        output.touch()

    monkeypatch.setattr(workflow, "plot_pca", fake_plot)
    runs = workflow._run_sample_count_pca(
        generated_file,
        reference_log,
        genes,
        "target",
        tmp_path / "target",
        params,
        simulated_is_normalized_log=True,
    )

    assert [run["sample_count"] for run in runs] == [3, 6]
    assert len(plotted) == 2
    assert len(plotted[0][0]) == 3 + len(reference_log)
    assert len(plotted[1][0]) == 6 + len(reference_log)
    assert plotted[0][1].count("TCGA") == len(reference_log)
    assert plotted[1][1].count("TCGA") == len(reference_log)
    assert set(plotted[0][0].index).intersection(sample_ids).issubset(
        set(plotted[1][0].index)
    )
    assert all(Path(run["pca_model_file"]).parent.name == "pca_by_sample_count" for run in runs)
    assert all(Path(run["pca_data_file"]).is_file() for run in runs)
    assert all(Path(run["pca_plot_file"]).is_file() for run in runs)
    assert all(Path(run["pca_gene_loadings_PC1_file"]).is_file() for run in runs)


def test_sample_count_pca_rejects_count_above_actual_h5ad_size(tmp_path, monkeypatch):
    generated_file = tmp_path / "small.h5ad"
    ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["s1", "s2", "s3"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(generated_file)
    params = SimpleNamespace(
        n_sample_for_pca=(4,),
        pca_sample_seed=1,
        incremental_pca_batch_size=4,
        pca_n_components=2,
        pca_figsize=(5.0, 5.0),
    )
    monkeypatch.setattr(workflow, "plot_pca", lambda **kwargs: None)
    reference = pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0]],
        index=["tcga_1", "tcga_2"],
        columns=["g1", "g2"],
    )

    with pytest.raises(ValueError, match="exceed the generated dataset sample count"):
        workflow._run_sample_count_pca(
            generated_file,
            reference,
            ["g1", "g2"],
            "target",
            tmp_path / "target",
            params,
            simulated_is_normalized_log=True,
        )


def test_fixed_filter_sample_pca_writes_filtered_h5ad_from_backed_input(
    tmp_path, monkeypatch
):
    genes = ["g1", "g2"]
    reference_linear = pd.DataFrame(
        [[2.0, 8.0], [6.0, 4.0], [9.0, 1.0]],
        index=["tcga_1", "tcga_2", "tcga_3"],
        columns=genes,
    )
    reference_path = tmp_path / "reference.csv"
    reference_linear.to_csv(reference_path)
    space = FixedGeneFilter(
        genes,
        reference_linear.iloc[:1],
        reference_path,
        ["A"],
        gep_params(
            reference_file=str(reference_path),
            ref_exp_type="TPM",
            filtering_ref_types=["A"],
        ),
        tmp_path / "target",
        len(genes),
        len(genes),
    )
    space.persist_genes()

    bulk_linear = np.array(
        [[1.0, 9.0], [3.0, 7.0], [7.0, 3.0], [9.0, 1.0]],
        dtype=np.float32,
    )
    bulk_log = np.log2(
        bulk_linear / bulk_linear.sum(axis=1, keepdims=True) * 1e6 + 1
    )
    generated_file = tmp_path / "bulk.h5ad"
    ad.AnnData(
        X=bulk_log,
        obs=pd.DataFrame(index=[f"sim_{i}" for i in range(4)]),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(generated_file)
    gene_params = SimpleNamespace(
        enabled=True,
        save_filtered_h5ad=True,
        filtered_dataset_postfix=None,
        plot_pca=True,
        pca_n_components=2,
        pca_figsize=(5.0, 5.0),
        n_sample_for_pca=(2,),
        pca_sample_seed=3,
        incremental_pca_batch_size=4,
    )

    def fake_plot(data, result_fp, color_code, **kwargs):
        Path(result_fp.replace(".png", "_PC0_PC1.png")).touch()

    monkeypatch.setattr(workflow, "plot_pca", fake_plot)
    outputs = workflow._postprocess_fixed_gene_filter(
        "target", generated_file, gene_params, space
    )

    filtered = ad.read_h5ad(outputs["filtered_bulk_gep_file"])
    assert filtered.shape == (4, 2)
    assert outputs["pca_runs"][0]["sample_count"] == 2
    coordinates = pd.read_csv(outputs["pca_runs"][0]["pca_data_file"], index_col=0)
    assert len(coordinates) == 2 + len(reference_linear)
    assert set(reference_linear.index).issubset(coordinates.index)


def test_unfiltered_comparison_uses_sample_count_pca(tmp_path, monkeypatch):
    genes = ["g1", "g2", "g3"]
    bulk = ad.AnnData(
        X=np.log2(
            np.array(
                [[1, 2, 7], [2, 5, 3], [8, 1, 1], [4, 4, 2]],
                dtype=np.float32,
            )
            + 1
        ),
        obs=pd.DataFrame(index=[f"sim_{i}" for i in range(4)]),
        var=pd.DataFrame(index=genes),
    )
    generated_file = tmp_path / "unfiltered.h5ad"
    bulk.write_h5ad(generated_file)
    reference_path = tmp_path / "reference.csv"
    pd.DataFrame(
        [[1, 6, 3], [2, 2, 6], [7, 2, 1]],
        index=["tcga_1", "tcga_2", "tcga_3"],
        columns=genes,
    ).to_csv(reference_path)
    params = SimpleNamespace(
        correlation_reference_dataset=None,
        enabled=False,
        plot_pca=True,
        n_sample_for_pca=(2,),
        pca_sample_seed=9,
        incremental_pca_batch_size=4,
        pca_n_components=2,
        pca_figsize=(5.0, 5.0),
        reference_file=str(reference_path),
    )

    def fake_plot(data, result_fp, color_code, **kwargs):
        Path(result_fp.replace(".png", "_PC0_PC1.png")).touch()

    monkeypatch.setattr(workflow, "plot_pca", fake_plot)
    outputs = workflow._run_gene_filtering_postprocess(
        "unfiltered",
        generated_file,
        params={},
        config=SimpleNamespace(),
        resolved_params=params,
    )

    assert outputs["pca_runs"][0]["sample_count"] == 2
    assert len(pd.read_csv(outputs["pca_runs"][0]["pca_data_file"])) == 5
