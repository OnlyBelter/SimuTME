import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import simutme.workflow as workflow
from simutme.simulation.filtering import FixedGeneFilter


QUANTILES = [0.05, 0.5, 0.95]
GENES = ["g1", "g2", "g3"]


def _make_filtering_space(tmp_path):
    reference = pd.DataFrame(
        {
            "g1": [100_000.0, 200_000.0, 300_000.0],
            "g2": [100_000.0, 200_000.0, 300_000.0],
            "g3": [800_000.0, 600_000.0, 400_000.0],
        },
        index=["tcga_1", "tcga_2", "tcga_3"],
    )
    reference.to_csv(tmp_path / "tcga.csv")
    gep = SimpleNamespace(
        enabled=True,
        reference_file=str(tmp_path / "tcga.csv"),
        ref_exp_type="TPM",
        filtering_method="median_gep",
        filtering_ref_types=None,
        gep_filtering_quantile=(None, 1.0),
        n_top=20,
        show_filtering_info=False,
        filtering_by_gene_range=False,
        min_percentage_within_gene_range=0.95,
        gene_quantile_range=None,
        filtering_in_pca_space=True,
        pca_n_components=1,
        norm_ord=1,
    )
    space = FixedGeneFilter(
        GENES,
        reference,
        tmp_path / "tcga.csv",
        None,
        gep,
        tmp_path / "Target",
        generation_n_genes=3,
        shared_n_genes=3,
    )
    space.persist_genes()
    return space


def _make_gene_params(**overrides):
    values = {
        "enabled": True,
        "reference_file": "tcga.csv",
        "filtering_type": "high_corr_gene_and_quantile_range",
        "quantile_range": QUANTILES,
        "q_col_name": workflow._resolve_quantile_col_names(
            None, QUANTILES, "gene_filtering_quantile_range"
        ),
        "corr_threshold": 0.3,
        "n_gene_max": 1000,
        "high_corr_gene_file": None,
        "save_filtered_h5ad": True,
        "filtered_dataset_postfix": None,
        "plot_pca": False,
        "pca_n_components": 2,
        "pca_figsize": (5.0, 5.0),
        "n_sample_for_pca": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_primary_h5ad(path, expression):
    fractions = pd.DataFrame(
        {"A": [0.2, 0.5, 0.3], "B": [0.8, 0.5, 0.7]},
        index=["d1_1", "d1_2", "d1_3"],
    )
    ad.AnnData(
        X=np.log2(expression + 1.0).astype(np.float32),
        obs=fractions,
        var=pd.DataFrame(index=GENES),
    ).write_h5ad(path)
    return fractions


def test_combined_mode_writes_separate_renormalized_h5ad_and_summary(tmp_path):
    space = _make_filtering_space(tmp_path)
    generated_file = tmp_path / "accepted_mixed.h5ad"
    expression = np.array(
        [
            [200_000.0, 400_000.0, 400_000.0],
            [200_000.0, 300_000.0, 500_000.0],
            [200_000.0, 500_000.0, 300_000.0],
        ]
    )
    fractions = _write_primary_h5ad(generated_file, expression)
    primary_before = generated_file.read_bytes()

    outputs = workflow._postprocess_fixed_gene_filter(
        "Target",
        generated_file,
        _make_gene_params(),
        space,
    )

    assert outputs["post_gep_quantile_pca_plot_file"] is None
    filtered_file = Path(outputs["filtered_bulk_gep_file"])
    filtered = ad.read_h5ad(filtered_file)
    assert filtered.var_names.tolist() == ["g1"]
    assert filtered.obs_names.tolist() == fractions.index.tolist()
    pd.testing.assert_frame_equal(filtered.obs, fractions)
    linear = np.expm1(np.asarray(filtered.X, dtype=np.float64) * np.log(2.0))
    np.testing.assert_allclose(linear.sum(axis=1), 1_000_000.0, rtol=1e-6)
    np.testing.assert_allclose(
        filtered.X[:, 0],
        np.full(3, np.log2(1_000_001.0)),
        rtol=1e-6,
    )
    assert outputs["fixed_gene_list_file"] == str(space.gene_list_file)
    assert pd.read_csv(outputs["gene_list_file"])["gene_name"].tolist() == ["g1"]
    assert generated_file.read_bytes() == primary_before
    assert filtered_file != generated_file

    metadata = json.loads(filtered_file.with_suffix(".metadata.json").read_text())
    assert metadata["identity"]["gene_filtering_quantile_range"] == QUANTILES
    assert metadata["identity"]["final_gene_list_hash"]
    filtered_before = filtered_file.read_bytes()
    reused = workflow._postprocess_fixed_gene_filter(
        "Target",
        generated_file,
        _make_gene_params(),
        space,
    )
    assert reused["filtered_bulk_gep_file"] == str(filtered_file)

    changed_range = [0.0, 0.5, 1.0]
    changed_params = _make_gene_params(
        quantile_range=changed_range,
        q_col_name=workflow._resolve_quantile_col_names(
            None, changed_range, "gene_filtering_quantile_range"
        ),
    )
    with pytest.raises(ValueError, match="Incompatible filtering metadata"):
        workflow._postprocess_fixed_gene_filter(
            "Target",
            generated_file,
            changed_params,
            space,
        )
    assert filtered_file.read_bytes() == filtered_before
    assert generated_file.read_bytes() == primary_before

    fraction_file = tmp_path / "cell_fractions.csv"
    fractions.to_csv(fraction_file, index_label="sample_id")
    summary_path = workflow._build_mixed_dataset_summary(
        dataset_name="Target",
        generated_bulk_gep_fp=generated_file,
        generated_cell_fraction_fp=fraction_file,
        gep_filtering=SimpleNamespace(enabled=False, filtering_method="median_gep"),
        gene_filtering=SimpleNamespace(
            enabled=True,
            filtering_type="high_corr_gene_and_quantile_range",
            correlation_reference_dataset="source",
        ),
        postprocess_outputs=outputs,
    )
    summary = json.loads(summary_path.read_text())
    assert summary["filtered_bulk_gep_file"] == str(filtered_file)
    assert summary["gene_list_file"] == outputs["gene_list_file"]
    assert summary["fixed_gene_list_file"] == outputs["fixed_gene_list_file"]
    assert summary["post_gep_gene_quantile_filter"]["gene_count_removed"] == 2


@pytest.mark.parametrize("expression", [1.0, 3.0])
def test_quantile_gene_selector_keeps_values_equal_to_reference_bounds(expression):
    reference = pd.DataFrame({"g1": [1.0, 2.0, 3.0]})
    d1 = pd.DataFrame({"g1": [expression, expression]})

    genes = workflow.get_gene_list_filtered_by_quantile_range(
        bulk_exp=d1,
        tcga_exp=reference,
        quantile_range=[0.0, 0.5, 1.0],
        q_col_name=workflow._resolve_quantile_col_names(
            None, [0.0, 0.5, 1.0], "gene_filtering_quantile_range"
        ),
    )

    assert genes == ["g1"]


def test_combined_mode_fails_when_no_genes_pass(tmp_path, monkeypatch):
    space = _make_filtering_space(tmp_path)
    generated_file = tmp_path / "accepted_mixed.h5ad"
    _write_primary_h5ad(
        generated_file,
        np.array(
            [
                [200_000.0, 400_000.0, 400_000.0],
                [200_000.0, 300_000.0, 500_000.0],
                [200_000.0, 500_000.0, 300_000.0],
            ]
        ),
    )
    monkeypatch.setattr(workflow, "get_gene_list_filtered_by_quantile_range", lambda **kwargs: [])

    with pytest.raises(ValueError, match="retained no genes"):
        workflow._postprocess_fixed_gene_filter(
            "Target",
            generated_file,
            _make_gene_params(),
            space,
        )
    assert not generated_file.with_name(
        f"{generated_file.stem}_filtered_by_high_corr_gene_and_quantile_range.h5ad"
    ).exists()


def test_combined_mode_fails_when_a_sample_has_zero_expression_on_final_genes(
    tmp_path,
):
    space = _make_filtering_space(tmp_path)
    generated_file = tmp_path / "accepted_mixed.h5ad"
    _write_primary_h5ad(
        generated_file,
        np.array(
            [
                [0.0, 600_000.0, 400_000.0],
                [200_000.0, 300_000.0, 500_000.0],
                [200_000.0, 500_000.0, 300_000.0],
            ]
        ),
    )

    with pytest.raises(ValueError, match="d1_1"):
        workflow._postprocess_fixed_gene_filter(
            "Target",
            generated_file,
            _make_gene_params(),
            space,
        )


def test_combined_mode_adds_separate_post_quantile_pca_figure(tmp_path, monkeypatch):
    space = _make_filtering_space(tmp_path)
    generated_file = tmp_path / "accepted_mixed.h5ad"
    expression = np.array(
        [
            [200_000.0, 400_000.0, 400_000.0],
            [200_000.0, 300_000.0, 500_000.0],
            [200_000.0, 500_000.0, 300_000.0],
        ]
    )
    fractions = _write_primary_h5ad(generated_file, expression)
    monkeypatch.setattr(
        workflow,
        "get_gene_list_filtered_by_quantile_range",
        lambda **kwargs: ["g1", "g2"],
    )

    gene_params = _make_gene_params(plot_pca=True)
    outputs = workflow._postprocess_fixed_gene_filter(
        "Target",
        generated_file,
        gene_params,
        space,
    )

    pre_plot = Path(outputs["pca_plot_file"])
    post_plot = Path(outputs["post_gep_quantile_pca_plot_file"])
    pre_model = Path(outputs["pca_model_file"])
    post_model = Path(outputs["post_gep_quantile_pca_model_file"])
    post_coordinates = pd.read_csv(
        outputs["post_gep_quantile_pca_data_file"],
        index_col=0,
    )
    assert pre_plot.is_file()
    assert post_plot.is_file()
    assert pre_plot != post_plot
    assert pre_model.is_file()
    assert post_model.is_file()
    assert pre_model != post_model
    assert workflow.load(post_model).n_features_in_ == 2
    assert len(post_coordinates) == len(fractions) + len(space.reference_log)
    assert post_coordinates.columns.tolist() == ["PC1", "PC2"]
    assert post_model.with_suffix(".metadata.json").is_file()
    reused = workflow._postprocess_fixed_gene_filter(
        "Target",
        generated_file,
        gene_params,
        space,
    )
    assert reused["post_gep_quantile_pca_plot_file"] == str(post_plot)

    fraction_file = tmp_path / "cell_fractions.csv"
    fractions.to_csv(fraction_file, index_label="sample_id")
    summary_path = workflow._build_mixed_dataset_summary(
        dataset_name="Target",
        generated_bulk_gep_fp=generated_file,
        generated_cell_fraction_fp=fraction_file,
        gep_filtering=SimpleNamespace(enabled=False, filtering_method="median_gep"),
        gene_filtering=SimpleNamespace(
            enabled=True,
            filtering_type="high_corr_gene_and_quantile_range",
            correlation_reference_dataset="source",
        ),
        postprocess_outputs=outputs,
    )
    summary = json.loads(summary_path.read_text())
    assert summary["pca_plot_file"] == str(pre_plot)
    assert summary["post_gep_quantile_pca_plot_file"] == str(post_plot)
    assert summary["post_gep_quantile_pca_model_file"] == str(post_model)


def test_high_corr_mode_does_not_create_post_quantile_pca_artifacts(tmp_path):
    space = _make_filtering_space(tmp_path)
    generated_file = tmp_path / "accepted_mixed.h5ad"
    _write_primary_h5ad(
        generated_file,
        np.array(
            [
                [200_000.0, 400_000.0, 400_000.0],
                [200_000.0, 300_000.0, 500_000.0],
                [200_000.0, 500_000.0, 300_000.0],
            ]
        ),
    )

    outputs = workflow._postprocess_fixed_gene_filter(
        "Target",
        generated_file,
        _make_gene_params(filtering_type="high_corr_gene"),
        space,
    )

    assert outputs["post_gep_quantile_pca_plot_file"] is None
    assert outputs["post_gep_quantile_pca_model_file"] is None
    assert outputs["post_gep_quantile_pca_data_file"] is None
    assert not list(space.result_dir.glob("*post_gep_quantile_range*"))
