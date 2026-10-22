from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import simutme.workflow as workflow
from simutme.simulation.filtering import normalize_selected, reference_variance_mask
from tests.test_correlation_first_filtering import workflow_case, gep_params


@pytest.mark.parametrize("threshold", [True, -1, np.nan, np.inf, "1"])
def test_invalid_variance_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        reference_variance_mask(pd.DataFrame({"a": [1, 2]}), threshold)


def test_linear_population_variance_and_normalization():
    reference = pd.DataFrame({"zero": [0, 0], "constant": [10, 10],
                              "boundary": [0, 2], "keep": [0, 4], "also_keep": [4, 0]})
    variance, keep = reference_variance_mask(reference, 1.0)
    assert variance["boundary"] == 1
    assert variance.index[keep].tolist() == ["keep", "also_keep"]
    normalized = normalize_selected(reference, variance.index[keep].tolist())
    np.testing.assert_allclose(normalized.sum(axis=1), 1e6)
    assert reference["constant"].tolist() == [10, 10]


def test_workflow_variance_metadata_and_cache_identity(workflow_case):
    config, params, _, _ = workflow_case
    params["gep_filtering"] = True
    params["gene_filtering_reference_variance_threshold"] = 1.0
    gene = workflow._resolve_gene_filtering_params("target", params, config)
    reference = pd.read_csv(config.tcga_merged_tpm_file_path, index_col=0)
    sct = params["sct_reference"]["file_path"]
    directory = Path(config.simu_bulk_dir) / "variance"
    space = workflow._prepare_fixed_gene_filter(
        "target", sct, ["CT1", "CT2"],
        gep_params(reference_file=str(config.tcga_merged_tpm_file_path)), gene, config, directory)
    assert space.genes == ["a", "b"]
    assert space.pca_identity["reference_variance_threshold"] == 1
    assert space.reference_variance_metadata["reference_variance_removed_n_genes"] == 1
    np.testing.assert_allclose(space.reference_linear.sum(1), 1e6)
    space.persist_genes()
    audit = pd.read_csv(directory / "reference_gene_variance.csv", index_col=0)
    assert not audit.loc["unused", "variance_pass"]
    space._prepare_pca()
    with pytest.raises(ValueError, match="Incompatible"):
        workflow._prepare_fixed_gene_filter(
            "target", sct, ["CT1", "CT2"],
            gep_params(reference_file=str(config.tcga_merged_tpm_file_path)),
            replace(gene, reference_variance_threshold=0.5), config, directory)
    pd.testing.assert_frame_equal(reference, pd.read_csv(config.tcga_merged_tpm_file_path, index_col=0))


def test_local_selection_excludes_heldout_information(workflow_case, monkeypatch):
    config, params, _, _ = workflow_case
    reference = pd.DataFrame({"a": [1, 3, 100], "b": [5, 1, 3],
                              "unused": [10, 10, 500]}, index=["r1", "r2", "r3"])
    reference.to_csv(config.tcga_merged_tpm_file_path)
    split = pd.DataFrame({"cancer_type": ["A"] * 3,
                          "role": ["fitting", "fitting", "held_out"]}, index=reference.index)
    monkeypatch.setattr(workflow, "split_tcga_reference",
                        lambda *args: (reference, split, pd.DataFrame()))
    params["gene_filtering_reference_variance_threshold"] = 1
    gene = workflow._resolve_gene_filtering_params("target", params, config)
    space = workflow._prepare_fixed_gene_filter(
        "target", params["sct_reference"]["file_path"], ["CT1", "CT2"],
        gep_params(filtering_method="knn", knn_n_neighbors=1,
                   reference_file=str(config.tcga_merged_tpm_file_path)), gene, config,
        Path(config.simu_bulk_dir) / "local_variance")
    assert space.genes == ["b"]  # a is exactly at the cutoff on fitting rows.
    assert space.calibration_identity["reference_variance_scope"] == "fitting-only"
    assert space.reference_variance_metadata["reference_variance_n_samples"] == 2


def test_pca_loading_export_order_and_values(tmp_path):
    model = SimpleNamespace(components_=np.array([[0.1, -0.9, 0.1], [-0.2, 0.8, 0.3]]))
    paths = workflow._export_pca_gene_loadings(model, ["b", "a", "c"], tmp_path / "comparison_pca_2.joblib")
    pc1 = pd.read_csv(paths["pca_gene_loadings_PC1_file"])
    assert pc1.gene_name.tolist() == ["b", "c", "a"]
    assert len(list(tmp_path.glob("*gene_loadings*.csv"))) == 1
    assert "pca_gene_loadings_PC2_file" not in paths
    np.testing.assert_allclose(pc1.set_index("gene_name").loc["a"], [-0.9, 0.8])


def test_end_to_end_variance_summary_and_cached_loading_exports(workflow_case):
    import json
    from joblib import load

    config, params, _, _ = workflow_case
    params.update(gep_filtering=True, gene_filtering_reference_variance_threshold=1.0,
                  plot_pca_for_comparison=True, gene_filtering_pca_n_components=2)
    output = workflow.run_bulk_simulation("target", params, config)
    directory = Path(output).parent / "target"
    summary = json.loads((directory / "mixed_dataset_summary.json").read_text())
    assert summary["reference_variance_removed_n_genes"] == 1
    model = load(summary["pca_model_file"])
    path = Path(summary["pca_gene_loadings_PC1_file"])
    values = pd.read_csv(path)
    assert len(values) == 2 and values["PC1_loading"].is_monotonic_decreasing
    np.testing.assert_allclose(values.set_index("gene_name").loc[["a", "b"], "PC1_loading"],
                               model.components_[0])
    np.testing.assert_allclose(values.set_index("gene_name").loc[["a", "b"], "PC2_loading"],
                               model.components_[1])
    assert "pca_gene_loadings_PC2_file" not in summary
    assert len(list(directory.glob("*gene_loadings*.csv"))) == 1
    path.unlink()
    workflow.run_bulk_simulation("target", params, config)
    assert Path(summary["pca_gene_loadings_PC1_file"]).exists()


def test_threshold_default_override_and_invalid_config(workflow_case):
    config, params, _, _ = workflow_case
    assert workflow._resolve_gene_filtering_params("target", params, config).reference_variance_threshold is None
    config.gene_filtering_reference_variance_threshold = 1
    assert workflow._resolve_gene_filtering_params("target", params, config).reference_variance_threshold == 1
    params["gene_filtering_reference_variance_threshold"] = None
    assert workflow._resolve_gene_filtering_params("target", params, config).reference_variance_threshold is None
    params["gene_filtering_reference_variance_threshold"] = True
    with pytest.raises(ValueError, match="threshold"):
        workflow._resolve_gene_filtering_params("target", params, config)
