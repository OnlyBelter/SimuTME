from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np
import pandas as pd
import pytest

import simutme.workflow as workflow_module
from simutme.workflow import (
    ResolvedGEPFilteringParams,
    ResolvedGeneFilteringParams,
    _build_mixed_dataset_summary,
    _resolve_gep_filtering_params,
    _resolve_gene_filtering_params,
    _run_gene_filtering_postprocess,
    filtering_by_gene_list_and_pca_plot,
)


def _make_filtering_config(tmp_path: Path):
    tcga_file = tmp_path / "tcga.csv"
    tcga_file.write_text("sample_id,G1,G2\ns1,1,2\n", encoding="utf-8")
    tcga_map_file = tmp_path / "tcga_map.csv"
    pd.DataFrame({"cancer_type": ["LUAD"]}, index=["s1"]).to_csv(tcga_map_file)
    high_corr_file = tmp_path / "high_corr.csv"
    pd.DataFrame({"gene_name": ["G1", "G2"]}).to_csv(high_corr_file, index=False)
    return SimpleNamespace(
        gep_filtering_enabled_default=False,
        gep_filtering_reference_file=None,
        gep_filtering_ref_exp_type=None,
        tcga_merged_tpm_file_path=tcga_file,
        high_corr_gene_file_path=high_corr_file,
        gep_filtering_quantile=(0.0, 0.95),
        gep_filtering_method="median_gep",
        gep_n_top=20,
        gep_show_filtering_info=False,
        gep_filtering_by_gene_range=False,
        gep_min_percentage_within_gene_range=0.95,
        gep_gene_quantile_range=None,
        gene_quantile_range=[0.005, 0.5, 0.995],
        gep_filtering_in_pca_space=True,
        gep_pca_n_components=0.9,
        gep_norm_ord=1,
        tcga2cancer_type_file_path=tcga_map_file,
        gene_filtering_enabled_default=False,
        gene_filtering_reference_file=None,
        gene_filtering_type=None,
        gene_list_type="high_corr_gene_and_quantile_range",
        gene_filtering_quantile_range=None,
        gene_filtering_q_col_name=None,
        gene_filtering_corr_threshold=0.3,
        gene_filtering_n_gene_max=1000,
        gene_filtering_save_filtered_h5ad=True,
        gene_filtering_filtered_dataset_postfix=None,
        calculate_gene_cell_prop_correlation=False,
        gene_filtering_correlation_reference_dataset=None,
        gene_filtering_plot_pca=True,
        gene_filtering_pca_n_components=2,
        gene_filtering_pca_figsize=(5.0, 5.0),
        _dataset_definitions={},
        _dataset_order=[],
        _generated_mixed_outputs={},
    )


def test_resolve_gep_filtering_params_uses_dataset_overrides_and_legacy_gene_range(tmp_path: Path):
    config = _make_filtering_config(tmp_path)

    resolved = _resolve_gep_filtering_params(
        dataset_name="Mixed_Test",
        params={
            "gep_filtering": True,
            "gep_filtering_method": "mean_gep",
            "gep_n_top": 33,
            "filtering_in_pca_space": False,
            "gep_filtering_by_gene_range": True,
        },
        config=config,
    )

    assert resolved.enabled is True
    assert resolved.reference_file == str(config.tcga_merged_tpm_file_path)
    assert resolved.ref_exp_type == "TPM"
    assert resolved.filtering_method == "mean_gep"
    assert resolved.n_top == 33
    assert resolved.filtering_in_pca_space is False
    assert resolved.gene_quantile_range == [0.005, 0.5, 0.995]
    assert resolved.high_corr_gene_list == ["G1", "G2"]


def test_resolve_gene_filtering_params_uses_explicit_gene_filtering_switch(tmp_path: Path):
    config = _make_filtering_config(tmp_path)

    resolved = _resolve_gene_filtering_params(
        dataset_name="Mixed_Test",
        params={"gene_filtering": True},
        config=config,
    )

    assert resolved.enabled is True
    assert resolved.reference_file == str(config.tcga_merged_tpm_file_path)
    assert resolved.filtering_type == "high_corr_gene_and_quantile_range"
    assert resolved.quantile_range == [0.005, 0.5, 0.995]
    assert resolved.q_col_name == ["q_0.5", "q_50", "q_99.5"]
    assert resolved.high_corr_gene_file == str(config.high_corr_gene_file_path)


def test_resolve_gene_filtering_params_prefers_dataset_overrides_over_global_defaults(tmp_path: Path):
    config = _make_filtering_config(tmp_path)
    config.gene_filtering_corr_threshold = 0.2
    config.gene_filtering_n_gene_max = 1600
    config.gene_filtering_plot_pca = False
    config.gene_filtering_pca_figsize = (5.0, 5.0)

    resolved = _resolve_gene_filtering_params(
        dataset_name="Mixed_Test",
        params={
            "gene_filtering": True,
            "gene_filtering_corr_threshold": 0.15,
            "gene_filtering_n_gene_max": 2000,
            "plot_pca_for_comparison": True,
            "gene_filtering_pca_figsize": [6, 4],
        },
        config=config,
    )

    assert resolved.corr_threshold == 0.15
    assert resolved.n_gene_max == 2000
    assert resolved.plot_pca is True
    assert resolved.pca_figsize == (6.0, 4.0)


def test_resolve_gene_filtering_params_validates_pca_components(tmp_path: Path):
    config = _make_filtering_config(tmp_path)

    with pytest.raises(ValueError, match="must be >= 2"):
        _resolve_gene_filtering_params(
            dataset_name="Mixed_Test",
            params={
                "gene_filtering": True,
                "gene_filtering_plot_pca": True,
                "gene_filtering_pca_n_components": 1,
            },
            config=config,
        )


def test_resolve_gep_filtering_params_allows_missing_optional_high_corr_file(tmp_path: Path):
    config = _make_filtering_config(tmp_path)
    config.high_corr_gene_file_path = tmp_path / "missing_high_corr.csv"

    resolved = _resolve_gep_filtering_params(
        dataset_name="Mixed_Test",
        params={},
        config=config,
    )

    assert resolved.high_corr_gene_list_file is None
    assert resolved.high_corr_gene_list is None


def test_resolve_gene_filtering_params_allows_missing_optional_high_corr_file(tmp_path: Path):
    config = _make_filtering_config(tmp_path)
    config.high_corr_gene_file_path = tmp_path / "missing_high_corr.csv"

    resolved = _resolve_gene_filtering_params(
        dataset_name="Mixed_Test",
        params={
            "gene_filtering": True,
            "gene_filtering_type": "high_corr_gene",
        },
        config=config,
    )

    assert resolved.enabled is True
    assert resolved.high_corr_gene_file is None


def test_resolve_gene_filtering_params_accepts_segment_correlation_reference(tmp_path: Path):
    config = _make_filtering_config(tmp_path)
    config.high_corr_gene_file_path = None
    config._dataset_order = ["Mixed_Correlation_Source", "Mixed_Target"]
    config._dataset_definitions = {
        "Mixed_Correlation_Source": {
            "dataset_type": "Mixed",
            "sampling_method_prop": "segment",
            "gep_filtering": False,
            "gene_filtering": False,
            "calculate_gene_cell_prop_correlation": True,
        },
        "Mixed_Target": {"dataset_type": "Mixed"},
    }

    resolved = _resolve_gene_filtering_params(
        dataset_name="Mixed_Target",
        params={
            "gene_filtering": True,
            "gene_filtering_type": "high_corr_gene",
            "gene_filtering_correlation_reference_dataset": "Mixed_Correlation_Source",
        },
        config=config,
    )

    assert resolved.correlation_reference_dataset == "Mixed_Correlation_Source"


def test_resolve_gene_filtering_params_rejects_non_segment_correlation_reference(tmp_path: Path):
    config = _make_filtering_config(tmp_path)
    config.high_corr_gene_file_path = None
    config._dataset_order = ["Mixed_Correlation_Source", "Mixed_Target"]
    config._dataset_definitions = {
        "Mixed_Correlation_Source": {
            "dataset_type": "Mixed",
            "sampling_method_prop": "dirichlet",
            "gep_filtering": False,
            "gene_filtering": False,
        },
        "Mixed_Target": {"dataset_type": "Mixed"},
    }

    with pytest.raises(ValueError, match="sampling_method_prop: segment"):
        _resolve_gene_filtering_params(
            dataset_name="Mixed_Target",
            params={
                "gene_filtering": True,
                "gene_filtering_type": "high_corr_gene",
                "gene_filtering_correlation_reference_dataset": "Mixed_Correlation_Source",
            },
            config=config,
        )


def test_full_gene_correlation_file_contains_unfiltered_genes_and_cell_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    generated_h5ad = tmp_path / "source.h5ad"
    gene_exp = pd.DataFrame(
        {"G1": [1.0, 2.0, 3.0, 4.0], "G2": [4.0, 3.0, 2.0, 1.0]},
        index=["s1", "s2", "s3", "s4"],
    )
    cell_frac = pd.DataFrame(
        {"A": [0.0, 0.25, 0.5, 0.75], "B": [0.75, 0.5, 0.25, 0.0]},
        index=gene_exp.index,
    )

    class FakeReadH5AD:
        def __init__(self, file_path):
            self.file_path = file_path

        def get_df(self, convert_to_tpm=False):
            return gene_exp

        def get_cell_fraction(self):
            return cell_frac

    monkeypatch.setattr(workflow_module, "ReadH5AD", FakeReadH5AD)
    correlation_file = workflow_module._calculate_full_gene_cell_prop_correlation(
        generated_bulk_gep_fp=generated_h5ad,
        result_file_path=tmp_path / "full_gene_corr_with_cell_fraction.csv",
    )

    result = pd.read_csv(correlation_file, index_col=0)
    assert result.index.tolist() == ["G1", "G2"]
    assert result.columns.tolist() == ["A", "B"]
    assert "n_at_least_one" not in result.columns


def test_gene_filtering_reuses_full_correlation_with_new_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    generated_h5ad = tmp_path / "target.h5ad"
    genes = ["G1", "G2", "G3", "G4"]
    bulk_exp = pd.DataFrame(
        {gene: [1.0, 2.0, 3.0, 4.0] for gene in genes},
        index=["s1", "s2", "s3", "s4"],
    )
    target_cell_frac = pd.DataFrame(
        {"A": [0.0, 0.3, 0.6, 0.9], "B": [0.9, 0.6, 0.3, 0.0]},
        index=bulk_exp.index,
    )
    tcga_exp = pd.DataFrame({gene: [1.0, 2.0, 4.0] for gene in genes}, index=["t1", "t2", "t3"])
    raw_corr = pd.DataFrame(
        {
            "A": [0.95, 0.82, 0.75, 0.20],
            "B": [0.10, 0.90, 0.40, 0.85],
        },
        index=genes,
    )
    raw_corr_file = tmp_path / "full_gene_corr_with_cell_fraction.csv"
    raw_corr.to_csv(raw_corr_file)

    class FakeReadH5AD:
        def __init__(self, file_path):
            self.file_path = file_path

        def get_df(self, convert_to_tpm=False):
            return bulk_exp

        def get_cell_fraction(self):
            return target_cell_frac

    class FakeReadExp:
        def __init__(self, exp_file, exp_type=None):
            self._df = tcga_exp.copy()

        def align_with_gene_list(self, gene_list, fill_not_exist=False):
            self._df = self._df.loc[:, [gene for gene in gene_list if gene in self._df.columns]]

        def to_log2cpm1p(self):
            self._df = np.log2(self._df + 1.0)

        def get_exp(self):
            return self._df

    monkeypatch.setattr(workflow_module, "ReadH5AD", FakeReadH5AD)
    monkeypatch.setattr(workflow_module, "ReadExp", FakeReadExp)

    low_threshold_genes = workflow_module.get_gene_list_for_filtering(
        bulk_exp_file=generated_h5ad,
        tcga_file=tmp_path / "tcga.csv",
        result_file_path=tmp_path / "low_threshold_genes.csv",
        filtering_type="high_corr_gene",
        corr_threshold=0.5,
        n_gene_max=2,
        corr_result_fp=str(tmp_path / "low_threshold_correlations.csv"),
        correlation_reference_file=raw_corr_file,
        force_recompute=True,
    )
    high_threshold_genes = workflow_module.get_gene_list_for_filtering(
        bulk_exp_file=generated_h5ad,
        tcga_file=tmp_path / "tcga.csv",
        result_file_path=tmp_path / "high_threshold_genes.csv",
        filtering_type="high_corr_gene",
        corr_threshold=0.9,
        n_gene_max=2,
        corr_result_fp=str(tmp_path / "high_threshold_correlations.csv"),
        correlation_reference_file=raw_corr_file,
        force_recompute=True,
    )

    assert set(low_threshold_genes) == {"G1", "G2", "G4"}
    assert set(high_threshold_genes) == {"G1", "G2"}
    pd.testing.assert_frame_equal(pd.read_csv(raw_corr_file, index_col=0), raw_corr)


def test_gene_filtering_postprocess_passes_configured_correlation_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    generated_h5ad = tmp_path / "target.h5ad"
    generated_h5ad.write_text("placeholder", encoding="utf-8")
    source_corr_file = tmp_path / "full_gene_corr_with_cell_fraction.csv"
    pd.DataFrame({"B Cells": [0.8]}, index=["G1"]).to_csv(source_corr_file)
    config = _make_filtering_config(tmp_path)
    config._generated_mixed_outputs = {
        "Mixed_Correlation_Source": {
            "full_gene_corr_with_cell_fraction_file": source_corr_file,
        }
    }
    captured = {}

    class FakeReadH5AD:
        def __init__(self, file_path):
            self.file_path = file_path

        def get_df(self):
            return pd.DataFrame({"G1": [1.0, 2.0]}, index=["s1", "s2"])

        def get_cell_fraction(self):
            return pd.DataFrame({"B Cells": [0.4, 0.6]}, index=["s1", "s2"])

    class FakeReadExp:
        def __init__(self, exp_file, exp_type=None):
            pass

        def get_exp(self):
            return pd.DataFrame({"G1": [1.0, 2.0]}, index=["t1", "t2"])

    def fake_get_gene_list_for_filtering(**kwargs):
        captured.update(kwargs)
        pd.DataFrame({"gene_name": ["G1"]}).to_csv(kwargs["result_file_path"], index=False)
        return ["G1"]

    monkeypatch.setattr(workflow_module, "ReadH5AD", FakeReadH5AD)
    monkeypatch.setattr(workflow_module, "ReadExp", FakeReadExp)
    monkeypatch.setattr(workflow_module, "get_gene_list_for_filtering", fake_get_gene_list_for_filtering)
    monkeypatch.setattr(workflow_module, "filtering_by_gene_list_and_pca_plot", lambda **kwargs: None)

    outputs = _run_gene_filtering_postprocess(
        dataset_name="Mixed_Target",
        generated_bulk_gep_fp=generated_h5ad,
        params={},
        config=config,
        resolved_params=ResolvedGeneFilteringParams(
            enabled=True,
            reference_file=str(tmp_path / "tcga.csv"),
            filtering_type="high_corr_gene",
            quantile_range=None,
            q_col_name=None,
            corr_threshold=0.2,
            n_gene_max=1,
            high_corr_gene_file=None,
            save_filtered_h5ad=False,
            filtered_dataset_postfix=None,
            plot_pca=False,
            pca_n_components=2,
            pca_figsize=(5.0, 5.0),
            correlation_reference_dataset="Mixed_Correlation_Source",
        ),
    )

    assert captured["correlation_reference_file"] == source_corr_file
    assert captured["full_corr_result_fp"] is None
    assert captured["force_recompute"] is True
    assert outputs["full_gene_corr_with_cell_fraction_file"] == str(source_corr_file)


def test_run_gene_filtering_postprocess_generates_high_corr_on_the_fly_when_file_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    generated_h5ad = tmp_path / "simu_gep_Mixed_Test_log2cpm1p.h5ad"
    generated_h5ad.write_text("placeholder", encoding="utf-8")
    tcga_file = tmp_path / "tcga.csv"
    tcga_file.write_text("sample_id,G1,G2\ns1,1,2\n", encoding="utf-8")
    config = _make_filtering_config(tmp_path)

    calls = {"get_gene_list_for_filtering": 0}

    def fake_get_gene_list_for_filtering(**kwargs):
        calls["get_gene_list_for_filtering"] += 1
        pd.DataFrame({"gene_name": ["G1", "G2"]}).to_csv(kwargs["result_file_path"], index=False)
        return ["G1", "G2"]

    class FakeReadH5AD:
        def __init__(self, file_path):
            self.file_path = file_path

        def get_df(self):
            return pd.DataFrame({"G1": [1.0], "G2": [2.0]}, index=["sample_1"])

        def get_cell_fraction(self):
            return pd.DataFrame({"B Cells": [1.0]}, index=["sample_1"])

    class FakeReadExp:
        def __init__(self, exp_file, exp_type=None):
            self.exp_file = exp_file

        def get_exp(self):
            return pd.DataFrame({"G1": [1.0], "G2": [2.0]}, index=["tcga_1"])

    monkeypatch.setattr(workflow_module, "get_gene_list_for_filtering", fake_get_gene_list_for_filtering)
    monkeypatch.setattr(workflow_module, "ReadH5AD", FakeReadH5AD)
    monkeypatch.setattr(workflow_module, "ReadExp", FakeReadExp)
    monkeypatch.setattr(workflow_module, "filtering_by_gene_list_and_pca_plot", lambda **kwargs: None)

    outputs = _run_gene_filtering_postprocess(
        dataset_name="Mixed_Test",
        generated_bulk_gep_fp=generated_h5ad,
        params={},
        config=config,
        resolved_params=ResolvedGeneFilteringParams(
            enabled=True,
            reference_file=str(tcga_file),
            filtering_type="high_corr_gene",
            quantile_range=None,
            q_col_name=None,
            corr_threshold=0.3,
            n_gene_max=1000,
            high_corr_gene_file=None,
            save_filtered_h5ad=True,
            filtered_dataset_postfix=None,
            plot_pca=False,
            pca_n_components=2,
            pca_figsize=(5.0, 5.0),
        ),
    )

    assert calls["get_gene_list_for_filtering"] == 1
    assert outputs["gene_list_file"] is not None
    assert Path(outputs["gene_list_file"]).exists()


def test_run_pca_comparison_when_gene_filtering_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    generated_h5ad = tmp_path / "simu_gep_Mixed_Test_log2cpm1p.h5ad"
    generated_h5ad.write_text("placeholder", encoding="utf-8")
    tcga_file = tmp_path / "tcga.csv"
    config = _make_filtering_config(tmp_path)
    captured = {}

    class FakeReadH5AD:
        def __init__(self, file_path):
            self.file_path = file_path

        def get_df(self):
            return pd.DataFrame({"G1": [1.0], "G2": [2.0], "G3": [3.0]}, index=["sample_1"])

    class FakeReadExp:
        def __init__(self, exp_file, exp_type=None):
            self.exp_file = exp_file

        def get_exp(self):
            return pd.DataFrame({"G1": [1.0], "G2": [2.0], "G4": [4.0]}, index=["tcga_1"])

    def fake_pca_plot(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(workflow_module, "ReadH5AD", FakeReadH5AD)
    monkeypatch.setattr(workflow_module, "ReadExp", FakeReadExp)
    monkeypatch.setattr(workflow_module, "filtering_by_gene_list_and_pca_plot", fake_pca_plot)

    outputs = _run_gene_filtering_postprocess(
        dataset_name="Mixed_Test",
        generated_bulk_gep_fp=generated_h5ad,
        params={},
        config=config,
        resolved_params=ResolvedGeneFilteringParams(
            enabled=False,
            reference_file=str(tcga_file),
            filtering_type="high_corr_gene",
            quantile_range=None,
            q_col_name=None,
            corr_threshold=0.3,
            n_gene_max=1000,
            high_corr_gene_file=None,
            save_filtered_h5ad=True,
            filtered_dataset_postfix=None,
            plot_pca=True,
            pca_n_components=2,
            pca_figsize=(5.0, 5.0),
        ),
    )

    assert pd.read_csv(outputs["gene_list_file"])["gene_name"].tolist() == ["G1", "G2"]
    assert outputs["pca_model_file"].endswith("_PCA_all_genes_for_pca.joblib")
    assert outputs["pca_data_file"].endswith("_PCA_with_TCGA_all_genes_for_pca.csv")
    assert outputs["pca_plot_file"].endswith("_PCA_with_TCGA_all_genes_for_pca_PC0_PC1.png")
    assert outputs["filtered_bulk_gep_file"] is None
    assert captured["gene_list"] == ["G1", "G2"]
    assert captured.get("h5ad_file_path") is None
    assert captured["if_plot_pca"] is True


def test_build_mixed_dataset_summary_writes_expected_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    generated_h5ad = tmp_path / "simu_gep_Mixed_Test_log2cpm1p.h5ad"
    filtered_h5ad = tmp_path / "simu_gep_Mixed_Test_log2cpm1p_filtered_by_high_corr_gene.h5ad"
    generated_h5ad.write_text("placeholder", encoding="utf-8")
    filtered_h5ad.write_text("placeholder", encoding="utf-8")
    result_dir = tmp_path / "Mixed_Test"
    result_dir.mkdir()
    gene_list_file = result_dir / "gene_list_filtered_by_high_corr_gene.csv"
    pd.DataFrame({"gene_name": ["G1", "G2", "G3"]}).to_csv(gene_list_file, index=False)

    class FakeReadH5AD:
        def __init__(self, file_path):
            self.file_path = str(file_path)

        def get_h5ad(self):
            if self.file_path.endswith("_filtered_by_high_corr_gene.h5ad"):
                return SimpleNamespace(shape=(8, 3))
            return SimpleNamespace(shape=(10, 5))

    monkeypatch.setattr(workflow_module, "ReadH5AD", FakeReadH5AD)

    summary_path = _build_mixed_dataset_summary(
        dataset_name="Mixed_Test",
        generated_bulk_gep_fp=generated_h5ad,
        generated_cell_fraction_fp=tmp_path / "cell_fractions.csv",
        gep_filtering=ResolvedGEPFilteringParams(
            enabled=True,
            reference_file="ref.csv",
            ref_exp_type="TPM",
            filtering_method="median_gep",
            filtering_ref_types=None,
            gep_filtering_quantile=(0.0, 0.95),
            n_top=20,
            show_filtering_info=False,
            filtering_by_gene_range=False,
            min_percentage_within_gene_range=0.95,
            gene_quantile_range=None,
            filtering_in_pca_space=True,
            pca_n_components=0.9,
            norm_ord=1,
            high_corr_gene_list_file=None,
            high_corr_gene_list=None,
        ),
        gene_filtering=ResolvedGeneFilteringParams(
            enabled=True,
            reference_file="ref.csv",
            filtering_type="high_corr_gene",
            quantile_range=None,
            q_col_name=None,
            corr_threshold=0.3,
            n_gene_max=1000,
            high_corr_gene_file=None,
            save_filtered_h5ad=True,
            filtered_dataset_postfix=None,
            plot_pca=True,
            pca_n_components=2,
            pca_figsize=(5.0, 5.0),
            correlation_reference_dataset="Mixed_Correlation_Source",
        ),
        postprocess_outputs={
            "filtered_bulk_gep_file": str(filtered_h5ad),
            "gene_list_file": str(gene_list_file),
            "pca_model_file": str(result_dir / "pca.joblib"),
            "pca_data_file": str(result_dir / "pca.csv"),
            "pca_plot_file": str(result_dir / "pca.png"),
            "full_gene_corr_with_cell_fraction_file": str(
                result_dir / "full_gene_corr_with_cell_fraction.csv"
            ),
        },
        gep_filtering_input_n_samples=14,
        gep_filtering_kept_n_samples=10,
        gep_filtering_removed_n_samples=4,
        gep_filtering_pca_target=0.9,
        gep_filtering_n_pcs=576,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["generated_n_samples"] == 10
    assert summary["generated_n_genes"] == 5
    assert summary["gep_filtering_input_n_samples"] == 14
    assert summary["gep_filtering_kept_n_samples"] == 10
    assert summary["gep_filtering_removed_n_samples"] == 4
    assert summary["gep_pca_n_components"] == 0.9
    assert summary["gep_filtering_n_pcs"] == 576
    assert summary["filtered_n_samples"] == 8
    assert summary["filtered_n_genes"] == 3
    assert summary["removed_n_genes"] == 2
    assert summary["filtered_gene_count"] == 3
    assert summary["pca_plot_file"] == str(result_dir / "pca.png")
    assert summary["gene_filtering_correlation_reference_dataset"] == "Mixed_Correlation_Source"
    assert summary["full_gene_corr_with_cell_fraction_file"] == str(
        result_dir / "full_gene_corr_with_cell_fraction.csv"
    )
    assert summary["gene_filtering_type"] == "high_corr_gene"


def test_build_mixed_dataset_summary_writes_null_gep_filtering_totals_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    generated_h5ad = tmp_path / "simu_gep_Mixed_Test_log2cpm1p.h5ad"
    generated_h5ad.write_text("placeholder", encoding="utf-8")
    all_gene_list = tmp_path / "gene_list_all_genes_for_pca.csv"
    pd.DataFrame({"gene_name": ["G1", "G2", "G3"]}).to_csv(all_gene_list, index=False)

    class FakeReadH5AD:
        def __init__(self, file_path):
            self.file_path = str(file_path)

        def get_h5ad(self):
            return SimpleNamespace(shape=(10, 5))

    monkeypatch.setattr(workflow_module, "ReadH5AD", FakeReadH5AD)

    summary_path = _build_mixed_dataset_summary(
        dataset_name="Mixed_Test",
        generated_bulk_gep_fp=generated_h5ad,
        generated_cell_fraction_fp=tmp_path / "cell_fractions.csv",
        gep_filtering=ResolvedGEPFilteringParams(
            enabled=False,
            reference_file=None,
            ref_exp_type=None,
            filtering_method="median_gep",
            filtering_ref_types=None,
            gep_filtering_quantile=(0.0, 0.95),
            n_top=20,
            show_filtering_info=False,
            filtering_by_gene_range=False,
            min_percentage_within_gene_range=0.95,
            gene_quantile_range=None,
            filtering_in_pca_space=True,
            pca_n_components=0.9,
            norm_ord=1,
            high_corr_gene_list_file=None,
            high_corr_gene_list=None,
        ),
        gene_filtering=ResolvedGeneFilteringParams(
            enabled=False,
            reference_file=None,
            filtering_type="all_genes",
            quantile_range=None,
            q_col_name=None,
            corr_threshold=0.3,
            n_gene_max=1000,
            high_corr_gene_file=None,
            save_filtered_h5ad=True,
            filtered_dataset_postfix=None,
            plot_pca=True,
            pca_n_components=2,
            pca_figsize=(5.0, 5.0),
            correlation_reference_dataset="Mixed_Correlation_Source",
        ),
        postprocess_outputs={
            "gene_list_file": str(all_gene_list),
            "full_gene_corr_with_cell_fraction_file": str(
                tmp_path / "full_gene_corr_with_cell_fraction.csv"
            ),
        },
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["gep_filtering_input_n_samples"] is None
    assert summary["gep_filtering_kept_n_samples"] is None
    assert summary["gep_filtering_removed_n_samples"] is None
    assert summary["gene_list_file"] == str(all_gene_list)
    assert summary["filtered_gene_count"] is None
    assert summary["removed_n_genes"] is None
    assert summary["gep_pca_n_components"] is None
    assert summary["gep_filtering_n_pcs"] is None
    assert summary["gene_filtering_correlation_reference_dataset"] == "Mixed_Correlation_Source"
    assert summary["full_gene_corr_with_cell_fraction_file"].endswith(
        "full_gene_corr_with_cell_fraction.csv"
    )


def test_build_mixed_dataset_summary_recovers_pca_metadata_from_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    generated_h5ad = tmp_path / "simu_gep_Mixed_Cached_log2cpm1p.h5ad"
    generated_h5ad.write_text("placeholder", encoding="utf-8")
    reference_file = tmp_path / "datasets" / "TCGA" / "tpm" / "reference.csv"
    pca_model_file = (
        reference_file.parent.parent
        / "pca_model_0.9"
        / "tcga_pca_model_for_gep_filtering.pkl"
    )
    pca_model_file.parent.mkdir(parents=True)
    pca_model_file.touch()

    class FakeReadH5AD:
        def __init__(self, file_path):
            self.file_path = str(file_path)

        def get_h5ad(self):
            return SimpleNamespace(shape=(10, 5))

    monkeypatch.setattr(workflow_module, "ReadH5AD", FakeReadH5AD)
    monkeypatch.setattr(
        workflow_module,
        "load",
        lambda _: SimpleNamespace(n_components=0.9, n_components_=576),
    )

    result_dir = generated_h5ad.parent / "Mixed_Cached"
    result_dir.mkdir()
    (result_dir / "mixed_dataset_summary.json").write_text(
        json.dumps({"generated_bulk_gep_file": str(generated_h5ad)}),
        encoding="utf-8",
    )
    summary_path = _build_mixed_dataset_summary(
        dataset_name="Mixed_Cached",
        generated_bulk_gep_fp=generated_h5ad,
        generated_cell_fraction_fp=tmp_path / "cell_fractions.csv",
        gep_filtering=ResolvedGEPFilteringParams(
            enabled=True,
            reference_file=str(reference_file),
            ref_exp_type="TPM",
            filtering_method="median_gep",
            filtering_ref_types=None,
            gep_filtering_quantile=(0.0, 0.95),
            n_top=20,
            show_filtering_info=False,
            filtering_by_gene_range=False,
            min_percentage_within_gene_range=0.95,
            gene_quantile_range=None,
            filtering_in_pca_space=True,
            pca_n_components=0.9,
            norm_ord=1,
            high_corr_gene_list_file=None,
            high_corr_gene_list=None,
        ),
        gene_filtering=ResolvedGeneFilteringParams(
            enabled=False,
            reference_file=None,
            filtering_type="all_genes",
            quantile_range=None,
            q_col_name=None,
            corr_threshold=0.3,
            n_gene_max=1000,
            high_corr_gene_file=None,
            save_filtered_h5ad=True,
            filtered_dataset_postfix=None,
            plot_pca=False,
            pca_n_components=2,
            pca_figsize=(5.0, 5.0),
            correlation_reference_dataset=None,
        ),
        postprocess_outputs={},
        preserve_existing_gep_pca_metadata=True,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["gep_pca_n_components"] == 0.9
    assert summary["gep_filtering_n_pcs"] == 576


def test_filtering_by_gene_list_and_pca_plot_recomputes_when_model_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = {"do_pca_analysis": 0, "plot_pca": 0}

    class FakeReadExp:
        def __init__(self, exp_file, exp_type=None):
            self._df = exp_file.copy()

        def align_with_gene_list(self, gene_list, fill_not_exist=False):
            columns = [gene for gene in gene_list if gene in self._df.columns]
            self._df = self._df.loc[:, columns].copy()

        def to_log2cpm1p(self):
            self._df = np.log2(self._df + 1)

        def get_exp(self):
            return self._df

    class FakePCAModel:
        explained_variance_ratio_ = np.array([0.6, 0.4])
        mean_ = np.zeros(2)
        components_ = np.eye(2)

        def transform(self, exp_df):
            return np.asarray(exp_df.iloc[:, :2], dtype=float)

    def fake_do_pca_analysis(exp_df, n_components, pca_result_fp, save_model=False):
        calls["do_pca_analysis"] += 1
        calls["save_model"] = save_model
        return FakePCAModel()

    def fake_plot_pca(data, figsize, result_fp, color_code, explained_variance_ratio):
        calls["plot_pca"] += 1

    monkeypatch.setattr(workflow_module, "ReadExp", FakeReadExp)
    monkeypatch.setattr(workflow_module, "do_pca_analysis", fake_do_pca_analysis)
    monkeypatch.setattr(workflow_module, "plot_pca", fake_plot_pca)

    pca_data_file = tmp_path / "existing_pca.csv"
    pd.DataFrame({"PC1": [1.0], "PC2": [2.0]}, index=["existing"]).to_csv(pca_data_file)
    pca_model_file = tmp_path / "missing_model.joblib"

    bulk_exp = pd.DataFrame({"G1": [1.0, 2.0], "G2": [3.0, 4.0]}, index=["s1", "s2"])
    tcga_exp = pd.DataFrame({"G1": [5.0, 6.0], "G2": [7.0, 8.0]}, index=["t1", "t2"])
    filtering_by_gene_list_and_pca_plot(
        bulk_exp=bulk_exp,
        tcga_exp=tcga_exp,
        gene_list=["G1", "G2"],
        result_dir=str(tmp_path),
        simu_dataset_name="Mixed_Test",
        n_components=2,
        pca_model_name_postfix="high_corr_gene",
        pca_model_file_path=str(pca_model_file),
        pca_data_file_path=str(pca_data_file),
        if_plot_pca=True,
    )

    assert calls["do_pca_analysis"] == 1
    assert calls["save_model"] is True
    assert calls["plot_pca"] == 1
