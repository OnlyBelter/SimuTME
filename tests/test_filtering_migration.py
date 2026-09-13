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
        gene_filtering_plot_pca=True,
        gene_filtering_pca_n_components=2,
        gene_filtering_pca_figsize=(5.0, 5.0),
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
            "gene_filtering_plot_pca": True,
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
        ),
        postprocess_outputs={
            "filtered_bulk_gep_file": str(filtered_h5ad),
            "gene_list_file": str(gene_list_file),
            "pca_model_file": str(result_dir / "pca.joblib"),
            "pca_data_file": str(result_dir / "pca.csv"),
        },
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["generated_n_samples"] == 10
    assert summary["generated_n_genes"] == 5
    assert summary["filtered_n_samples"] == 8
    assert summary["filtered_n_genes"] == 3
    assert summary["filtered_gene_count"] == 3
    assert summary["gene_filtering_type"] == "high_corr_gene"


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

        def transform(self, exp_df):
            return np.asarray(exp_df.iloc[:, :2], dtype=float)

    def fake_do_pca_analysis(exp_df, n_components, pca_result_fp):
        calls["do_pca_analysis"] += 1
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
    assert calls["plot_pca"] == 1
