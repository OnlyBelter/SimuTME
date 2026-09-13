from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from simutme.workflow import (
    SimuConfig,
    _resolve_mixed_sct_dataset_path,
    _resolve_output_dir_naming_metadata,
    _resolve_sct_aux_output_dir,
    run_bulk_simulation,
)


def _make_config(
    dataset_definitions: dict[str, dict],
    generated_sct_outputs: dict[str, dict[str, Path | None]] | None = None,
    output_suffix: str = "_sorted_bulk_imputed",
):
    return SimpleNamespace(
        sorted_bulk_imputation_output_suffix=output_suffix,
        _dataset_definitions=dataset_definitions,
        _dataset_order=list(dataset_definitions.keys()),
        _generated_sct_outputs=generated_sct_outputs or {},
    )


def test_resolve_mixed_sct_dataset_path_supports_explicit_path_reference(tmp_path: Path):
    original_sct_path = tmp_path / "toy_sct.h5ad"
    original_sct_path.write_text("placeholder", encoding="utf-8")
    config = _make_config(
        dataset_definitions={"Mixed_Test": {"dataset_type": "Mixed"}},
    )

    resolved_path = _resolve_mixed_sct_dataset_path(
        dataset_name="Mixed_Test",
        params={
            "sct_reference": {
                "source_type": "path",
                "file_path": str(original_sct_path),
                "version": "original",
            }
        },
        config=config,
    )

    assert resolved_path == original_sct_path


def test_resolve_mixed_sct_dataset_path_rejects_version_for_explicit_path_reference(tmp_path: Path):
    sct_path = tmp_path / "toy_sct_sorted_bulk_imputed.h5ad"
    sct_path.write_text("placeholder", encoding="utf-8")
    config = _make_config(
        dataset_definitions={"Mixed_Test": {"dataset_type": "Mixed"}},
    )

    with pytest.raises(ValueError, match='must not provide "sct_reference.version"'):
        _resolve_mixed_sct_dataset_path(
            dataset_name="Mixed_Test",
            params={
                "sct_reference": {
                    "source_type": "path",
                    "file_path": str(sct_path),
                    "version": "sorted_bulk_imputed",
                }
            },
            config=config,
        )


def test_resolve_mixed_sct_dataset_path_supports_generated_dataset_reference(tmp_path: Path):
    original_sct_path = tmp_path / "generated_sct.h5ad"
    original_sct_path.write_text("placeholder", encoding="utf-8")
    config = _make_config(
        dataset_definitions={
            "SCT_A": {"dataset_type": "SCT", "apply_sorted_bulk_imputation": False},
            "Mixed_Test": {"dataset_type": "Mixed"},
        },
        generated_sct_outputs={"SCT_A": {"original": original_sct_path, "sorted_bulk_imputed": None}},
    )

    resolved_path = _resolve_mixed_sct_dataset_path(
        dataset_name="Mixed_Test",
        params={
            "sct_reference": {
                "source_type": "generated_dataset",
                "dataset_name": "SCT_A",
                "version": "original",
            }
        },
        config=config,
    )

    assert resolved_path == original_sct_path


def test_resolve_mixed_sct_dataset_path_rejects_forward_generated_reference():
    config = _make_config(
        dataset_definitions={
            "Mixed_Test": {"dataset_type": "Mixed"},
            "SCT_Later": {"dataset_type": "SCT", "apply_sorted_bulk_imputation": False},
        },
    )

    with pytest.raises(ValueError, match="appears later in the YAML order"):
        _resolve_mixed_sct_dataset_path(
            dataset_name="Mixed_Test",
            params={
                "sct_reference": {
                    "source_type": "generated_dataset",
                    "dataset_name": "SCT_Later",
                    "version": "original",
                }
            },
            config=config,
        )


def test_resolve_mixed_sct_dataset_path_rejects_non_sct_reference():
    config = _make_config(
        dataset_definitions={
            "Mixed_Source": {"dataset_type": "Mixed"},
            "Mixed_Test": {"dataset_type": "Mixed"},
        },
    )

    with pytest.raises(ValueError, match="must reference an SCT dataset"):
        _resolve_mixed_sct_dataset_path(
            dataset_name="Mixed_Test",
            params={
                "sct_reference": {
                    "source_type": "generated_dataset",
                    "dataset_name": "Mixed_Source",
                    "version": "original",
                }
            },
            config=config,
        )


def test_resolve_mixed_sct_dataset_path_rejects_imputed_generated_reference_without_output(tmp_path: Path):
    original_sct_path = tmp_path / "generated_sct.h5ad"
    original_sct_path.write_text("placeholder", encoding="utf-8")
    config = _make_config(
        dataset_definitions={
            "SCT_A": {"dataset_type": "SCT", "apply_sorted_bulk_imputation": False},
            "Mixed_Test": {"dataset_type": "Mixed"},
        },
        generated_sct_outputs={"SCT_A": {"original": original_sct_path, "sorted_bulk_imputed": None}},
    )

    with pytest.raises(ValueError, match='version: "original"'):
        _resolve_mixed_sct_dataset_path(
            dataset_name="Mixed_Test",
            params={
                "sct_reference": {
                    "source_type": "generated_dataset",
                    "dataset_name": "SCT_A",
                    "version": "sorted_bulk_imputed",
                }
            },
            config=config,
        )


def test_resolve_mixed_sct_dataset_path_rejects_legacy_sct_dataset_version():
    config = _make_config(
        dataset_definitions={"Mixed_Test": {"dataset_type": "Mixed"}},
    )

    with pytest.raises(ValueError, match='deprecated "sct_dataset_version"'):
        _resolve_mixed_sct_dataset_path(
            dataset_name="Mixed_Test",
            params={"sct_dataset_version": "original"},
            config=config,
        )


def test_simu_config_from_yaml_rejects_legacy_top_level_sct_dataset_path(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "simu_bulk_dir": "./datasets",
                    "sct_dataset_file_path": "./datasets/toy_sct.h5ad",
                    "tcga2cancer_type_file_path": "./datasets/tcga_id.csv",
                    "tcga_merged_tpm_file_path": "./datasets/merged_tpm.csv",
                    "high_corr_gene_file_path": "./datasets/high_corr.csv",
                    "merged_sc_dataset_file_path": "./datasets/merged_sc.h5ad",
                },
                "constants": {
                    "sc_dataset_ids": ["ds1"],
                    "cell_type2subtypes": {"B Cells": ["B Cells"]},
                    "total_rna_coefficient": {"B Cells": 1.0},
                    "output_types": ["B Cells"],
                },
                "parameters": {
                    "gene_quantile_range": [0.005, 0.5, 0.995],
                    "gep_filtering_quantile": [0.0, 0.95],
                    "pca_n_components": 0.9,
                    "gene_list_type": "all_genes",
                    "n_base": 30,
                },
                "datasets": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match='paths.sct_dataset_file_path'):
        SimuConfig.from_yaml(str(config_path))


def test_output_dir_naming_metadata_uses_unique_union_of_sct_subset_lists():
    config = SimpleNamespace(
        n_base=30,
        sc_dataset_ids=["ds1", "ds2", "ds3", "ds4"],
        _dataset_definitions={},
    )

    n_datasets, n_base = _resolve_output_dir_naming_metadata(
        params={
            "dataset_type": "SCT",
            "cancer_cell_dataset_ids": ["ds2"],
            "non_cancer_cell_dataset_ids": ["ds1", "ds2", "ds3"],
        },
        config=config,
        dataset_type="sct",
    )

    assert n_datasets == 3
    assert n_base == 30


def test_output_dir_naming_metadata_falls_back_to_sc_dataset_ids_when_sct_subset_lists_incomplete():
    config = SimpleNamespace(
        n_base=30,
        sc_dataset_ids=["ds1", "ds2", "ds3", "ds4"],
        _dataset_definitions={},
    )

    n_datasets, n_base = _resolve_output_dir_naming_metadata(
        params={
            "dataset_type": "SCT",
            "cancer_cell_dataset_ids": ["ds2"],
        },
        config=config,
        dataset_type="sct",
    )

    assert n_datasets == 4
    assert n_base == 30


def test_output_dir_naming_metadata_for_mixed_generated_dataset_uses_referenced_sct_subset_lists():
    config = SimpleNamespace(
        n_base=30,
        sc_dataset_ids=["ds1", "ds2", "ds3", "ds4", "ds5"],
        _dataset_definitions={
            "SCT_A": {
                "dataset_type": "SCT",
                "cancer_cell_dataset_ids": ["ds2"],
                "non_cancer_cell_dataset_ids": ["ds1", "ds2", "ds3"],
            }
        },
    )

    n_datasets, n_base = _resolve_output_dir_naming_metadata(
        params={
            "dataset_type": "Mixed",
            "sct_reference": {
                "source_type": "generated_dataset",
                "dataset_name": "SCT_A",
                "version": "original",
            }
        },
        config=config,
        dataset_type="mixed",
    )

    assert n_datasets == 3
    assert n_base == 30


def test_resolve_sct_aux_output_dir_places_outputs_beside_sct_dataset(tmp_path: Path):
    sct_path = tmp_path / "generated_sct_dataset_11ds_n_base30_all_subtypes" / "toy.h5ad"
    resolved = _resolve_sct_aux_output_dir(
        sct_dataset_file_path=sct_path,
        dataset_name="SCT_N10_hnscc_07Jan_01",
    )

    assert resolved == sct_path.parent / "SCT_N10_hnscc_07Jan_01"


def test_simu_config_from_yaml_ignores_deprecated_sorted_bulk_imputation_out_dir(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "simu_bulk_dir": "./datasets",
                    "sorted_bulk_imputation_out_dir": "./datasets/generated_sct_dataset_outputs",
                    "tcga2cancer_type_file_path": "./datasets/tcga_id.csv",
                    "tcga_merged_tpm_file_path": "./datasets/merged_tpm.csv",
                    "high_corr_gene_file_path": "./datasets/high_corr.csv",
                    "merged_sc_dataset_file_path": "./datasets/merged_sc.h5ad",
                },
                "constants": {
                    "sc_dataset_ids": ["ds1"],
                    "cell_type2subtypes": {"B Cells": ["B Cells"]},
                    "total_rna_coefficient": {"B Cells": 1.0},
                    "output_types": ["B Cells"],
                },
                "parameters": {
                    "gene_quantile_range": [0.005, 0.5, 0.995],
                    "gep_filtering_quantile": [0.0, 0.95],
                    "pca_n_components": 0.9,
                    "gene_list_type": "all_genes",
                    "n_base": 30,
                },
                "datasets": {},
            }
        ),
        encoding="utf-8",
    )

    config = SimuConfig.from_yaml(str(config_path))

    assert config.simu_bulk_dir == Path("./datasets")


def test_run_bulk_simulation_requires_explicit_sampling_method_gep_for_mixed(tmp_path: Path):
    sct_path = tmp_path / "toy_sct.h5ad"
    sct_path.write_text("placeholder", encoding="utf-8")
    config = SimpleNamespace(
        simu_bulk_dir=tmp_path / "simu_bulk",
        n_base=30,
        sc_dataset_ids=["ds1", "ds2"],
        stratify_by_dataset="none",
        _dataset_definitions={"Mixed_Test": {"dataset_type": "Mixed"}},
        _dataset_order=["Mixed_Test"],
        _generated_sct_outputs={},
        gep_filtering_enabled_default=False,
        gep_filtering_reference_file=None,
        tcga_merged_tpm_file_path=tmp_path / "tcga.csv",
        gep_filtering_ref_exp_type=None,
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
        high_corr_gene_file_path=tmp_path / "high_corr.csv",
        tcga2cancer_type_file_path=tmp_path / "tcga2type.csv",
        gene_filtering_enabled_default=False,
        gene_filtering_reference_file=None,
        gene_filtering_type="all_genes",
        gene_list_type="all_genes",
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

    with pytest.raises(ValueError, match='Missing "sampling_method_gep"'):
        run_bulk_simulation(
            dataset_name="Mixed_Test",
            params={
                "dataset_type": "Mixed",
                "sct_reference": {
                    "source_type": "path",
                    "file_path": str(sct_path),
                },
                "n_samples": 10,
                "sampling_method_prop": "segment",
            },
            config=config,
        )


def test_run_bulk_simulation_rejects_legacy_filtering_key_for_mixed(tmp_path: Path):
    sct_path = tmp_path / "toy_sct.h5ad"
    sct_path.write_text("placeholder", encoding="utf-8")
    config = SimpleNamespace(
        simu_bulk_dir=tmp_path / "simu_bulk",
        n_base=30,
        sc_dataset_ids=["ds1", "ds2"],
        stratify_by_dataset="none",
        _dataset_definitions={"Mixed_Test": {"dataset_type": "Mixed"}},
        _dataset_order=["Mixed_Test"],
        _generated_sct_outputs={},
        gep_filtering_enabled_default=False,
        gep_filtering_reference_file=None,
        tcga_merged_tpm_file_path=tmp_path / "tcga.csv",
        gep_filtering_ref_exp_type=None,
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
        high_corr_gene_file_path=tmp_path / "high_corr.csv",
        tcga2cancer_type_file_path=tmp_path / "tcga2type.csv",
        gene_filtering_enabled_default=False,
        gene_filtering_reference_file=None,
        gene_filtering_type="all_genes",
        gene_list_type="all_genes",
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

    with pytest.raises(ValueError, match='"filtering" is no longer supported'):
        run_bulk_simulation(
            dataset_name="Mixed_Test",
            params={
                "dataset_type": "Mixed",
                "sct_reference": {
                    "source_type": "path",
                    "file_path": str(sct_path),
                },
                "sampling_method_gep": "random",
                "n_samples": 10,
                "sampling_method_prop": "segment",
                "filtering": False,
            },
            config=config,
        )


def test_run_bulk_simulation_mixed_prefers_dataset_total_rna_coefficient(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sct_path = tmp_path / "toy_sct.h5ad"
    sct_path.write_text("placeholder", encoding="utf-8")

    tcga_path = tmp_path / "tcga.csv"
    tcga_path.write_text("sample_id,G1,G2\ns1,1,2\n", encoding="utf-8")
    tcga2type_path = tmp_path / "tcga2type.csv"
    tcga2type_path.write_text("sample_id,cancer_type\ns1,LUAD\n", encoding="utf-8")
    high_corr_path = tmp_path / "high_corr.csv"
    high_corr_path.write_text("gene_name\nG1\n", encoding="utf-8")

    config = SimpleNamespace(
        simu_bulk_dir=tmp_path / "simu_bulk",
        n_base=30,
        sc_dataset_ids=["ds1", "ds2"],
        stratify_by_dataset="none",
        _dataset_definitions={"Mixed_Test": {"dataset_type": "Mixed"}},
        _dataset_order=["Mixed_Test"],
        _generated_sct_outputs={},
        merged_sc_dataset_file_path=tmp_path / "merged_sc.h5ad",
        tcga2cancer_type_file_path=tcga2type_path,
        tcga_merged_tpm_file_path=tcga_path,
        high_corr_gene_file_path=high_corr_path,
        sorted_bulk_imputation_seed=123,
        cell_type2subtypes={"B Cells": ["B Cells"]},
        total_rna_coefficient={"B Cells": 1.0},
        gep_filtering_enabled_default=False,
        gep_filtering_reference_file=None,
        gep_filtering_ref_exp_type=None,
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
        gene_filtering_enabled_default=False,
        gene_filtering_reference_file=None,
        gene_filtering_type="all_genes",
        gene_list_type="all_genes",
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

    captured: dict[str, object] = {}

    class FakeBulkGEPGenerator:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.generated_bulk_gep_fp = tmp_path / "existing_bulk.h5ad"
            self.generated_bulk_gep_fp.write_text("placeholder", encoding="utf-8")
            self.generated_cell_fraction_fp = tmp_path / "cell_fractions.csv"
            self.generated_cell_fraction_fp.write_text("sample_id\n", encoding="utf-8")

    monkeypatch.setattr("simutme.workflow.BulkGEPGenerator", FakeBulkGEPGenerator)
    monkeypatch.setattr("simutme.workflow._save_cancer_cell_fraction_distribution", lambda **kwargs: None)
    monkeypatch.setattr("simutme.workflow._run_gene_filtering_postprocess", lambda **kwargs: {})
    monkeypatch.setattr("simutme.workflow._build_mixed_dataset_summary", lambda **kwargs: None)

    run_bulk_simulation(
        dataset_name="Mixed_Test",
        params={
            "dataset_type": "Mixed",
            "sct_reference": {
                "source_type": "path",
                "file_path": str(sct_path),
            },
            "n_samples": 10,
            "sampling_method_prop": "segment",
            "sampling_method_gep": "random",
            "total_rna_coefficient": {"B Cells": 2.5},
        },
        config=config,
    )

    assert captured["total_rna_coefficient"] == {"B Cells": 2.5}
