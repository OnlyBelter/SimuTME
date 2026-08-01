from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from simutme.workflow import SimuConfig, _resolve_mixed_sct_dataset_path


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
