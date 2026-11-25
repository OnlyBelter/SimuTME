import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

import simutme.simulation.parallel_candidates as parallel
import simutme.simulation.simulator as simulator
import simutme.workflow as workflow
from simutme.workflow import SimuConfig


@pytest.fixture
def stat_case(tmp_path, monkeypatch):
    sct = tmp_path / "sct.h5ad"
    ad.AnnData(
        X=np.log2(np.array([[1., 9.], [3., 7.], [8., 2.], [9., 1.]]) + 1),
        obs=pd.DataFrame(
            {"A": [1., 1., 0., 0.], "B": [0., 0., 1., 1.]},
            index=["a1", "a2", "b1", "b2"],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(sct)
    reference = tmp_path / "reference.csv"
    pd.DataFrame(
        {"g1": [1., 3., 8., 9.], "g2": [9., 7., 2., 1.]},
        index=["r1", "r2", "r3", "r4"],
    ).to_csv(reference)
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame(
        {"cancer_type": ["X", "X", "Y", "Y"]},
        index=["r1", "r2", "r3", "r4"],
    ).to_csv(mapping)
    config = SimuConfig(
        simu_bulk_dir=tmp_path / "output",
        tcga2cancer_type_file_path=mapping,
        tcga_merged_tpm_file_path=reference,
        high_corr_gene_file_path=None,
        merged_sc_dataset_file_path=tmp_path / "unused.h5ad",
        sc_dataset_ids=[],
        cell_type2subtypes={"A": ["A"], "B": ["B"]},
        total_rna_coefficient={"A": 1., "B": 1.},
        n_base=1,
        output_types=["A", "B"],
        gep_filtering_quantile=(None, 1.0),
        plot_pca_for_comparison=False,
        mixed_sct_loading_mode="eager",
    )
    params = {
        "dataset_type": "Mixed",
        "simu_mode": "simulation_for_stat",
        "n_samples": 100,
        "max_batches": 4,
        "sct_reference": {"source_type": "path", "file_path": str(sct)},
        "sampling_method_prop": "segment",
        "gep_filtering": True,
        "gep_filtering_method": "median_gep",
        "gep_filtering_in_pca_space": True,
        "gep_pca_n_components": 1,
        "gep_norm_ord": 1,
        "n_subgroup": 2,
        "gene_filtering": False,
    }
    monkeypatch.setattr(simulator, "_candidate_batch_size", lambda *args: 5)
    monkeypatch.setattr(parallel, "_candidate_batch_size", lambda *args: 5)
    return config, params


def test_statistics_serial_writes_only_summary(stat_case):
    config, params = stat_case
    path = Path(workflow.run_bulk_simulation("target", params, config))
    summary = json.loads(path.read_text())
    assert path.name == "mixed_dataset_summary.json"
    assert path.parent.name == "simulation_for_stat"
    assert summary["processed_batches"] == 4
    assert summary["candidate_batch_size"] == 5
    assert summary["gep_filtering_n_subgroup"] == 2
    assert summary["gep_filtering_method"] == "median_gep"
    assert summary["stop_reason"] == "max_batches_reached"
    assert summary["gep_filtering_input_n_samples"] == 20
    assert sum(summary["gep_filtering_input_n_samples_by_cancer_type"].values()) == 20
    assert sum(summary["gep_filtering_kept_n_samples_by_cancer_type"].values()) == (
        summary["gep_filtering_kept_n_samples"]
    )
    assert [p for p in Path(config.simu_bulk_dir).rglob("*") if p.is_file()] == [path]
    with pytest.raises(FileExistsError, match="already exists"):
        workflow.run_bulk_simulation("target", params, config)


@pytest.mark.parametrize("max_batches", [None, 1])
def test_statistics_target_counts_entire_final_batch(stat_case, max_batches, monkeypatch):
    config, params = stat_case
    params.update(n_samples=3, max_batches=max_batches)
    monkeypatch.setattr(
        workflow.FixedGeneFilter, "filter",
        lambda self, expression, **kwargs: _accept_all(self, expression),
    )
    path = Path(workflow.run_bulk_simulation("target", params, config))
    summary = json.loads(path.read_text())
    assert summary["processed_batches"] == 1
    assert summary["stop_reason"] == "n_samples_reached"
    assert summary["gep_filtering_input_n_samples"] == 5
    assert summary["gep_filtering_kept_n_samples"] == 5


def _accept_all(space, expression):
    space.last_batch_cancer_type = next(iter(space.reference_cancer_type_groups))
    return expression


def test_parallel_statistics_are_worker_count_independent(stat_case):
    config, params = stat_case
    params.update(mixed_candidate_workers=1, mixed_candidate_seed=42)
    first = json.loads(Path(workflow.run_bulk_simulation("one", params, config)).read_text())
    params["mixed_candidate_workers"] = 2
    second = json.loads(Path(workflow.run_bulk_simulation("two", params, config)).read_text())
    for key in (
        "processed_batches",
        "gep_filtering_input_n_samples",
        "gep_filtering_kept_n_samples",
        "gep_filtering_input_n_samples_by_cancer_type",
        "gep_filtering_kept_n_samples_by_cancer_type",
        "gep_filtering_acceptance_rate_by_cancer_type",
    ):
        assert first[key] == second[key]
    assert first["processed_batches"] == 4
    assert len([p for p in Path(config.simu_bulk_dir).rglob("*") if p.is_file()]) == 2


@pytest.mark.parametrize("max_batches", [0, -1, True, 1.5, "1"])
def test_statistics_rejects_invalid_batch_limit(stat_case, max_batches):
    config, params = stat_case
    params["max_batches"] = max_batches
    with pytest.raises(ValueError, match="max_batches"):
        workflow.run_bulk_simulation("target", params, config)


def test_statistics_requires_subgroup(stat_case):
    config, params = stat_case
    params["n_subgroup"] = None
    with pytest.raises(ValueError, match="n_subgroup"):
        workflow.run_bulk_simulation("target", params, config)


def test_statistics_reports_zero_acceptance_and_unselected_types(stat_case, monkeypatch):
    config, params = stat_case
    params["max_batches"] = 2

    def reject_all(space, expression, **kwargs):
        space.last_batch_cancer_type = "X"
        return expression.iloc[:0]

    monkeypatch.setattr(workflow.FixedGeneFilter, "filter", reject_all)
    path = Path(workflow.run_bulk_simulation("zero", params, config))
    summary = json.loads(path.read_text())
    assert summary["stop_reason"] == "max_batches_reached"
    assert summary["gep_filtering_kept_n_samples"] == 0
    assert summary["gep_filtering_acceptance_rate_by_cancer_type"] == {
        "X": 0.0, "Y": None,
    }


def test_statistics_reuses_existing_upstream_inputs(stat_case, tmp_path):
    config, params = stat_case
    original_sct = Path(params["sct_reference"]["file_path"])
    source_sct = {
        "dataset_type": "SCT", "sampling_method_gep": "random", "n_samples": 10,
    }
    source_mixed = {
        "dataset_type": "Mixed", "sampling_method_prop": "segment",
        "gep_filtering": False, "gene_filtering": False,
        "sct_reference": {"source_type": "generated_dataset",
                          "dataset_name": "sct_source", "version": "original"},
    }
    config._dataset_definitions = {
        "sct_source": source_sct, "corr_source": source_mixed, "target": params,
    }
    config._dataset_order = list(config._dataset_definitions)
    sct_dir = Path(config.simu_bulk_dir) / "generated_sct_dataset_0ds_n_base1_all_subtypes"
    sct_dir.mkdir(parents=True)
    cached_sct = sct_dir / "simu_gep_sct_source_gepsamp-random_log2cpm1p.h5ad"
    shutil.copyfile(original_sct, cached_sct)
    corr = (Path(config.simu_bulk_dir) / "segment_0ds_n_base1_no_filtering"
            / "corr_source" / "full_gene_corr_with_cell_fraction.csv")
    corr.parent.mkdir(parents=True)
    pd.DataFrame({"A": [0.8, 0.1], "B": [0.1, 0.8]}, index=["g1", "g2"]).to_csv(corr)
    before = (cached_sct.read_bytes(), corr.read_bytes())
    params.update(
        sct_reference={"source_type": "generated_dataset",
                       "dataset_name": "sct_source", "version": "original"},
        gene_filtering=True,
        gene_filtering_correlation_reference_dataset="corr_source",
        max_batches=1,
    )
    path = Path(workflow.run_bulk_simulation("target", params, config))
    assert path.is_file()
    assert (cached_sct.read_bytes(), corr.read_bytes()) == before
    assert {p for p in Path(config.simu_bulk_dir).rglob("*") if p.is_file()} == {
        path, cached_sct, corr,
    }


def test_statistics_dirichlet_sampling(stat_case):
    config, params = stat_case
    params.update(
        sampling_method_prop="dirichlet",
        dirichlet_sampling_seed=42,
        max_batches=1,
    )
    path = Path(workflow.run_bulk_simulation("dirichlet", params, config))
    summary = json.loads(path.read_text())
    assert summary["sampling_method_prop"] == "dirichlet"
    assert summary["generation"]["dirichlet_sampling_seed"] == 42
    assert summary["generation"]["dirichlet_alpha_resolved"] == {"A": 1.0, "B": 1.0}


def test_statistics_failure_does_not_write_summary(stat_case, monkeypatch):
    config, params = stat_case

    def fail(*args, **kwargs):
        raise RuntimeError("injected batch failure")

    monkeypatch.setattr(workflow.FixedGeneFilter, "filter", fail)
    with pytest.raises(RuntimeError, match="injected batch failure"):
        workflow.run_bulk_simulation("failed", params, config)
    assert not [p for p in Path(config.simu_bulk_dir).rglob("*") if p.is_file()]


def test_statistics_does_not_touch_existing_full_output(stat_case):
    config, params = stat_case
    output_dir = Path(config.simu_bulk_dir) / "segment_0ds_n_base1_with_filtering_1.0_median_gep"
    output_dir.mkdir(parents=True)
    existing = output_dir / "simu_gep_target_gepsamp-random_log2cpm1p.h5ad"
    existing.write_bytes(b"existing full output")
    params["max_batches"] = 1
    summary = Path(workflow.run_bulk_simulation("target", params, config))
    assert summary.exists()
    assert existing.read_bytes() == b"existing full output"


def test_global_statistics_mode_is_used_by_mixed_dataset(stat_case):
    config, params = stat_case
    config.simu_mode = "simulation_for_stat"
    params.pop("simu_mode")
    params["max_batches"] = 1
    summary = json.loads(Path(workflow.run_bulk_simulation("global", params, config)).read_text())
    assert summary["simu_mode"] == "simulation_for_stat"
    assert summary["processed_batches"] == 1


def test_cli_skips_existing_statistics_dependencies(tmp_path, monkeypatch):
    import simutme
    from simutme import workflow as workflow_module

    datasets = {
        "sct_source": {"dataset_type": "SCT"},
        "corr_source": {"dataset_type": "Mixed"},
        "target": {
            "dataset_type": "Mixed",
            "simu_mode": "simulation_for_stat",
            "sct_reference": {"source_type": "generated_dataset",
                              "dataset_name": "sct_source"},
            "gene_filtering_correlation_reference_dataset": "corr_source",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"datasets": datasets}), encoding="utf-8")
    fake_config = SimpleNamespace(simu_mode="full_simulation")
    monkeypatch.setattr(simutme.SimuConfig, "from_yaml", lambda path: fake_config)
    monkeypatch.setattr(workflow_module, "preflight_segment_cell_prop_priors", lambda *args: None)
    registered = []
    monkeypatch.setattr(workflow_module, "_register_statistics_inputs",
                        lambda name, *args: registered.append(name))
    invoked = []
    monkeypatch.setattr(simutme, "run_bulk_simulation",
                        lambda name, *args: invoked.append(name))
    module_spec = importlib.util.spec_from_file_location(
        "statistics_cli", Path(__file__).resolve().parents[1] / "src/main_simu_bulk.py",
    )
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)

    assert cli.main(["-c", str(config_path)]) == 0
    assert registered == ["target"]
    assert invoked == ["target"]


@pytest.mark.parametrize("override", [
    {"simu_mode": "invalid"},
    {"gep_filtering": False},
    {"gep_filtering_method": "mean_gep"},
    {"gep_filtering_in_pca_space": False},
])
def test_statistics_rejects_invalid_mode_or_filter(stat_case, override):
    config, params = stat_case
    params.update(override)
    with pytest.raises(ValueError):
        workflow.run_bulk_simulation("target", params, config)


def test_statistics_full_single_type_does_not_save_pca(stat_case):
    config, params = stat_case
    params["filtering_ref_types"] = ["X"]
    params["max_batches"] = 1
    path = Path(workflow.run_bulk_simulation("single", params, config))
    summary = json.loads(path.read_text())
    assert summary["gep_filtering_input_n_samples_by_cancer_type"] == {"X": 5}
    assert summary["gep_filtering_n_pcs_min"] == 1
    assert [p for p in Path(config.simu_bulk_dir).rglob("*") if p.is_file()] == [path]


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_statistics_reports_cancer_specific_pca_settings(stat_case, workers):
    config, params = stat_case
    params["gep_filtering_by_cancer_type"] = {
        "X": {"n_subgroup": 3, "gep_pca_n_components": 0.9},
        "Y": {"gep_pca_n_components": 1},
    }
    if workers:
        params.update(mixed_candidate_workers=workers, mixed_candidate_seed=42)
    path = Path(workflow.run_bulk_simulation(f"overrides_{workers}", params, config))
    summary = json.loads(path.read_text())
    expected = {
        "X": {"n_subgroup": 3, "gep_pca_n_components": 0.9},
        "Y": {"n_subgroup": 2, "gep_pca_n_components": 1},
    }
    assert summary["gep_filtering_parameters_by_cancer_type"] == expected
    assert summary["filtering_identity"]["gep_filtering_parameters_by_cancer_type"] == expected
    assert summary["gep_filtering_n_subgroup"] == 2
    assert summary["gep_pca_n_components"] == 1
    assert sum(summary["gep_filtering_input_n_samples_by_cancer_type"].values()) == 20


def test_statistics_overrides_are_stable_across_worker_counts(stat_case):
    config, params = stat_case
    params.update(
        mixed_candidate_workers=1, mixed_candidate_seed=42,
        gep_filtering_by_cancer_type={
            "X": {"gep_pca_n_components": 0.9},
            "Y": {"n_subgroup": 3},
        },
    )
    one = json.loads(Path(workflow.run_bulk_simulation("overrides_one", params, config)).read_text())
    params["mixed_candidate_workers"] = 2
    two = json.loads(Path(workflow.run_bulk_simulation("overrides_two", params, config)).read_text())
    for key in (
        "gep_filtering_parameters_by_cancer_type",
        "gep_filtering_input_n_samples_by_cancer_type",
        "gep_filtering_kept_n_samples_by_cancer_type",
        "gep_filtering_acceptance_rate_by_cancer_type",
    ):
        assert one[key] == two[key]


def test_statistics_rejects_override_for_excluded_type(stat_case):
    config, params = stat_case
    params["filtering_ref_types"] = ["X"]
    params["gep_filtering_by_cancer_type"] = {"Y": {"n_subgroup": 3}}
    with pytest.raises(ValueError, match="unknown or excluded"):
        workflow.run_bulk_simulation("excluded", params, config)
    assert not [p for p in Path(config.simu_bulk_dir).rglob("*") if p.is_file()]


def test_full_simulation_summary_keeps_defaults_with_single_type_override(stat_case, monkeypatch):
    config, params = stat_case
    monkeypatch.setattr(workflow, "_save_cancer_cell_fraction_distribution", lambda *args, **kwargs: None)
    params.update(
        simu_mode="full_simulation", n_samples=3, filtering_ref_types=["X"],
        gep_filtering_by_cancer_type={
            "X": {"n_subgroup": 3, "gep_pca_n_components": 0.9}
        },
    )
    generated = Path(workflow.run_bulk_simulation("full_override", params, config))
    assert generated.is_file()
    summaries = list(Path(config.simu_bulk_dir).rglob("mixed_dataset_summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text())
    assert summary["gep_filtering_n_subgroup"] == 2
    assert summary["gep_pca_n_components"] == 1
    assert summary["gep_filtering_parameters_by_cancer_type"] == {
        "X": {"n_subgroup": 3, "gep_pca_n_components": 0.9}
    }
    assert summary["filtering_identity"]["pca_target"] == 0.9


def test_statistics_ignores_full_mode_cancer_type_limit(stat_case):
    config, params = stat_case
    params["max_samples_per_cancer_type"] = "ignored"
    summary = json.loads(Path(workflow.run_bulk_simulation("stat_cap_ignored", params, config)).read_text())
    assert summary["processed_batches"] == 4
    assert "max_samples_per_cancer_type" not in summary


def test_full_simulation_summary_reports_retained_counts(stat_case, monkeypatch):
    config, params = stat_case
    monkeypatch.setattr(workflow, "_save_cancer_cell_fraction_distribution", lambda *args, **kwargs: None)
    params.update(simu_mode="full_simulation", n_samples=4,
                  max_samples_per_cancer_type=2)
    generated = Path(workflow.run_bulk_simulation("full_cap", params, config))
    summary_path = workflow._resolve_mixed_dataset_result_dir(generated, "full_cap") / "mixed_dataset_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["max_samples_per_cancer_type"] == 2
    assert summary["gep_retained_n_samples_by_cancer_type"] == {"X": 2, "Y": 2}
    assert summary["gep_filtering_kept_n_samples"] == (
        4 + summary["gep_discarded_by_cancer_type_limit"]
        + summary["gep_discarded_by_total_target"]
    )
    assert summary["filtering_identity"]["generation"]["max_samples_per_cancer_type"] == 2


def test_full_simulation_limit_supports_single_reference_type(stat_case, monkeypatch):
    config, params = stat_case
    monkeypatch.setattr(workflow, "_save_cancer_cell_fraction_distribution", lambda *args, **kwargs: None)
    params.update(simu_mode="full_simulation", n_samples=3,
                  filtering_ref_types=["X"], max_samples_per_cancer_type=3)
    generated = Path(workflow.run_bulk_simulation("single_cap", params, config))
    summary_path = workflow._resolve_mixed_dataset_result_dir(generated, "single_cap") / "mixed_dataset_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["gep_retained_n_samples_by_cancer_type"] == {"X": 3}
