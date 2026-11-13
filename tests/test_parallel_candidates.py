import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import simutme.simulation.parallel_candidates as parallel
import simutme.simulation.simulator as simulator
import simutme.workflow as workflow
from simutme.simulation.filtering import FixedGeneFilter
from simutme.simulation.sampling import generate_cell_fraction
from simutme.simulation.simulator import BulkGEPGenerator, GEPGenerationConfig
from simutme.workflow import SimuConfig, _resolve_parallel_candidate_settings
from tests.test_correlation_first_filtering import gep_params


@pytest.fixture
def setup_parallel(tmp_path, monkeypatch):
    monkeypatch.setattr(simulator, "_candidate_batch_size", lambda *args: 5)
    monkeypatch.setattr(parallel, "_candidate_batch_size", lambda *args: 5)
    source = tmp_path / "sct.h5ad"
    ad.AnnData(
        X=np.log2(np.array([[1., 9., 4.], [3., 7., 8.],
                            [8., 2., 6.], [9., 1., 3.]]) + 1),
        obs=pd.DataFrame({"A": [1., 1., 0., 0.], "B": [0., 0., 1., 1.]},
                         index=["a1", "a2", "b1", "b2"]),
        var=pd.DataFrame(index=["g1", "g2", "unused"]),
    ).write_h5ad(source)
    reference = pd.DataFrame({"g1": [0., 5., 10.], "g2": [10., 5., 0.]})

    def create(name, workers=1, seed=42, method="mgep", quantiles=(None, 1.0)):
        output = tmp_path / name
        generator = BulkGEPGenerator(
            simu_bulk_dir=output, merged_sc_dataset_file_path=None,
            sct_dataset_file_path=source, cell_type2subtype={"A": ["A"], "B": ["B"]},
            sc_dataset_ids=[], bulk_dataset_name="test", check_basic_info=False,
            total_rna_coefficient={"A": 1., "B": 1.}, subtype_col_name="cell_type",
            output_types=["A", "B"], mixed_sct_loading_mode="eager",
        )
        space = FixedGeneFilter(
            ["g1", "g2"], reference, tmp_path / "reference.csv", None,
            gep_params(
                filtering_method=method,
                filtering_in_pca_space=(method == "median_gep"),
                gep_filtering_quantile=quantiles,
            ),
            output / "test", 3, 2,
        )
        config = GEPGenerationConfig(
            n_samples=12, simu_method="mul", sampling_method_prop="segment",
            filtering=True, filtering_method=method, filtering_space=space,
            filtering_in_pca_space=(method == "median_gep"), norm_ord=1,
            mixed_candidate_workers=workers, mixed_candidate_seed=seed,
        )
        return generator, config

    return create


@pytest.mark.parametrize(
    "method,quantiles", [("mgep", (None, 1.0)), ("median_gep", (0.05, 0.95))],
)
def test_parallel_workers_produce_identical_ordered_outputs(setup_parallel, method, quantiles):
    first, first_config = setup_parallel("one", workers=1, method=method, quantiles=quantiles)
    second, second_config = setup_parallel("two", workers=2, method=method, quantiles=quantiles)
    first.generate_gep(first_config)
    second.generate_gep(second_config)
    for first_path, second_path in [
        (first.generated_bulk_gep_csv_fp, second.generated_bulk_gep_csv_fp),
        (first.generated_cell_fraction_fp, second.generated_cell_fraction_fp),
        (first.sampled_sc_cell_id_file_path, second.sampled_sc_cell_id_file_path),
    ]:
        pd.testing.assert_frame_equal(pd.read_csv(first_path), pd.read_csv(second_path))
    assert first.generated_bulk_gep_counter == second.generated_bulk_gep_counter == 12
    assert first.gep_filtering_input_n_samples == second.gep_filtering_input_n_samples >= 15
    assert first.gep_filtering_kept_n_samples == second.gep_filtering_kept_n_samples >= 12
    assert ad.read_h5ad(first.generated_bulk_gep_fp).n_obs == 12
    checkpoint = json.loads((first.filtering_space.result_dir
                             / "parallel_candidate_checkpoint.json").read_text())
    assert checkpoint["accepted"] == 12
    assert checkpoint["next_batch"] == first.n_round
    if method == "median_gep":
        assert first.gep_filtering_input_n_samples > first.gep_filtering_kept_n_samples
        assert first.filtering_space.stats()["pc_distance_input_n_samples"] == first.gep_filtering_input_n_samples
        assert first.filtering_space.stats()["pc_distance_kept_n_samples"] == first.gep_filtering_kept_n_samples
        assert first.filtering_space.stats() == second.filtering_space.stats()
        assert checkpoint["generator_stats"]["gep_filtering_n_pcs"] == 1
    else:
        assert first.gep_filtering_input_n_samples == first.gep_filtering_kept_n_samples == 15
        assert checkpoint["next_batch"] == 3


@pytest.mark.parametrize("method", ["mgep", "median_gep"])
def test_worker_matches_serial_candidate_replay(setup_parallel, monkeypatch, method):
    generator, config = setup_parallel("replay", method=method)
    generator.filtering_space = config.filtering_space
    generator.filtering_space.prepare_distance_reference()
    obs, _, sc_dataset = generator._load_sc_data_for_generation("mul")
    worker_config = {
        "seed": 42, "batch_size": 5, "prior": None,
        "add_noise": False, "noise_params": (),
    }
    monkeypatch.setattr(parallel, "_WORKER", (generator, obs, worker_config))
    actual = parallel._run_batch(2)
    np.random.seed(np.random.SeedSequence([42, 2]).generate_state(4))
    fractions = generate_cell_fraction(
        sampling_method="segment", n_cell_frac=5, sample_prefix="s_segment_2",
        cell_type2subtype=generator.cell_fraction_type2subtype,
    )
    selected = generator._sc_sampling(fractions, obs, n_threads=1, sc_dataset=sc_dataset)
    expression = generator._map_cell_id2exp(
        selected, sc_dataset=sc_dataset, simu_method="mul", cell_frac=fractions,
    )
    expected = generator.filtering_space.filter(expression)
    pd.testing.assert_frame_equal(actual["expression"], expected)
    pd.testing.assert_frame_equal(actual["fractions"], fractions.loc[expected.index])
    pd.testing.assert_frame_equal(actual["selected"], selected.loc[expected.index])


@pytest.mark.parametrize(
    "method,quantiles", [("mgep", (None, 1.0)), ("median_gep", (0.05, 0.95))],
)
def test_parallel_resume_after_committed_batch(setup_parallel, monkeypatch, method, quantiles):
    baseline, baseline_config = setup_parallel("baseline", method=method, quantiles=quantiles)
    baseline.generate_gep(baseline_config)
    interrupted, config = setup_parallel("resume", workers=2, method=method, quantiles=quantiles)
    original = interrupted._save_parallel_checkpoint

    def interrupt_after_batch():
        original()
        if interrupted.n_round == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(interrupted, "_save_parallel_checkpoint", interrupt_after_batch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted.generate_gep(config)
    resumed, resumed_config = setup_parallel("resume", workers=1, method=method, quantiles=quantiles)
    resumed.generate_gep(resumed_config)
    pd.testing.assert_frame_equal(
        pd.read_csv(baseline.generated_bulk_gep_csv_fp),
        pd.read_csv(resumed.generated_bulk_gep_csv_fp),
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(baseline.generated_cell_fraction_fp),
        pd.read_csv(resumed.generated_cell_fraction_fp),
    )
    assert baseline.filtering_space.stats() == resumed.filtering_space.stats()


def test_parallel_rejects_partial_output_and_changed_seed(setup_parallel):
    generator, config = setup_parallel("partial")
    generator.generate_gep(config)
    changed, changed_config = setup_parallel("partial", seed=43)
    with pytest.raises(ValueError, match="Incompatible filtering metadata"):
        changed.generate_gep(changed_config)
    with generator.generated_bulk_gep_csv_fp.open("a") as handle:
        handle.write("partial")
    broken, broken_config = setup_parallel("partial")
    with pytest.raises(ValueError, match="checkpoint or intermediate files"):
        broken.generate_gep(broken_config)


@pytest.mark.parametrize("method", ["mgep", "median_gep"])
def test_checkpoint_without_survivors_preserves_batch_progress(setup_parallel, method):
    generator, config = setup_parallel("empty", method=method)
    generator.filtering_space = config.filtering_space
    generator.parallel_worker_count = 1
    generator._reset_gep_filtering_stats(True)
    generator._prepare_filtering_artifacts(config)
    generator.n_round = 2
    generator._record_gep_filtering_counts(10, 0)
    if method == "median_gep":
        generator.filtering_space.distance_input = 10
    else:
        generator.filtering_space.mgep_distance_input = 10
    generator._save_filtering_metadata()
    generator._save_parallel_checkpoint()
    resumed, resumed_config = setup_parallel("empty", method=method)
    resumed.generate_gep(resumed_config)
    assert resumed.n_round == 5
    assert resumed.gep_filtering_input_n_samples == 25
    assert resumed.generated_bulk_gep_counter == 12
    if method == "median_gep":
        assert resumed.filtering_space.stats()["pc_distance_input_n_samples"] == 25


def test_parallel_worker_failure_preserves_checkpoint(setup_parallel):
    generator, config = setup_parallel("failure")
    source = ad.read_h5ad(generator.sct_dataset_file_path)
    source.obs["A"] = 1.0
    source.obs["B"] = 0.0
    source.write_h5ad(generator.sct_dataset_file_path)
    with pytest.raises(RuntimeError, match="Parallel candidate batch 0 failed"):
        generator.generate_gep(config)
    checkpoint = json.loads((generator.filtering_space.result_dir
                             / "parallel_candidate_checkpoint.json").read_text())
    assert checkpoint["next_batch"] == checkpoint["accepted"] == 0
    assert all(size is None for size in checkpoint["output_sizes"].values())


def test_parallel_refuses_existing_run_lock(setup_parallel):
    generator, config = setup_parallel("locked")
    lock = generator.simu_bulk_dir / ".test.parallel.lock"
    lock.write_text("existing lock")
    with pytest.raises(ValueError, match="lock already exists"):
        generator.generate_gep(config)
    assert lock.read_text() == "existing lock"


@pytest.mark.parametrize("workers,seed", [(-1, 42), (True, 42), (2, None), (2, -1)])
def test_parallel_invalid_settings(workers, seed):
    config = SimpleNamespace(
        mixed_candidate_workers=workers, mixed_candidate_seed=seed,
        gep_filtering_enabled_default=True, gep_filtering_method="mgep",
        mixed_sct_loading_mode="eager",
    )
    with pytest.raises(ValueError, match="mixed_candidate"):
        _resolve_parallel_candidate_settings("target", {"dataset_type": "Mixed"}, config)


def test_parallel_dataset_override_and_restrictions():
    config = SimpleNamespace(
        mixed_candidate_workers=0, mixed_candidate_seed=None,
        gep_filtering_enabled_default=True, gep_filtering_method="mgep",
        mixed_sct_loading_mode="eager",
    )
    params = {"dataset_type": "Mixed", "mixed_candidate_workers": 4,
              "mixed_candidate_seed": 42}
    assert _resolve_parallel_candidate_settings("target", params, config) == (4, 42)
    with pytest.raises(ValueError, match="segment sampling"):
        _resolve_parallel_candidate_settings(
            "target", {**params, "sampling_method_prop": "dirichlet"}, config
        )
    config.mixed_sct_loading_mode = "lazy"
    with pytest.raises(ValueError, match="eager"):
        _resolve_parallel_candidate_settings("target", params, config)


def test_parallel_median_requires_pca_l1_and_respects_overrides(setup_parallel):
    config = SimpleNamespace(
        mixed_candidate_workers=0, mixed_candidate_seed=None,
        gep_filtering_enabled_default=True, gep_filtering_method="median_gep",
        gep_filtering_in_pca_space=True, gep_norm_ord=1, mixed_sct_loading_mode="eager",
    )
    params = {"dataset_type": "Mixed", "sampling_method_prop": "segment",
              "mixed_candidate_workers": 2, "mixed_candidate_seed": 42}
    assert _resolve_parallel_candidate_settings("target", params, config) == (2, 42)
    for override in (
        {"gep_filtering_in_pca_space": False},
        {"filtering_in_pca_space": False},
        {"gep_norm_ord": 2},
    ):
        with pytest.raises(ValueError, match="PCA-space L1"):
            _resolve_parallel_candidate_settings("target", {**params, **override}, config)
    for override in ({"gep_filtering_method": "mean_gep"}, {"gep_filtering": False}):
        with pytest.raises(ValueError, match="segment sampling"):
            _resolve_parallel_candidate_settings("target", {**params, **override}, config)
    assert _resolve_parallel_candidate_settings(
        "target", {**params, "mixed_candidate_workers": 0, "gep_norm_ord": 2}, config,
    ) == (0, 42)
    _, generation = setup_parallel("invalid-median", method="median_gep")
    with pytest.raises(ValueError, match="PCA-space L1"):
        replace(generation, norm_ord=2)
    with pytest.raises(ValueError, match="PCA-space L1"):
        replace(generation, filtering_in_pca_space=False)


@pytest.mark.parametrize("gene_range", [False, True])
def test_median_parallel_worker_resets_distance_stats(setup_parallel, monkeypatch, gene_range):
    generator, config = setup_parallel("median-counters", method="median_gep")
    generator.filtering_space = config.filtering_space
    generator.filtering_space.gep = replace(
        generator.filtering_space.gep, filtering_by_gene_range=gene_range,
    )
    generator.filtering_space.prepare_distance_reference()

    def unexpected_refit():
        pytest.fail("A worker must not refit the prepared reference PCA")

    monkeypatch.setattr(generator.filtering_space, "_prepare_pca", unexpected_refit)
    obs, _, _ = generator._load_sc_data_for_generation("mul")
    worker_config = {
        "seed": 42, "batch_size": 5, "prior": None,
        "add_noise": False, "noise_params": (),
    }
    monkeypatch.setattr(parallel, "_WORKER", (generator, obs, worker_config))
    for batch_id in range(2):
        result = parallel._run_batch(batch_id)
        assert result["stats"]["pc_distance_input_n_samples"] == result["input"] == 5
        assert result["stats"]["pc_distance_kept_n_samples"] == result["kept"]
        if gene_range:
            assert result["stats"]["gene_range_input_n_samples"] == result["input"]
            assert result["stats"]["gene_range_kept_n_samples"] == result["input"]


def test_parallel_setup_precomputes_gene_range_bounds_in_log_space(setup_parallel, monkeypatch):
    generator, config = setup_parallel("parallel-log-range", method="median_gep")
    space = config.filtering_space
    space.gep = replace(
        space.gep,
        filtering_by_gene_range=True,
        gene_quantile_range=[0.25, 0.5, 0.75],
    )
    generator.filtering_space = space
    generator._parallel_checkpoint = {"next_batch": 0, "accepted": 0}
    generator.n_samples = config.n_samples
    generator.generated_bulk_gep_counter = 0
    generator.n_round = 0
    generator.sct_dataset_df = pd.DataFrame({"g1": [1.0], "g2": [1.0]})
    generator._load_sc_data_for_generation = lambda method: (
        pd.DataFrame(), ["g1", "g2"], simulator.DatasetType.SCT,
    )
    generator._save_parallel_checkpoint = lambda: None
    space.prepare_distance_reference = lambda: None
    captured = {}

    def capture_worker_filter(_generator, _config, _obs, worker_filter):
        captured["range_bounds"] = worker_filter.range_bounds.copy()
        assert worker_filter.reference_log is None

    monkeypatch.setattr(parallel, "run_parallel_candidates", capture_worker_filter)
    generator._generate_parallel_candidates(config)

    expected = space.reference_log.quantile([0.25, 0.75])
    pd.testing.assert_frame_equal(captured["range_bounds"], expected)


def test_median_worker_replays_filtered_candidates(setup_parallel, monkeypatch):
    generator, config = setup_parallel("median-filtered-replay", method="median_gep")
    generator.filtering_space = config.filtering_space
    generator.filtering_space.gep = replace(
        generator.filtering_space.gep, gep_filtering_quantile=(0.05, 0.95),
    )
    generator.filtering_space.prepare_distance_reference()
    obs, _, sc_dataset = generator._load_sc_data_for_generation("mul")
    worker_config = {
        "seed": 42, "batch_size": 50, "prior": None,
        "add_noise": False, "noise_params": (),
    }
    monkeypatch.setattr(parallel, "_WORKER", (generator, obs, worker_config))
    actual = parallel._run_batch(2)
    np.random.seed(np.random.SeedSequence([42, 2]).generate_state(4))
    fractions = generate_cell_fraction(
        sampling_method="segment", n_cell_frac=50, sample_prefix="s_segment_2",
        cell_type2subtype=generator.cell_fraction_type2subtype,
    )
    selected = generator._sc_sampling(fractions, obs, n_threads=1, sc_dataset=sc_dataset)
    expression = generator._map_cell_id2exp(
        selected, sc_dataset=sc_dataset, simu_method="mul", cell_frac=fractions,
    )
    expected = generator.filtering_space.filter(expression)
    assert 0 < len(expected) < len(expression)
    pd.testing.assert_frame_equal(actual["expression"], expected)
    pd.testing.assert_frame_equal(actual["fractions"], fractions.loc[expected.index])
    pd.testing.assert_frame_equal(actual["selected"], selected.loc[expected.index])
    assert actual["stats"]["pc_distance_input_n_samples"] == len(expression)
    assert actual["stats"]["pc_distance_kept_n_samples"] == len(expected)


def test_median_pca_l1_uses_both_reference_quantiles(setup_parallel):
    _, config = setup_parallel("median-quantiles", method="median_gep")
    space = config.filtering_space
    space.gep = replace(space.gep, gep_filtering_quantile=(0.1, 0.4))
    reference_distances = space.prepare_distance_reference()
    simulated = pd.DataFrame(
        {"g1": [0.0, 0.001, 2.0, 5.0, 8.0, 9.999, 10.0],
         "g2": [10.0, 9.999, 8.0, 5.0, 2.0, 0.001, 0.0]},
        index=["far-left", "left", "inner-left", "center", "inner-right", "right", "far-right"],
    )
    normalized = simulated.div(simulated.sum(axis=1), axis=0) * 1e6
    coordinates = space.pca.transform(np.log2(normalized.to_numpy() + 1))
    center = np.median(space.reference_coordinates, axis=0)
    expected_distances = np.abs(coordinates - center).sum(axis=1)
    np.testing.assert_allclose(
        space.distance_from_normalized(normalized), expected_distances, atol=1e-12,
    )
    lower, upper = np.quantile(reference_distances, [0.1, 0.4])
    expected = simulated.index[(expected_distances >= lower) & (expected_distances <= upper)]
    assert len(expected) > 0
    assert (expected_distances < lower).any()
    assert (expected_distances > upper).any()
    assert space.filter(simulated).index.tolist() == expected.tolist()


def test_benchmark_prepares_median_pca_worker_state(setup_parallel, tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/benchmark_parallel_candidates.py"
    spec = importlib.util.spec_from_file_location("candidate_benchmark", script)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    generator, _ = setup_parallel("benchmark-source", method="median_gep")
    genes = tmp_path / "genes.csv"
    pd.DataFrame({"gene_name": ["g1", "g2"]}).to_csv(genes, index=False)
    reference = pd.DataFrame({"g1": [0., 5., 10.], "g2": [10., 5., 0.]})
    config = SimpleNamespace(
        _dataset_definitions={"target": {"sampling_method_prop": "segment"}},
        cell_type2subtypes={"A": ["A"], "B": ["B"]}, output_types=["A", "B"],
        total_rna_coefficient={"A": 1., "B": 1.}, pred_cell_prop_file_path=None,
    )
    gep = gep_params(
        reference_file=str(tmp_path / "reference.csv"), filtering_method="median_gep",
        filtering_in_pca_space=True, norm_ord=1,
    )
    monkeypatch.setattr(benchmark.SimuConfig, "from_yaml", lambda path: config)
    monkeypatch.setattr(benchmark, "_resolve_gep_filtering_params", lambda *args: gep)
    monkeypatch.setattr(benchmark, "_resolve_configured_segment_prior", lambda *args: None)
    monkeypatch.setattr(benchmark, "ReadExp", lambda *args, **kwargs: SimpleNamespace(exp=reference))
    args = SimpleNamespace(
        config="config.yaml", dataset="target", gene_list=str(genes), sct=str(generator.sct_dataset_file_path),
        seed=42, batch_size=5,
    )
    _, _, space, _, _ = benchmark._workload(args, tmp_path)
    assert space.pca is not None
    assert space.reference_coordinates is None
    assert space.reference_distances is not None
    assert not space.filter(pd.DataFrame({"g1": [4.], "g2": [6.]})).empty
    capsys.readouterr()
    monkeypatch.setattr(
        sys, "argv",
        [str(script), "--config", "config.yaml", "--dataset", "target", "--sct", args.sct,
         "--gene-list", str(genes), "--workers", "0", "1", "--batch-size", "5",
         "--batches", "2", "--repeats", "1"],
    )
    benchmark.main()
    report = json.loads(capsys.readouterr().out)
    assert report["filtering_method"] == "median_gep"
    assert [result["workers"] for result in report["results"]] == [0, 1]
    assert all("filter_acceptance_rate" in result["repeats"][0] for result in report["results"])


@pytest.mark.parametrize("method", ["mgep", "median_gep"])
def test_workflow_resolves_and_runs_parallel_dataset(setup_parallel, tmp_path, monkeypatch, method):
    generator, _ = setup_parallel("unused")
    reference = tmp_path / "reference.csv"
    pd.DataFrame({"g1": [0., 5., 10.], "g2": [10., 5., 0.],
                  "unused": [5., 5., 5.]}, index=["r1", "r2", "r3"]).to_csv(reference)
    config = SimuConfig(
        simu_bulk_dir=tmp_path / "workflow", tcga2cancer_type_file_path=tmp_path / "mapping.csv",
        tcga_merged_tpm_file_path=reference, high_corr_gene_file_path=None,
        merged_sc_dataset_file_path=tmp_path / "unused.h5ad", sc_dataset_ids=[],
        cell_type2subtypes={"A": ["A"], "B": ["B"]},
        total_rna_coefficient={"A": 1., "B": 1.}, n_base=1,
        output_types=["A", "B"], gep_filtering_quantile=(None, 1.0),
        plot_pca_for_comparison=False, mixed_sct_loading_mode="eager",
    )
    params = {
        "dataset_type": "Mixed", "sampling_method_prop": "segment", "n_samples": 3,
        "sct_reference": {"source_type": "path", "file_path": str(generator.sct_dataset_file_path)},
        "gep_filtering": True, "gep_filtering_method": method,
        "gep_filtering_in_pca_space": method == "median_gep",
        "gep_pca_n_components": 1, "gep_norm_ord": 1, "gene_filtering": False,
        "mixed_candidate_workers": 1, "mixed_candidate_seed": 42,
    }
    monkeypatch.setattr(workflow, "_save_cancer_cell_fraction_distribution", lambda **kwargs: None)
    workflow.preflight_segment_cell_prop_priors(config, {"target": params})
    output = workflow.run_bulk_simulation("target", params, config)
    assert ad.read_h5ad(output).n_obs == 3
    metadata_path = workflow._resolve_mixed_dataset_result_dir(output, "target") / "filtering_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["parallel_execution"]["workers"] == 1
    assert metadata["identity"]["generation"]["parallel_candidates"]["seed"] == 42
    if method == "median_gep":
        assert metadata["stats"]["pc_distance_input_n_samples"] >= 3
        assert metadata["generator_stats"]["gep_filtering_n_pcs"] == 1
