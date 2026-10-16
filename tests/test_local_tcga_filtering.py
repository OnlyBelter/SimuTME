import json
from dataclasses import replace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import simutme.workflow as workflow
import simutme.simulation.filtering as base_filtering
from simutme.simulation.filtering import normalize_selected
from simutme.simulation.local_filtering import LocalTCGAFilter, split_tcga_reference
from simutme.simulation.simulator import GEPGenerationConfig
from tests.test_correlation_first_filtering import gep_params, read_summary, workflow_case


def reference_data(n=20):
    values = np.linspace(1, 9, n)
    ids = [f"TCGA-AA-{i:04d}-01A" for i in range(n)]
    expression = pd.DataFrame(
        {"a": values, "b": 10 - values, "unused": np.full(n, 100.0)}, index=ids
    )
    annotation = pd.DataFrame(
        {"cancer_type": ["A"] * (n // 2) + ["B"] * (n - n // 2)}, index=ids
    )
    return expression, annotation


def local_params(**overrides):
    values = dict(filtering_method="knn_residual", norm_ord=2, knn_n_neighbors=2)
    values.update(overrides)
    return replace(gep_params(), **values)


def make_local(tmp_path, expression=None, annotation=None, **overrides):
    default_expression, default_annotation = reference_data()
    expression = default_expression if expression is None else expression
    annotation = default_annotation if annotation is None else annotation
    gep = local_params(**overrides)
    eligible, split, summary = split_tcga_reference(
        expression, annotation, gep.reference_holdout_fraction, gep.reference_holdout_seed
    )
    return LocalTCGAFilter(
        ["a", "b", "unused"], eligible, tmp_path / "reference.csv", None,
        gep, tmp_path / "result", 3, 3, split, summary, tmp_path / "annotation.csv",
    )


def test_stratified_ceiling_counts_and_complete_disjoint_roles():
    reference, annotation = reference_data(16)
    annotation.cancer_type = ["A"] * 11 + ["B"] * 5
    eligible, split, summary = split_tcga_reference(reference, annotation)
    counts = summary.set_index("cancer_type")
    assert counts.loc["A", "n_held_out"] == 3
    assert counts.loc["B", "n_held_out"] == 1
    fitting = set(split.index[split.role == "fitting"])
    heldout = set(split.index[split.role == "held_out"])
    assert not fitting & heldout
    assert fitting | heldout == set(reference.index) == set(eligible.index)


def test_split_stable_under_row_order_and_unrelated_strata():
    expression, annotation = reference_data()
    _, first, _ = split_tcga_reference(expression, annotation)
    _, reordered, _ = split_tcga_reference(expression.iloc[::-1], annotation.iloc[::-1])
    pd.testing.assert_frame_equal(first, reordered)
    additional = expression.iloc[:5].copy()
    additional.index = [f"extra_{i}" for i in range(5)]
    extra_annotation = pd.DataFrame({"cancer_type": ["0_unrelated"] * 5}, index=additional.index)
    _, expanded, _ = split_tcga_reference(
        pd.concat([expression, additional]), pd.concat([annotation, extra_annotation])
    )
    pd.testing.assert_frame_equal(first, expanded.loc[first.index])
    _, subset, _ = split_tcga_reference(expression, annotation, ref_types=["A"])
    pd.testing.assert_frame_equal(first.loc[annotation.cancer_type == "A"], subset)


@pytest.mark.parametrize("problem,match", [
    ("missing_id", "missing reference sample"),
    ("duplicate_reference", "duplicate"),
    ("duplicate_annotation", "duplicate"),
    ("missing_label", "missing or blank"),
    ("blank_label", "missing or blank"),
    ("missing_column", "cancer_type column"),
    ("singleton_type", "infeasible"),
])
def test_invalid_annotation_fails_fast(problem, match):
    expression, annotation = reference_data()
    if problem == "missing_id":
        annotation = annotation.iloc[1:]
    elif problem == "duplicate_reference":
        expression = pd.concat([expression, expression.iloc[:1]])
    elif problem == "duplicate_annotation":
        annotation = pd.concat([annotation, annotation.iloc[:1]])
    elif problem == "missing_label":
        annotation.iloc[0, 0] = np.nan
    elif problem == "blank_label":
        annotation.iloc[0, 0] = " "
    elif problem == "missing_column":
        annotation.columns = ["other"]
    else:
        annotation.iloc[0, 0] = "rare"
    with pytest.raises(ValueError, match=match):
        split_tcga_reference(expression, annotation)


@pytest.mark.parametrize("override,match", [
    ({"reference_holdout_fraction": 0}, "holdout_fraction"),
    ({"reference_holdout_fraction": np.nan}, "holdout_fraction"),
    ({"reference_holdout_seed": -1}, "holdout_seed"),
    ({"reference_holdout_seed": True}, "holdout_seed"),
    ({"knn_n_neighbors": 0}, "knn_n_neighbors"),
    ({"knn_n_neighbors": 100}, "fewer samples"),
    ({"knn_distance_quantile": 1.01}, "knn_distance_quantile"),
    ({"pca_residual_quantile": -0.1}, "pca_residual_quantile"),
    ({"max_generation_batches": 0}, "max_generation_batches"),
    ({"filtering_in_pca_space": False}, "in_pca_space"),
])
def test_local_parameter_validation(tmp_path, override, match):
    with pytest.raises(ValueError, match=match):
        make_local(tmp_path, **override)


def test_low_level_api_cannot_silently_bypass_local_calibration(tmp_path):
    with pytest.raises(ValueError, match="prepared LocalTCGAFilter"):
        GEPGenerationConfig(n_samples=1, filtering_method="knn_residual")
    disabled = GEPGenerationConfig(n_samples=1, filtering=False, filtering_method="knn_residual")
    assert disabled.filtering_space is None
    space = make_local(tmp_path)
    with pytest.raises(ValueError, match="requires filtering"):
        GEPGenerationConfig(n_samples=1, filtering=False, filtering_method="knn_residual",
                            filtering_space=space)


def test_only_fitting_samples_reach_pca_and_ranges(tmp_path, monkeypatch):
    space = make_local(
        tmp_path, filtering_by_gene_range=True, gene_quantile_range=[0.1, 0.5, 0.9]
    )
    captured = []
    original_fit = base_filtering.PCA.fit

    def capture(model, data, *args, **kwargs):
        captured.append(np.asarray(data).copy())
        return original_fit(model, data, *args, **kwargs)

    monkeypatch.setattr(base_filtering.PCA, "fit", capture)
    space.prepare_calibration()
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0], space.reference_log)
    assert space.neighbors.reference_sample_ids_ == space.fitting_ids
    assert not set(space.neighbors.reference_sample_ids_) & set(space.heldout_ids)
    pd.testing.assert_frame_equal(
        space.range_bounds, space.reference_linear.quantile([0.1, 0.9])
    )
    assert len(space.calibration_scores) == len(space.heldout_ids)
    assert space.knn_threshold == space.calibration_scores.knn_distance.quantile(
        space.gep.knn_distance_quantile
    )


def test_heldout_changes_do_not_change_fitted_pca(tmp_path):
    expression, annotation = reference_data()
    _, split, _ = split_tcga_reference(expression, annotation)
    modified = expression.copy()
    modified.loc[split.index[split.role == "held_out"], "unused"] *= 100
    first = make_local(tmp_path / "first", expression, annotation)
    second = make_local(tmp_path / "second", modified, annotation)
    first.prepare_calibration()
    second.prepare_calibration()
    np.testing.assert_allclose(first.pca.mean_, second.pca.mean_)
    np.testing.assert_allclose(first.pca.components_, second.pca.components_)
    assert second.residual_threshold > first.residual_threshold


def test_scores_equal_manual_distances_and_reconstruction(tmp_path):
    space = make_local(tmp_path)
    space.prepare_calibration()
    query = space.heldout_log.iloc[:1]
    actual = space.score_log_profiles(query)
    centered = query.to_numpy() - space.pca.mean_
    scores = np.sum(centered[:, None, :] * space.pca.components_[None, :, :], axis=2)
    all_distances = np.linalg.norm(space.reference_coordinates - scores[0], axis=1)
    expected_distance = np.sort(all_distances)[:space.gep.knn_n_neighbors].mean()
    reconstructed = np.sum(scores[:, :, None] * space.pca.components_[None, :, :], axis=1) + space.pca.mean_
    expected_residual = np.linalg.norm(query.to_numpy() - reconstructed, axis=1)
    np.testing.assert_allclose(actual.knn_distance, expected_distance)
    np.testing.assert_allclose(actual.pca_residual, expected_residual)


def test_cached_calibration_has_identical_scores_and_no_refit(tmp_path, monkeypatch):
    first = make_local(tmp_path)
    first.prepare_calibration()
    query = first.heldout_log.copy()
    expected = first.score_log_profiles(query)
    cached = make_local(tmp_path)
    monkeypatch.setattr(base_filtering.PCA, "fit", lambda *args, **kwargs: pytest.fail("PCA refitted"))
    cached.prepare_calibration()
    pd.testing.assert_frame_equal(cached.score_log_profiles(query), expected)
    assert cached.knn_threshold == first.knn_threshold
    assert cached.residual_threshold == first.residual_threshold


def test_empty_and_partial_batches_keep_ids_and_export_flags(tmp_path):
    space = make_local(tmp_path)
    space.prepare_calibration()
    query = space.reference_linear.iloc[:3].copy()
    query.index = ["s3", "s1", "s2"]
    expected = space.score_log_profiles(np.log2(normalize_selected(query, space.genes) + 1))
    space.knn_threshold = float(expected.knn_distance.median())
    space.residual_threshold = float(expected.pca_residual.max() + 1)
    filtered = space.filter(query)
    expected_ids = expected.index[expected.knn_distance <= space.knn_threshold].tolist()
    assert filtered.index.tolist() == expected_ids
    space.save_batch_scores(filtered.index[:1])
    saved = pd.read_csv(space.candidate_scores_path, index_col=0)
    assert saved.index.tolist() == query.index.tolist()
    assert saved.exported.sum() == 1
    assert space.filter(query.iloc[:0]).empty
    assert space.generation_batches == 2


@pytest.fixture
def local_workflow_case(workflow_case):
    config, params, source, batches = workflow_case
    expression, annotation = reference_data()
    expression.to_csv(config.tcga_merged_tpm_file_path)
    annotation.to_csv(config.tcga2cancer_type_file_path)
    params.update(
        gep_filtering=True, gep_filtering_method="knn_residual",
        gep_knn_n_neighbors=2, gep_knn_distance_quantile=1.0,
        gep_pca_residual_quantile=1.0, gep_norm_ord=2,
    )
    return config, params, source, batches


def test_local_workflow_calibrates_before_generation_and_reuses_cache(local_workflow_case, monkeypatch):
    config, params, _, batches = local_workflow_case
    original_filter = LocalTCGAFilter.filter

    def checked_filter(space, profiles):
        assert space.calibration_scores is not None
        assert space.calibration_scores_path.exists()
        return original_filter(space, profiles)

    monkeypatch.setattr(LocalTCGAFilter, "filter", checked_filter)
    output = workflow.run_bulk_simulation("target", params, config)
    summary = read_summary(output)
    assert summary["filtering_workflow_version"] == "correlation-first-local-v2"
    assert summary["reference_fitting_n_samples"] == 16
    assert summary["reference_heldout_n_samples"] == 4
    assert summary["reference_split_unit"] == "sample"
    assert summary["exported_n_samples"] == 3
    assert ad.read_h5ad(output).var_names.tolist() == ["a", "b"]
    scores = pd.read_csv(summary["gep_filtering_candidate_scores_file"])
    assert scores.exported.sum() == 3
    workflow.run_bulk_simulation("target", params, config)
    assert len(batches) == 1
    Path(output).unlink()
    workflow.run_bulk_simulation("target", params, config)
    assert len(batches) == 1
    assert read_summary(output)["generation_batches"] == 1


@pytest.mark.parametrize("change", ["seed", "neighbors", "quantile", "bound", "annotation", "missing_scores"])
def test_new_artifacts_reject_changed_settings_without_overwrite(local_workflow_case, change):
    config, params, _, _ = local_workflow_case
    output = workflow.run_bulk_simulation("target", params, config)
    before = Path(output).read_bytes()
    if change == "seed":
        params["gep_reference_holdout_seed"] = 123
    elif change == "neighbors":
        params["gep_knn_n_neighbors"] = 3
    elif change == "quantile":
        params["gep_pca_residual_quantile"] = 0.95
    elif change == "bound":
        params["gep_max_generation_batches"] = 99
    elif change == "annotation":
        labels = pd.read_csv(config.tcga2cancer_type_file_path, index_col=0)
        labels.iloc[[0, -1], 0] = labels.iloc[[-1, 0], 0].to_numpy()
        labels.to_csv(config.tcga2cancer_type_file_path)
    else:
        Path(read_summary(output)["tcga_heldout_calibration_scores_file"]).unlink()
    with pytest.raises(ValueError):
        workflow.run_bulk_simulation("target", params, config)
    assert Path(output).read_bytes() == before


def test_missing_batch_counter_cannot_reset_zero_survivor_run(local_workflow_case, monkeypatch):
    config, params, _, batches = local_workflow_case
    params.update(gep_max_generation_batches=1, gep_filtering_by_gene_range=True,
                  gep_min_percentage_within_gene_range=1.0)
    original = LocalTCGAFilter.filter

    def reject_by_range(space, profiles):
        profiles = profiles.copy()
        profiles.loc[:, "a"] = 0
        profiles.loc[:, "b"] = 1e6
        return original(space, profiles)

    monkeypatch.setattr(LocalTCGAFilter, "filter", reject_by_range)
    with pytest.raises(RuntimeError, match="max_generation_batches=1"):
        workflow.run_bulk_simulation("target", params, config)
    metadata_file = next(config.simu_bulk_dir.rglob("filtering_metadata.json"))
    metadata = json.loads(metadata_file.read_text())
    assert metadata["stats"]["pc_distance_input_n_samples"] == 0
    metadata["stats"].pop("generation_batches")
    metadata_file.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checkpoint statistics"):
        workflow.run_bulk_simulation("target", params, config)
    assert len(batches) == 1


def test_batch_limit_and_zero_survivor_resume_preserve_artifacts(local_workflow_case, monkeypatch):
    config, params, _, batches = local_workflow_case
    params["gep_max_generation_batches"] = 2
    original = LocalTCGAFilter.filter

    def reject_all(space, profiles):
        profiles = profiles.copy()
        profiles.loc[:, "a"] = 0
        profiles.loc[:, "b"] = 1e6
        return original(space, profiles)

    monkeypatch.setattr(LocalTCGAFilter, "filter", reject_all)
    with pytest.raises(RuntimeError, match="max_generation_batches=2.*exported=0"):
        workflow.run_bulk_simulation("target", params, config)
    assert len(batches) == 2
    metadata_files = list(config.simu_bulk_dir.rglob("filtering_metadata.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text())
    assert metadata["stats"]["generation_batches"] == 2
    assert metadata["generator_stats"]["n_round"] == 2
    candidate_file = metadata_files[0].parent / "gep_filtering_candidate_scores.csv"
    before = candidate_file.read_bytes()
    with pytest.raises(RuntimeError, match="max_generation_batches=2"):
        workflow.run_bulk_simulation("target", params, config)
    assert len(batches) == 2
    assert candidate_file.read_bytes() == before


def test_disabled_new_method_does_not_split_or_fit(workflow_case, monkeypatch):
    config, params, _, _ = workflow_case
    params["gep_filtering_method"] = "knn_residual"
    monkeypatch.setattr(workflow, "split_tcga_reference", lambda *args, **kwargs: pytest.fail("Split ran"))
    output = workflow.run_bulk_simulation("target", params, config)
    assert read_summary(output)["gep_filtering_n_pcs"] is None
    assert not list(config.simu_bulk_dir.rglob("tcga_reference_split.csv"))


def test_diagnostics_use_all_tcga_and_actual_filtering_model(local_workflow_case, monkeypatch):
    config, params, _, _ = local_workflow_case
    params.update(plot_pca_for_comparison=True, gep_pca_n_components=2)
    joint_inputs = []
    original = workflow.do_pca_analysis

    def capture(exp_df, **kwargs):
        joint_inputs.append(exp_df.copy())
        return original(exp_df=exp_df, **kwargs)

    monkeypatch.setattr(workflow, "do_pca_analysis", capture)
    output = workflow.run_bulk_simulation("target", params, config)
    summary = read_summary(output)
    assert len(joint_inputs[0]) == 23
    coordinates = pd.read_csv(summary["reference_pca_data_file"])
    assert coordinates.source.value_counts().to_dict() == {
        "Fitting TCGA": 16, "Held-out TCGA": 4, "Simulated": 3,
    }
    for key in ("reference_pca_plot_file", "reference_pca_pdf_file",
                "score_distribution_plot_file", "score_distribution_pdf_file"):
        assert Path(summary[key]).exists()
    assert summary["pca_model_file"] != summary["gep_filtering_pca_model_file"]


def test_individual_sample_split_reports_patient_overlap(tmp_path):
    expression, annotation = reference_data(10)
    ids = expression.index.tolist()
    ids[1] = "TCGA-AA-0000-02A"
    expression.index = annotation.index = ids
    seed = next(
        seed for seed in range(100)
        if split_tcga_reference(expression, annotation, seed=seed)[1].loc[ids[0], "role"]
        != split_tcga_reference(expression, annotation, seed=seed)[1].loc[ids[1], "role"]
    )
    space = make_local(tmp_path, expression, annotation, reference_holdout_seed=seed)
    space.prepare_calibration()
    assert space.summary()["reference_patient_overlap_n"] == 1
    assert space.split.loc[ids[0], "role"] != space.split.loc[ids[1], "role"]


def test_local_configuration_overrides_and_invalid_norm(local_workflow_case):
    config, params, _, _ = local_workflow_case
    config.gep_reference_holdout_seed = 123
    params["gep_reference_holdout_seed"] = 7
    resolved = workflow._resolve_gep_filtering_params("target", params, config)
    assert resolved.reference_holdout_seed == 7
    assert resolved.knn_n_neighbors == 2
    with pytest.raises(ValueError, match="norm_ord"):
        workflow._resolve_gep_filtering_params("target", {**params, "gep_norm_ord": 2.5}, config)


def test_one_filtering_pc_skips_only_reference_plot(local_workflow_case):
    config, params, _, _ = local_workflow_case
    params["plot_pca_for_comparison"] = True
    output = workflow.run_bulk_simulation("target", params, config)
    summary = read_summary(output)
    assert summary["gep_filtering_n_pcs"] == 1
    assert "reference_pca_plot_file" not in summary
    assert Path(summary["score_distribution_plot_file"]).exists()
    assert Path(summary["pca_plot_file"]).exists()


def test_candidate_flag_corruption_fails_before_reuse(local_workflow_case):
    config, params, _, _ = local_workflow_case
    output = workflow.run_bulk_simulation("target", params, config)
    score_file = Path(read_summary(output)["gep_filtering_candidate_scores_file"])
    scores = pd.read_csv(score_file, index_col=0)
    scores.iloc[0, scores.columns.get_loc("joint_pass")] = False
    scores.to_csv(score_file)
    before = Path(output).read_bytes()
    with pytest.raises(ValueError, match="inconsistent"):
        workflow.run_bulk_simulation("target", params, config)
    assert Path(output).read_bytes() == before
