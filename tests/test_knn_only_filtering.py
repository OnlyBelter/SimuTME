from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import simutme.workflow as workflow
from simutme.simulation.local_filtering import LocalTCGAFilter
from simutme.simulation.simulator import GEPGenerationConfig
from tests.test_correlation_first_filtering import read_summary, workflow_case
from tests.test_local_tcga_filtering import local_workflow_case, make_local


@pytest.mark.parametrize("method,expected_ids", [
    ("knn", ["close"]), ("knn_residual", []),
])
def test_residual_failure_rejects_only_in_strict_mode(tmp_path, monkeypatch, method, expected_ids):
    space = make_local(tmp_path, filtering_method=method)
    space.prepare_calibration()
    profiles = space.reference_linear.iloc[:2].copy()
    profiles.index = ["close", "far"]

    def controlled_scores(log):
        return pd.DataFrame(
            {"knn_distance": [0.0, space.knn_threshold + 100],
             "pca_residual": [space.residual_threshold + 100] * 2},
            index=log.index,
        )

    monkeypatch.setattr(space, "score_log_profiles", controlled_scores)
    accepted = space.filter(profiles)
    assert accepted.index.tolist() == expected_ids
    assert not space.last_scores.residual_pass.any()
    assert space.residual_after_knn_kept == 0
    assert space.distance_kept == len(expected_ids)
    space.save_batch_scores(accepted.index)
    space.validate_saved_scores(len(expected_ids))
    if method == "knn":
        summary = space.summary()
        assert summary["residual_filter_enabled"] is False
        assert summary["residual_diagnostic_pass_n_samples"] == 0
        assert summary["local_filter_kept_n_samples"] == 1


def test_knn_calibration_joint_flag_ignores_diagnostic_residual(tmp_path):
    space = make_local(
        tmp_path, filtering_method="knn", knn_distance_quantile=1.0, pca_residual_quantile=0.0,
    )
    space.prepare_calibration()
    assert space.calibration_scores.joint_pass.all()
    assert not space.calibration_scores.residual_pass.all()
    assert space.workflow_version == "correlation-first-knn-v3"
    assert space.identity["residual_filter_enabled"] is False


@pytest.mark.parametrize("flag", ["knn_pass", "residual_pass"])
def test_knn_checkpoint_rejects_flags_that_contradict_scores(tmp_path, flag):
    space = make_local(tmp_path, filtering_method="knn")
    space.prepare_calibration()
    accepted = space.filter(space.reference_linear.iloc[:2])
    space.save_batch_scores(accepted.index)
    scores = pd.read_csv(space.candidate_scores_path, index_col=0)
    scores.loc[scores.index[0], flag] = not scores.loc[scores.index[0], flag]
    if flag == "knn_pass":
        scores["joint_pass"] = scores.knn_pass
        scores["exported"] = False
    scores.to_csv(space.candidate_scores_path, float_format="%.17g")
    with pytest.raises(ValueError, match="score comparisons"):
        space.validate_saved_scores(int(scores.exported.sum()))


@pytest.mark.parametrize("first_method,second_method", [
    ("knn", "knn_residual"), ("knn_residual", "knn"),
])
def test_cross_method_model_and_calibration_cache_is_rejected(tmp_path, first_method, second_method):
    first = make_local(tmp_path, filtering_method=first_method)
    first.prepare_calibration()
    second = make_local(tmp_path, filtering_method=second_method)
    with pytest.raises(ValueError, match="Incompatible"):
        second.validate_model()


def test_knn_exports_with_failing_residual_can_be_cached_and_resumed(local_workflow_case, monkeypatch):
    config, params, _, batches = local_workflow_case
    params["gep_filtering_method"] = "knn"
    original = LocalTCGAFilter.score_log_profiles

    def high_simulated_residual(space, expression):
        scores = original(space, expression)
        if expression.index.astype(str).str.startswith("s_").all():
            scores.pca_residual = space.residual_threshold + 100
        return scores

    monkeypatch.setattr(LocalTCGAFilter, "score_log_profiles", high_simulated_residual)
    output = workflow.run_bulk_simulation("target", params, config)
    assert "with_filtering_knn/" in output
    summary = read_summary(output)
    assert summary["filtering_workflow_version"] == "correlation-first-knn-v3"
    assert summary["residual_filter_enabled"] is False
    assert summary["local_filter_kept_n_samples"] == 3
    assert summary["knn_residual_joint_kept_n_samples"] == 0
    scores = pd.read_csv(summary["gep_filtering_candidate_scores_file"])
    assert scores.exported.all() and scores.joint_pass.all()
    assert not scores.residual_pass.any()
    assert len(ad.read_h5ad(output)) == 3
    workflow.run_bulk_simulation("target", params, config)
    Path(output).unlink()
    workflow.run_bulk_simulation("target", params, config)
    assert len(batches) == 1
    assert read_summary(output)["local_filter_kept_n_samples"] == 3


def test_low_level_generator_requires_matching_knn_context(tmp_path):
    with pytest.raises(ValueError, match="prepared LocalTCGAFilter"):
        GEPGenerationConfig(n_samples=1, filtering_method="knn")
    strict = make_local(tmp_path)
    with pytest.raises(ValueError, match="matching"):
        GEPGenerationConfig(n_samples=1, filtering_method="knn", filtering_space=strict)
    knn = make_local(tmp_path / "knn", filtering_method="knn")
    config = GEPGenerationConfig(n_samples=1, filtering_method="knn", filtering_space=knn)
    assert config.filtering_space is knn


def test_disabled_knn_does_not_split_or_calibrate(workflow_case, monkeypatch):
    config, params, _, _ = workflow_case
    params["gep_filtering_method"] = "knn"
    monkeypatch.setattr(workflow, "split_tcga_reference", lambda *args, **kwargs: pytest.fail("Split ran"))
    output = workflow.run_bulk_simulation("target", params, config)
    assert read_summary(output)["gep_filtering_n_pcs"] is None


def test_knn_generation_bound_still_uses_neighbor_acceptance(local_workflow_case, monkeypatch):
    config, params, _, batches = local_workflow_case
    params.update(gep_filtering_method="knn", gep_max_generation_batches=1)
    original = LocalTCGAFilter.score_log_profiles

    def distant_candidates(space, expression):
        scores = original(space, expression)
        if expression.index.astype(str).str.startswith("s_").all():
            scores.knn_distance = space.knn_threshold + 100
            scores.pca_residual = 0.0
        return scores

    monkeypatch.setattr(LocalTCGAFilter, "score_log_profiles", distant_candidates)
    with pytest.raises(RuntimeError, match="knn reached.*residual_filter_enabled=False"):
        workflow.run_bulk_simulation("target", params, config)
    assert len(batches) == 1


def test_knn_residual_plot_is_labeled_diagnostic_only(local_workflow_case, monkeypatch):
    from matplotlib.axes import Axes

    config, params, _, _ = local_workflow_case
    params.update(gep_filtering_method="knn", plot_pca_for_comparison=True)
    titles, line_labels = [], []
    original_set, original_line = Axes.set, Axes.axvline

    def recording_set(ax, **kwargs):
        if "title" in kwargs:
            titles.append(kwargs["title"])
        return original_set(ax, **kwargs)

    def recording_line(ax, *args, **kwargs):
        line_labels.append(kwargs.get("label"))
        return original_line(ax, *args, **kwargs)

    monkeypatch.setattr(Axes, "set", recording_set)
    monkeypatch.setattr(Axes, "axvline", recording_line)
    output = workflow.run_bulk_simulation("target", params, config)
    assert "PCA residual (diagnostic only)" in titles
    assert "Residual percentile (diagnostic only)" in line_labels
    assert Path(read_summary(output)["score_distribution_plot_file"]).exists()
