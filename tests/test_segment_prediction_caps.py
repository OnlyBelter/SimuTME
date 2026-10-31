from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

pytest_plugins = ("tests.test_correlation_first_filtering",)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simutme.simulation.sampling import (
    estimate_cell_prop_maxima_from_pred_cell_props,
    format_segment_fraction_caps,
    generate_cell_fraction,
    merge_cell_prop_prior_with_maxima,
    resolve_segment_cell_prop_prior,
    segment_sampling,
)
from simutme.simulation.simulator import GEPGenerationConfig
from simutme.workflow import _save_cancer_cell_fraction_distribution
from simutme.config import DESIDE_TCGA_CELL_PROPORTION_RANGES


def _write_predictions(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_empirical_caps_normalize_rows_and_ignore_metadata(tmp_path):
    path = _write_predictions(tmp_path / "pred.csv", {
        "sample_id": ["s1", "s2"],
        "cancer_type": ["A", "B"],
        "A": [0.2, 2.0],
        "B": [0.3, 1.0],
        "C": [0.5, 1.0],
    })

    caps, metadata = estimate_cell_prop_maxima_from_pred_cell_props(path, ["A", "B", "C"])

    assert caps == {"A": 0.5, "B": 0.3, "C": 0.5}
    assert metadata["segment_fraction_cap_source_n_rows"] == 2
    assert metadata["segment_fraction_cap_scope"] == "global-normalized-row-maximum"
    assert metadata["segment_fraction_cap_source"] == str(path.resolve())


def test_segment_fraction_caps_log_format_uses_three_significant_digits():
    caps = {"Plasma B cells": 0.23923923923923923, "Mast Cells": 0.05}

    assert format_segment_fraction_caps(caps) == (
        "{'Plasma B cells': 0.239, 'Mast Cells': 0.05}"
    )
    assert caps["Plasma B cells"] == 0.23923923923923923


@pytest.mark.parametrize(
    "rows, message",
    [
        ({"A": [0.5], "B": [0.5]}, "missing required"),
        ({"A": [-0.1], "B": [1.1]}, "negative"),
        ({"A": ["bad"], "B": [1]}, "nonnumeric or non-finite"),
        ({"A": [np.inf], "B": [1]}, "nonnumeric or non-finite"),
        ({"A": [0], "B": [0]}, "non-positive totals"),
        ({"A": [], "B": []}, "no sample rows"),
    ],
)
def test_invalid_prediction_file_fails_fast(tmp_path, rows, message):
    path = _write_predictions(tmp_path / "invalid.csv", rows)
    cell_types = ["A", "B", "C"] if message == "missing required" else ["A", "B"]
    with pytest.raises(ValueError, match=message):
        estimate_cell_prop_maxima_from_pred_cell_props(path, cell_types)


def test_empirical_caps_require_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="pred_cell_prop_file_path"):
        estimate_cell_prop_maxima_from_pred_cell_props(tmp_path / "missing.csv", ["A"])


def test_empirical_caps_require_nonempty_feasible_cell_types(tmp_path):
    path = _write_predictions(tmp_path / "pred.csv", {"A": [1.0], "B": [0.0]})
    with pytest.raises(ValueError, match="unique output"):
        estimate_cell_prop_maxima_from_pred_cell_props(path, ["A", "A"])
    with pytest.raises(ValueError, match="sum to less than one"):
        merge_cell_prop_prior_with_maxima(["A", "B"], None, {"A": 0.2, "B": 0.3})


def test_manual_segment_ranges_intersect_empirical_caps():
    caps = {"A": 0.7, "B": 0.6, "C": 0.5}
    combined = merge_cell_prop_prior_with_maxima(
        ["A", "B", "C"], {"A": (0.1, 0.9), "C": (0.0, 0.4)}, caps,
    )
    assert combined == {"A": (0.1, 0.7), "B": (0.0, 0.6), "C": (0.0, 0.4)}
    with pytest.raises(ValueError, match="exceeds its empirical maximum"):
        merge_cell_prop_prior_with_maxima(["A", "B"], {"A": (0.8, 0.9)}, {"A": 0.7, "B": 0.6})
    with pytest.raises(ValueError, match="sum to more than one"):
        merge_cell_prop_prior_with_maxima(
            ["A", "B"], {"A": (0.7, 1.0), "B": (0.6, 1.0)}, {"A": 0.8, "B": 0.8})
    with pytest.raises(ValueError, match="maxima sum to less than one"):
        merge_cell_prop_prior_with_maxima(
            ["A", "B"], {"A": (0.0, 0.4), "B": (0.0, 0.4)}, {"A": 0.8, "B": 0.8})
    with pytest.raises(ValueError, match="absent from segment outputs"):
        merge_cell_prop_prior_with_maxima(["A"], {"other": (0, 1)}, {"A": 1.0})


def test_deside_preset_expands_and_clips_without_changing_constant():
    original = DESIDE_TCGA_CELL_PROPORTION_RANGES.copy()
    resolved = resolve_segment_cell_prop_prior(
        list(original), "DeSide_predicted_TCGA_range", 0.15,
    )

    assert resolved["Cancer Cells"] == pytest.approx((0.0, 1.0))
    assert resolved["CD4 T"] == pytest.approx((0.0, 0.256))
    assert resolved["CAFs"] == pytest.approx((0.0, 0.866))
    assert DESIDE_TCGA_CELL_PROPORTION_RANGES == original


def test_manual_segment_prior_expands_partial_mapping():
    source = {"A": [0.2, 0.6]}
    resolved = resolve_segment_cell_prop_prior(["A", "B"], source, 0.15)

    assert resolved["A"] == pytest.approx((0.05, 0.75))
    assert "B" not in resolved
    assert source == {"A": [0.2, 0.6]}


@pytest.mark.parametrize(
    "cell_types, prior, enlargement, message",
    [
        (["A"], "unknown", 0, "Unknown cell_prop_prior preset"),
        (["unknown"], "DeSide_predicted_TCGA_range", 0, "missing output cell types"),
        (["A"], {"B": (0, 1)}, 0, "absent from segment outputs"),
        (["A"], {"A": (0.8, 0.2)}, 0, "0 <= minimum <= maximum"),
        (["A"], {"A": (0, np.nan)}, 0, "finite"),
        (["A"], {"A": (0, 1)}, -0.1, "finite nonnegative"),
        (["A"], {"A": (0, 1)}, np.inf, "finite nonnegative"),
        (["A"], {"A": (0, 1)}, True, "finite nonnegative"),
        (["A", "B"], {"A": (0.8, 1), "B": (0.4, 1)}, 0, "minima sum"),
        (["A", "B"], {"A": (0, 0.4), "B": (0, 0.4)}, 0, "maxima sum"),
    ],
)
def test_segment_prior_rejects_invalid_inputs(cell_types, prior, enlargement, message):
    with pytest.raises(ValueError, match=message):
        resolve_segment_cell_prop_prior(cell_types, prior, enlargement)


def test_expanded_segment_prior_constrains_samples():
    prior = resolve_segment_cell_prop_prior(
        ["A", "B", "C"], {"A": (0.1, 0.6), "B": (0.1, 0.5)}, 0.05,
    )
    np.random.seed(42)
    values = segment_sampling(100, ["A", "B", "C"], cell_prop_prior=prior, max_attempts=50000)
    for cell_type, (lower, upper) in prior.items():
        assert values[cell_type].between(lower, upper).all()


def test_segment_sampling_respects_caps_and_reports_rejections():
    caps = {"A": 0.7, "B": 0.6, "C": 0.5}
    prior = merge_cell_prop_prior_with_maxima(["A", "B", "C"], None, caps)
    np.random.seed(17)
    result = segment_sampling(500, ["A", "B", "C"], cell_prop_prior=prior, max_attempts=50000)
    assert (result <= pd.Series(caps)).all().all()
    stats = result.attrs["segment_sampling_stats"]
    assert stats["requested"] == stats["accepted"] == len(result)
    assert stats["attempted"] == stats["accepted"] + stats["rejected"]
    assert stats["rejected"] > 0


def test_segment_sampling_attempt_limit_is_bounded():
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        segment_sampling(
            1, ["A", "B"], cell_prop_prior={"A": (0, 0), "B": (0, 0)},
            max_attempts=3,
        )


def test_generate_cell_fraction_preserves_segment_attempt_stats():
    np.random.seed(21)
    result = generate_cell_fraction(
        sampling_method="segment",
        n_cell_frac=10,
        cell_type2subtype={"A": ["A"], "B": ["B"]},
        cell_prop_prior={"A": (0, 1), "B": (0, 1)},
        max_segment_sampling_attempts=100,
    )
    assert result.attrs["segment_sampling_stats"]["requested"] == 10


def test_generator_config_allows_only_filtered_segment_caps(tmp_path):
    path = tmp_path / "pred.csv"
    path.write_text("A,B\n0.5,0.5\n", encoding="utf-8")
    caps = {"A": 0.5, "B": 0.5}
    metadata = {"segment_fraction_cap_source": str(path)}
    config = GEPGenerationConfig(
        n_samples=1, sampling_method_prop="segment", filtering=True,
        pred_cell_prop_file_path=path, segment_fraction_max_caps=caps,
        segment_fraction_cap_metadata=metadata,
    )
    assert config.segment_fraction_max_caps == caps
    with pytest.raises(ValueError, match="require GEP filtering"):
        GEPGenerationConfig(
            n_samples=1, sampling_method_prop="segment", filtering=False,
            pred_cell_prop_file_path=path, segment_fraction_max_caps=caps,
            segment_fraction_cap_metadata=metadata,
        )
    with pytest.raises(ValueError, match="not resolved"):
        GEPGenerationConfig(n_samples=1, sampling_method_prop="segment", filtering=True,
                            pred_cell_prop_file_path=path)
    with pytest.raises(ValueError, match="segment maximum caps only"):
        GEPGenerationConfig(
            n_samples=1, sampling_method_prop="random", filtering=True,
            pred_cell_prop_file_path=path,
        )


def test_mixed_workflow_resolves_caps_only_for_filtered_segment(
    workflow_case, tmp_path, monkeypatch,
):
    import simutme.workflow as workflow

    config, params, _, _ = workflow_case
    predictions = _write_predictions(tmp_path / "pred.csv", {
        "sample_id": ["p1", "p2"],
        "cancer_type": ["A", "B"],
        "CT1": [0.8, 0.2],
        "CT2": [0.2, 0.8],
    })
    config.pred_cell_prop_file_path = _write_predictions(
        tmp_path / "shared_pred.csv",
        {
            "CT1": [0.4, 0.4],
            "CT2": [0.6, 0.6],
        },
    )
    params.update(
        gep_filtering=True,
        pred_cell_prop_file_path=str(predictions),
        cell_prop_prior=None,
    )
    captured = {}

    class DummySpace:
        additional_artifacts = []

    class FakeBulkGEPGenerator:
        def __init__(self, **kwargs):
            self.generated_bulk_gep_fp = tmp_path / "generated.h5ad"
            self.generated_cell_fraction_fp = tmp_path / "fractions.csv"
            self.gep_filtering_input_n_samples = None
            self.gep_filtering_kept_n_samples = None
            self.gep_filtering_removed_n_samples = None
            self.gep_filtering_pca_target = None
            self.gep_filtering_n_pcs = None

        def generate_gep(self, config):
            captured["generation_config"] = config

    monkeypatch.setattr(workflow, "BulkGEPGenerator", FakeBulkGEPGenerator)
    monkeypatch.setattr(workflow, "_prepare_fixed_gene_filter",
                        lambda *args, **kwargs: DummySpace())
    monkeypatch.setattr(workflow, "_save_gep_distance_distribution", lambda *args: {})
    monkeypatch.setattr(workflow, "_save_cancer_cell_fraction_distribution",
                        lambda **kwargs: None)
    monkeypatch.setattr(workflow, "_run_gene_filtering_postprocess",
                        lambda **kwargs: {})
    monkeypatch.setattr(workflow, "_build_mixed_dataset_summary",
                        lambda **kwargs: None)

    workflow.run_bulk_simulation("target", params, config)
    generation = captured["generation_config"]
    assert generation.sampling_method_prop == "segment"
    assert generation.filtering is True
    assert generation.segment_fraction_max_caps == {"CT1": 0.8, "CT2": 0.8}
    assert generation.cell_prop_prior == {"CT1": (0.0, 0.8), "CT2": (0.0, 0.8)}
    assert generation.segment_fraction_cap_metadata["segment_fraction_cap_source_n_rows"] == 2


@pytest.mark.parametrize("source", ["global", "dataset"])
@pytest.mark.parametrize("prior", ["preset", "manual", "empty"])
def test_filtered_segment_prior_overrides_prediction_caps(
    workflow_case, tmp_path, monkeypatch, source, prior,
):
    import simutme.workflow as workflow

    config, params, _, _ = workflow_case
    config.cell_type2subtypes = {"Cancer Cells": ["Cancer Cells"], "CD4 T": ["CD4 T"]}
    config.output_types = ["Cancer Cells", "CD4 T"]
    unused_prediction = tmp_path / "missing_predictions.csv"
    if source == "global":
        config.pred_cell_prop_file_path = unused_prediction
    else:
        params["pred_cell_prop_file_path"] = str(unused_prediction)
    configured_prior = {
        "preset": "DeSide_predicted_TCGA_range",
        "manual": {"Cancer Cells": (0.2, 0.9)},
        "empty": {},
    }[prior]
    params.update(
        sampling_method_prop="segment",
        gep_filtering=True,
        gene_filtering=False,
        cell_prop_prior=configured_prior,
        enlarge_cell_prop_prior=0.15,
    )
    captured = {}

    class DummySpace:
        additional_artifacts = []

    class FakeBulkGEPGenerator:
        def __init__(self, **kwargs):
            self.generated_bulk_gep_fp = tmp_path / "generated.h5ad"
            self.generated_cell_fraction_fp = tmp_path / "fractions.csv"
            self.gep_filtering_input_n_samples = None
            self.gep_filtering_kept_n_samples = None
            self.gep_filtering_removed_n_samples = None
            self.gep_filtering_pca_target = None
            self.gep_filtering_n_pcs = None

        def generate_gep(self, config):
            captured["generation"] = config

    def unexpected_caps(*args, **kwargs):
        raise AssertionError("Prediction-file caps must not be computed when a prior is provided.")

    monkeypatch.setattr(workflow, "BulkGEPGenerator", FakeBulkGEPGenerator)
    monkeypatch.setattr(workflow, "_prepare_fixed_gene_filter",
                        lambda *args, **kwargs: DummySpace())
    monkeypatch.setattr(workflow, "_save_gep_distance_distribution", lambda *args: {})
    monkeypatch.setattr(workflow, "estimate_cell_prop_maxima_from_pred_cell_props", unexpected_caps)
    monkeypatch.setattr(workflow, "_save_cancer_cell_fraction_distribution", lambda **kwargs: None)
    monkeypatch.setattr(workflow, "_run_gene_filtering_postprocess", lambda **kwargs: {})
    monkeypatch.setattr(workflow, "_build_mixed_dataset_summary", lambda **kwargs: None)

    workflow.run_bulk_simulation("target", params, config)

    generation = captured["generation"]
    assert generation.pred_cell_prop_file_path is None
    assert generation.segment_fraction_max_caps is None
    assert generation.segment_fraction_cap_metadata is None
    if prior == "preset":
        assert generation.cell_prop_prior["Cancer Cells"] == pytest.approx((0, 1))
        assert generation.cell_prop_prior["CD4 T"] == pytest.approx((0, 0.256))
    elif prior == "manual":
        assert generation.cell_prop_prior["Cancer Cells"] == pytest.approx((0.05, 1))
        assert "CD4 T" not in generation.cell_prop_prior
    else:
        assert generation.cell_prop_prior == {}


@pytest.mark.parametrize(
    "params, message",
    [
        ({"cell_prop_prior": "DeSide_predicted_TCGA_range", "gep_filtering": False}, "require segment"),
        ({"cell_prop_prior": "DeSide_predicted_TCGA_range", "sampling_method_prop": "dirichlet"}, "require segment"),
        ({"enlarge_cell_prop_prior": 0.15}, "requires an explicit cell_prop_prior"),
        ({"cell_prop_prior": {"CT1": (0, 1)}, "enlarge_cell_prop_prior": -0.1}, "finite nonnegative"),
    ],
)
def test_segment_prior_configuration_fails_early(workflow_case, params, message):
    import simutme.workflow as workflow

    config, default_params, _, _ = workflow_case
    default_params.update({"gep_filtering": True, **params})
    with pytest.raises(ValueError, match=message):
        workflow.preflight_segment_cell_prop_priors(config, {"target": default_params})


def test_cli_preflights_segment_prior_before_running_any_dataset(workflow_case, tmp_path, monkeypatch):
    import simutme

    config, _, _, _ = workflow_case
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "datasets": {
            "first": {"dataset_type": "SCT"},
            "target": {
                "dataset_type": "Mixed", "sampling_method_prop": "segment",
                "gep_filtering": True, "cell_prop_prior": "DeSide_predicted_TCGA_range",
            },
        },
    }, sort_keys=False), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "simutme_cli", Path(__file__).resolve().parents[1] / "src/main_simu_bulk.py",
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    called = []
    monkeypatch.setattr(simutme.SimuConfig, "from_yaml", lambda path: config)
    monkeypatch.setattr(simutme, "run_bulk_simulation", lambda *args: called.append(args))

    with pytest.raises(ValueError, match="missing output cell types"):
        cli.main(["-c", str(config_path)])
    assert called == []


@pytest.mark.parametrize(
    "sampling_method_prop, gep_filtering, path_is_used",
    [
        ("dirichlet", True, True),
        ("segment", True, True),
        ("segment", False, False),
        ("random", True, False),
    ],
)
def test_global_prediction_path_is_only_used_by_supported_modes(
    workflow_case,
    tmp_path,
    monkeypatch,
    sampling_method_prop,
    gep_filtering,
    path_is_used,
):
    import simutme.workflow as workflow

    config, params, _, _ = workflow_case
    predictions = _write_predictions(tmp_path / "shared_pred.csv", {
        "sample_id": ["p1", "p2"],
        "cancer_type": ["A", "B"],
        "CT1": [0.8, 0.2],
        "CT2": [0.2, 0.8],
    })
    config.pred_cell_prop_file_path = predictions
    params.pop("pred_cell_prop_file_path", None)
    params.update(
        sampling_method_prop=sampling_method_prop,
        gep_filtering=gep_filtering,
        gene_filtering=False,
        cell_prop_prior=None,
    )
    captured = {}

    class DummySpace:
        additional_artifacts = []

    class FakeBulkGEPGenerator:
        def __init__(self, **kwargs):
            self.generated_bulk_gep_fp = tmp_path / "generated.h5ad"
            self.generated_cell_fraction_fp = tmp_path / "fractions.csv"
            self.gep_filtering_input_n_samples = None
            self.gep_filtering_kept_n_samples = None
            self.gep_filtering_removed_n_samples = None
            self.gep_filtering_pca_target = None
            self.gep_filtering_n_pcs = None

        def generate_gep(self, config):
            captured["generation_config"] = config

    monkeypatch.setattr(workflow, "BulkGEPGenerator", FakeBulkGEPGenerator)
    monkeypatch.setattr(workflow, "_prepare_fixed_gene_filter",
                        lambda *args, **kwargs: DummySpace())
    monkeypatch.setattr(workflow, "_save_gep_distance_distribution", lambda *args: {})
    monkeypatch.setattr(workflow, "_save_cancer_cell_fraction_distribution",
                        lambda **kwargs: None)
    monkeypatch.setattr(workflow, "_run_gene_filtering_postprocess",
                        lambda **kwargs: {})
    monkeypatch.setattr(workflow, "_build_mixed_dataset_summary",
                        lambda **kwargs: None)
    monkeypatch.setattr(
        workflow,
        "_save_estimated_dirichlet_alpha_from_pred_cell_props",
        lambda **kwargs: captured.update(
            alpha_estimation_path=kwargs["pred_cell_prop_file_path"]
        ),
    )

    workflow.run_bulk_simulation("target", params, config)

    generation = captured["generation_config"]
    expected_path = predictions if path_is_used else None
    assert generation.pred_cell_prop_file_path == expected_path
    if sampling_method_prop == "segment" and gep_filtering:
        assert generation.segment_fraction_max_caps == {"CT1": 0.8, "CT2": 0.8}
    else:
        assert generation.segment_fraction_max_caps is None
    if sampling_method_prop == "dirichlet":
        assert captured["alpha_estimation_path"] == predictions


@pytest.mark.parametrize(
    "sampling_method, global_filter, dataset_filter, gene_filter, expected",
    [
        ("dirichlet", False, False, False, True),
        ("dirichlet", False, True, False, True),
        ("segment", False, True, False, True),
        ("segment", True, False, False, False),
        ("sparse", False, True, False, True),
        ("sparse", False, False, False, False),
        ("random", True, None, False, True),
        ("segment", False, False, True, False),
    ],
)
def test_mixed_workflow_enables_all_cell_plots_for_dirichlet_or_gep_filtering(
    workflow_case, tmp_path, monkeypatch, sampling_method, global_filter,
    dataset_filter, gene_filter, expected,
):
    import simutme.workflow as workflow

    config, params, _, _ = workflow_case
    config.gep_filtering_enabled_default = global_filter
    params.update(sampling_method_prop=sampling_method, gene_filtering=gene_filter)
    if dataset_filter is None:
        params.pop("gep_filtering", None)
    else:
        params["gep_filtering"] = dataset_filter
    captured = {}

    class DummySpace:
        additional_artifacts = []

    class FakeBulkGEPGenerator:
        def __init__(self, **kwargs):
            self.generated_bulk_gep_fp = tmp_path / "generated.h5ad"
            self.generated_cell_fraction_fp = tmp_path / "fractions.csv"
            self.gep_filtering_input_n_samples = None
            self.gep_filtering_kept_n_samples = None
            self.gep_filtering_removed_n_samples = None
            self.gep_filtering_pca_target = None
            self.gep_filtering_n_pcs = None

        def generate_gep(self, config):
            captured["generation_filtering"] = config.filtering

    monkeypatch.setattr(workflow, "BulkGEPGenerator", FakeBulkGEPGenerator)
    monkeypatch.setattr(workflow, "_prepare_fixed_gene_filter",
                        lambda *args, **kwargs: DummySpace())
    monkeypatch.setattr(
        workflow, "_save_gep_distance_distribution",
        lambda *args: captured.update(distance_plot=True) or {},
    )
    monkeypatch.setattr(
        workflow, "_save_cancer_cell_fraction_distribution",
        lambda **kwargs: captured.update(plot_all_cell_types=kwargs["plot_all_cell_types"]),
    )
    monkeypatch.setattr(workflow, "_run_gene_filtering_postprocess", lambda **kwargs: {})
    monkeypatch.setattr(workflow, "_build_mixed_dataset_summary", lambda **kwargs: None)

    workflow.run_bulk_simulation("target", params, config)

    assert captured["plot_all_cell_types"] is expected
    assert captured["generation_filtering"] is (
        global_filter if dataset_filter is None else dataset_filter
    )
    assert captured.get("distance_plot", False) is captured["generation_filtering"]


def test_gep_filtered_mixed_workflow_writes_non_cancer_plots(
    workflow_case, tmp_path, monkeypatch,
):
    import simutme.workflow as workflow

    config, params, _, _ = workflow_case
    params.update(sampling_method_prop="segment", gep_filtering=True, gene_filtering=False)
    captured = {}

    class DummySpace:
        additional_artifacts = []

    class FakeBulkGEPGenerator:
        def __init__(self, **kwargs):
            self.generated_bulk_gep_fp = tmp_path / "generated.h5ad"
            self.generated_cell_fraction_fp = tmp_path / "fractions.csv"
            self.gep_filtering_input_n_samples = None
            self.gep_filtering_kept_n_samples = None
            self.gep_filtering_removed_n_samples = None
            self.gep_filtering_pca_target = None
            self.gep_filtering_n_pcs = None

        def generate_gep(self, config):
            assert config.filtering
            pd.DataFrame(
                {"Cancer Cells": [0.2, 0.6, 0.8], "CT1": [0.8, 0.4, 0.2]},
                index=["accepted1", "accepted2", "accepted3"],
            ).to_csv(self.generated_cell_fraction_fp)

    def save_plots(**kwargs):
        captured["output_dir"] = kwargs["output_dir"]
        _save_cancer_cell_fraction_distribution(**kwargs)

    monkeypatch.setattr(workflow, "BulkGEPGenerator", FakeBulkGEPGenerator)
    monkeypatch.setattr(workflow, "_prepare_fixed_gene_filter",
                        lambda *args, **kwargs: DummySpace())
    monkeypatch.setattr(workflow, "_save_gep_distance_distribution", lambda *args: {})
    monkeypatch.setattr(workflow, "_save_cancer_cell_fraction_distribution", save_plots)
    monkeypatch.setattr(workflow, "_run_gene_filtering_postprocess", lambda **kwargs: {})
    monkeypatch.setattr(workflow, "_build_mixed_dataset_summary", lambda **kwargs: None)

    workflow.run_bulk_simulation("target", params, config)

    output_dir = captured["output_dir"]
    summary = pd.read_csv(output_dir / "cell_fraction_summary_all_cell_types.csv")
    assert summary.loc[0, "cell_type"] == "CT1"
    assert summary.loc[0, "n_samples"] == 3
    assert summary.loc[0, "mean"] == pytest.approx((0.8 + 0.4 + 0.2) / 3, abs=1e-6)
    assert (output_dir / "CT1_fraction_distribution.png").exists()
    assert (output_dir / "CT1_fraction_distribution.pdf").exists()


def test_example_config_loads_shared_file_paths_from_paths():
    from simutme.workflow import SimuConfig

    config_path = Path(__file__).resolve().parents[1] / "src" / "config_example.yaml"
    with config_path.open(encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    shared_path_names = {
        "gep_filtering_reference_file",
        "gene_filtering_reference_file",
        "pred_cell_prop_file_path",
    }
    assert shared_path_names.issubset(yaml_data["paths"])
    assert shared_path_names.isdisjoint(yaml_data["parameters"])
    assert "pred_cell_prop_file_path" not in yaml_data["datasets"]["Mixed_N100_dirichlet"]

    config = SimuConfig.from_yaml(str(config_path))
    assert config.gep_filtering_reference_file == Path(
        yaml_data["paths"]["gep_filtering_reference_file"]
    )
    assert config.gene_filtering_reference_file == Path(
        yaml_data["paths"]["gene_filtering_reference_file"]
    )
    assert config.pred_cell_prop_file_path == Path(
        yaml_data["paths"]["pred_cell_prop_file_path"]
    )


def test_segment_prediction_caps_require_gep_filtering(workflow_case, tmp_path, monkeypatch):
    import simutme.workflow as workflow

    config, params, _, _ = workflow_case
    predictions = _write_predictions(tmp_path / "pred.csv", {"CT1": [0.5], "CT2": [0.5]})
    params.update(gep_filtering=False, pred_cell_prop_file_path=str(predictions))
    monkeypatch.setattr(workflow, "BulkGEPGenerator",
                        lambda **kwargs: pytest.fail("Generator initialized before validation"))
    with pytest.raises(ValueError, match="only when.*gep_filtering"):
        workflow.run_bulk_simulation("target", params, config)
