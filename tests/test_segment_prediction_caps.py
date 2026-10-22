from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

pytest_plugins = ("tests.test_correlation_first_filtering",)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simutme.simulation.sampling import (
    estimate_cell_prop_maxima_from_pred_cell_props,
    generate_cell_fraction,
    merge_cell_prop_prior_with_maxima,
    segment_sampling,
)
from simutme.simulation.simulator import GEPGenerationConfig


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
    params.update(
        gep_filtering=True,
        pred_cell_prop_file_path=str(predictions),
        cell_prop_prior={"CT1": (0.1, 0.7)},
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
    assert generation.cell_prop_prior == {"CT1": (0.1, 0.7), "CT2": (0.0, 0.8)}
    assert generation.segment_fraction_cap_metadata["segment_fraction_cap_source_n_rows"] == 2


def test_segment_prediction_caps_require_gep_filtering(workflow_case, tmp_path, monkeypatch):
    import simutme.workflow as workflow

    config, params, _, _ = workflow_case
    predictions = _write_predictions(tmp_path / "pred.csv", {"CT1": [0.5], "CT2": [0.5]})
    params.update(gep_filtering=False, pred_cell_prop_file_path=str(predictions))
    monkeypatch.setattr(workflow, "BulkGEPGenerator",
                        lambda **kwargs: pytest.fail("Generator initialized before validation"))
    with pytest.raises(ValueError, match="only when.*gep_filtering"):
        workflow.run_bulk_simulation("target", params, config)
