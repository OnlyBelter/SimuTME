import json
from dataclasses import replace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import simutme.workflow as workflow
import simutme.simulation.filtering as filtering
from simutme.simulation.filtering import FixedGeneFilter, normalize_selected, select_fixed_genes
from simutme.simulation.simulator import BulkGEPGenerator
from simutme.workflow import SimuConfig, ResolvedGEPFilteringParams


def gep_params(**overrides):
    values = dict(
        enabled=True, reference_file="reference.csv", ref_exp_type="TPM",
        filtering_method="mean_gep", filtering_ref_types=None,
        gep_filtering_quantile=(None, 1.0), n_top=20, show_filtering_info=False,
        filtering_by_gene_range=False, min_percentage_within_gene_range=1.0,
        gene_quantile_range=[0.0, 0.5, 1.0], filtering_in_pca_space=True,
        pca_n_components=1, norm_ord=1, high_corr_gene_list_file=None,
        high_corr_gene_list=None,
    )
    values.update(overrides)
    return ResolvedGEPFilteringParams(**values)


def make_space(tmp_path, **overrides):
    reference = pd.DataFrame(
        {"b": [8.0, 6.0, 4.0], "a": [2.0, 4.0, 6.0], "unused": [100.0] * 3},
        index=["r1", "r2", "r3"],
    )
    return FixedGeneFilter(
        ["a", "b"], reference, tmp_path / "reference.csv", None,
        gep_params(**overrides), tmp_path / "target", 3, 3,
    )


def test_selection_uses_only_target_types_and_deterministic_ties():
    correlations = pd.DataFrame(
        {"CT1": [0.8, 0.8, np.nan, 0.1], "CT2": [0.0, 0.0, np.inf, 0.7],
         "not_output": [0.1, 0.1, 0.99, 0.1]}, index=["b", "a", "c", "d"],
    )
    original = correlations.copy(deep=True)
    genes, shared = select_fixed_genes(
        ["d", "c", "b", "a"], ["a", "b", "c", "d"], ["CT1", "CT2"],
        correlations=correlations, n_gene_max=1,
    )
    assert genes == ["a", "d"] and shared == 4
    other, _ = select_fixed_genes(
        ["a", "b", "c", "d"], ["d", "c", "b", "a"], ["CT1", "CT2"],
        correlations=correlations.iloc[::-1], n_gene_max=1, corr_threshold=0.75,
    )
    assert other == ["a"]
    pd.testing.assert_frame_equal(correlations, original)
    with pytest.raises(ValueError, match="missing target cell types"):
        select_fixed_genes(["a"], ["a"], ["missing"], correlations=correlations)


def test_supplied_list_precedence_and_shared_universe():
    genes, shared = select_fixed_genes(
        ["a", "b", "c"], ["b", "a", "d"], ["missing"],
        supplied_genes=["c", "b", "a"], correlations=pd.DataFrame(),
        corr_threshold=100, n_gene_max=0,
    )
    assert genes == ["a", "b"] and shared == 2
    assert select_fixed_genes(["b", "a", "c"], ["a", "b"], ["CT1"])[0] == genes


def test_normalization_aligns_orders_and_sums_to_one_million():
    expression = pd.DataFrame({"b": [30.0, 20.0], "a": [10.0, 20.0], "c": [999.0, 999.0]})
    result = normalize_selected(expression, ["a", "b"])
    np.testing.assert_allclose(result.sum(axis=1), 1e6)
    pd.testing.assert_frame_equal(
        result, normalize_selected(expression.loc[:, ["c", "a", "b"]], ["a", "b"])
    )
    np.testing.assert_allclose(result.iloc[0], [250000.0, 750000.0])


@pytest.mark.parametrize("expression,genes,message", [
    (pd.DataFrame([[1, 2]], columns=["a", "a"]), ["a"], "duplicate"),
    (pd.DataFrame({"a": [1]}), [], "empty"),
    (pd.DataFrame({"a": [1]}), ["a", "a"], "duplicate"),
    (pd.DataFrame({"a": [1]}), ["b"], "missing selected"),
    (pd.DataFrame({"a": [-1]}), ["a"], "negative"),
    (pd.DataFrame({"a": [np.inf]}), ["a"], "non-finite"),
    (pd.DataFrame({"a": [0], "b": [1]}), ["a"], "row sums"),
])
def test_invalid_normalization_fails_fast(expression, genes, message):
    with pytest.raises(ValueError, match=message):
        normalize_selected(expression, genes)


def test_fresh_and_cached_pca_use_same_reference_log_inputs(tmp_path, monkeypatch):
    fit_inputs, transforms = [], []
    original_fit = filtering.PCA.fit
    original_transform = filtering.safe_pca_transform

    def capture_fit(model, data, *args, **kwargs):
        fit_inputs.append(np.asarray(data).copy())
        return original_fit(model, data, *args, **kwargs)

    def capture_transform(model, data, label):
        transforms.append((label, np.asarray(data).copy()))
        return original_transform(model, data, label=label)

    monkeypatch.setattr(filtering.PCA, "fit", capture_fit)
    monkeypatch.setattr(filtering, "safe_pca_transform", capture_transform)
    space = make_space(tmp_path)
    simulated = pd.DataFrame({"unused": [999], "b": [6], "a": [4]}, index=["s1"])
    first = space.filter(simulated)
    cached = make_space(tmp_path)
    second = cached.filter(simulated.loc[:, ["a", "unused", "b"]])
    assert len(fit_inputs) == 1
    np.testing.assert_allclose(fit_inputs[0], space.reference_log)
    assert [label for label, _ in transforms] == [
        "Reference log2(TPM + 1)", "Simulated log2(TPM + 1)",
        "Reference log2(TPM + 1)", "Simulated log2(TPM + 1)",
    ]
    np.testing.assert_allclose(transforms[1][1], np.log2(normalize_selected(simulated, space.genes) + 1))
    np.testing.assert_allclose(transforms[0][1], transforms[2][1])
    pd.testing.assert_frame_equal(first, second)
    assert cached.pca.n_components_ == 1
    assert cached.pca.filtering_genes_ == space.genes


def test_range_denominator_and_survivor_order(tmp_path):
    space = make_space(tmp_path, filtering_by_gene_range=True)
    simulated = pd.DataFrame(
        {"a": [4, 10, 5], "b": [6, 0, 5], "unused": [1000] * 3},
        index=["s3", "s1", "s2"],
    )
    result = space.filter(simulated)
    assert result.index.tolist() == ["s3", "s2"]
    pd.testing.assert_frame_equal(result, normalize_selected(simulated, space.genes).loc[["s3", "s2"]])
    assert space.range_input == 3 and space.range_kept == 2


def test_empty_and_range_rejected_batches_skip_pca(tmp_path):
    space = make_space(tmp_path, filtering_by_gene_range=True)
    assert space.filter(pd.DataFrame(columns=["a", "b", "unused"], dtype=float)).empty
    assert space.filter(pd.DataFrame({"a": [10], "b": [0], "unused": [1000]})).empty
    assert not space.model_path.exists()


def test_disabled_gep_normalizes_without_rejection(tmp_path):
    space = make_space(tmp_path, enabled=False, filtering_by_gene_range=True)
    simulated = pd.DataFrame({"a": [10, 0], "b": [0, 10]}, index=["s1", "s2"])
    result = space.filter(simulated)
    assert result.index.tolist() == simulated.index.tolist()
    np.testing.assert_allclose(result.sum(axis=1), 1e6)
    assert not space.model_path.exists()


def test_model_cache_rejects_changed_genes_reference_and_target(tmp_path):
    space = make_space(tmp_path)
    space.filter(pd.DataFrame({"a": [4], "b": [6]}))
    for changed in [
        make_space(tmp_path, pca_n_components=0.9),
        FixedGeneFilter(["b", "a"], space.reference_linear, tmp_path / "reference.csv",
                        None, gep_params(), space.result_dir, 3, 3),
        FixedGeneFilter(["a", "b"], space.reference_linear * pd.Series([2, 1], index=["a", "b"]),
                        tmp_path / "reference.csv", None, gep_params(), space.result_dir, 3, 3),
    ]:
        with pytest.raises(ValueError, match="new dataset name/output location"):
            changed.validate_model()
    space.model_metadata_path.unlink()
    with pytest.raises(ValueError, match="Missing"):
        make_space(tmp_path).validate_model()


@pytest.fixture
def workflow_case(tmp_path, monkeypatch):
    sct = tmp_path / "sct.h5ad"
    ad.AnnData(
        X=np.log2(np.array([[2.0, 8.0, 100], [6.0, 4.0, 100]]) + 1),
        obs=pd.DataFrame({"cell_type": ["CT1", "CT2"]}, index=["c1", "c2"]),
        var=pd.DataFrame(index=["a", "b", "unused"]),
    ).write_h5ad(sct)
    reference = tmp_path / "reference.csv"
    pd.DataFrame({"a": [2, 4, 6], "b": [8, 6, 4], "unused": [100] * 3},
                 index=["r1", "r2", "r3"]).to_csv(reference)
    correlations = tmp_path / "source_correlations.csv"
    pd.DataFrame({"CT1": [0.8, 0.1, 0.1], "CT2": [0.1, 0.8, 0.1],
                  "not_output": [0.1, 0.1, 0.99]}, index=["a", "b", "unused"]).to_csv(correlations)
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame({"cancer_type": ["A", "A", "B"]}, index=["r1", "r2", "r3"]).to_csv(mapping)
    config = SimuConfig(
        simu_bulk_dir=tmp_path / "outputs", tcga2cancer_type_file_path=mapping,
        tcga_merged_tpm_file_path=reference, high_corr_gene_file_path=None,
        merged_sc_dataset_file_path=tmp_path / "unused.h5ad",
        sc_dataset_ids=[], cell_type2subtypes={"CT1": ["CT1"], "CT2": ["CT2"]},
        total_rna_coefficient={"CT1": 1, "CT2": 1}, n_base=1,
        output_types=["CT1", "CT2"], gep_filtering_quantile=(None, 1.0),
        plot_pca_for_comparison=False, gep_pca_n_components=1,
    )
    config._dataset_order = ["source", "target"]
    config._dataset_definitions = {
        "source": {"dataset_type": "Mixed", "sampling_method_prop": "segment",
                   "gep_filtering": False, "gene_filtering": False},
        "target": {"dataset_type": "Mixed"},
    }
    config._generated_mixed_outputs["source"] = {
        "full_gene_corr_with_cell_fraction_file": correlations,
    }
    params = dict(
        dataset_type="Mixed", sampling_method_prop="segment", n_samples=3,
        sct_reference={"source_type": "path", "file_path": str(sct)},
        gene_filtering=True, gep_filtering=False,
        gene_filtering_correlation_reference_dataset="source",
    )
    monkeypatch.setattr(workflow, "_save_cancer_cell_fraction_distribution", lambda **kwargs: None)
    monkeypatch.setattr(BulkGEPGenerator, "_load_sc_data_for_generation",
                        lambda self, method: (pd.DataFrame(), ["a", "b", "unused"], "SCT"))
    monkeypatch.setattr(BulkGEPGenerator, "_sc_sampling",
                        lambda self, cell_frac, **kwargs: pd.DataFrame(
                            {"selected_cell_id": ["c1"] * len(cell_frac)}, index=cell_frac.index))
    generated_batches = []

    def generate_batch(self, cell_frac, **kwargs):
        assert self.filtering_space.gene_list_file.exists()
        generated_batches.append(True)
        return pd.DataFrame({"unused": [100] * len(cell_frac), "b": [6] * len(cell_frac),
                             "a": [4] * len(cell_frac)}, index=cell_frac.index)

    monkeypatch.setattr(BulkGEPGenerator, "_map_cell_id2exp", generate_batch)
    monkeypatch.setattr(workflow, "get_gene_list_for_filtering",
                        lambda *args, **kwargs: pytest.fail("Post-generation gene selection ran"))
    return config, params, correlations, generated_batches


def read_summary(output):
    return json.loads((workflow._resolve_mixed_dataset_result_dir(output, "target")
                       / "mixed_dataset_summary.json").read_text())


def test_workflow_selection_export_and_cache_reuse(workflow_case):
    config, params, source_file, batches = workflow_case
    source_before = source_file.read_bytes()
    output = workflow.run_bulk_simulation("target", params, config)
    accepted = ad.read_h5ad(output)
    assert accepted.var_names.tolist() == ["a", "b"]
    np.testing.assert_allclose(accepted.X, np.tile(np.log2([400001, 600001]), (3, 1)), rtol=1e-7)
    assert len(batches) == 1 and source_file.read_bytes() == source_before
    summary = read_summary(output)
    assert summary["generated_n_genes"] == summary["exported_n_genes"] == 2
    assert summary["pre_selection_n_genes"] == 3 and summary["removed_n_genes"] == 1
    assert summary["full_gene_corr_with_cell_fraction_file"] == str(source_file)
    fractions = pd.read_csv(summary["generated_cell_fraction_file"], index_col=0)
    assert fractions.index.tolist() == accepted.obs_names.tolist()
    assert ad.read_h5ad(summary["filtered_bulk_gep_file"]).var_names.tolist() == ["a", "b"]
    assert workflow.run_bulk_simulation("target", params, config) == output
    assert len(batches) == 1


def test_filtered_segment_prior_is_saved_in_cache_identity(workflow_case):
    config, params, _, _ = workflow_case
    params.update(
        gep_filtering=True,
        gep_filtering_method="mgep",
        filtering_in_pca_space=False,
        cell_prop_prior={"CT1": [0.1, 0.9]},
        enlarge_cell_prop_prior=0.05,
    )

    output = workflow.run_bulk_simulation("target", params, config)
    result_dir = workflow._resolve_mixed_dataset_result_dir(output, "target")
    metadata = json.loads((result_dir / "filtering_metadata.json").read_text())
    generation = metadata["identity"]["generation"]
    assert generation["cell_prop_prior"]["CT1"] == pytest.approx([0.05, 0.95])
    assert generation["pred_cell_prop_file_path"] is None
    assert generation["segment_fraction_cap_metadata"] is None
    assert workflow.run_bulk_simulation("target", params, config) == output

    params["enlarge_cell_prop_prior"] = 0.1
    with pytest.raises(ValueError, match="new dataset name/output location"):
        workflow.run_bulk_simulation("target", params, config)


@pytest.mark.parametrize("change", ["gene_list", "range", "distance", "missing_metadata"])
@pytest.mark.parametrize("resume", [False, True])
def test_workflow_rejects_incompatible_outputs(workflow_case, change, resume):
    config, params, source_file, _ = workflow_case
    output = workflow.run_bulk_simulation("target", params, config)
    artifact = Path(output).with_suffix(".csv") if resume else Path(output)
    before = artifact.read_bytes()
    if resume:
        Path(output).unlink()
    if change == "gene_list":
        correlations = pd.read_csv(source_file, index_col=0)
        correlations.loc["b", "CT2"] = 0.1
        correlations.to_csv(source_file)
    elif change == "range":
        params["gep_min_percentage_within_gene_range"] = 0.5
    elif change == "distance":
        params["gep_norm_ord"] = 2
    else:
        (workflow._resolve_mixed_dataset_result_dir(output, "target") / "filtering_metadata.json").unlink()
    with pytest.raises(ValueError, match="new dataset name/output location"):
        workflow.run_bulk_simulation("target", params, config)
    assert artifact.read_bytes() == before


def test_legacy_modes_and_missing_sources_rejected(workflow_case):
    config, params, _, _ = workflow_case
    with pytest.raises(ValueError, match="Migrate"):
        workflow._resolve_gene_filtering_params(
            "target", {**params, "gene_filtering_type": "high_corr_gene_and_quantile_range"}, config
        )
    with pytest.raises(ValueError, match="requires"):
        workflow._resolve_gene_filtering_params("target", {"gene_filtering": True}, config)


def test_reference_subset_and_empty_subset(workflow_case):
    config, params, _, _ = workflow_case
    gep = workflow._resolve_gep_filtering_params(
        "target", {**params, "gep_filtering": True, "filtering_ref_types": ["A"]}, config
    )
    gene = workflow._resolve_gene_filtering_params("target", params, config)
    args = ("target", params["sct_reference"]["file_path"], config.output_types)
    space = workflow._prepare_fixed_gene_filter(*args, gep, gene, config, config.simu_bulk_dir / "target")
    assert space.reference_log.index.tolist() == ["r1", "r2"]
    assert len(space.prepare_distance_reference()) == 2
    with pytest.raises(ValueError, match="No reference samples"):
        workflow._prepare_fixed_gene_filter(
            *args, replace(gep, filtering_ref_types=["missing"]), gene, config,
            config.simu_bulk_dir / "target",
        )


def test_workflow_pca_summary_and_cached_totals(workflow_case):
    config, params, _, batches = workflow_case
    params["gep_filtering"] = True
    output = workflow.run_bulk_simulation("target", params, config)
    summary = read_summary(output)
    assert summary["gep_pca_n_components"] == summary["gep_filtering_n_pcs"] == 1
    assert summary["gep_filtering_input_n_samples"] == 90
    assert ad.read_h5ad(output).n_obs == 3
    png = Path(summary["gep_filtering_distance_plot_png"])
    svg = Path(summary["gep_filtering_distance_plot_svg"])
    assert png.exists() and svg.exists()
    png.unlink()
    workflow.run_bulk_simulation("target", params, config)
    assert png.exists()
    assert read_summary(output)["gep_filtering_input_n_samples"] == 90
    assert ad.read_h5ad(output).n_obs == 3
    assert len(batches) == 1


def test_gene_filtering_disabled_uses_common_universe(workflow_case):
    config, params, _, _ = workflow_case
    params.update(gene_filtering=False, gep_filtering=True)
    output = workflow.run_bulk_simulation("target", params, config)
    assert ad.read_h5ad(output).var_names.tolist() == ["a", "b", "unused"]
    assert read_summary(output)["removed_n_genes"] == 0


def test_intermediate_metadata_and_mismatched_rows_fail_without_deletion(workflow_case):
    config, params, _, _ = workflow_case
    output = workflow.run_bulk_simulation("target", params, config)
    Path(output).unlink()
    summary = read_summary(output)
    fractions_file = Path(summary["generated_cell_fraction_file"])
    fractions = pd.read_csv(fractions_file, index_col=0).iloc[:1]
    fractions.to_csv(fractions_file)
    before = fractions_file.read_bytes()
    with pytest.raises(ValueError, match="not deleted"):
        workflow.run_bulk_simulation("target", params, config)
    assert fractions_file.read_bytes() == before


def test_diagnostic_pca_is_separate_and_uses_fixed_log_space(workflow_case, monkeypatch):
    config, params, _, _ = workflow_case
    params.update(gep_filtering=True, plot_pca_for_comparison=True)
    captured = []
    original = workflow.do_pca_analysis

    def capture(exp_df, **kwargs):
        captured.append(exp_df.copy())
        return original(exp_df=exp_df, **kwargs)

    monkeypatch.setattr(workflow, "do_pca_analysis", capture)
    output = workflow.run_bulk_simulation("target", params, config)
    summary = read_summary(output)
    assert len(captured) == 1
    assert captured[0].columns.tolist() == ["a", "b"]
    np.testing.assert_allclose(captured[0].iloc[-3:], np.log2(
        np.array([[200000, 800000], [400000, 600000], [600000, 400000]]) + 1))
    assert summary["pca_model_file"] != summary["gep_filtering_pca_model_file"]
    assert Path(summary["pca_plot_file"]).exists()
    workflow.run_bulk_simulation("target", params, config)
    assert len(captured) == 1


def test_compatible_intermediate_resume_keeps_normalization_and_stats(workflow_case):
    config, params, _, batches = workflow_case
    params["gep_filtering"] = True
    output = workflow.run_bulk_simulation("target", params, config)
    original = ad.read_h5ad(output)
    Path(output).unlink()
    assert workflow.run_bulk_simulation("target", params, config) == output
    restored = ad.read_h5ad(output)
    np.testing.assert_allclose(restored.X, original.X)
    assert restored.obs_names.tolist() == original.obs_names.tolist()
    assert len(batches) == 1
    assert read_summary(output)["gep_filtering_input_n_samples"] == 90
    assert restored.n_obs == 3


def test_supplied_list_skips_source_and_legacy_range_selection(workflow_case):
    config, params, source_file, _ = workflow_case
    supplied = source_file.with_name("supplied_genes.csv")
    pd.DataFrame({"gene_name": ["b", "a", "not_shared"]}).to_csv(supplied, index=False)
    params.update(
        gene_filtering_high_corr_gene_file=str(supplied),
        gene_filtering_correlation_reference_dataset="does_not_exist",
        gene_filtering_type="high_corr_gene_and_quantile_range",
        gene_filtering_corr_threshold=100,
        gene_filtering_n_gene_max=0,
    )
    config._generated_mixed_outputs.clear()
    output = workflow.run_bulk_simulation("target", params, config)
    assert ad.read_h5ad(output).var_names.tolist() == ["a", "b"]
    assert read_summary(output)["full_gene_corr_with_cell_fraction_file"] is None


def test_missing_source_matrix_fails_before_first_batch(workflow_case):
    config, params, source_file, batches = workflow_case
    source_file.unlink()
    with pytest.raises(ValueError, match="calculate_gene_cell_prop_correlation"):
        workflow.run_bulk_simulation("target", params, config)
    assert batches == []


def test_mgep_filters_linear_tpm_by_reference_l1_quantile(tmp_path, monkeypatch):
    space = make_space(
        tmp_path,
        filtering_method="mgep",
        filtering_in_pca_space=False,
        gep_filtering_quantile=(0.0, 0.5),
    )
    reference = space.reference_linear.to_numpy()
    expected_center = np.median(reference, axis=0)
    expected_reference_distances = np.linalg.norm(
        reference - expected_center, ord=1, axis=1
    )
    expected_cutoff = np.quantile(expected_reference_distances, 0.5)
    captured_norm_inputs = []
    original_norm = np.linalg.norm

    def capture_norm(values, ord=None, axis=None):
        captured_norm_inputs.append((np.asarray(values).copy(), ord))
        return original_norm(values, ord=ord, axis=axis)

    monkeypatch.setattr(filtering.np.linalg, "norm", capture_norm)
    simulated = pd.DataFrame(
        {"a": [2.0, 6.0, 1.0, 9.0], "b": [8.0, 4.0, 9.0, 1.0]},
        index=["boundary_low", "boundary_high", "outside_low", "outside_high"],
    )

    result = space.filter(simulated)

    assert result.index.tolist() == ["boundary_low", "boundary_high"]
    np.testing.assert_allclose(space.mgep_center, expected_center)
    assert space.mgep_cutoff == expected_cutoff
    assert [ord_value for _, ord_value in captured_norm_inputs] == [1]
    np.testing.assert_allclose(
        captured_norm_inputs[0][0],
        normalize_selected(simulated, space.genes).to_numpy() - expected_center,
    )
    assert space.stats()["mgep_distance_input_n_samples"] == 4
    assert space.stats()["mgep_distance_kept_n_samples"] == 2
    assert not space.model_path.exists()


def test_mgep_requires_pca_space_filtering_to_be_disabled(tmp_path):
    with pytest.raises(ValueError, match='mgep.*filtering_in_pca_space: false'):
        make_space(tmp_path, filtering_method="mgep")


def test_mgep_uses_inclusive_lower_and_upper_distance_cutoffs(tmp_path):
    space = make_space(
        tmp_path, filtering_method="mgep", filtering_in_pca_space=False,
        gep_filtering_quantile=(0.25, 0.75),
    )
    simulated = pd.DataFrame(
        {"a": [4.0, 5.0, 6.0, 7.0], "b": [6.0, 5.0, 4.0, 3.0]},
        index=["below", "lower", "upper", "above"],
    )

    assert space.filter(simulated).index.tolist() == ["lower", "upper"]
    assert space.mgep_lower_cutoff == pytest.approx(200000)
    assert space.mgep_cutoff == pytest.approx(400000)
    assert space.stats()["mgep_distance_kept_n_samples"] == 2


@pytest.mark.parametrize("lower", [None, 0.0])
def test_mgep_zero_or_null_lower_quantile_keeps_center(tmp_path, lower):
    space = make_space(
        tmp_path, filtering_method="mgep", filtering_in_pca_space=False,
        gep_filtering_quantile=(lower, 0.5),
    )
    simulated = pd.DataFrame({"a": [4.0, 7.0], "b": [6.0, 3.0]},
                             index=["center", "outside"])
    assert space.filter(simulated).index.tolist() == ["center"]
    assert space.mgep_lower_cutoff is None


@pytest.mark.parametrize("quantiles", [
    (-0.1, 0.95), (0.96, 0.95), (0.1, 1.1), (0.1, None),
    (np.nan, 0.95), (0.1, np.nan), (np.inf, 1.0), (0.1, np.inf),
])
def test_mgep_rejects_invalid_distance_quantiles(tmp_path, quantiles):
    with pytest.raises(ValueError, match="gep_filtering_quantile must satisfy"):
        make_space(
            tmp_path,
            filtering_method="mgep",
            filtering_in_pca_space=False,
            gep_filtering_quantile=quantiles,
        )


@pytest.mark.parametrize(
    "method,in_pca,norm,lower",
    [
        ("median_gep", True, 2, 0.0),
        ("median_gep", False, 1, 0.0),
        ("mean_gep", True, 1, 0.0),
        ("linear_mmd", False, 1, 0.0),
        ("mgep", False, 2, None),
        ("mgep", False, 2, 0.0),
        ("mgep", False, 2, 0.25),
    ],
)
def test_distance_diagnostic_uses_filter_reference_and_saved_samples(
    tmp_path, monkeypatch, method, in_pca, norm, lower
):
    from matplotlib.axes import Axes
    from scipy.sparse import csr_matrix

    space = make_space(
        tmp_path, filtering_method=method, filtering_in_pca_space=in_pca,
        norm_ord=norm, gep_filtering_quantile=(lower, 1.0),
    )
    profiles = pd.DataFrame({"a": [2.0, 6.0], "b": [8.0, 4.0]}, index=["s1", "s2"])
    normalized = normalize_selected(profiles, space.genes)
    assert space.filter(profiles).index.tolist() == profiles.index.tolist()
    reference = (
        space.reference_linear.to_numpy() if method == "mgep"
        else space.reference_coordinates if in_pca
        else space.reference_log.to_numpy()
    )
    center = (
        np.median(reference, axis=0) if method in ("mgep", "median_gep")
        else np.mean(reference, axis=0)
    )
    expected_norm = 1 if method == "mgep" else 2 if method == "linear_mmd" else norm
    expected_reference = np.linalg.norm(reference - center, ord=expected_norm, axis=1)
    if method == "linear_mmd":
        expected_reference **= 2
    np.testing.assert_allclose(space.prepare_distance_reference(), expected_reference)

    saved = tmp_path / "accepted.h5ad"
    ad.AnnData(
        X=csr_matrix(np.log2(normalized.to_numpy() + 1).astype(np.float32)),
        obs=pd.DataFrame(index=profiles.index),
        var=pd.DataFrame(index=space.genes),
    ).write_h5ad(saved)
    lines, histograms, simulated_chunks = [], [], []
    original_line, original_stairs = Axes.axvline, Axes.stairs
    original_distance = space.distance_from_normalized

    def capture_distance(values):
        distances = original_distance(values)
        simulated_chunks.extend(distances.tolist())
        return distances

    def capture_line(ax, x, *args, **kwargs):
        lines.append((x, kwargs["label"]))
        return original_line(ax, x, *args, **kwargs)

    def capture_stairs(ax, values, edges, *args, **kwargs):
        histograms.append((np.asarray(values), kwargs["label"]))
        return original_stairs(ax, values, edges, *args, **kwargs)

    monkeypatch.setattr(Axes, "axvline", capture_line)
    monkeypatch.setattr(Axes, "stairs", capture_stairs)
    monkeypatch.setattr(space, "distance_from_normalized", capture_distance)
    paths = workflow._save_gep_distance_distribution(saved, space, batch_size=1)
    assert all(Path(path).is_file() for path in paths.values())
    assert [label for _, label in histograms] == [
        "Reference (n=3)", "Kept simulated (n=2)"
    ]
    assert lines[0][0] == pytest.approx(np.quantile(expected_reference, 1.0))
    has_lower = lower is not None and (method != "mgep" or lower > 0)
    assert len(lines) == 1 + has_lower
    if has_lower:
        assert lines[1][0] == pytest.approx(np.quantile(expected_reference, lower))
    saved_log = np.log2(normalized.to_numpy() + 1).astype(np.float32).astype(np.float64)
    saved_linear = np.expm1(saved_log * np.log(2))
    saved_linear *= 1e6 / saved_linear.sum(axis=1, keepdims=True)
    expected_profiles = (
        saved_linear if method == "mgep"
        else space.pca.transform(np.log2(saved_linear + 1)) if in_pca
        else np.log2(saved_linear + 1)
    )
    expected_simulated = np.linalg.norm(expected_profiles - center, ord=expected_norm, axis=1)
    if method == "linear_mmd":
        expected_simulated **= 2
    np.testing.assert_allclose(simulated_chunks, expected_simulated, rtol=1e-5, atol=1e-5)


def test_distance_plot_rejects_mismatched_saved_gene_order(tmp_path):
    space = make_space(tmp_path, filtering_in_pca_space=False)
    saved = tmp_path / "wrong.h5ad"
    ad.AnnData(
        X=np.array([[1.0, 2.0]]),
        var=pd.DataFrame(index=["b", "a"]),
    ).write_h5ad(saved)
    with pytest.raises(ValueError, match="gene order"):
        workflow._save_gep_distance_distribution(saved, space)


def test_workflow_mgep_uses_linear_filter_and_distinct_output_tag(workflow_case):
    config, params, _, _ = workflow_case
    params.update(
        gep_filtering=True,
        gep_filtering_method="mgep",
        filtering_in_pca_space=False,
    )

    output = workflow.run_bulk_simulation("target", params, config)
    summary = read_summary(output)

    assert "_mgep" in output
    assert summary["gep_filtering_method"] == "mgep"
    assert summary["gep_filtering_n_pcs"] is None
    first_input = summary["mgep_distance_input_n_samples"]
    first_kept = summary["mgep_distance_kept_n_samples"]
    assert first_input > 0 and first_kept > 0

    Path(output).unlink()
    assert workflow.run_bulk_simulation("target", params, config) == output
    resumed_summary = read_summary(output)
    assert resumed_summary["mgep_distance_input_n_samples"] == first_input
    assert resumed_summary["mgep_distance_kept_n_samples"] == first_kept


@pytest.mark.parametrize("lower", [None, 0.0])
def test_workflow_mgep_cache_preserves_upper_only_identity(workflow_case, lower):
    config, params, _, batches = workflow_case
    params.update(
        gep_filtering=True, gep_filtering_method="mgep",
        filtering_in_pca_space=False, gep_filtering_quantile=[lower, 1.0],
    )
    output = workflow.run_bulk_simulation("target", params, config)
    result_dir = workflow._resolve_mixed_dataset_result_dir(output, "target")
    metadata = json.loads((result_dir / "filtering_metadata.json").read_text())
    assert metadata["identity"]["distance_quantiles"] == [lower, 1.0]
    assert workflow.run_bulk_simulation("target", params, config) == output
    assert len(batches) == 1

    params["gep_filtering_quantile"] = [0.25, 1.0]
    with pytest.raises(ValueError, match="Incompatible filtering metadata"):
        workflow.run_bulk_simulation("target", params, config)
    assert len(batches) == 1


def test_workflow_without_gep_filtering_uses_method_independent_output_tag(workflow_case):
    config, params, _, _ = workflow_case
    params.update(
        gep_filtering=False,
        gep_filtering_method="mgep",
        filtering_in_pca_space=False,
    )

    output = workflow.run_bulk_simulation("target", params, config)
    summary = read_summary(output)

    assert "segment_0ds_n_base1_no_filtering" in output
    assert summary["gep_filtering_enabled"] is False
    assert summary["gep_filtering_method"] is None
    assert summary["mgep_distance_input_n_samples"] == 0
    assert summary["mgep_distance_kept_n_samples"] == 0
