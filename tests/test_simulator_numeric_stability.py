import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simutme.simulation.simu_helper import DatasetType, SimuMethod
from simutme.simulation.simulator import BulkGEPGenerator
from simutme.utility.pub_func import MAX_LOG2_TPM, do_pca_analysis, log_exp2cpm, log2_transform, non_log2cpm, safe_pca_transform


def _make_generator_with_sct_matrix(sct_dataset_df: pd.DataFrame) -> BulkGEPGenerator:
    generator = object.__new__(BulkGEPGenerator)
    generator.sct_dataset_df = sct_dataset_df
    generator.sct_dataset_reader = None
    generator.merged_sc_dataset_df = None
    generator.generated_sc_dataset_df = None
    generator.cell_type2subtype = {"CT1": ["CT1"], "CT2": ["CT2"]}
    generator.total_rna_coefficient = {"CT1": 1.0e12, "CT2": 2.0e12}
    return generator


class _FakeLazySCTReader:
    def __init__(self, source_df: pd.DataFrame):
        self.source_df = source_df
        self.calls: list[tuple[list[str], bool]] = []

    def get_rows_df(self, row_ids: list[str], convert_to_tpm: bool = False) -> pd.DataFrame:
        self.calls.append((list(row_ids), bool(convert_to_tpm)))
        return self.source_df.loc[row_ids, :].copy()


class _FakeReaderWithShape:
    def __init__(self, n_obs: int, n_vars: int):
        self._h5ad = type("FakeH5AD", (), {"n_obs": n_obs, "n_vars": n_vars})()

    def get_h5ad(self):
        return self._h5ad


def test_map_cell_id2exp_uses_stable_float64_aggregation_for_weighted_bulk():
    sct_dataset_df = pd.DataFrame(
        {
            "gene_a": np.array([1.0e30, 3.0e30], dtype=np.float32),
            "gene_b": np.array([2.0e30, 4.0e30], dtype=np.float32),
        },
        index=["cell_1", "cell_2"],
    )
    generator = _make_generator_with_sct_matrix(sct_dataset_df)

    selected_cell_id = pd.DataFrame(
        {
            "cell_type": ["CT1", "CT2"],
            "selected_cell_id": ["cell_1", "cell_2"],
        },
        index=pd.Index(["sample_1", "sample_1"], name="sample_id"),
    )
    cell_frac = pd.DataFrame(
        {"CT1": np.array([0.4], dtype=np.float32), "CT2": np.array([0.6], dtype=np.float32)},
        index=["sample_1"],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        simulated = generator._map_cell_id2exp(
            selected_cell_id=selected_cell_id,
            sc_dataset=DatasetType.SCT,
            simu_method=SimuMethod.MULTIPLY,
            cell_frac=cell_frac,
            gep_type="MCT",
        )

    assert caught == []
    assert np.isfinite(simulated.to_numpy()).all()
    np.testing.assert_allclose(simulated.sum(axis=1).to_numpy(), np.array([1.0e6]), rtol=1e-6)

    expression_matrix = sct_dataset_df.to_numpy(dtype=np.float64, copy=True)
    expression_matrix *= np.array([1.0e12, 2.0e12], dtype=np.float64).reshape(-1, 1)
    expected = expression_matrix.T @ np.array([0.4, 0.6], dtype=np.float64).reshape(-1, 1)
    expected = expected / expected.sum() * 1.0e6
    np.testing.assert_allclose(
        simulated.loc["sample_1"].to_numpy(),
        expected.flatten(),
        rtol=1e-6,
    )


def test_map_cell_id2exp_supports_lazy_sct_row_loading():
    sct_dataset_df = pd.DataFrame(
        {
            "gene_a": np.array([10.0, 30.0], dtype=np.float32),
            "gene_b": np.array([20.0, 40.0], dtype=np.float32),
        },
        index=["cell_1", "cell_2"],
    )
    generator = _make_generator_with_sct_matrix(None)
    generator.sct_dataset_reader = _FakeLazySCTReader(sct_dataset_df)

    selected_cell_id = pd.DataFrame(
        {
            "cell_type": ["CT1", "CT2"],
            "selected_cell_id": ["cell_1", "cell_2"],
        },
        index=pd.Index(["sample_1", "sample_1"], name="sample_id"),
    )
    cell_frac = pd.DataFrame(
        {"CT1": np.array([0.25], dtype=np.float32), "CT2": np.array([0.75], dtype=np.float32)},
        index=["sample_1"],
    )

    simulated = generator._map_cell_id2exp(
        selected_cell_id=selected_cell_id,
        sc_dataset=DatasetType.SCT,
        simu_method=SimuMethod.MULTIPLY,
        cell_frac=cell_frac,
        gep_type="MCT",
    )

    assert generator.sct_dataset_reader.calls == [(["cell_1", "cell_2"], True)]
    np.testing.assert_allclose(simulated.sum(axis=1).to_numpy(), np.array([1.0e6]), rtol=1e-6)


def test_log_exp2cpm_converts_large_float32_values_without_overflow():
    log_space = np.array([[130.0, 0.0]], dtype=np.float32)

    cpm = log_exp2cpm(log_space)
    capped_cpm = log_exp2cpm(np.array([[MAX_LOG2_TPM, 0.0]], dtype=np.float32))

    assert np.isfinite(cpm).all()
    np.testing.assert_allclose(cpm.sum(axis=1), np.array([1.0e6]), rtol=1e-6)
    np.testing.assert_allclose(cpm, capped_cpm, rtol=1e-6)


def test_log2_transform_caps_log_values_at_20():
    tpm = pd.DataFrame({"gene_a": [2 ** 25], "gene_b": [1.0]}, index=["sample_1"])

    log_gep = log2_transform(tpm)

    assert float(log_gep.max().max()) == MAX_LOG2_TPM
    assert np.isfinite(log_gep.to_numpy()).all()


def test_non_log2cpm_rejects_zero_sum_rows():
    exp = pd.DataFrame({"gene_a": [0.0], "gene_b": [0.0]}, index=["sample_1"])

    try:
        non_log2cpm(exp)
    except ValueError as exc:
        assert "row sums" in str(exc)
    else:
        raise AssertionError("Expected non_log2cpm to reject zero-sum rows.")


def test_filter_simulated_geps_keeps_full_reference_when_filter_subset_is_empty():
    generator = object.__new__(BulkGEPGenerator)
    generator.m_gep_ref = None
    generator.marker_ratio_ref = None
    generator.q_dis_nn_ref_upper = 0.0
    generator.q_dis_nn_ref_lower = 0.0
    generator.filtering_quantile_upper = 1.0
    generator.filtering_quantile_lower = None
    generator.cell_type_used = ["CT1", "CT2"]
    generator.n_samples = 2
    generator.n_neighbors_each_ref = 1
    generator.ref_neighbor_counter = {}

    simulated_gep = pd.DataFrame(
        {
            "gene_a": [140.0, 160.0],
            "gene_b": [360.0, 340.0],
        },
        index=["sim_1", "sim_2"],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        reference_dir = Path(tmp_dir) / "datasets" / "TCGA"
        reference_dir.mkdir(parents=True, exist_ok=True)
        reference_file = reference_dir / "reference_tpm.csv"
        pd.DataFrame(
            {
                "gene_a": [100.0, 150.0, 200.0],
                "gene_b": [400.0, 350.0, 300.0],
            },
            index=["ref_1", "ref_2", "ref_3"],
        ).to_csv(reference_file)

        filtered = generator._filter_simulated_geps(
            simulated_gep=simulated_gep,
            simulated_gep_bak=simulated_gep.copy(),
            reference_file=str(reference_file),
            ref_exp_type="TPM",
            filtering_method="mean_gep",
            sample_id_for_filtering=[],
            gene_list_in_sc_ds=simulated_gep.columns.to_list(),
            high_corr_gene_list=None,
            filtering_by_gene_range=False,
            min_percentage_within_gene_range=0.95,
            gene_quantile_range=None,
            filtering_in_pca_space=True,
            pca_n_components=1,
            norm_ord=1,
            n_top=5,
            show_filtering_info=False,
        )

    assert filtered is not None
    assert not filtered.empty
    assert generator.gep_filtering_pca_target == 1
    assert generator.gep_filtering_n_pcs == 1


def test_filter_simulated_geps_reloads_reference_for_cached_pca_state():
    generator = object.__new__(BulkGEPGenerator)
    generator.m_gep_ref = np.array([[150.0, 350.0]])
    generator.marker_ratio_ref = None
    generator.q_dis_nn_ref_upper = 1.0e9
    generator.q_dis_nn_ref_lower = 0.0
    generator.filtering_quantile_upper = 1.0
    generator.filtering_quantile_lower = None
    generator.cell_type_used = ["CT1", "CT2"]
    generator.n_samples = 2
    generator.n_neighbors_each_ref = 1
    generator.ref_neighbor_counter = {}

    simulated_gep = pd.DataFrame(
        {
            "gene_a": [140.0, 160.0],
            "gene_b": [360.0, 340.0],
        },
        index=["sim_1", "sim_2"],
    )
    exp_ref_df = pd.DataFrame(
        {
            "gene_a": [100.0, 150.0, 200.0],
            "gene_b": [400.0, 350.0, 300.0],
        },
        index=["ref_1", "ref_2", "ref_3"],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        pca_model_dir = Path(tmp_dir) / "datasets" / "TCGA" / "pca_model_1"
        pca_model_dir.mkdir(parents=True, exist_ok=True)
        reference_file = pca_model_dir.parent / "reference_tpm.csv"
        exp_ref_df.to_csv(reference_file)

        pca = PCA(n_components=1, random_state=42)
        pca.fit(exp_ref_df.to_numpy(dtype=np.float64))
        dump(pca, pca_model_dir / "tcga_pca_model_for_gep_filtering.pkl")
        pd.DataFrame(["gene_a", "gene_b"]).to_csv(pca_model_dir / "gene_list_for_pca.csv")

        filtered = generator._filter_simulated_geps(
            simulated_gep=simulated_gep,
            simulated_gep_bak=simulated_gep.copy(),
            reference_file=str(reference_file),
            ref_exp_type="TPM",
            filtering_method="mean_gep",
            sample_id_for_filtering=[],
            gene_list_in_sc_ds=simulated_gep.columns.to_list(),
            high_corr_gene_list=None,
            filtering_by_gene_range=False,
            min_percentage_within_gene_range=0.95,
            gene_quantile_range=None,
            filtering_in_pca_space=True,
            pca_n_components=1,
            norm_ord=1,
            n_top=5,
            show_filtering_info=False,
        )

    assert filtered is not None
    assert not filtered.empty
    assert generator.gep_filtering_pca_target == 1
    assert generator.gep_filtering_n_pcs == 1


def test_safe_pca_transform_handles_large_values_without_sklearn_matmul():
    exp_df = pd.DataFrame(
        {
            "gene_a": [1.0e19, 1.1e19, 0.9e19],
            "gene_b": [0.5e19, 0.6e19, 0.4e19],
        },
        index=["s1", "s2", "s3"],
    )
    pca = PCA(n_components=1, random_state=42)
    pca.fit(exp_df.to_numpy(dtype=np.float64))

    with np.errstate(all="raise"):
        coords = safe_pca_transform(pca, exp_df, label="test PCA input")

    assert coords.shape == (3, 1)
    assert np.isfinite(coords).all()


def test_format_total_rna_coefficient_message_is_compact_for_all_default_values():
    message = BulkGEPGenerator._format_total_rna_coefficient_message(
        {"CT1": 1.0, "CT2": 1.0, "CT3": 1.0}
    )

    assert message == "Total RNA coefficients: all 3 cell types use 1.0."


def test_format_total_rna_coefficient_message_lists_only_non_default_values():
    message = BulkGEPGenerator._format_total_rna_coefficient_message(
        {"CT1": 1.0, "CT2": 2.5, "CT3": 0.5}
    )

    assert message == "Total RNA coefficients: 2/3 non-default (CT2=2.5, CT3=0.5)."


def test_do_pca_analysis_uses_full_solver_for_stable_fit():
    exp_df = pd.DataFrame(
        {
            "gene_a": [18.0, 18.5, 19.0, 19.5],
            "gene_b": [10.0, 10.5, 11.0, 11.5],
            "gene_c": [5.0, 5.5, 6.0, 6.5],
        },
        index=["s1", "s2", "s3", "s4"],
    )

    pca = do_pca_analysis(exp_df=exp_df, n_components=2, pca_result_fp="")

    assert getattr(pca, "_fit_svd_solver", None) == "full"


def test_gep_filtering_counters_accumulate_totals():
    generator = object.__new__(BulkGEPGenerator)

    generator._reset_gep_filtering_stats(True)
    generator._record_gep_filtering_counts(input_n_samples=12, kept_n_samples=9)
    generator._record_gep_filtering_counts(input_n_samples=5, kept_n_samples=0)

    assert generator.gep_filtering_input_n_samples == 17
    assert generator.gep_filtering_kept_n_samples == 9
    assert generator.gep_filtering_removed_n_samples == 8


def test_gep_filtering_counters_stay_null_when_disabled():
    generator = object.__new__(BulkGEPGenerator)

    generator._reset_gep_filtering_stats(False)
    generator._record_gep_filtering_counts(input_n_samples=12, kept_n_samples=9)

    assert generator.gep_filtering_input_n_samples is None
    assert generator.gep_filtering_kept_n_samples is None
    assert generator.gep_filtering_removed_n_samples is None
    assert generator.gep_filtering_pca_target is None
    assert generator.gep_filtering_n_pcs is None


def test_resolve_sct_loading_mode_uses_eager_for_smaller_auto_sct_matrix():
    generator = object.__new__(BulkGEPGenerator)
    generator.mixed_sct_loading_mode = "auto"
    generator.mixed_sct_auto_eager_max_gb = 4.0

    resolved = generator._resolve_sct_loading_mode(_FakeReaderWithShape(n_obs=1_000, n_vars=2_000))

    assert resolved == "eager"


def test_resolve_sct_loading_mode_uses_lazy_for_larger_auto_sct_matrix():
    generator = object.__new__(BulkGEPGenerator)
    generator.mixed_sct_loading_mode = "auto"
    generator.mixed_sct_auto_eager_max_gb = 4.0

    resolved = generator._resolve_sct_loading_mode(_FakeReaderWithShape(n_obs=200_000, n_vars=20_000))

    assert resolved == "lazy"


def test_resolve_sct_loading_mode_honors_explicit_override():
    generator = object.__new__(BulkGEPGenerator)
    generator.mixed_sct_loading_mode = "lazy"
    generator.mixed_sct_auto_eager_max_gb = 4.0

    resolved = generator._resolve_sct_loading_mode(_FakeReaderWithShape(n_obs=1_000, n_vars=2_000))

    assert resolved == "lazy"


def test_resolve_sct_loading_mode_uses_configured_auto_threshold():
    generator = object.__new__(BulkGEPGenerator)
    generator.mixed_sct_loading_mode = "auto"
    generator.mixed_sct_auto_eager_max_gb = 0.001

    resolved = generator._resolve_sct_loading_mode(_FakeReaderWithShape(n_obs=1_000, n_vars=2_000))

    assert resolved == "lazy"
