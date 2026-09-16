import sys
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import pytest  # noqa: F401
except ModuleNotFoundError:
    # Direct-run mode: pytest not available; run the __main__ block only.
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simutme.workflow import SimuConfig, _resolve_stratify_by_dataset
from simutme.simulation.simulator import BulkGEPGenerator
from simutme.simulation.simulator_sct import SingleCellTypeGEPGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_obs(n_ct1_ds1: int = 100,
              n_ct1_ds2: int = 10,
              n_ct2_ds1: int = 30,
              n_ct2_ds2: int = 60,
              n_ct3_ds1: int = 5,
              n_ct3_ds2: int = 5) -> pd.DataFrame:
    """Construct a minimal obs DataFrame for stratification testing.

    Three cell types, two datasets with intentionally imbalanced counts.
    """
    rows = []
    for cell_type, ds, n in [
        ("CT1", "ds1", n_ct1_ds1),
        ("CT1", "ds2", n_ct1_ds2),
        ("CT2", "ds1", n_ct2_ds1),
        ("CT2", "ds2", n_ct2_ds2),
        ("CT3", "ds1", n_ct3_ds1),
        ("CT3", "ds2", n_ct3_ds2),
    ]:
        for i in range(n):
            rows.append({
                "cell_id": f"cell_{cell_type}_{ds}_{i}",
                "cell_type": cell_type,
                "cell_subtype": cell_type,
                "dataset_id": ds,
                "patient_id": f"{ds}_p{i % 3}",
            })
    df = pd.DataFrame(rows).set_index("cell_id")
    return df


def _make_generator(stratify_by_dataset: str = "balanced_quota",
                    stratify_seed: int = 123,
                    cell_type_col_name: str = "cell_type") -> BulkGEPGenerator:
    """Build a lightweight BulkGEPGenerator without invoking the merged h5ad loader.

    We only need the __init__ attributes used by the private stratify method.
    """
    allowed = {"none", "balanced_quota"}
    if stratify_by_dataset not in allowed:
        raise ValueError
    g = object.__new__(BulkGEPGenerator)
    g.stratify_by_dataset = stratify_by_dataset
    g.stratify_seed = int(stratify_seed)
    g.cell_type_col_name = cell_type_col_name
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStratifyByDatasetBalancedQuota:
    def test_balanced_quota_uses_min_per_dataset(self):
        obs = _make_obs()
        g = _make_generator("balanced_quota", 7)
        out = g._stratify_obs_by_dataset_balanced_quota(obs)

        # CT1: min(100,10)=10 per dataset, total 20
        # CT2: min(30,60)=30 per dataset, total 60
        # CT3: min(5,5)=5 per dataset, total 10
        ct1 = out[out["cell_type"] == "CT1"]
        ct2 = out[out["cell_type"] == "CT2"]
        ct3 = out[out["cell_type"] == "CT3"]

        assert ct1.groupby("dataset_id").size().tolist() == [10, 10]
        assert ct2.groupby("dataset_id").size().tolist() == [30, 30]
        assert ct3.groupby("dataset_id").size().tolist() == [5, 5]

        assert ct1.groupby("dataset_id").size().tolist() == [10, 10]
        assert len(out) == 20 + 60 + 10

    def test_none_mode_no_op(self):
        obs = _make_obs()
        g = _make_generator("none")
        out = g._stratify_obs_by_dataset_balanced_quota(obs)
        pd.testing.assert_frame_equal(out.sort_index(), obs.sort_index())

    def test_reproducibility_same_seed(self):
        obs = _make_obs()
        g1 = _make_generator("balanced_quota", 123)
        g2 = _make_generator("balanced_quota", 123)
        out1 = g1._stratify_obs_by_dataset_balanced_quota(obs)
        out2 = g2._stratify_obs_by_dataset_balanced_quota(obs)
        pd.testing.assert_index_equal(out1.sort_index().index, out2.sort_index().index)

    def test_different_seed_different_draw(self):
        obs = _make_obs(n_ct1_ds1=1000, n_ct1_ds2=200, n_ct2_ds1=300, n_ct2_ds2=600, n_ct3_ds1=20, n_ct3_ds2=20)
        g1 = _make_generator("balanced_quota", 1)
        g2 = _make_generator("balanced_quota", 2)
        # CT1 quota = min(1000,200) = 200 from each dataset.
        ct1_ds1_ids_a = g1._stratify_obs_by_dataset_balanced_quota(obs).query("cell_type=='CT1' and dataset_id=='ds1'").index
        ct1_ds1_ids_b = g2._stratify_obs_by_dataset_balanced_quota(obs).query("cell_type=='CT1' and dataset_id=='ds1'").index
        # With probability ~1 they differ across independent draws.
        assert set(ct1_ds1_ids_a) != set(ct1_ds1_ids_b)

    def test_bulk_gep_init_rejects_bad_enum(self):
        tmp = Path('/tmp/simutme_test_stratify')
        tmp.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError):
            BulkGEPGenerator(
                simu_bulk_dir=str(tmp),
                merged_sc_dataset_file_path=str(tmp / "x.h5ad"),
                sct_dataset_file_path=None,
                cell_type2subtype={"CT": ["CT"]},
                sc_dataset_ids=["ds1"],
                stratify_by_dataset="not_allowed",
                check_basic_info=False,
            )

    def test_single_cell_type_init_accepts_knobs(self):
        tmp = Path('/tmp/simutme_test_stratify_sct')
        tmp.mkdir(parents=True, exist_ok=True)
        # Check default: should instantiate without error (stratify defaults to none).
        g = SingleCellTypeGEPGenerator(
            simu_bulk_dir=str(tmp),
            bulk_dataset_name="dummy",
            merged_sc_dataset_file_path=str(tmp / "x.h5ad"),
            cell_type2subtype={"CT": ["CT"], "DC": ["pDC", "mDC"]},
            sc_dataset_ids=["ds1"],
            cancer_cell_dataset_ids=None,
            non_cancer_cell_dataset_ids=None,
            selected_subtypes_by_cell_type={"DC": ["pDC"]},
            subtype_col_name="cell_subtype",
            check_basic_info=False,
        )
        assert g.stratify_by_dataset == "none"
        assert g.selected_subtypes_by_cell_type == {"DC": ["pDC"]}

    def test_get_obs_for_type_applies_selected_subtypes_for_parent_cell_type(self):
        g = object.__new__(SingleCellTypeGEPGenerator)
        g.cell_subtype_used = []
        g.cell_type_col_name = "cell_type"
        g.subtype_col_name = "cell_subtype"
        g.selected_subtypes_by_cell_type = {"DC": ["pDC"]}
        g.merged_sc_dataset_obs = pd.DataFrame(
            [
                {"cell_type": "DC", "cell_subtype": "pDC", "dataset_id": "ds1"},
                {"cell_type": "DC", "cell_subtype": "mDC", "dataset_id": "ds1"},
                {"cell_type": "DC", "cell_subtype": "pDC", "dataset_id": "ds2"},
                {"cell_type": "B Cells", "cell_subtype": "B Cells", "dataset_id": "ds1"},
            ],
            index=["dc_p_1", "dc_m_1", "dc_p_2", "b_1"],
        )

        filtered = g._get_obs_for_type("DC", min_n=1, warn_n=5)

        assert filtered.index.tolist() == ["dc_p_1", "dc_p_2"]
        assert set(filtered["cell_type"]) == {"DC"}
        assert set(filtered["cell_subtype"]) == {"pDC"}

    def test_resolve_stratify_default_none(self):
        # Construct a dataclass with just the required subset of SimuConfig fields.
        from dataclasses import dataclass
        @dataclass
        class _Cfg:
            stratify_by_dataset: str = "none"
        cfg = _Cfg()
        assert _resolve_stratify_by_dataset({}, cfg) == "none"
        assert _resolve_stratify_by_dataset({"stratify_by_dataset": "balanced_quota"}, cfg) == "balanced_quota"
        with pytest.raises(ValueError):
            _resolve_stratify_by_dataset({"stratify_by_dataset": "bad"}, cfg)
        # Global bad value
        cfg_bad = _Cfg(stratify_by_dataset="banana")
        with pytest.raises(ValueError):
            _resolve_stratify_by_dataset({}, cfg_bad)


if __name__ == "__main__":
    # Direct runner equivalent for environments without pytest.
    np.random.seed(0)
    failures: list[str] = []

    def check(name, cond):
        if cond:
            print(f"[PASS] {name}")
        else:
            failures.append(name)
            print(f"[FAIL] {name}")

    t = TestStratifyByDatasetBalancedQuota()
    # test_balanced_quota_uses_min_per_dataset
    obs = _make_obs()
    g = _make_generator("balanced_quota", 7)
    out = g._stratify_obs_by_dataset_balanced_quota(obs)
    ct1 = out[out["cell_type"] == "CT1"]
    ct2 = out[out["cell_type"] == "CT2"]
    ct3 = out[out["cell_type"] == "CT3"]
    check("quota per dataset: CT1 sizes [10,10]", ct1.groupby("dataset_id").size().tolist() == [10, 10])
    check("quota per dataset: CT2 sizes [30,30]", ct2.groupby("dataset_id").size().tolist() == [30, 30])
    check("quota per dataset: CT3 sizes [5,5]", ct3.groupby("dataset_id").size().tolist() == [5, 5])
    check("total stratified rows 90", len(out) == 90)

    # test_none_mode_no_op
    out_none = _make_generator("none")._stratify_obs_by_dataset_balanced_quota(obs)
    check("none mode is no-op", out_none.sort_index().equals(obs.sort_index()))

    # test_reproducibility_same_seed
    g1 = _make_generator("balanced_quota", 123)
    g2 = _make_generator("balanced_quota", 123)
    o1 = g1._stratify_obs_by_dataset_balanced_quota(obs).sort_index().index
    o2 = g2._stratify_obs_by_dataset_balanced_quota(obs).sort_index().index
    check("reproducibility same seed", list(o1) == list(o2))

    # test_different_seed_different_draw
    obs_big = _make_obs(1000, 200, 300, 600, 20, 20)
    ga = _make_generator("balanced_quota", 1)
    gb = _make_generator("balanced_quota", 2)
    oa = ga._stratify_obs_by_dataset_balanced_quota(obs_big).query("cell_type=='CT1' and dataset_id=='ds1'").index
    ob = gb._stratify_obs_by_dataset_balanced_quota(obs_big).query("cell_type=='CT1' and dataset_id=='ds1'").index
    check("different seed → different draw", set(oa) != set(ob))

    # BulkGEPGenerator enum guard
    try:
        BulkGEPGenerator(
            simu_bulk_dir="/tmp/simutme_strat_check",
            merged_sc_dataset_file_path="/tmp/simutme_strat_check/x.h5ad",
            sct_dataset_file_path=None,
            cell_type2subtype={"CT": ["CT"]},
            sc_dataset_ids=["ds1"],
            stratify_by_dataset="not_allowed",
            check_basic_info=False,
        )
        check("BulkGEP bad enum raises", False)
    except ValueError:
        check("BulkGEP bad enum raises", True)

    # SingleCellTypeGEPGenerator default none
    try:
        g = SingleCellTypeGEPGenerator(
            simu_bulk_dir="/tmp/simutme_strat_check2",
            bulk_dataset_name="dummy",
            merged_sc_dataset_file_path="/tmp/simutme_strat_check2/x.h5ad",
            cell_type2subtype={"CT": ["CT"]},
            sc_dataset_ids=["ds1"],
            cancer_cell_dataset_ids=None,
            non_cancer_cell_dataset_ids=None,
            check_basic_info=False,
        )
        check("SCT default stratify none", g.stratify_by_dataset == "none")
    except Exception as exc:
        print(f"SCT default init unexpected exc: {exc}")
        check("SCT default stratify none", False)

    # resolve_stratify_by_dataset tests via dataclass
    from dataclasses import dataclass
    @dataclass
    class _Cfg: stratify_by_dataset: str = "none"
    cfg = _Cfg()
    check("resolve default none", _resolve_stratify_by_dataset({}, cfg) == "none")
    check("resolve override balanced", _resolve_stratify_by_dataset({"stratify_by_dataset": "balanced_quota"}, cfg) == "balanced_quota")
    try:
        _resolve_stratify_by_dataset({"stratify_by_dataset": "bad"}, cfg); check("bad override raises", False)
    except ValueError: check("bad override raises", True)
    try:
        _resolve_stratify_by_dataset({}, _Cfg("banana")); check("bad global raises", False)
    except ValueError: check("bad global raises", True)

    if failures:
        print(f"\nFAILED {len(failures)} tests: {failures}")
        sys.exit(1)
    print("\nALL DIRECT TESTS PASSED")
