import numpy as np
import pandas as pd
import pytest

from simutme.simulation.filtering import FixedGeneFilter, identity_hash
from simutme.workflow import _resolve_gep_filtering_params
from tests.test_correlation_first_filtering import gep_params
from tests.test_filtering_migration import _make_filtering_config


def _space(tmp_path, overrides=None, *, cancer_types=None, subgroup=4, target=1):
    reference = pd.DataFrame(
        np.random.default_rng(12).uniform(1, 10, size=(8, 3)),
        index=[f"r{i}" for i in range(8)], columns=["g1", "g2", "g3"],
    )
    if cancer_types is None:
        cancer_types = ["A"] * 4 + ["B"] * 4
    labels = pd.Series(cancer_types, index=reference.index)
    gep = gep_params(
        filtering_method="median_gep", n_subgroup=subgroup,
        pca_n_components=target, gep_filtering_by_cancer_type=overrides,
    )
    return FixedGeneFilter(
        ["g1", "g2", "g3"], reference, tmp_path / "ref.csv", None,
        gep, tmp_path / "result", 3, 3, reference_cancer_types=labels,
        dataset_name="Mixed_Test",
    )


def test_override_configuration_uses_dataset_and_global_defaults(tmp_path):
    config = _make_filtering_config(tmp_path)
    config.n_subgroup = 10
    params = {
        "gep_filtering": True, "gep_filtering_method": "median_gep",
        "gep_filtering_in_pca_space": True, "n_subgroup": 4,
        "gep_pca_n_components": 0.9,
        "gep_filtering_by_cancer_type": {
            "B": {"gep_pca_n_components": 0.95},
            "A": {"n_subgroup": 2},
        },
    }
    resolved = _resolve_gep_filtering_params("Mixed_Test", params, config)
    assert resolved.n_subgroup == 4
    assert resolved.pca_n_components == 0.9
    assert resolved.gep_filtering_by_cancer_type == {
        "A": {"n_subgroup": 2}, "B": {"gep_pca_n_components": 0.95},
    }
    params.pop("n_subgroup")
    assert _resolve_gep_filtering_params("Mixed_Test", params, config).n_subgroup == 10


@pytest.mark.parametrize("overrides", [
    "A", [], {"": {"n_subgroup": 2}}, {" A": {"n_subgroup": 2}},
    {"A": None}, {"A": {}}, {"A": {"typo": 1}}, {"A": {"n_subgroup": None}},
    {"A": {"n_subgroup": True}}, {"A": {"n_subgroup": 0}},
    {"A": {"n_subgroup": 2.0}}, {"A": {"gep_pca_n_components": None}},
    {"A": {"gep_pca_n_components": True}}, {"A": {"gep_pca_n_components": 0}},
    {"A": {"gep_pca_n_components": 1.0}}, {"A": {"gep_pca_n_components": float("nan")}},
])
def test_invalid_override_configuration_fails(tmp_path, overrides):
    config = _make_filtering_config(tmp_path)
    params = {
        "gep_filtering": True, "gep_filtering_method": "median_gep",
        "gep_filtering_in_pca_space": True, "n_subgroup": 4,
        "gep_filtering_by_cancer_type": overrides,
    }
    with pytest.raises(ValueError, match="Mixed dataset"):
        _resolve_gep_filtering_params("Mixed_Test", params, config)


def test_override_requires_positive_default_and_supported_filter(tmp_path):
    config = _make_filtering_config(tmp_path)
    params = {
        "gep_filtering": True, "gep_filtering_method": "median_gep",
        "gep_filtering_in_pca_space": True,
        "gep_filtering_by_cancer_type": {"A": {"n_subgroup": 2}},
    }
    with pytest.raises(ValueError, match="positive default"):
        _resolve_gep_filtering_params("Mixed_Test", params, config)
    with pytest.raises(ValueError, match="enabled PCA-space"):
        _resolve_gep_filtering_params(
            "Mixed_Test", {**params, "n_subgroup": 4, "gep_filtering_in_pca_space": False}, config
        )
    with pytest.raises(ValueError, match="default GEP PCA"):
        _resolve_gep_filtering_params(
            "Mixed_Test", {**params, "n_subgroup": 4, "gep_pca_n_components": 1.0}, config
        )
    with pytest.raises(ValueError, match="default GEP PCA"):
        _resolve_gep_filtering_params(
            "Mixed_Test", {**params, "n_subgroup": 4, "gep_pca_n_components": None}, config
        )


def test_selected_batch_uses_type_specific_cap_and_pca(tmp_path):
    space = _space(tmp_path, {
        "A": {"n_subgroup": 2, "gep_pca_n_components": 1},
        "B": {"n_subgroup": 3, "gep_pca_n_components": 2},
    })
    observed = set()
    for seed in range(30):
        selected = space._select_batch_cancer_type(seed)
        space.prepare_batch_reference(batch_id=seed)
        expected = space.gep_filtering_parameters_by_cancer_type[selected]
        assert space.last_batch_reference_sample_count == expected["n_subgroup"]
        assert space.pca.n_components == expected["gep_pca_n_components"]
        assert space.pca.n_components_ == expected["gep_pca_n_components"]
        observed.add(selected)
        space.clear_batch_reference()
    assert observed == {"A", "B"}


def test_partial_overrides_inherit_defaults_and_clamp_to_type_size(tmp_path):
    space = _space(tmp_path, {"A": {"n_subgroup": 7}, "B": {"gep_pca_n_components": 0.9}})
    assert space.gep_filtering_parameters_by_cancer_type == {
        "A": {"n_subgroup": 7, "gep_pca_n_components": 1},
        "B": {"n_subgroup": 4, "gep_pca_n_components": 0.9},
    }
    for seed in range(20):
        indices = space._sample_subgroup_indices(seed=seed)
        assert len(indices) == 4
        assert len(set(indices)) == 4
        assert all(i in space.reference_cancer_type_groups[space.last_sampled_cancer_type]
                   for i in indices)


@pytest.mark.parametrize("overrides, match", [
    ({"Z": {"n_subgroup": 2}}, "unknown or excluded"),
    ({"A": {"n_subgroup": 1}}, 'Cancer type "A".*too few'),
    ({"B": {"n_subgroup": 2, "gep_pca_n_components": 3}}, 'Cancer type "B".*too few'),
    ({"B": {"gep_pca_n_components": 4}}, 'Cancer type "B".*too few'),
])
def test_ineligible_types_and_incompatible_pca_fail_early(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        _space(tmp_path, overrides)


def test_identity_is_canonical_and_scalar_only_is_unchanged(tmp_path):
    scalar = _space(tmp_path)
    empty = _space(tmp_path, {})
    assert scalar.identity == empty.identity
    assert "gep_filtering_parameters_by_cancer_type" not in scalar.identity
    settings = {"A": {"n_subgroup": 2}, "B": {"gep_pca_n_components": 0.9}}
    first = _space(tmp_path, settings)
    reordered = _space(tmp_path, {"B": settings["B"], "A": settings["A"]})
    changed = _space(tmp_path, {"A": {"n_subgroup": 3}, "B": settings["B"]})
    assert identity_hash(first.identity) == identity_hash(reordered.identity)
    assert identity_hash(first.identity) != identity_hash(changed.identity)
    assert first.identity["gep_filtering_parameters_by_cancer_type"] == (
        first.gep_filtering_parameters_by_cancer_type
    )


def test_single_type_full_reference_uses_override_and_validates_model_identity(tmp_path):
    labels = ["A"] * 8
    settings = {"A": {"n_subgroup": 10, "gep_pca_n_components": 2}}
    space = _space(tmp_path, settings, cancer_types=labels)
    assert not space.uses_subgroup_reference
    assert space.pca_identity["pca_target"] == 2
    space.prepare_distance_reference()
    assert space.pca.n_components == 2
    changed = _space(
        tmp_path, {"A": {"n_subgroup": 10, "gep_pca_n_components": 1}}, cancer_types=labels
    )
    with pytest.raises(ValueError, match="Incompatible filtering metadata"):
        changed.validate_model()


def test_single_type_override_can_activate_per_batch_reference(tmp_path):
    space = _space(
        tmp_path, {"A": {"n_subgroup": 3}}, cancer_types=["A"] * 8, subgroup=20
    )
    assert space.uses_subgroup_reference
    space.prepare_batch_reference(seed=42)
    assert space.last_batch_reference_sample_count == 3
