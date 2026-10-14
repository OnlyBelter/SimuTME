from types import SimpleNamespace

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from simutme.simulation.simulator import GEPGenerationConfig
from simutme.simulation.sampling import generate_cell_fraction
from simutme.workflow import _resolve_dataset_override


CELL_TYPE2SUBTYPE = {
    "B cells": ["B cells"],
    "T cells": ["T cells"],
    "Cancer cells": ["Cancer cells"],
}
DIRICHLET_ALPHA = {
    "B cells": 0.5,
    "T cells": 1.0,
    "Cancer cells": 2.0,
}


def _sample_two_batches(seed: int):
    rng = np.random.default_rng(seed)
    batches = []
    for batch_index in range(2):
        batches.append(
            generate_cell_fraction(
                sampling_method="dirichlet",
                n_cell_frac=64,
                cell_type2subtype=CELL_TYPE2SUBTYPE,
                sample_prefix=f"batch_{batch_index}",
                dirichlet_alpha=DIRICHLET_ALPHA,
                rng=rng,
            )
        )
    return batches


def test_dirichlet_seed_reproduces_distinct_batches():
    first_run = _sample_two_batches(seed=123)
    second_run = _sample_two_batches(seed=123)

    for first, second in zip(first_run, second_run):
        assert_frame_equal(first, second)
    assert not first_run[0].equals(first_run[1])


def test_different_dirichlet_seeds_change_the_draws():
    first_run = _sample_two_batches(seed=123)
    second_run = _sample_two_batches(seed=456)

    assert not first_run[0].equals(second_run[0])


def test_dirichlet_without_generator_uses_global_numpy_rng():
    np.random.seed(123)
    first = generate_cell_fraction(
        sampling_method="dirichlet",
        n_cell_frac=64,
        cell_type2subtype=CELL_TYPE2SUBTYPE,
        dirichlet_alpha=DIRICHLET_ALPHA,
    )

    np.random.seed(123)
    second = generate_cell_fraction(
        sampling_method="dirichlet",
        n_cell_frac=64,
        cell_type2subtype=CELL_TYPE2SUBTYPE,
        dirichlet_alpha=DIRICHLET_ALPHA,
    )

    assert_frame_equal(first, second)


def test_dataset_seed_overrides_global_seed():
    config = SimpleNamespace(dirichlet_sampling_seed=123)

    assert _resolve_dataset_override(
        {"dirichlet_sampling_seed": 456},
        config,
        "dirichlet_sampling_seed",
        config.dirichlet_sampling_seed,
    ) == 456
    assert _resolve_dataset_override(
        {},
        config,
        "dirichlet_sampling_seed",
        config.dirichlet_sampling_seed,
    ) == 123


def test_negative_dirichlet_seed_is_rejected():
    with pytest.raises(ValueError, match="non-negative integer"):
        GEPGenerationConfig(n_samples=1, dirichlet_sampling_seed=-1)
