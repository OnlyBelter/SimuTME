import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simutme.simulation.simulator import _candidate_batch_size


@pytest.mark.parametrize(
    ("n_samples", "filtering", "expected"),
    [
        (300, True, 5000),
        (100, True, 3000),
        (167, True, 5000),
        (10000, True, 5000),
        (100, False, 100),
        (300, False, 300),
        (1500, False, 1000),
    ],
)
def test_candidate_batch_size(n_samples, filtering, expected):
    assert _candidate_batch_size(n_samples, filtering) == expected
