import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simutme.sct_imputation import (
    _assign_nearest_bulk_samples_for_sct_cells,
    _log2cpm1p_to_tpm,
    _tpm_to_log2cpm1p,
)
from simutme.utility.pub_func import MAX_LOG2_TPM


def test_log2cpm1p_to_tpm_caps_large_log_values():
    restored = _log2cpm1p_to_tpm(np.array([[130.0, 0.0]], dtype=np.float32))
    capped = _log2cpm1p_to_tpm(np.array([[MAX_LOG2_TPM, 0.0]], dtype=np.float32))

    assert np.isfinite(restored).all()
    np.testing.assert_allclose(restored, capped, rtol=1e-6)


def test_assign_nearest_bulk_samples_avoids_matmul_overflow_for_large_log_values():
    x_log2cpm1p = np.array(
        [
            [130.0, 0.0],
            [0.0, 130.0],
        ],
        dtype=np.float32,
    )
    sct_cell_types = pd.Series(["SCT_A", "SCT_A"])
    mapping_df = pd.DataFrame(
        {
            "cell_type_SCT_N6K_13Jul": ["SCT_A"],
            "cell_type_Kassandra_SortedBulk": ["Bulk_A"],
            "included": [1],
        }
    )
    bulk_tpm = np.array(
        [
            [1.0e6, 0.0],
            [0.0, 1.0e6],
        ],
        dtype=np.float32,
    )
    bulk_indices = {"Bulk_A": np.array([0, 1], dtype=int)}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        assigned, included = _assign_nearest_bulk_samples_for_sct_cells(
            x_log2cpm1p=x_log2cpm1p,
            sct_cell_types=sct_cell_types,
            mapping_df=mapping_df,
            bulk_tpm=bulk_tpm,
            bulk_indices=bulk_indices,
            batch_size=2,
        )

    assert caught == []
    np.testing.assert_array_equal(assigned, np.array([0, 1]))
    np.testing.assert_array_equal(included, np.array([1, 1], dtype=np.int8))


def test_tpm_to_log2cpm1p_caps_large_tpm_values():
    restored = _tpm_to_log2cpm1p(np.array([[2.0 ** 25, 1.0]], dtype=np.float64))

    assert np.isfinite(restored).all()
    assert float(restored.max()) == MAX_LOG2_TPM
