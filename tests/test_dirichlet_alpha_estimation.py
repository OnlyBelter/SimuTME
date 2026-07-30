import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from simutme.simulation.sampling import estimate_dirichlet_alpha_from_pred_cell_props
from simutme.workflow import _save_estimated_dirichlet_alpha_from_pred_cell_props


def test_estimate_dirichlet_alpha_from_pred_cell_props_uses_requested_columns(tmp_path: Path):
    csv_path = tmp_path / "pred_cell_props.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "Cancer Cells": [0.7, 0.4],
            "CD4 T": [0.2, 0.3],
            "NK": [0.1, 0.3],
            "unused_meta": ["a", "b"],
        }
    ).to_csv(csv_path, index=False)

    fake_dirichlet = types.SimpleNamespace(
        mle=lambda x, method="meanversion": np.array([2.0, 3.0, 4.0])
    )
    original_module = sys.modules.get("dirichlet")
    sys.modules["dirichlet"] = fake_dirichlet
    try:
        alpha = estimate_dirichlet_alpha_from_pred_cell_props(
            pred_cell_prop_file_path=csv_path,
            cell_types=["Cancer Cells", "CD4 T", "NK"],
        )
    finally:
        if original_module is None:
            del sys.modules["dirichlet"]
        else:
            sys.modules["dirichlet"] = original_module

    assert alpha == {"Cancer Cells": 2.0, "CD4 T": 3.0, "NK": 4.0}


def test_save_estimated_dirichlet_alpha_from_pred_cell_props_writes_csv(tmp_path: Path):
    csv_path = tmp_path / "pred_cell_props.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "Cancer Cells": [0.7, 0.4],
            "CD4 T": [0.2, 0.3],
            "NK": [0.1, 0.3],
        }
    ).to_csv(csv_path, index=False)

    fake_dirichlet = types.SimpleNamespace(
        mle=lambda x, method="meanversion": np.array([2.0, 3.0, 4.0])
    )
    original_module = sys.modules.get("dirichlet")
    sys.modules["dirichlet"] = fake_dirichlet
    try:
        out_path = _save_estimated_dirichlet_alpha_from_pred_cell_props(
            pred_cell_prop_file_path=csv_path,
            cell_types=["Cancer Cells", "CD4 T", "NK"],
            output_dir=tmp_path,
        )
    finally:
        if original_module is None:
            del sys.modules["dirichlet"]
        else:
            sys.modules["dirichlet"] = original_module

    assert out_path.exists()
    alpha_df = pd.read_csv(out_path)
    assert list(alpha_df["cell_type"]) == ["Cancer Cells", "CD4 T", "NK"]
