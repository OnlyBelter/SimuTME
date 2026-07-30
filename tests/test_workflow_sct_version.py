from pathlib import Path
from types import SimpleNamespace

import pytest

from simutme.workflow import _resolve_mixed_sct_dataset_path


def test_resolve_mixed_sct_dataset_path_requires_original_when_imputation_not_applied(tmp_path: Path):
    original_sct_path = tmp_path / "toy_sct.h5ad"
    original_sct_path.write_text("placeholder", encoding="utf-8")

    config = SimpleNamespace(
        sct_dataset_file_path=original_sct_path,
        sorted_bulk_imputation_output_suffix="_sorted_bulk_imputed",
    )

    with pytest.raises(ValueError, match='set "sct_dataset_version" to "original"'):
        _resolve_mixed_sct_dataset_path(
            dataset_name="Mixed_Test",
            params={"sct_dataset_version": "sorted_bulk_imputed"},
            config=config,
        )
