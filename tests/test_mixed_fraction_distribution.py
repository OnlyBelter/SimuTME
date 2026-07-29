from pathlib import Path

import pandas as pd

from simutme.workflow import _save_cancer_cell_fraction_distribution


def test_save_cancer_cell_fraction_distribution_writes_outputs(tmp_path: Path):
    cell_frac = pd.DataFrame(
        {
            "Cancer Cells": [0.1, 0.2, 0.4, 0.6],
            "B Cells": [0.9, 0.8, 0.6, 0.4],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    cell_frac_path = tmp_path / "cell_fractions_mixed.csv"
    cell_frac.to_csv(cell_frac_path)

    output_dir = tmp_path / "Mixed_Test"
    _save_cancer_cell_fraction_distribution(
        dataset_name="Mixed_Test",
        cell_fraction_file_path=cell_frac_path,
        output_dir=output_dir,
    )

    summary_df = pd.read_csv(output_dir / "cancer_cell_fraction_summary.csv")
    assert summary_df.loc[0, "dataset_id"] == "Mixed_Test"
    assert summary_df.loc[0, "cell_type"] == "Cancer Cells"
    assert int(summary_df.loc[0, "n_samples"]) == 4
    assert (output_dir / "cancer_cell_fraction_distribution.png").exists()
    assert (output_dir / "cancer_cell_fraction_distribution.pdf").exists()
