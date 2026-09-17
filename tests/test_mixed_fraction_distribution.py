from pathlib import Path

import pandas as pd

from simutme.simulation.sampling import (
    generate_cell_fraction,
    restrict_cell_type2subtype_to_output_types,
)
from simutme.workflow import _save_cancer_cell_fraction_distribution


def test_mixed_fraction_mapping_uses_parent_label_when_requested():
    mapping = restrict_cell_type2subtype_to_output_types(
        {"B Cells": ["Non-plasma B cells", "Plasma B cells"], "DC": ["pDC", "mDC"]},
        ["Non-plasma B cells", "Plasma B cells", "DC"],
    )

    cell_fractions = generate_cell_fraction(
        sampling_method="segment",
        n_cell_frac=4,
        cell_type2subtype=mapping,
    )

    assert mapping["DC"] == ["DC"]
    assert cell_fractions.columns.tolist() == ["Non-plasma B cells", "Plasma B cells", "DC"]


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


def test_save_cancer_cell_fraction_distribution_writes_all_cell_types_when_enabled(tmp_path: Path):
    cell_frac = pd.DataFrame(
        {
            "Cancer Cells": [0.1, 0.2, 0.4, 0.6],
            "B Cells": [0.9, 0.8, 0.6, 0.4],
            "CD4 T": [0.0, 0.0, 0.0, 0.1],
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
        plot_all_cell_types=True,
    )

    assert (output_dir / "cell_fraction_summary_all_cell_types.csv").exists()
    assert (output_dir / "B_Cells_fraction_distribution.png").exists()
    assert (output_dir / "B_Cells_fraction_distribution.pdf").exists()
