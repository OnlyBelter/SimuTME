from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from simutme.simulation.sample_diversity import run_sct_sample_diversity_analysis


def test_run_sct_sample_diversity_analysis_writes_expected_outputs(tmp_path: Path):
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.2, 2.2, 3.2],
            [4.0, 5.0, 6.0],
            [4.1, 5.1, 6.1],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "B Cells": [1.0, 1.0, 0.0, 0.0],
            "Cancer Cells": [0.0, 0.0, 1.0, 1.0],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    var = pd.DataFrame(index=["G1", "G2", "G3"])
    adata = ad.AnnData(X=x, obs=obs, var=var)

    sct_path = tmp_path / "toy_sct.h5ad"
    adata.write_h5ad(sct_path)

    output_dir = tmp_path / "toy_dataset" / "sample_diversity"
    run_sct_sample_diversity_analysis(
        sct_dataset_file_path=sct_path,
        dataset_name="toy_dataset",
        output_dir=output_dir,
        gene_mode="all",
        distance_metric="cosine",
        min_samples_per_cell_type=2,
    )

    distance_df = pd.read_csv(output_dir / "per_cell_type_distance.csv")
    summary_df = pd.read_csv(output_dir / "per_cell_type_summary.csv")
    centroid_df = pd.read_csv(output_dir / "per_cell_type_mGEP_log.csv", index_col=0)

    assert set(distance_df["cell_type"]) == {"B Cells", "Cancer Cells"}
    assert set(summary_df["cell_type"]) == {"B Cells", "Cancer Cells"}
    assert set(centroid_df.index) == {"B Cells", "Cancer Cells"}
    assert (output_dir / "config_used.yaml").exists()
    assert (output_dir / "figures" / "distance_boxplot_by_cell_type.png").exists()
