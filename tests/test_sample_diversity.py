from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import pytest

from simutme.simulation.sample_diversity import (
    _get_distance_axis_label,
    compare_sct_sample_diversity_across_datasets,
    run_sct_sample_diversity_analysis,
)


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


def test_run_sct_sample_diversity_analysis_hvg_uses_reference_gene_variance(tmp_path: Path):
    ref_x = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 3.0, 1.0],
            [1.0, 4.0, 1.0],
            [1.0, 5.0, 1.0],
        ],
        dtype=np.float32,
    )
    ref_adata = ad.AnnData(X=ref_x, obs=pd.DataFrame(index=[f"c{i}" for i in range(ref_x.shape[0])]), var=pd.DataFrame(index=["G1", "G2", "G3"]))
    ref_path = tmp_path / "merged_sc.h5ad"
    ref_adata.write_h5ad(ref_path)

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

    output_dir = tmp_path / "toy_dataset_hvg" / "sample_diversity"
    run_sct_sample_diversity_analysis(
        sct_dataset_file_path=sct_path,
        dataset_name="toy_dataset_hvg",
        output_dir=output_dir,
        gene_mode="hvg",
        n_top_genes=2,
        merged_sc_dataset_file_path=ref_path,
        distance_metric="cosine",
        min_samples_per_cell_type=2,
    )

    distance_df = pd.read_csv(output_dir / "per_cell_type_distance.csv")
    assert distance_df["n_genes"].nunique() == 1
    assert int(distance_df["n_genes"].iloc[0]) == 2
    assert (ref_path.with_name(f"{ref_path.stem}.gene_variance.csv")).exists()


def test_get_distance_axis_label():
    assert _get_distance_axis_label("cosine") == "1-cosine distance to mGEP_log"
    assert _get_distance_axis_label("l2") == "l2 distance to mGEP_log"


def test_compare_sct_sample_diversity_across_datasets_writes_expected_outputs(tmp_path: Path):
    dataset_a = pd.DataFrame(
        {
            "dataset_id": ["A", "A", "A", "A"],
            "cell_type": ["B Cells", "B Cells", "Cancer Cells", "Cancer Cells"],
            "sample_id": ["a1", "a2", "a3", "a4"],
            "distance": [0.1, 0.2, 0.3, 0.4],
            "distance_metric": ["cosine"] * 4,
            "n_genes": [100, 100, 100, 100],
        }
    )
    dataset_b = pd.DataFrame(
        {
            "dataset_id": ["B", "B", "B", "B"],
            "cell_type": ["B Cells", "B Cells", "Cancer Cells", "Cancer Cells"],
            "sample_id": ["b1", "b2", "b3", "b4"],
            "distance": [0.15, 0.25, 0.35, 0.45],
            "distance_metric": ["cosine"] * 4,
            "n_genes": [120, 120, 120, 120],
        }
    )
    dataset_c = pd.DataFrame(
        {
            "dataset_id": ["C", "C"],
            "cell_type": ["B Cells", "Cancer Cells"],
            "sample_id": ["c1", "c2"],
            "distance": [0.12, 0.32],
            "distance_metric": ["cosine"] * 2,
            "n_genes": [110, 110],
        }
    )
    csv_a = tmp_path / "dataset_a.csv"
    csv_b = tmp_path / "dataset_b.csv"
    csv_c = tmp_path / "dataset_c.csv"
    dataset_a.to_csv(csv_a, index=False)
    dataset_b.to_csv(csv_b, index=False)
    dataset_c.to_csv(csv_c, index=False)

    output_dir = tmp_path / "comparison"
    compare_sct_sample_diversity_across_datasets(
        distance_csv_paths=[csv_a, csv_b, csv_c],
        output_dir=output_dir,
        y_max=0.1,
    )

    merged_df = pd.read_csv(output_dir / "per_cell_type_distance_all_datasets.csv")
    summary_df = pd.read_csv(output_dir / "per_dataset_per_cell_type_summary.csv")
    config_used = (output_dir / "config_used.yaml").read_text(encoding="utf-8")
    assert set(merged_df["dataset_id"]) == {"A", "B", "C"}
    assert set(summary_df["dataset_id"]) == {"A", "B", "C"}
    assert (output_dir / "config_used.yaml").exists()
    assert "y_max: 0.1" in config_used
    assert (output_dir / "figures" / "distance_boxplot_by_cell_type.png").exists()
    assert (output_dir / "figures" / "B_Cells_distance_distribution.png").exists()


def test_compare_sct_sample_diversity_across_datasets_rejects_mixed_distance_metrics(tmp_path: Path):
    dataset_a = pd.DataFrame(
        {
            "dataset_id": ["A", "A"],
            "cell_type": ["B Cells", "Cancer Cells"],
            "sample_id": ["a1", "a2"],
            "distance": [0.1, 0.2],
            "distance_metric": ["cosine", "cosine"],
            "n_genes": [100, 100],
        }
    )
    dataset_b = pd.DataFrame(
        {
            "dataset_id": ["B", "B"],
            "cell_type": ["B Cells", "Cancer Cells"],
            "sample_id": ["b1", "b2"],
            "distance": [0.3, 0.4],
            "distance_metric": ["l2", "l2"],
            "n_genes": [100, 100],
        }
    )
    csv_a = tmp_path / "dataset_a.csv"
    csv_b = tmp_path / "dataset_b.csv"
    dataset_a.to_csv(csv_a, index=False)
    dataset_b.to_csv(csv_b, index=False)

    with pytest.raises(ValueError, match="same distance metric"):
        compare_sct_sample_diversity_across_datasets(
            distance_csv_paths=[csv_a, csv_b],
            output_dir=tmp_path / "comparison",
        )
