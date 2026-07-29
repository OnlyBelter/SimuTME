from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

logger = logging.getLogger(__name__)


def _ensure_dense_float32(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x


def _infer_cell_type_labels(obs_df: pd.DataFrame) -> pd.Series:
    if obs_df.empty:
        raise ValueError("SCT dataset obs is empty; cannot infer sample cell types.")
    return obs_df.idxmax(axis=1).astype(str)


def _read_gene_list(gene_list_path: Path) -> list[str]:
    gene_df = pd.read_csv(gene_list_path)
    if gene_df.empty:
        raise ValueError(f"Provided gene list file is empty: {gene_list_path}")
    gene_series = gene_df.iloc[:, 0].dropna().astype(str)
    gene_list = gene_series.drop_duplicates().tolist()
    if not gene_list:
        raise ValueError(f"No valid genes were found in: {gene_list_path}")
    return gene_list


def _select_genes_for_cell_type(
    cell_type_df: pd.DataFrame,
    gene_mode: str,
    n_top_genes: int,
    provided_gene_list: list[str] | None,
) -> list[str]:
    if cell_type_df.columns.has_duplicates:
        cell_type_df = cell_type_df.loc[:, ~cell_type_df.columns.duplicated()].copy()

    if gene_mode == "all":
        selected = cell_type_df.columns.astype(str).tolist()
    elif gene_mode == "provided_gene_list":
        if provided_gene_list is None:
            raise ValueError('"provided_gene_list" mode requires a gene list file.')
        selected = [gene for gene in provided_gene_list if gene in cell_type_df.columns]
    elif gene_mode == "hvg":
        gene_var = cell_type_df.var(axis=0, ddof=1).fillna(0.0)
        gene_var = gene_var.sort_values(ascending=False)
        selected = gene_var.head(n_top_genes).index.astype(str).tolist()
    else:
        raise ValueError(f"Unsupported sample diversity gene mode: {gene_mode}")

    if not selected:
        raise ValueError(
            f"No genes remained after gene selection for cell type {cell_type_df.index.name or 'unknown'}."
        )
    return selected


def _compute_distance_to_centroid(
    x: np.ndarray,
    centroid: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric == "cosine":
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm == 0.0:
            raise ValueError("Cannot compute cosine distance with a zero-norm centroid.")
        x_norm = np.linalg.norm(x, axis=1)
        valid = x_norm > 0
        cosine_similarity = np.ones(x.shape[0], dtype=np.float32)
        cosine_similarity[valid] = (x[valid] @ centroid) / (x_norm[valid] * centroid_norm)
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        return 1.0 - cosine_similarity
    if metric == "l2":
        return np.linalg.norm(x - centroid[None, :], axis=1)
    raise ValueError(f"Unsupported distance metric: {metric}")


def _safe_filename(text: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(text))
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_") or "unknown"


def _save_config_used(
    output_dir: Path,
    dataset_name: str,
    sct_dataset_file_path: str | Path,
    gene_mode: str,
    n_top_genes: int,
    gene_list_path: Path | None,
    distance_metric: str,
    min_samples_per_cell_type: int,
) -> None:
    config_used = {
        "dataset_name": dataset_name,
        "sct_dataset_file_path": str(sct_dataset_file_path),
        "sample_diversity_gene_mode": gene_mode,
        "sample_diversity_n_top_genes": int(n_top_genes),
        "sample_diversity_gene_list_path": None if gene_list_path is None else str(gene_list_path),
        "sample_diversity_distance_metric": distance_metric,
        "sample_diversity_min_samples_per_cell_type": int(min_samples_per_cell_type),
    }
    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_used, fh, sort_keys=False, allow_unicode=False)


def _plot_distance_boxplot(distance_df: pd.DataFrame, figure_dir: Path) -> None:
    if distance_df.empty:
        return
    plt.figure(figsize=(max(6, distance_df["cell_type"].nunique() * 0.6), 5))
    sns.boxplot(data=distance_df, x="cell_type", y="distance", color="#8fb3d9")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Cell type")
    plt.ylabel("Distance to mGEP_log")
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig(figure_dir / f"distance_boxplot_by_cell_type.{suffix}", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_per_cell_type_distributions(distance_df: pd.DataFrame, figure_dir: Path) -> None:
    if distance_df.empty:
        return
    for cell_type, current_df in distance_df.groupby("cell_type", sort=False):
        plt.figure(figsize=(5, 4))
        sns.histplot(current_df["distance"], bins=min(30, max(10, current_df.shape[0] // 2)), kde=True, color="#4c78a8")
        plt.xlabel("Distance to mGEP_log")
        plt.ylabel("Sample count")
        plt.title(cell_type)
        plt.tight_layout()
        safe_cell_type = _safe_filename(cell_type)
        for suffix in ("png", "pdf"):
            plt.savefig(
                figure_dir / f"{safe_cell_type}_distance_distribution.{suffix}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close()


def run_sct_sample_diversity_analysis(
    sct_dataset_file_path: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    gene_mode: str = "all",
    n_top_genes: int = 2000,
    gene_list_path: str | Path | None = None,
    distance_metric: str = "cosine",
    min_samples_per_cell_type: int = 5,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    gene_list_file = None if gene_list_path in (None, "") else Path(gene_list_path)
    provided_gene_list = None if gene_list_file is None else _read_gene_list(gene_list_file)

    adata = ad.read_h5ad(sct_dataset_file_path)
    sample_x = _ensure_dense_float32(adata.X)
    exp_df = pd.DataFrame(sample_x, index=adata.obs_names.astype(str), columns=adata.var_names.astype(str))
    cell_type_labels = _infer_cell_type_labels(adata.obs)

    distance_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    centroid_records: list[pd.Series] = []

    for cell_type, sample_ids in cell_type_labels.groupby(cell_type_labels, sort=False).groups.items():
        cell_type_df = exp_df.loc[list(sample_ids)].copy()
        if cell_type_df.shape[0] < min_samples_per_cell_type:
            logger.warning(
                "Skipping sample diversity analysis for cell type %s in dataset %s: only %d samples (< %d).",
                cell_type,
                dataset_name,
                cell_type_df.shape[0],
                min_samples_per_cell_type,
            )
            continue

        selected_genes = _select_genes_for_cell_type(
            cell_type_df=cell_type_df,
            gene_mode=gene_mode,
            n_top_genes=n_top_genes,
            provided_gene_list=provided_gene_list,
        )
        current_df = cell_type_df.loc[:, selected_genes].copy()
        if current_df.columns.has_duplicates:
            current_df = current_df.loc[:, ~current_df.columns.duplicated()].copy()
        if current_df.shape[1] == 0:
            logger.warning(
                "Skipping sample diversity analysis for cell type %s in dataset %s: no genes remained after filtering.",
                cell_type,
                dataset_name,
            )
            continue

        centroid = current_df.median(axis=0)
        distances = _compute_distance_to_centroid(
            x=current_df.to_numpy(dtype=np.float32, copy=False),
            centroid=centroid.to_numpy(dtype=np.float32, copy=False),
            metric=distance_metric,
        )

        distance_frames.append(
            pd.DataFrame(
                {
                    "dataset_id": dataset_name,
                    "cell_type": str(cell_type),
                    "sample_id": current_df.index.astype(str),
                    "distance": distances.astype(float),
                    "distance_metric": distance_metric,
                    "n_genes": int(current_df.shape[1]),
                }
            )
        )
        summary_rows.append(
            {
                "dataset_id": dataset_name,
                "cell_type": str(cell_type),
                "n_samples": int(current_df.shape[0]),
                "n_genes": int(current_df.shape[1]),
                "median_distance": float(np.median(distances)),
                "mean_distance": float(np.mean(distances)),
                "iqr_distance": float(np.percentile(distances, 75) - np.percentile(distances, 25)),
                "p90_distance": float(np.percentile(distances, 90)),
            }
        )
        centroid_records.append(pd.Series(centroid, name=str(cell_type)))

    if not distance_frames:
        raise ValueError(
            f"No cell types in SCT dataset {dataset_name} met the minimum sample requirement "
            f"for sample diversity analysis."
        )

    distance_df = pd.concat(distance_frames, axis=0, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    centroid_df = pd.DataFrame(centroid_records)
    centroid_df.index.name = "cell_type"

    distance_df.to_csv(output_dir / "per_cell_type_distance.csv", index=False, float_format="%.6f")
    summary_df.to_csv(output_dir / "per_cell_type_summary.csv", index=False, float_format="%.6f")
    centroid_df.to_csv(output_dir / "per_cell_type_mGEP_log.csv", float_format="%.6f")

    _save_config_used(
        output_dir=output_dir,
        dataset_name=dataset_name,
        sct_dataset_file_path=sct_dataset_file_path,
        gene_mode=gene_mode,
        n_top_genes=n_top_genes,
        gene_list_path=gene_list_file,
        distance_metric=distance_metric,
        min_samples_per_cell_type=min_samples_per_cell_type,
    )
    _plot_distance_boxplot(distance_df=distance_df, figure_dir=figure_dir)
    _plot_per_cell_type_distributions(distance_df=distance_df, figure_dir=figure_dir)

    logger.info("Sample diversity analysis outputs saved under: %s", output_dir)
    return output_dir
