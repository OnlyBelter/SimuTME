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


def _default_reference_gene_variance_path(merged_sc_dataset_file_path: Path) -> Path:
    return merged_sc_dataset_file_path.with_name(f"{merged_sc_dataset_file_path.stem}.gene_variance.csv")


def _compute_gene_variance_table(
    merged_sc_dataset_file_path: Path,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    import scipy.sparse as sp

    adata = ad.read_h5ad(merged_sc_dataset_file_path, backed="r")
    n_obs = int(adata.n_obs)
    n_vars = int(adata.n_vars)
    if n_obs <= 1:
        raise ValueError(f"Reference scRNA-seq dataset has too few cells: {merged_sc_dataset_file_path}")

    col_sum = np.zeros(n_vars, dtype=np.float64)
    col_sumsq = np.zeros(n_vars, dtype=np.float64)

    x = adata.X
    for start in range(0, n_obs, int(chunk_size)):
        end = min(n_obs, start + int(chunk_size))
        x_chunk = x[start:end, :]
        if sp.issparse(x_chunk):
            col_sum += np.asarray(x_chunk.sum(axis=0)).ravel()
            col_sumsq += np.asarray(x_chunk.multiply(x_chunk).sum(axis=0)).ravel()
        else:
            x_arr = np.asarray(x_chunk, dtype=np.float64)
            col_sum += x_arr.sum(axis=0)
            col_sumsq += np.square(x_arr).sum(axis=0)

    mean = col_sum / float(n_obs)
    var = (col_sumsq / float(n_obs)) - np.square(mean)
    var = np.clip(var, a_min=0.0, a_max=None)

    gene_names = pd.Index(adata.var_names.astype(str))
    var_series = pd.Series(var.astype(float), index=gene_names, name="variance")
    var_series = var_series.sort_values(ascending=False)

    out_df = var_series.reset_index()
    out_df.columns = ["gene_name", "variance"]
    out_df["rank"] = np.arange(1, out_df.shape[0] + 1, dtype=int)
    return out_df


def _load_or_compute_reference_gene_variance(
    merged_sc_dataset_file_path: Path,
    cache_file_path: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    cache_path = _default_reference_gene_variance_path(merged_sc_dataset_file_path) if cache_file_path is None else cache_file_path
    if cache_path.exists():
        gene_var_df = pd.read_csv(cache_path)
        if gene_var_df.empty or "gene_name" not in gene_var_df.columns:
            raise ValueError(f"Invalid cached gene-variance table: {cache_path}")
        gene_var_df["gene_name"] = gene_var_df["gene_name"].astype(str)
        if "variance" in gene_var_df.columns:
            gene_var_df["variance"] = pd.to_numeric(gene_var_df["variance"], errors="coerce").fillna(0.0)
            gene_var_df = gene_var_df.sort_values("variance", ascending=False)
        if "rank" not in gene_var_df.columns:
            gene_var_df = gene_var_df.reset_index(drop=True)
            gene_var_df["rank"] = np.arange(1, gene_var_df.shape[0] + 1, dtype=int)
        return gene_var_df.reset_index(drop=True), cache_path

    gene_var_df = _compute_gene_variance_table(merged_sc_dataset_file_path=merged_sc_dataset_file_path)
    gene_var_df.to_csv(cache_path, index=False, float_format="%.8f")
    return gene_var_df, cache_path


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


def _get_distance_axis_label(distance_metric: str) -> str:
    if distance_metric == "cosine":
        return "1-cosine distance to mGEP_log"
    if distance_metric == "l2":
        return "l2 distance to mGEP_log"
    return f"{distance_metric} distance to mGEP_log"


def _save_config_used(
    output_dir: Path,
    dataset_name: str,
    sct_dataset_file_path: str | Path,
    gene_mode: str,
    n_top_genes: int,
    gene_list_path: Path | None,
    distance_metric: str,
    min_samples_per_cell_type: int,
    merged_sc_dataset_file_path: Path | None = None,
    reference_gene_variance_table_path: Path | None = None,
) -> None:
    config_used = {
        "dataset_name": dataset_name,
        "sct_dataset_file_path": str(sct_dataset_file_path),
        "sample_diversity_gene_mode": gene_mode,
        "sample_diversity_n_top_genes": int(n_top_genes),
        "sample_diversity_gene_list_path": None if gene_list_path is None else str(gene_list_path),
        "sample_diversity_distance_metric": distance_metric,
        "sample_diversity_min_samples_per_cell_type": int(min_samples_per_cell_type),
        "merged_sc_dataset_file_path": None if merged_sc_dataset_file_path is None else str(merged_sc_dataset_file_path),
        "reference_gene_variance_table_path": None
        if reference_gene_variance_table_path is None
        else str(reference_gene_variance_table_path),
    }
    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_used, fh, sort_keys=False, allow_unicode=False)


def _plot_distance_boxplot(distance_df: pd.DataFrame, figure_dir: Path, distance_metric: str) -> None:
    if distance_df.empty:
        return
    plt.figure(figsize=(max(6, distance_df["cell_type"].nunique() * 0.6), 5))
    ax = sns.boxplot(data=distance_df, x="cell_type", y="distance", color="#8fb3d9")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Cell type")
    plt.ylabel(_get_distance_axis_label(distance_metric))
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
    merged_sc_dataset_file_path: str | Path | None = None,
    distance_metric: str = "cosine",
    min_samples_per_cell_type: int = 5,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    gene_list_file = None if gene_list_path in (None, "") else Path(gene_list_path)
    provided_gene_list = None if gene_list_file is None else _read_gene_list(gene_list_file)

    reference_gene_variance_table_path = None
    shared_hvg_genes: list[str] | None = None
    merged_sc_path = None if merged_sc_dataset_file_path in (None, "") else Path(merged_sc_dataset_file_path)
    if gene_mode == "hvg":
        if merged_sc_path is None:
            raise ValueError('"hvg" gene mode requires "merged_sc_dataset_file_path".')
        if not merged_sc_path.exists():
            raise FileNotFoundError(f"merged_sc_dataset_file_path does not exist: {merged_sc_path}")
        gene_var_df, reference_gene_variance_table_path = _load_or_compute_reference_gene_variance(
            merged_sc_dataset_file_path=merged_sc_path
        )
        shared_hvg_genes = (
            gene_var_df.loc[:, "gene_name"].astype(str).drop_duplicates().head(int(n_top_genes)).tolist()
        )
        if not shared_hvg_genes:
            raise ValueError(f"No HVGs were derived from reference dataset: {merged_sc_path}")

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

        if gene_mode == "hvg":
            selected_genes = [gene for gene in (shared_hvg_genes or []) if gene in cell_type_df.columns]
            if not selected_genes:
                logger.warning(
                    "Skipping sample diversity analysis for cell type %s in dataset %s: no HVGs are present.",
                    cell_type,
                    dataset_name,
                )
                continue
        else:
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
        merged_sc_dataset_file_path=merged_sc_path,
        reference_gene_variance_table_path=reference_gene_variance_table_path,
    )
    _plot_distance_boxplot(distance_df=distance_df, figure_dir=figure_dir, distance_metric=distance_metric)
    _plot_per_cell_type_distributions(distance_df=distance_df, figure_dir=figure_dir)

    logger.info("Sample diversity analysis outputs saved under: %s", output_dir)
    return output_dir
