import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _detect_delimiter(file_path: Path) -> str:
    sample = file_path.read_text(encoding="utf-8", errors="replace")[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        return dialect.delimiter
    except csv.Error:
        comma_count = sample.count(",")
        tab_count = sample.count("\t")
        if comma_count == 0 and tab_count == 0:
            raise ValueError(f"Could not detect a delimiter in {file_path}.")
        return "," if comma_count >= tab_count else "\t"


def _read_cell_type_mapping(mapping_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(mapping_csv, dtype={"cell_type_SCT_N6K_13Jul": str, "cell_type_Kassandra_SortedBulk": str})
    required = {"cell_type_SCT_N6K_13Jul", "cell_type_Kassandra_SortedBulk", "included"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {mapping_csv}: {sorted(missing)}")
    df["cell_type_SCT_N6K_13Jul"] = df["cell_type_SCT_N6K_13Jul"].astype(str).str.strip()
    df["cell_type_Kassandra_SortedBulk"] = df["cell_type_Kassandra_SortedBulk"].astype(str).str.strip()
    df["included"] = df["included"].astype(int)
    return df


def _infer_sct_cell_types(obs: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.Series:
    valid_types = set(mapping_df["cell_type_SCT_N6K_13Jul"].astype(str).tolist())
    cols = [c for c in obs.columns.astype(str).tolist() if c in valid_types]
    if not cols:
        raise ValueError("No one-hot cell-type columns found in SCT adata.obs that match cell_type_mapping.csv.")
    scores = obs.loc[:, cols]
    return scores.idxmax(axis=1).astype(str)


def _log2cpm1p_to_tpm(x: np.ndarray) -> np.ndarray:
    return np.exp2(x) - 1.0


def _tpm_to_log2cpm1p(tpm: np.ndarray) -> np.ndarray:
    return np.log2(tpm + 1.0)


def _row_normalize_to_1e6(x: np.ndarray) -> np.ndarray:
    lib = x.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return (x / lib) * 1e6


@dataclass(frozen=True)
class BulkReference:
    sample_ids: List[str]
    tpm: np.ndarray


def _load_bulk_reference_as_tpm(expr_csv: Path, target_genes: List[str]) -> BulkReference:
    delimiter = _detect_delimiter(expr_csv)
    logging.info("Detected Kassandra expression delimiter: %r", delimiter)

    gene_to_idx = {gene: i for i, gene in enumerate(target_genes)}
    n_genes = len(target_genes)
    present = 0

    with expr_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError(f"Expression file {expr_csv} has no usable header.")
        sample_ids = [str(x).strip() for x in header[1:]]
        n_samples = len(sample_ids)
        values = np.zeros((n_samples, n_genes), dtype=np.float32)

        for row in reader:
            if not row:
                continue
            gene = str(row[0]).strip()
            col = gene_to_idx.get(gene)
            if col is None:
                continue
            if len(row) != n_samples + 1:
                raise ValueError(f"Row for gene {gene!r} has {len(row)} columns, expected {n_samples + 1}.")
            try:
                values[:, col] = np.asarray(row[1:], dtype=np.float32)
            except ValueError as exc:
                raise ValueError(f"Non-numeric values encountered in gene row {gene!r}.") from exc
            present += 1

    logging.info("Loaded %d/%d target genes from Kassandra bulk file.", present, n_genes)
    return BulkReference(sample_ids=sample_ids, tpm=_row_normalize_to_1e6(values))


def _read_kassandra_annotation(annot_tsv: Path, cell_type_col: str) -> pd.Series:
    annot = pd.read_csv(annot_tsv, sep=_detect_delimiter(annot_tsv), index_col=0, dtype=str)
    annot.index = annot.index.astype(str).str.strip()
    annot.columns = annot.columns.astype(str).str.strip()
    if cell_type_col not in annot.columns:
        raise ValueError(f"Column {cell_type_col!r} not found in {annot_tsv}.")
    return annot[cell_type_col].fillna("").astype(str).str.strip()


def _compute_zero_ratio_matrix_tpm(tpm: np.ndarray, eps: float) -> np.ndarray:
    return np.mean(tpm <= eps, axis=1).astype(np.float32)


def _compute_zero_ratio_from_log2cpm1p_in_batches(x: np.ndarray, eps: float, batch_size: int) -> np.ndarray:
    zero_ratio = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], batch_size):
        end = min(x.shape[0], start + batch_size)
        tpm_batch = _row_normalize_to_1e6(_log2cpm1p_to_tpm(x[start:end, :].astype(np.float32, copy=False)))
        zero_ratio[start:end] = _compute_zero_ratio_matrix_tpm(tpm_batch, eps=eps)
    return zero_ratio


def _boxplot_whiskers(values: pd.Series, value_col_prefix: str) -> pd.Series:
    clean = values.dropna().astype(float).sort_values()
    if clean.empty:
        return pd.Series(
            {
                f"bottom_whisker_{value_col_prefix}": np.nan,
                f"top_whisker_{value_col_prefix}": np.nan,
            }
        )
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    inlier = clean[(clean >= lower_fence) & (clean <= upper_fence)]
    if inlier.empty:
        return pd.Series(
            {
                f"bottom_whisker_{value_col_prefix}": np.nan,
                f"top_whisker_{value_col_prefix}": np.nan,
            }
        )
    return pd.Series(
        {
            f"bottom_whisker_{value_col_prefix}": float(inlier.min()),
            f"top_whisker_{value_col_prefix}": float(inlier.max()),
        }
    )


def _summarize_zero_ratio(per_sample: pd.DataFrame, cell_type_order: List[str]) -> pd.DataFrame:
    summary = (
        per_sample.groupby("cell_type", as_index=False, observed=False)
        .agg(
            n_repeats=("sample_id", "count"),
            n_cells_total=("sample_id", lambda s: np.nan),
            n_cells_target=("sample_id", lambda s: np.nan),
            mean_zero_ratio=("zero_ratio", "mean"),
            median_zero_ratio=("zero_ratio", "median"),
            q1_zero_ratio=("zero_ratio", lambda s: s.quantile(0.25)),
            q3_zero_ratio=("zero_ratio", lambda s: s.quantile(0.75)),
            sd_zero_ratio=("zero_ratio", "std"),
            min_zero_ratio=("zero_ratio", "min"),
            max_zero_ratio=("zero_ratio", "max"),
        )
        .copy()
    )
    whisker_df = per_sample.groupby("cell_type", observed=False)["zero_ratio"].apply(
        lambda s: _boxplot_whiskers(s, "zero_ratio")
    ).reset_index()
    whisker_df = whisker_df.pivot(index="cell_type", columns="level_1", values="zero_ratio").reset_index()
    whisker_df.columns.name = None
    summary = summary.merge(whisker_df, on="cell_type", how="left")
    summary["iqr_zero_ratio"] = summary["q3_zero_ratio"] - summary["q1_zero_ratio"]
    summary["whisker_range_zero_ratio"] = summary["top_whisker_zero_ratio"] - summary["bottom_whisker_zero_ratio"]
    order_rank = {ct: i for i, ct in enumerate(cell_type_order)}
    summary["order_rank"] = summary["cell_type"].map(order_rank)
    summary = summary.sort_values(["order_rank", "cell_type"]).drop(columns=["order_rank"])
    return summary


def _plot_boxplot(
    per_sample: pd.DataFrame,
    cell_type_order: List[str],
    out_png: Path,
    seed: int,
    value_col: str,
    y_label: str,
) -> None:
    present_order = [ct for ct in cell_type_order if ct in set(per_sample["cell_type"])]
    plt.figure(figsize=(max(8.0, 0.55 * len(present_order) + 2.0), 4.0))
    ax = plt.gca()
    data = [
        per_sample.loc[per_sample["cell_type"] == ct, value_col].to_numpy(dtype=float, copy=True)
        for ct in present_order
    ]
    positions = np.arange(len(present_order), dtype=float)
    ax.boxplot(
        data,
        positions=positions,
        widths=0.65,
        showfliers=True,
        flierprops={"markersize": 2.0},
        patch_artist=True,
        boxprops={"facecolor": "#cfe2f3", "edgecolor": "black"},
        medianprops={"color": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
    )
    rng = np.random.default_rng(seed)
    for pos, values in zip(positions, data):
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=values.size)
        ax.scatter(
            np.full(values.shape, pos) + jitter,
            values,
            s=8,
            alpha=0.65,
            color="black",
            linewidths=0,
            zorder=3,
        )
    ax.yaxis.grid(True)
    plt.ylabel(y_label)
    plt.xlabel("")
    plt.xticks(positions, present_order, rotation=35, ha="right")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def _sample_per_sample_for_boxplot(per_sample: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or per_sample.shape[0] <= max_samples:
        return per_sample.copy()

    rng = np.random.default_rng(seed)
    group_sizes = per_sample.groupby("cell_type", observed=False).size()
    present_sizes = group_sizes[group_sizes > 0]
    proportional = (present_sizes / present_sizes.sum()) * max_samples
    alloc = np.floor(proportional).astype(int)
    alloc[alloc == 0] = 1
    alloc = np.minimum(alloc, present_sizes)

    remaining = max_samples - int(alloc.sum())
    if remaining > 0:
        residual = (proportional - np.floor(proportional)).sort_values(ascending=False)
        for cell_type in residual.index:
            if remaining == 0:
                break
            if alloc.loc[cell_type] < present_sizes.loc[cell_type]:
                alloc.loc[cell_type] += 1
                remaining -= 1
    elif remaining < 0:
        residual = (proportional - np.floor(proportional)).sort_values(ascending=True)
        for cell_type in residual.index:
            if remaining == 0:
                break
            if alloc.loc[cell_type] > 1:
                alloc.loc[cell_type] -= 1
                remaining += 1

    sampled_parts = []
    for cell_type, group_df in per_sample.groupby("cell_type", observed=False, sort=False):
        if group_df.empty:
            continue
        n_take = min(int(alloc.get(cell_type, 0)), group_df.shape[0])
        if n_take <= 0:
            continue
        sampled_parts.append(group_df.sample(n=n_take, random_state=int(rng.integers(0, 2**31 - 1))))

    sampled_df = pd.concat(sampled_parts, axis=0, ignore_index=True)
    sampled_df["cell_type"] = pd.Categorical(
        sampled_df["cell_type"],
        categories=per_sample["cell_type"].cat.categories,
        ordered=True,
    )
    return sampled_df.sort_values(["cell_type", "sample_id"]).reset_index(drop=True)


def _save_sct_sparsity_outputs(
    sample_ids: List[str],
    cell_types: pd.Series,
    zero_ratio: np.ndarray,
    cell_type_order: List[str],
    out_dir: Path,
    prefix: str,
    seed: int,
    boxplot_sample_size: int,
) -> None:
    per_sct = pd.DataFrame({"sample_id": sample_ids, "cell_type": cell_types.astype(str).tolist(), "zero_ratio": zero_ratio.astype(float)})
    per_sct["cell_type"] = pd.Categorical(per_sct["cell_type"], categories=cell_type_order, ordered=True)
    per_sct = per_sct.sort_values(["cell_type", "sample_id"]).reset_index(drop=True)
    per_sct.to_csv(out_dir / f"{prefix}_sct_sparsity_per_cell.csv", index=False, float_format="%.6f")
    _summarize_zero_ratio(per_sct, cell_type_order).to_csv(
        out_dir / f"{prefix}_sct_sparsity_summary.csv",
        index=False,
        float_format="%.6f",
    )
    sampled = _sample_per_sample_for_boxplot(per_sct, max_samples=boxplot_sample_size, seed=seed)
    _plot_boxplot(sampled, cell_type_order, out_dir / f"{prefix}_sct_sparsity_boxplot.png", seed=seed, value_col="zero_ratio", y_label="Zero ratio across genes")


def _summarize_added_expression_fraction(per_sample: pd.DataFrame, cell_type_order: List[str]) -> pd.DataFrame:
    summary = (
        per_sample.groupby("cell_type", as_index=False, observed=False)
        .agg(
            n_samples=("sample_id", "count"),
            mean_y_fraction_before_adjustment=("y_fraction_before_adjustment", "mean"),
            median_y_fraction_before_adjustment=("y_fraction_before_adjustment", "median"),
            q1_y_fraction_before_adjustment=("y_fraction_before_adjustment", lambda s: s.quantile(0.25)),
            q3_y_fraction_before_adjustment=("y_fraction_before_adjustment", lambda s: s.quantile(0.75)),
            sd_y_fraction_before_adjustment=("y_fraction_before_adjustment", "std"),
            min_y_fraction_before_adjustment=("y_fraction_before_adjustment", "min"),
            max_y_fraction_before_adjustment=("y_fraction_before_adjustment", "max"),
            mean_y_fraction_after_adjustment=("y_fraction_after_adjustment", "mean"),
            median_y_fraction_after_adjustment=("y_fraction_after_adjustment", "median"),
            q1_y_fraction_after_adjustment=("y_fraction_after_adjustment", lambda s: s.quantile(0.25)),
            q3_y_fraction_after_adjustment=("y_fraction_after_adjustment", lambda s: s.quantile(0.75)),
            sd_y_fraction_after_adjustment=("y_fraction_after_adjustment", "std"),
            min_y_fraction_after_adjustment=("y_fraction_after_adjustment", "min"),
            max_y_fraction_after_adjustment=("y_fraction_after_adjustment", "max"),
        )
        .copy()
    )
    order_rank = {ct: i for i, ct in enumerate(cell_type_order)}
    summary["order_rank"] = summary["cell_type"].map(order_rank)
    return summary.sort_values(["order_rank", "cell_type"]).drop(columns=["order_rank"])


def _save_added_expression_outputs(
    sample_ids: List[str],
    cell_types: pd.Series,
    y_added_expression_before: np.ndarray,
    y_added_expression_after: np.ndarray,
    cell_type_order: List[str],
    out_dir: Path,
    prefix: str,
    seed: int,
    boxplot_sample_size: int,
) -> None:
    per_sample = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "cell_type": cell_types.astype(str).tolist(),
            "y_added_expression_before_adjustment": y_added_expression_before.astype(float),
            "y_added_expression_after_adjustment": y_added_expression_after.astype(float),
        }
    )
    per_sample["y_fraction_before_adjustment"] = per_sample["y_added_expression_before_adjustment"] / 1e6
    per_sample["y_fraction_after_adjustment"] = per_sample["y_added_expression_after_adjustment"] / 1e6
    per_sample["cell_type"] = pd.Categorical(per_sample["cell_type"], categories=cell_type_order, ordered=True)
    per_sample = per_sample.sort_values(["cell_type", "sample_id"]).reset_index(drop=True)
    per_sample.to_csv(out_dir / f"{prefix}_sct_added_expression_y_per_cell.csv", index=False, float_format="%.6f")
    _summarize_added_expression_fraction(per_sample, cell_type_order).to_csv(
        out_dir / f"{prefix}_sct_added_expression_y_summary.csv",
        index=False,
        float_format="%.6f",
    )

    before_boxplot = per_sample.loc[:, ["sample_id", "cell_type", "y_fraction_before_adjustment"]].rename(
        columns={"y_fraction_before_adjustment": "value"}
    )
    before_boxplot = _sample_per_sample_for_boxplot(before_boxplot, max_samples=boxplot_sample_size, seed=seed)
    _plot_boxplot(
        before_boxplot,
        cell_type_order,
        out_dir / f"{prefix}_sct_added_expression_y_fraction_before_adjustment_boxplot.png",
        seed=seed,
        value_col="value",
        y_label="y_fraction_of_total before adjustment",
    )

    after_boxplot = per_sample.loc[:, ["sample_id", "cell_type", "y_fraction_after_adjustment"]].rename(
        columns={"y_fraction_after_adjustment": "value"}
    )
    after_boxplot = _sample_per_sample_for_boxplot(after_boxplot, max_samples=boxplot_sample_size, seed=seed)
    _plot_boxplot(
        after_boxplot,
        cell_type_order,
        out_dir / f"{prefix}_sct_added_expression_y_fraction_after_adjustment_boxplot.png",
        seed=seed,
        value_col="value",
        y_label="y_fraction_of_total after adjustment",
    )


def _build_bulk_cell_type_index(
    bulk_ref: BulkReference,
    sample_to_cell_type: pd.Series,
    allowed_bulk_cell_types: List[str],
) -> Dict[str, np.ndarray]:
    sample_to_idx = {sid: i for i, sid in enumerate(bulk_ref.sample_ids)}
    out: Dict[str, List[int]] = {ct: [] for ct in allowed_bulk_cell_types}
    for sample_id, cell_type in sample_to_cell_type.items():
        if cell_type not in out:
            continue
        idx = sample_to_idx.get(sample_id)
        if idx is None:
            continue
        out[cell_type].append(idx)
    result = {}
    for ct, idxs in out.items():
        if not idxs:
            raise ValueError(f"No Kassandra samples found for bulk cell type {ct!r}.")
        result[ct] = np.asarray(idxs, dtype=int)
    return result


def _assign_nearest_bulk_samples_for_sct_cells(
    x_log2cpm1p: np.ndarray,
    sct_cell_types: pd.Series,
    mapping_df: pd.DataFrame,
    bulk_tpm: np.ndarray,
    bulk_indices: Dict[str, np.ndarray],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    sct_to_bulk = dict(zip(mapping_df["cell_type_SCT_N6K_13Jul"], mapping_df["cell_type_Kassandra_SortedBulk"]))
    sct_to_included = dict(zip(mapping_df["cell_type_SCT_N6K_13Jul"], mapping_df["included"]))
    assigned = np.empty(sct_cell_types.shape[0], dtype=int)
    included_flags = np.empty(sct_cell_types.shape[0], dtype=np.int8)
    sct_type_array = sct_cell_types.astype(str).to_numpy()
    for sct_ct in pd.unique(sct_type_array).tolist():
        rows = np.flatnonzero(sct_type_array == sct_ct)
        if rows.size == 0:
            continue
        bulk_ct = sct_to_bulk.get(sct_ct)
        if bulk_ct is None:
            raise ValueError(f"SCT cell type {sct_ct!r} not found in cell_type_mapping.csv.")
        included = sct_to_included.get(sct_ct)
        if included is None:
            raise ValueError(f"SCT cell type {sct_ct!r} is missing its included flag in cell_type_mapping.csv.")
        choices = bulk_indices.get(bulk_ct)
        if choices is None:
            raise ValueError(f"Bulk cell type {bulk_ct!r} has no sample index list.")
        included_flags[rows] = int(included)
        if choices.size == 1:
            assigned[rows] = int(choices[0])
            continue

        candidate_tpm = bulk_tpm[choices, :].astype(np.float32, copy=False)
        candidate_sq = np.einsum("ij,ij->i", candidate_tpm, candidate_tpm).astype(np.float32, copy=False)
        for start in range(0, rows.size, batch_size):
            batch_rows = rows[start:start + batch_size]
            sct_tpm = _row_normalize_to_1e6(
                _log2cpm1p_to_tpm(x_log2cpm1p[batch_rows, :].astype(np.float32, copy=False))
            ).astype(np.float32, copy=False)
            sct_sq = np.einsum("ij,ij->i", sct_tpm, sct_tpm).astype(np.float32, copy=False)
            dot = sct_tpm @ candidate_tpm.T
            dist_sq = sct_sq[:, None] + candidate_sq[None, :] - (2.0 * dot)
            np.maximum(dist_sq, 0.0, out=dist_sq)
            nearest_choice = np.argmin(dist_sq, axis=1)
            assigned[batch_rows] = choices[nearest_choice]
        logging.info(
            "Matched SCT cell type %r to nearest sorted-bulk donors within bulk cell type %r (%d candidates, %d rows).",
            sct_ct,
            bulk_ct,
            choices.size,
            rows.size,
        )
    return assigned, included_flags


def _impute_batch(
    x_batch_log2cpm1p: np.ndarray,
    bulk_tpm: np.ndarray,
    bulk_idx_batch: np.ndarray,
    included_batch: np.ndarray,
    eps: float,
    bulk_weight: float,
    y_frac_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tpm_batch = _row_normalize_to_1e6(_log2cpm1p_to_tpm(x_batch_log2cpm1p.astype(np.float32, copy=False)))
    mask0 = tpm_batch <= eps
    out_tpm = np.empty_like(tpm_batch, dtype=np.float32)
    added_before_per_row = np.zeros(tpm_batch.shape[0], dtype=np.float32)
    added_after_per_row = np.zeros(tpm_batch.shape[0], dtype=np.float32)
    threshold_total = np.float32(y_frac_threshold * 1e6)
    sct_weight = np.float32(1.0 - bulk_weight)
    bulk_weight = np.float32(bulk_weight)

    for bulk_idx in np.unique(bulk_idx_batch).tolist():
        rows = np.flatnonzero(bulk_idx_batch == bulk_idx)
        if rows.size == 0:
            continue
        bulk_vec = bulk_tpm[bulk_idx].astype(np.float32, copy=False)
        include_rows = rows[included_batch[rows] == 1]
        if include_rows.size > 0:
            averaged = sct_weight * tpm_batch[include_rows, :] + bulk_weight * bulk_vec[None, :]
            out_tpm[include_rows, :] = averaged.astype(np.float32, copy=False)

        zero_guided_rows = rows[included_batch[rows] != 1]
        if zero_guided_rows.size == 0:
            continue

        bulk_nonzero = bulk_vec > eps
        impute_mask = mask0[zero_guided_rows, :] & bulk_nonzero[None, :]
        added_before = (bulk_vec[None, :] * impute_mask).sum(axis=1).astype(np.float32)
        added_before_per_row[zero_guided_rows] = added_before

        cap_scale = np.ones_like(added_before, dtype=np.float32)
        over_cap = added_before > threshold_total
        if np.any(over_cap):
            cap_scale[over_cap] = threshold_total / added_before[over_cap]
        adjusted_bulk = bulk_vec[None, :] * cap_scale[:, None]
        added_after = (adjusted_bulk * impute_mask).sum(axis=1).astype(np.float32)
        added_after_per_row[zero_guided_rows] = added_after

        scale = ((1e6 - added_after) / 1e6).astype(np.float32, copy=False)
        scaled = tpm_batch[zero_guided_rows, :] * scale[:, None]
        merged = np.where(impute_mask, adjusted_bulk, scaled).astype(np.float32, copy=False)

        overflow = added_after >= 1e6
        if np.any(overflow):
            overflow_mask = impute_mask[overflow, :]
            overflow_vals = (adjusted_bulk[overflow, :] * overflow_mask).astype(np.float32, copy=False)
            denom = overflow_vals.sum(axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            merged[overflow, :] = ((overflow_vals / denom) * 1e6).astype(np.float32, copy=False)
            added_after_per_row[zero_guided_rows[overflow]] = 1e6

        out_tpm[zero_guided_rows, :] = merged

    return _tpm_to_log2cpm1p(out_tpm).astype(np.float32, copy=False), added_before_per_row, added_after_per_row


def run_sorted_bulk_guided_sct_imputation(
    input_h5ad: str | Path,
    output_h5ad: str | Path,
    bulk_expr: str | Path,
    bulk_annot: str | Path,
    cell_type_mapping: str | Path,
    out_dir: str | Path,
    output_prefix: str,
    bulk_cell_type_col: str = "Tumor_model_annot",
    eps: float = 1e-8,
    seed: int = 123,
    batch_size: int = 128,
    sct_boxplot_sample_size: int = 1000,
    bulk_weight: float = 0.5,
    y_frac_threshold: float = 0.15,
) -> Path:
    input_h5ad = Path(input_h5ad)
    output_h5ad = Path(output_h5ad)
    bulk_expr = Path(bulk_expr)
    bulk_annot = Path(bulk_annot)
    cell_type_mapping = Path(cell_type_mapping)
    out_dir = Path(out_dir)

    for fp in [input_h5ad, bulk_expr, bulk_annot, cell_type_mapping]:
        if not fp.exists():
            raise FileNotFoundError(f"Required input file not found: {fp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)

    mapping_df = _read_cell_type_mapping(cell_type_mapping)
    sct_cell_type_order = mapping_df["cell_type_SCT_N6K_13Jul"].astype(str).tolist()
    bulk_cell_types = sorted(mapping_df["cell_type_Kassandra_SortedBulk"].astype(str).unique().tolist())

    logging.info("Loading SCT h5ad for sorted-bulk-guided imputation: %s", input_h5ad)
    adata = ad.read_h5ad(input_h5ad)
    if adata.X is None:
        raise ValueError("adata.X is empty.")
    if not isinstance(adata.X, np.ndarray):
        adata.X = np.asarray(adata.X)
    if adata.X.dtype != np.float32:
        adata.X = adata.X.astype(np.float32, copy=False)

    sct_cell_types = _infer_sct_cell_types(adata.obs, mapping_df)
    target_genes = [str(g) for g in adata.var_names.astype(str).tolist()]
    logging.info("SCT cells: %d; genes: %d.", adata.n_obs, adata.n_vars)

    bulk_ref = _load_bulk_reference_as_tpm(bulk_expr, target_genes)
    sample_to_cell_type = _read_kassandra_annotation(bulk_annot, bulk_cell_type_col)
    bulk_indices = _build_bulk_cell_type_index(bulk_ref, sample_to_cell_type, bulk_cell_types)
    assigned_bulk_idx, included_flags = _assign_nearest_bulk_samples_for_sct_cells(
        adata.X,
        sct_cell_types,
        mapping_df,
        bulk_ref.tpm,
        bulk_indices,
        batch_size=batch_size,
    )
    n_average = int(np.sum(included_flags == 1))
    n_zero_guided = int(np.sum(included_flags != 1))
    logging.info(
        "SCT post-imputation branch counts: included==1 averaging for %d cells; included==0 zero-guided replacement for %d cells.",
        n_average,
        n_zero_guided,
    )

    sct_sample_ids = adata.obs_names.astype(str).tolist()
    logging.info("Computing and saving pre-imputation SCT sparsity outputs.")
    pre_zero_ratio = _compute_zero_ratio_from_log2cpm1p_in_batches(adata.X, eps=eps, batch_size=batch_size)
    _save_sct_sparsity_outputs(
        sample_ids=sct_sample_ids,
        cell_types=sct_cell_types,
        zero_ratio=pre_zero_ratio,
        cell_type_order=sct_cell_type_order,
        out_dir=out_dir,
        prefix=f"{output_prefix}_pre_imputation",
        seed=seed,
        boxplot_sample_size=sct_boxplot_sample_size,
    )

    logging.info("Running sorted-bulk-guided SCT imputation in batches (batch_size=%d).", batch_size)
    y_added_expression_before = np.zeros(adata.n_obs, dtype=np.float32)
    y_added_expression_after = np.zeros(adata.n_obs, dtype=np.float32)
    for start in range(0, adata.n_obs, batch_size):
        end = min(adata.n_obs, start + batch_size)
        imputed_batch, y_before_batch, y_after_batch = _impute_batch(
            adata.X[start:end, :],
            bulk_ref.tpm,
            assigned_bulk_idx[start:end],
            included_flags[start:end],
            eps=eps,
            bulk_weight=bulk_weight,
            y_frac_threshold=y_frac_threshold,
        )
        adata.X[start:end, :] = imputed_batch
        y_added_expression_before[start:end] = y_before_batch
        y_added_expression_after[start:end] = y_after_batch
        if (start // batch_size) % 25 == 0:
            logging.info("Processed %d/%d cells for sorted-bulk-guided imputation.", end, adata.n_obs)

    logging.info("Saving imputed SCT h5ad: %s", output_h5ad)
    adata.write_h5ad(output_h5ad, compression="gzip")

    logging.info("Computing and saving post-imputation SCT sparsity outputs.")
    post_zero_ratio = _compute_zero_ratio_from_log2cpm1p_in_batches(adata.X, eps=eps, batch_size=batch_size)
    _save_sct_sparsity_outputs(
        sample_ids=sct_sample_ids,
        cell_types=sct_cell_types,
        zero_ratio=post_zero_ratio,
        cell_type_order=sct_cell_type_order,
        out_dir=out_dir,
        prefix=f"{output_prefix}_post_imputation",
        seed=seed,
        boxplot_sample_size=sct_boxplot_sample_size,
    )

    logging.info("Saving y-fraction diagnostics for the imputed SCT dataset.")
    _save_added_expression_outputs(
        sample_ids=sct_sample_ids,
        cell_types=sct_cell_types,
        y_added_expression_before=y_added_expression_before,
        y_added_expression_after=y_added_expression_after,
        cell_type_order=sct_cell_type_order,
        out_dir=out_dir,
        prefix=output_prefix,
        seed=seed,
        boxplot_sample_size=sct_boxplot_sample_size,
    )

    bulk_zero = _compute_zero_ratio_matrix_tpm(bulk_ref.tpm, eps=eps)
    bulk_cell_type_series = pd.Series([sample_to_cell_type.get(sid, "") for sid in bulk_ref.sample_ids], index=bulk_ref.sample_ids, dtype=str)
    bulk_keep = bulk_cell_type_series.isin(bulk_cell_types)
    per_bulk = pd.DataFrame(
        {
            "sample_id": bulk_cell_type_series.index[bulk_keep].astype(str).tolist(),
            "cell_type": bulk_cell_type_series.loc[bulk_keep].astype(str).tolist(),
            "zero_ratio": bulk_zero[np.flatnonzero(bulk_keep.to_numpy())].astype(float),
        }
    )
    per_bulk["cell_type"] = pd.Categorical(per_bulk["cell_type"], categories=bulk_cell_types, ordered=True)
    per_bulk = per_bulk.sort_values(["cell_type", "sample_id"]).reset_index(drop=True)
    per_bulk.to_csv(out_dir / f"{output_prefix}_kassandra_bulk_sparsity_per_sample.csv", index=False, float_format="%.6f")
    _summarize_zero_ratio(per_bulk, bulk_cell_types).to_csv(
        out_dir / f"{output_prefix}_kassandra_bulk_sparsity_summary.csv",
        index=False,
        float_format="%.6f",
    )
    _plot_boxplot(
        per_bulk,
        bulk_cell_types,
        out_dir / f"{output_prefix}_kassandra_bulk_sparsity_boxplot.png",
        seed=seed,
        value_col="zero_ratio",
        y_label="Zero ratio across genes",
    )

    logging.info("Sorted-bulk-guided SCT imputation finished. Outputs saved under: %s", out_dir)
    return output_h5ad
