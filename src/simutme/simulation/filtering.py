"""Fixed-gene filtering space and versioned artifact compatibility."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.decomposition import PCA

from ..utility import safe_pca_transform

logger = logging.getLogger(__name__)
WORKFLOW_VERSION = "correlation-first-v1"
NORMALIZATION_VERSION = "subset-row-sum-1e6-log2p1-v1"


def identity_hash(identity):
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def validate_gene_labels(genes, label):
    index = pd.Index(genes)
    if index.has_duplicates or index.isna().any():
        raise ValueError(f"{label} contains duplicate or missing gene labels.")
    if not len(index):
        raise ValueError(f"{label} is empty.")


def normalize_selected(expression, genes, label="Expression"):
    """Subset before normalization; never impute selected genes."""
    validate_gene_labels(genes, "Fixed gene list G")
    validate_gene_labels(expression.columns, f"{label} gene list")
    missing = pd.Index(genes).difference(expression.columns)
    if len(missing):
        raise ValueError(f"{label} is missing selected genes: {missing.tolist()}.")
    values = expression.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{label} contains negative or non-finite expression.")
    selected = expression.loc[:, genes].astype(np.float64)
    totals = selected.sum(axis=1)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError(f"{label} has nonpositive or non-finite selected-gene row sums.")
    return selected.div(totals, axis=0) * 1e6


def select_fixed_genes(sct_genes, reference_genes, cell_types, supplied_genes=None,
                       correlations=None, corr_threshold=0.3, n_gene_max=1000):
    validate_gene_labels(sct_genes, "Target SCT gene list")
    validate_gene_labels(reference_genes, "Reference gene list")
    shared = set(sct_genes).intersection(reference_genes)
    if supplied_genes is not None:
        validate_gene_labels(supplied_genes, "Supplied gene list")
        genes = sorted(shared.intersection(supplied_genes))
    elif correlations is not None:
        validate_gene_labels(correlations.index, "Correlation gene list")
        if correlations.columns.has_duplicates:
            raise ValueError("Correlation matrix contains duplicate cell-type labels.")
        missing = set(cell_types).difference(correlations.columns)
        if missing:
            raise ValueError(f"Correlation reference is missing target cell types: {sorted(missing)}.")
        if not np.isfinite(corr_threshold) or not -1 <= corr_threshold <= 1:
            raise ValueError("gene_filtering_corr_threshold must be finite and in [-1, 1].")
        if n_gene_max <= 0:
            raise ValueError("gene_filtering_n_gene_max must be positive.")
        eligible = correlations.loc[sorted(shared.intersection(correlations.index)), cell_types]
        selected = set()
        for cell_type in cell_types:
            scores = eligible[cell_type].astype(float)
            scores = scores.loc[np.isfinite(scores) & (scores >= corr_threshold)]
            # The pre-sorted gene index breaks correlation ties deterministically.
            selected.update(scores.sort_values(ascending=False, kind="stable").head(n_gene_max).index)
        genes = sorted(selected)
    else:
        genes = sorted(shared)
    validate_gene_labels(genes, "Fixed gene list G")
    return genes, len(shared)


def validate_metadata(path, expected):
    path = Path(path)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Missing or unreadable filtering metadata: {path}. "
            "Use a new dataset name/output location; old artifacts are not overwritten."
        ) from exc
    if metadata.get("identity") != expected:
        raise ValueError(
            f"Incompatible filtering metadata: {path}. "
            "Use a new dataset name/output location; old artifacts are not overwritten."
        )
    return metadata


def write_metadata(path, identity, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"identity": identity, **metadata}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


class FixedGeneFilter:
    """Reference-only filter reused across generation batches."""

    workflow_version = WORKFLOW_VERSION
    supported_methods = ("median_gep", "mean_gep", "linear_mmd")

    def __init__(self, genes, reference_linear, reference_file, reference_subset,
                 gep_params, result_dir, generation_n_genes, shared_n_genes,
                 correlation_file=None):
        self.genes = list(genes)
        self.reference_linear = normalize_selected(reference_linear, self.genes, "Reference")
        self.reference_log = np.log2(self.reference_linear + 1)
        self.gep = gep_params
        self.result_dir = Path(result_dir)
        self.gene_list_file = self.result_dir / "fixed_filtering_gene_list.csv"
        self.correlation_file = str(correlation_file) if correlation_file else None
        self.generation_n_genes = int(generation_n_genes)
        self.shared_n_genes = int(shared_n_genes)
        self.gene_hash = identity_hash(self.genes)
        reference_hash = hashlib.sha256(
            pd.util.hash_pandas_object(self.reference_log, index=True).values.tobytes()
        ).hexdigest()
        self.pca_identity = {
            "workflow_version": self.workflow_version,
            "gene_list_hash": self.gene_hash,
            "normalization_version": NORMALIZATION_VERSION,
            "reference_file": str(Path(reference_file).resolve()),
            "reference_hash": reference_hash,
            "reference_subset": reference_subset,
            "pca_target": self.gep.pca_n_components,
        }
        self.identity = {
            **self.pca_identity,
            "output_definition": "fixed-G-log2-TPM-plus-1",
            "gep_filtering": self.gep.enabled,
            "range_enabled": self.gep.filtering_by_gene_range,
            "range_quantiles": self.gep.gene_quantile_range,
            "range_min_fraction": self.gep.min_percentage_within_gene_range,
            "pc_distance_enabled": self.gep.filtering_in_pca_space,
            "distance_method": self.gep.filtering_method,
            "distance_quantiles": list(self.gep.gep_filtering_quantile),
            "distance_norm": self.gep.norm_ord,
        }
        self.model_path = self.result_dir / "gep_filtering_pca.joblib"
        self.model_metadata_path = self.model_path.with_suffix(".metadata.json")
        self.additional_artifacts = []
        self.pca = None
        self.reference_coordinates = None
        self.range_bounds = None
        self.range_input = self.range_kept = 0
        self.distance_input = self.distance_kept = 0
        if self.gep.enabled:
            if self.gep.filtering_method not in self.supported_methods:
                raise ValueError(
                    "Correlation-first GEP filtering supports median_gep, mean_gep, "
                    "and linear_mmd; migrate marker_ratio filtering to a reference-distance method."
                )
            q_low, q_high = self.gep.gep_filtering_quantile
            if self.gep.filtering_method != "knn_residual" and (
                q_high is None or not 0 <= q_high <= 1
                or (q_low is not None and not 0 <= q_low <= q_high)
            ):
                raise ValueError("gep_filtering_quantile must satisfy 0 <= lower <= upper <= 1.")
            if not 0 <= self.gep.min_percentage_within_gene_range <= 1:
                raise ValueError("gep_min_percentage_within_gene_range must be in [0, 1].")
            if self.gep.filtering_by_gene_range:
                quantiles = self.gep.gene_quantile_range
                if quantiles is None or len(quantiles) != 3 or not (
                    0 <= quantiles[0] <= quantiles[1] <= quantiles[2] <= 1
                ):
                    raise ValueError("gep_gene_quantile_range must be an ordered quantile triplet.")

    def persist_genes(self):
        self.result_dir.mkdir(parents=True, exist_ok=True)
        if self.gene_list_file.exists():
            if pd.read_csv(self.gene_list_file)["gene_name"].tolist() != self.genes:
                raise ValueError(
                    "Stored fixed gene list differs from G. Use a new dataset name/output location."
                )
        else:
            pd.DataFrame({"gene_name": self.genes}).to_csv(self.gene_list_file, index=False)
        logger.info("Filtering genes: %d selected, %d shared, %d before selection.",
                    len(self.genes), self.shared_n_genes, self.generation_n_genes)

    def validate_model(self):
        if self.model_path.exists() or self.model_metadata_path.exists():
            validate_metadata(self.model_metadata_path, self.pca_identity)
            if not self.model_path.exists():
                raise ValueError("PCA model is missing. Use a new dataset name/output location.")
            self.pca = load(self.model_path)
            if (getattr(self.pca, "filtering_genes_", None) != self.genes
                    or self.pca.n_features_in_ != len(self.genes)):
                raise ValueError("Stored PCA gene order differs from G. Use a new dataset name/output location.")

    def _prepare_pca(self):
        if self.reference_coordinates is not None:
            return
        if self.pca is None:
            self.validate_model()
        if self.pca is None:
            self.pca = PCA(n_components=self.gep.pca_n_components, svd_solver="full", random_state=42)
            self.pca.fit(self.reference_log.to_numpy(dtype=np.float64))
            self.pca.filtering_genes_ = self.genes
            self.result_dir.mkdir(parents=True, exist_ok=True)
            dump(self.pca, self.model_path)
            write_metadata(self.model_metadata_path, self.pca_identity)
        self.reference_coordinates = safe_pca_transform(
            self.pca, self.reference_log, label="Reference log2(TPM + 1)"
        )
        logger.info("GEP-filtering PCA: target=%s, fitted=%d PCs.",
                    self.gep.pca_n_components, self.pca.n_components_)

    def filter(self, simulated_linear):
        normalized = normalize_selected(simulated_linear, self.genes, "Simulated profiles")
        if normalized.empty or not self.gep.enabled:
            return normalized
        if self.gep.filtering_by_gene_range:
            if self.range_bounds is None:
                lower, _, upper = self.gep.gene_quantile_range
                self.range_bounds = self.reference_linear.quantile([lower, upper])
            mask = (
                normalized.ge(self.range_bounds.iloc[0], axis=1)
                & normalized.le(self.range_bounds.iloc[1], axis=1)
            ).mean(axis=1) >= self.gep.min_percentage_within_gene_range
            self.range_input += len(normalized)
            self.range_kept += int(mask.sum())
            logger.info("Gene-range acceptance: %d/%d profiles on %d genes.",
                        mask.sum(), len(mask), len(self.genes))
            normalized = normalized.loc[mask]
        if normalized.empty:
            return normalized
        if self.gep.filtering_in_pca_space:
            self._prepare_pca()
            reference = self.reference_coordinates
            simulated = safe_pca_transform(
                self.pca, np.log2(normalized + 1), label="Simulated log2(TPM + 1)"
            )
        else:
            reference = self.reference_log.to_numpy()
            simulated = np.log2(normalized + 1).to_numpy()
        center = (np.median(reference, axis=0) if self.gep.filtering_method == "median_gep"
                  else np.mean(reference, axis=0))
        norm = 2 if self.gep.filtering_method == "linear_mmd" else self.gep.norm_ord
        ref_distance = np.linalg.norm(reference - center, ord=norm, axis=1)
        sim_distance = np.linalg.norm(simulated - center, ord=norm, axis=1)
        if self.gep.filtering_method == "linear_mmd":
            ref_distance = ref_distance ** 2
            sim_distance = sim_distance ** 2
        lower, upper = self.gep.gep_filtering_quantile
        mask = sim_distance <= np.quantile(ref_distance, upper)
        if lower is not None:
            mask &= sim_distance >= np.quantile(ref_distance, lower)
        self.distance_input += len(normalized)
        self.distance_kept += int(mask.sum())
        logger.info("%s-distance acceptance: %d/%d profiles.",
                    "PC" if self.gep.filtering_in_pca_space else "Log-expression",
                    mask.sum(), len(mask))
        return normalized.loc[mask].copy()

    def stats(self):
        return {
            "gene_range_input_n_samples": self.range_input,
            "gene_range_kept_n_samples": self.range_kept,
            "pc_distance_input_n_samples": self.distance_input,
            "pc_distance_kept_n_samples": self.distance_kept,
        }

    def restore_stats(self, stats):
        self.range_input = stats.get("gene_range_input_n_samples", 0)
        self.range_kept = stats.get("gene_range_kept_n_samples", 0)
        self.distance_input = stats.get("pc_distance_input_n_samples", 0)
        self.distance_kept = stats.get("pc_distance_kept_n_samples", 0)
