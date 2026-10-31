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
SUPPORTED_GEP_FILTERING_METHODS = ("median_gep", "mean_gep", "mgep", "linear_mmd")
REMOVED_GEP_FILTERING_METHODS = ("knn", "knn_residual")


def validate_gep_filtering_method(method):
    supported = ", ".join(SUPPORTED_GEP_FILTERING_METHODS)
    if method in REMOVED_GEP_FILTERING_METHODS:
        raise ValueError(
            f'GEP filtering method "{method}" has been removed. '
            f"Supported methods are: {supported}."
        )
    if method not in SUPPORTED_GEP_FILTERING_METHODS:
        raise ValueError(
            f'Unsupported GEP filtering method "{method}". '
            f"Supported methods are: {supported}."
        )


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


def reference_variance_mask(reference, threshold):
    """Select on linear-reference population variance before gene subsetting."""
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold) or threshold < 0):
        raise ValueError("gene_filtering_reference_variance_threshold must be finite and >= 0.")
    values = reference.to_numpy(dtype=np.float64)
    if len(reference) < 2:
        raise ValueError("Reference variance selection requires at least two selected reference samples.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Variance reference contains negative or non-finite linear expression.")
    variance = reference.astype(np.float64).var(axis=0, ddof=0)
    if not np.isfinite(variance.to_numpy()).all():
        raise ValueError("Reference variance contains non-finite values.")
    return variance, variance > threshold


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
    supported_methods = SUPPORTED_GEP_FILTERING_METHODS

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
        if self.gep.filtering_method == "mgep":
            self.identity["distance_space"] = "linear-tpm-selected-G-v1"
            self.identity["distance_norm"] = 1
        self.model_path = self.result_dir / "gep_filtering_pca.joblib"
        self.model_metadata_path = self.model_path.with_suffix(".metadata.json")
        self.additional_artifacts = []
        self.pca = None
        self.reference_coordinates = None
        self.range_bounds = None
        self.range_input = self.range_kept = 0
        self.distance_input = self.distance_kept = 0
        self.mgep_center = None
        self.mgep_cutoff = None
        self.distance_center = None
        self.reference_distances = None
        self.mgep_distance_input = self.mgep_distance_kept = 0
        if self.gep.enabled:
            if self.gep.filtering_method not in self.supported_methods:
                raise ValueError(
                    "Correlation-first GEP filtering supports median_gep, mgep, "
                    "mean_gep, and linear_mmd; migrate marker_ratio filtering to "
                    "a reference-distance method."
                )
            if self.gep.filtering_method == "mgep" and self.gep.filtering_in_pca_space:
                raise ValueError(
                    'GEP filtering method "mgep" requires '
                    '"filtering_in_pca_space: false".'
                )
            q_low, q_high = self.gep.gep_filtering_quantile
            if self.gep.filtering_method == "mgep" and q_low not in (None, 0, 0.0):
                raise ValueError(
                    'GEP filtering method "mgep" uses only the upper quantile; '
                    'set the lower quantile to null or 0.'
                )
            if (
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
            if self.gep.filtering_method == "mgep" and self.reference_linear.empty:
                raise ValueError('"mgep" requires at least one reference profile.')

    def persist_genes(self):
        self.result_dir.mkdir(parents=True, exist_ok=True)
        if self.gene_list_file.exists():
            if pd.read_csv(self.gene_list_file)["gene_name"].tolist() != self.genes:
                raise ValueError(
                    "Stored fixed gene list differs from G. Use a new dataset name/output location."
                )
        else:
            pd.DataFrame({"gene_name": self.genes}).to_csv(self.gene_list_file, index=False)
        variance_table = getattr(self, "reference_variance_table", None)
        if variance_table is not None:
            variance_table.to_csv(
                self.result_dir / "reference_gene_variance.csv", float_format="%.17g")
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

    def _prepare_mgep(self):
        if self.mgep_center is not None:
            return
        reference = self.reference_linear.to_numpy(dtype=np.float64)
        self.mgep_center = np.median(reference, axis=0)
        reference_distances = np.empty(len(reference), dtype=np.float64)
        for start in range(0, len(reference), 256):
            stop = min(start + 256, len(reference))
            reference_distances[start:stop] = np.abs(
                reference[start:stop] - self.mgep_center
            ).sum(axis=1)
        self.reference_distances = reference_distances
        upper_quantile = self.gep.gep_filtering_quantile[1]
        self.mgep_cutoff = float(np.quantile(reference_distances, upper_quantile))
        logger.info(
            "Linear-TPM mGEP L1 cutoff at q=%.3g: %.3g",
            upper_quantile,
            self.mgep_cutoff,
        )

    def prepare_distance_reference(self):
        """Prepare the same reference distances and center used for filtering."""
        if self.reference_distances is not None:
            return self.reference_distances
        if self.gep.filtering_method == "mgep":
            self._prepare_mgep()
            return self.reference_distances
        if self.gep.filtering_in_pca_space:
            self._prepare_pca()
            reference = self.reference_coordinates
        else:
            reference = self.reference_log.to_numpy()
        self.distance_center = (
            np.median(reference, axis=0) if self.gep.filtering_method == "median_gep"
            else np.mean(reference, axis=0)
        )
        norm = 2 if self.gep.filtering_method == "linear_mmd" else self.gep.norm_ord
        self.reference_distances = np.linalg.norm(
            reference - self.distance_center, ord=norm, axis=1
        )
        if self.gep.filtering_method == "linear_mmd":
            self.reference_distances = self.reference_distances ** 2
        return self.reference_distances

    def distance_from_normalized(self, normalized):
        """Calculate profile distances from selected-gene linear TPM."""
        self.prepare_distance_reference()
        if self.gep.filtering_method == "mgep":
            return np.linalg.norm(
                normalized.to_numpy(dtype=np.float64) - self.mgep_center, ord=1, axis=1
            )
        if self.gep.filtering_in_pca_space:
            simulated = safe_pca_transform(
                self.pca, np.log2(normalized + 1), label="Simulated log2(TPM + 1)"
            )
        else:
            simulated = np.log2(normalized + 1).to_numpy()
        norm = 2 if self.gep.filtering_method == "linear_mmd" else self.gep.norm_ord
        distances = np.linalg.norm(simulated - self.distance_center, ord=norm, axis=1)
        return distances ** 2 if self.gep.filtering_method == "linear_mmd" else distances

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
        if self.gep.filtering_method == "mgep":
            simulated_distances = self.distance_from_normalized(normalized)
            mask = simulated_distances <= self.mgep_cutoff
            self.mgep_distance_input += len(normalized)
            self.mgep_distance_kept += int(mask.sum())
            logger.info(
                "Linear-TPM mGEP L1 acceptance: %d/%d profiles.",
                mask.sum(),
                len(mask),
            )
            return normalized.loc[mask].copy()
        ref_distance = self.prepare_distance_reference()
        sim_distance = self.distance_from_normalized(normalized)
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
        stats = {
            "gene_range_input_n_samples": self.range_input,
            "gene_range_kept_n_samples": self.range_kept,
            "pc_distance_input_n_samples": self.distance_input,
            "pc_distance_kept_n_samples": self.distance_kept,
        }
        if self.gep.filtering_method == "mgep":
            stats.update(
                mgep_distance_input_n_samples=self.mgep_distance_input,
                mgep_distance_kept_n_samples=self.mgep_distance_kept,
            )
        return stats

    def restore_stats(self, stats):
        self.range_input = stats.get("gene_range_input_n_samples", 0)
        self.range_kept = stats.get("gene_range_kept_n_samples", 0)
        self.distance_input = stats.get("pc_distance_input_n_samples", 0)
        self.distance_kept = stats.get("pc_distance_kept_n_samples", 0)
        self.mgep_distance_input = stats.get("mgep_distance_input_n_samples", 0)
        self.mgep_distance_kept = stats.get("mgep_distance_kept_n_samples", 0)
