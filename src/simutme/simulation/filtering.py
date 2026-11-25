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


def validate_cancer_type_pca_target(value, context):
    """Validate and normalize one PCA target for a cancer-specific override."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context}: PCA target must be a positive integer or a fraction in (0, 1).")
    if isinstance(value, (int, np.integer)) and value > 0:
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and 0 < value < 1:
        return float(value)
    raise ValueError(f"{context}: PCA target must be a positive integer or a fraction in (0, 1).")


def normalize_cancer_type_overrides(value, context):
    """Validate the optional dataset-level cancer-type override mapping."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f'{context}: "gep_filtering_by_cancer_type" must be a mapping.')
    resolved = {}
    allowed = {"n_subgroup", "gep_pca_n_components"}
    for cancer_type, settings in value.items():
        if not isinstance(cancer_type, str) or not cancer_type.strip() or cancer_type != cancer_type.strip():
            raise ValueError(f"{context}: cancer-type override labels must be nonempty exact strings.")
        label = f'{context}, cancer type "{cancer_type}"'
        if not isinstance(settings, dict) or not settings:
            raise ValueError(f"{label}: override must be a nonempty mapping.")
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(f"{label}: unknown override fields: {sorted(map(str, unknown))}.")
        entry = {}
        if "n_subgroup" in settings:
            subgroup = settings["n_subgroup"]
            if (isinstance(subgroup, (bool, np.bool_))
                    or not isinstance(subgroup, (int, np.integer)) or subgroup <= 0):
                raise ValueError(f'{label}: "n_subgroup" must be a positive integer.')
            entry["n_subgroup"] = int(subgroup)
        if "gep_pca_n_components" in settings:
            entry["gep_pca_n_components"] = validate_cancer_type_pca_target(
                settings["gep_pca_n_components"], label
            )
        resolved[cancer_type] = entry
    return {name: resolved[name] for name in sorted(resolved)}


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
                 correlation_file=None, reference_cancer_types=None, dataset_name=None):
        self.genes = list(genes)
        self.reference_file = Path(reference_file)
        self.reference_linear = normalize_selected(reference_linear, self.genes, "Reference")
        self.reference_log = np.log2(self.reference_linear + 1)
        self.reference_log_memmap = None
        self.gep = gep_params
        self.reference_sample_count = len(self.reference_log)
        requested_n_subgroup = getattr(self.gep, "n_subgroup", None)
        context = f'Mixed dataset "{dataset_name or Path(result_dir).name}"'
        overrides = normalize_cancer_type_overrides(
            getattr(self.gep, "gep_filtering_by_cancer_type", None), context
        )
        if overrides and requested_n_subgroup is None:
            raise ValueError(
                f'{context}: "gep_filtering_by_cancer_type" requires '
                'a positive default "n_subgroup".'
            )
        if requested_n_subgroup is not None and (
            isinstance(requested_n_subgroup, bool)
            or not isinstance(requested_n_subgroup, (int, np.integer))
            or requested_n_subgroup <= 0
        ):
            raise ValueError('"n_subgroup" must be a positive integer or null.')
        self.n_subgroup = (
            self.reference_sample_count
            if requested_n_subgroup is None else int(requested_n_subgroup)
        )
        self.reference_cancer_type_groups = {}
        self.reference_cancer_type_hash = None
        if requested_n_subgroup is not None:
            if reference_cancer_types is None:
                raise ValueError(
                    '"n_subgroup" requires cancer-type labels for every reference sample.'
                )
            if isinstance(reference_cancer_types, pd.Series):
                cancer_types = reference_cancer_types.copy()
                if cancer_types.index.has_duplicates:
                    raise ValueError(
                        "Reference cancer-type labels contain duplicate sample IDs."
                    )
                cancer_types = cancer_types.reindex(self.reference_log.index)
            else:
                cancer_types = pd.Series(reference_cancer_types)
                if len(cancer_types) != self.reference_sample_count:
                    raise ValueError(
                        "Reference cancer-type labels must match the number of "
                        "eligible GEP reference samples."
                    )
                cancer_types.index = self.reference_log.index
            if cancer_types.isna().any():
                missing_count = int(cancer_types.isna().sum())
                raise ValueError(
                    "Cancer-type labels are missing for "
                    f"{missing_count} eligible GEP reference samples."
                )
            cancer_type_values = cancer_types.astype(str).to_numpy()
            for cancer_type in sorted(set(cancer_type_values)):
                self.reference_cancer_type_groups[cancer_type] = np.flatnonzero(
                    cancer_type_values == cancer_type
                )
            self.reference_cancer_type_hash = hashlib.sha256(
                pd.util.hash_pandas_object(
                    pd.Series(
                        cancer_type_values,
                        index=self.reference_log.index,
                        name="cancer_type",
                    ),
                    index=True,
                ).values.tobytes()
            ).hexdigest()
        self.gep_filtering_parameters_by_cancer_type = None
        if overrides:
            validate_cancer_type_pca_target(self.gep.pca_n_components, f"{context}, default GEP PCA")
            unknown = sorted(set(overrides) - set(self.reference_cancer_type_groups))
            if unknown:
                raise ValueError(
                    f"{context}: cancer-type overrides are unknown or excluded "
                    f"from the eligible reference: {unknown}."
                )
            self.gep_filtering_parameters_by_cancer_type = {}
            for cancer_type in sorted(self.reference_cancer_type_groups):
                selected = overrides.get(cancer_type, {})
                target = validate_cancer_type_pca_target(
                    selected.get("gep_pca_n_components", self.gep.pca_n_components),
                    f'{context}, cancer type "{cancer_type}"',
                )
                self.gep_filtering_parameters_by_cancer_type[cancer_type] = {
                    "n_subgroup": selected.get("n_subgroup", self.n_subgroup),
                    "gep_pca_n_components": target,
                }
        single_type = (
            next(iter(self.reference_cancer_type_groups))
            if len(self.reference_cancer_type_groups) == 1 else None
        )
        single_type_cap = (
            self.gep_filtering_parameters_by_cancer_type[single_type]["n_subgroup"]
            if single_type is not None and self.gep_filtering_parameters_by_cancer_type is not None
            else self.n_subgroup
        )
        self.uses_subgroup_reference = (
            requested_n_subgroup is not None
            and (
                len(self.reference_cancer_type_groups) > 1
                or single_type_cap < self.reference_sample_count
            )
        )
        self.full_reference_pca_target = self.gep.pca_n_components
        if self.gep_filtering_parameters_by_cancer_type is not None and not self.uses_subgroup_reference:
            self.full_reference_pca_target = self.gep_filtering_parameters_by_cancer_type[
                single_type
            ]["gep_pca_n_components"]
        self.last_batch_pca_n_components = None
        self.last_batch_id = None
        self.last_batch_cancer_type = None
        self.last_batch_reference_sample_count = None
        self.last_sampled_cancer_type = None
        self._batch_subgroup_sampler = None
        if requested_n_subgroup is not None:
            if (not self.gep.enabled or self.gep.filtering_method != "median_gep"
                    or not self.gep.filtering_in_pca_space):
                raise ValueError(
                    '"n_subgroup" requires enabled PCA-space "median_gep" filtering.'
                )
            for cancer_type, indices in self.reference_cancer_type_groups.items():
                type_settings = (
                    self.gep_filtering_parameters_by_cancer_type[cancer_type]
                    if self.gep_filtering_parameters_by_cancer_type is not None else {}
                )
                component_target = type_settings.get("gep_pca_n_components", self.gep.pca_n_components)
                subgroup_cap = type_settings.get("n_subgroup", self.n_subgroup)
                minimum_samples = (
                    max(2, int(component_target))
                    if isinstance(component_target, (int, np.integer))
                    else 2
                )
                sample_count = min(subgroup_cap, len(indices))
                maximum_components = min(sample_count, len(self.genes))
                if sample_count < minimum_samples or (
                    isinstance(component_target, (int, np.integer))
                    and component_target > maximum_components
                ):
                    raise ValueError(
                        f'{context}: Cancer type "{cancer_type}" has {len(indices)} eligible GEP '
                        f"reference samples; n_subgroup={subgroup_cap} yields "
                        f"{sample_count} samples, too few for the configured "
                        "GEP-filtering PCA components."
                    )
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
            "pca_target": self.full_reference_pca_target,
        }
        if self.gep_filtering_parameters_by_cancer_type is not None:
            self.pca_identity["gep_filtering_parameters_by_cancer_type"] = (
                self.gep_filtering_parameters_by_cancer_type
            )
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
            "n_subgroup": self.n_subgroup,
            "reference_sample_count": self.reference_sample_count,
            "reference_sampling": (
                "one-cancer-type-per-candidate-batch" if self.uses_subgroup_reference
                else "full-reference"
            ),
        }
        if self.reference_cancer_type_hash is not None:
            self.identity.update(
                reference_cancer_type_count=len(self.reference_cancer_type_groups),
                reference_cancer_type_hash=self.reference_cancer_type_hash,
                cancer_type_kept_counts="accepted-candidates-v2",
            )
        if self.gep.filtering_by_gene_range:
            self.identity["range_space"] = "normalized-log2-tpm-plus-1-v1"
        if self.gep.filtering_method == "mgep":
            self.identity["distance_space"] = "linear-tpm-selected-G-v1"
            self.identity["distance_norm"] = 1
        self.model_path = self.result_dir / "gep_filtering_pca.joblib"
        self.model_metadata_path = self.model_path.with_suffix(".metadata.json")
        self.persist_pca = True
        self.additional_artifacts = []
        self.pca = None
        self.reference_coordinates = None
        self.range_bounds = None
        self.range_input = self.range_kept = 0
        self.distance_input = self.distance_kept = 0
        self.mgep_center = None
        self.mgep_cutoff = None
        self.mgep_lower_cutoff = None
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
        if self.uses_subgroup_reference:
            return
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
        if self.pca is None and self.persist_pca:
            self.validate_model()
        if self.pca is None:
            self.pca = PCA(n_components=self.full_reference_pca_target, svd_solver="full", random_state=42)
            self.pca.fit(self.reference_log.to_numpy(dtype=np.float64))
            self.pca.filtering_genes_ = self.genes
            if self.persist_pca:
                self.result_dir.mkdir(parents=True, exist_ok=True)
                dump(self.pca, self.model_path)
                write_metadata(self.model_metadata_path, self.pca_identity)
        self.reference_coordinates = safe_pca_transform(
            self.pca, self.reference_log, label="Reference log2(TPM + 1)"
        )
        logger.info("GEP-filtering PCA: target=%s, fitted=%d PCs.",
                    self.full_reference_pca_target, self.pca.n_components_)

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
        lower_quantile, upper_quantile = self.gep.gep_filtering_quantile
        self.mgep_cutoff = float(np.quantile(reference_distances, upper_quantile))
        if lower_quantile is not None and lower_quantile > 0:
            self.mgep_lower_cutoff = float(np.quantile(reference_distances, lower_quantile))
            logger.info(
                "Linear-TPM mGEP L1 lower cutoff at q=%.3g: %.3g",
                lower_quantile,
                self.mgep_lower_cutoff,
            )
        logger.info(
            "Linear-TPM mGEP L1 cutoff at q=%.3g: %.3g",
            upper_quantile,
            self.mgep_cutoff,
        )

    def prepare_distance_reference(self):
        """Prepare the same reference distances and center used for filtering."""
        if self.reference_distances is not None:
            return self.reference_distances
        if self.uses_subgroup_reference:
            raise RuntimeError(
                "Reduced n_subgroup filtering requires a candidate batch reference."
            )
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

    def _reference_log_rows(self, indices):
        if self.reference_log is not None:
            return self.reference_log.iloc[indices].to_numpy(dtype=np.float64, copy=False)
        if self.reference_log_memmap is not None:
            return np.asarray(self.reference_log_memmap[indices], dtype=np.float64)
        raise RuntimeError("The full GEP reference is unavailable for subgroup sampling.")

    def _select_batch_cancer_type(self, seed=None, cancer_type=None):
        if not self.reference_cancer_type_groups:
            return None
        self._batch_subgroup_sampler = (
            np.random if seed is None else np.random.default_rng(seed)
        )
        if cancer_type is not None and cancer_type not in self.reference_cancer_type_groups:
            raise ValueError(f"Unknown GEP reference cancer type: {cancer_type}")
        self.last_sampled_cancer_type = (
            self._batch_subgroup_sampler.choice(list(self.reference_cancer_type_groups))
            if cancer_type is None else cancer_type
        )
        self.last_batch_cancer_type = self.last_sampled_cancer_type
        return self.last_sampled_cancer_type

    def _sample_subgroup_indices(self, seed=None):
        if not self.reference_cancer_type_groups:
            sampler = np.random if seed is None else np.random.default_rng(seed)
            return sampler.choice(
                self.reference_sample_count,
                size=min(self.n_subgroup, self.reference_sample_count),
                replace=False,
            )

        if seed is not None or self._batch_subgroup_sampler is None:
            self._select_batch_cancer_type(seed)
        sampler = self._batch_subgroup_sampler
        cancer_type = self.last_sampled_cancer_type
        cancer_type_indices = self.reference_cancer_type_groups[cancer_type]
        subgroup_cap = (
            self.gep_filtering_parameters_by_cancer_type[cancer_type]["n_subgroup"]
            if self.gep_filtering_parameters_by_cancer_type is not None
            else self.n_subgroup
        )
        return sampler.choice(
            cancer_type_indices,
            size=min(subgroup_cap, len(cancer_type_indices)),
            replace=False,
        )

    def prepare_batch_reference(self, batch_id=None, seed=None):
        """Fit a transient PCA reference for one candidate batch."""
        if not self.uses_subgroup_reference:
            return self.prepare_distance_reference()
        if self.reference_sample_count <= 0:
            raise ValueError("GEP filtering reference has no samples for subgroup sampling.")

        sampling_seed = seed if self._batch_subgroup_sampler is None else None
        indices = self._sample_subgroup_indices(sampling_seed)
        reference = self._reference_log_rows(indices)
        if not np.isfinite(reference).all():
            raise ValueError("Sampled GEP reference contains non-finite log expression.")

        type_settings = (
            self.gep_filtering_parameters_by_cancer_type[self.last_sampled_cancer_type]
            if self.gep_filtering_parameters_by_cancer_type is not None
            else {"n_subgroup": self.n_subgroup, "gep_pca_n_components": self.gep.pca_n_components}
        )
        self.pca = PCA(
            n_components=type_settings["gep_pca_n_components"],
            svd_solver="full",
            random_state=42,
        )
        self.pca.fit(reference)
        self.reference_coordinates = safe_pca_transform(
            self.pca, reference, label="Sampled reference log2(TPM + 1)"
        )
        self.distance_center = np.median(self.reference_coordinates, axis=0)
        self.reference_distances = np.linalg.norm(
            self.reference_coordinates - self.distance_center,
            ord=self.gep.norm_ord,
            axis=1,
        )
        self.last_batch_pca_n_components = int(self.pca.n_components_)
        self.last_batch_id = batch_id
        self.last_batch_cancer_type = self.last_sampled_cancer_type
        self.last_batch_reference_sample_count = len(indices)
        logger.info(
            "GEP PCA subgroup for batch %s: cancer type %s, %d/%d reference "
            "samples; n_subgroup=%s, PCA target=%s; %d PCs.",
            batch_id if batch_id is not None else "unknown",
            self.last_batch_cancer_type,
            len(indices),
            len(self.reference_cancer_type_groups[self.last_batch_cancer_type]),
            type_settings["n_subgroup"],
            type_settings["gep_pca_n_components"],
            self.last_batch_pca_n_components,
        )
        return self.reference_distances

    def clear_batch_reference(self):
        """Release transient PCA state after filtering a subgroup batch."""
        self._batch_subgroup_sampler = None
        if not self.uses_subgroup_reference:
            return
        self.pca = None
        self.reference_coordinates = None
        self.distance_center = None
        self.reference_distances = None

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

    def prepare_range_reference(self):
        """Prepare inclusive per-gene bounds in normalized log2(TPM + 1) space."""
        if self.range_bounds is None:
            lower, _, upper = self.gep.gene_quantile_range
            self.range_bounds = self.reference_log.quantile([lower, upper])
        return self.range_bounds

    def filter(self, simulated_linear, batch_id=None, subgroup_seed=None, cancer_type=None):
        if self.uses_subgroup_reference:
            self.last_batch_pca_n_components = None
            self.last_batch_id = batch_id
            self.last_batch_cancer_type = None
            self.last_batch_reference_sample_count = None
            self._select_batch_cancer_type(subgroup_seed, cancer_type)
        elif self.gep.n_subgroup is not None and len(self.reference_cancer_type_groups) == 1:
            self.last_batch_cancer_type = next(iter(self.reference_cancer_type_groups))
            if self.gep_filtering_parameters_by_cancer_type is not None:
                settings = self.gep_filtering_parameters_by_cancer_type[self.last_batch_cancer_type]
                logger.info(
                    "GEP PCA batch %s: cancer type %s, n_subgroup=%s, PCA target=%s; "
                    "using the full reference.",
                    batch_id if batch_id is not None else "unknown",
                    self.last_batch_cancer_type, settings["n_subgroup"],
                    settings["gep_pca_n_components"],
                )
        normalized = normalize_selected(simulated_linear, self.genes, "Simulated profiles")
        if normalized.empty or not self.gep.enabled:
            self._batch_subgroup_sampler = None
            return normalized
        if self.gep.filtering_by_gene_range:
            range_values = np.log2(normalized + 1)
            range_bounds = self.prepare_range_reference()
            mask = (
                range_values.ge(range_bounds.iloc[0], axis=1)
                & range_values.le(range_bounds.iloc[1], axis=1)
            ).mean(axis=1) >= self.gep.min_percentage_within_gene_range
            self.range_input += len(normalized)
            self.range_kept += int(mask.sum())
            logger.info("Gene-range acceptance: %d/%d profiles on %d genes.",
                        mask.sum(), len(mask), len(self.genes))
            normalized = normalized.loc[mask]
        if normalized.empty:
            self._batch_subgroup_sampler = None
            return normalized
        if self.gep.filtering_method == "mgep":
            simulated_distances = self.distance_from_normalized(normalized)
            mask = simulated_distances <= self.mgep_cutoff
            if self.mgep_lower_cutoff is not None:
                mask &= simulated_distances >= self.mgep_lower_cutoff
            self.mgep_distance_input += len(normalized)
            self.mgep_distance_kept += int(mask.sum())
            logger.info(
                "Linear-TPM mGEP L1 acceptance: %d/%d profiles.",
                mask.sum(),
                len(mask),
            )
            return normalized.loc[mask].copy()
        if self.uses_subgroup_reference:
            self.prepare_batch_reference(batch_id=batch_id, seed=subgroup_seed)
        try:
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
        finally:
            self.clear_batch_reference()

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
