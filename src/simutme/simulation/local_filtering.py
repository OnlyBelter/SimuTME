"""Cancer-stratified calibration for local TCGA neighbor/residual filtering."""

import hashlib
import logging

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.neighbors import NearestNeighbors

from ..utility import safe_pca_transform
from .filtering import (
    FixedGeneFilter, identity_hash, normalize_selected, validate_metadata, write_metadata,
)

logger = logging.getLogger(__name__)
LOCAL_WORKFLOW_VERSION = "correlation-first-local-v2"
SPLIT_VERSION = "cancer-sample-ceiling-sha256-v1"


def validate_local_parameters(gep):
    for name in ("reference_holdout_fraction", "knn_distance_quantile", "pca_residual_quantile"):
        value = getattr(gep, name)
        valid = np.isfinite(value) and (
            0 < value < 1 if name == "reference_holdout_fraction" else 0 <= value <= 1
        )
        if not valid:
            raise ValueError(f"gep_{name} must be finite and in "
                             f"{'(0, 1)' if name == 'reference_holdout_fraction' else '[0, 1]'}.")
    for name in ("reference_holdout_seed", "knn_n_neighbors", "max_generation_batches"):
        value = getattr(gep, name)
        minimum = 0 if name == "reference_holdout_seed" else 1
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"gep_{name} must be an integer >= {minimum}.")
    if not gep.filtering_in_pca_space:
        raise ValueError("knn_residual requires gep_filtering_in_pca_space: true.")
    if isinstance(gep.norm_ord, bool) or gep.norm_ord not in (1, 2):
        raise ValueError("knn_residual requires gep_norm_ord to be 1 or 2.")


def split_tcga_reference(reference, annotation, fraction=0.2, seed=42, ref_types=None):
    """Select calibration samples independently within each cancer stratum."""
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("gep_reference_holdout_fraction must be in (0, 1).")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("gep_reference_holdout_seed must be a nonnegative integer.")
    for index, label in ((reference.index, "Reference"), (annotation.index, "Annotation")):
        if index.has_duplicates or index.isna().any():
            raise ValueError(f"{label} contains duplicate or missing sample IDs.")
    if "cancer_type" not in annotation.columns:
        raise ValueError("TCGA annotation must contain a cancer_type column.")
    missing = reference.index.difference(annotation.index)
    if len(missing):
        raise ValueError(f"TCGA annotation is missing reference sample IDs: {missing[:10].tolist()}.")
    labels = annotation.loc[reference.index, "cancer_type"]
    if labels.isna().any() or labels.astype(str).str.strip().eq("").any():
        raise ValueError("TCGA reference samples have missing or blank cancer_type labels.")
    labels = labels.astype(str).str.strip()
    logger.info("Ignoring %d TCGA annotation rows absent from expression.",
                len(annotation.index.difference(reference.index)))
    if ref_types:
        labels = labels.loc[labels.isin(ref_types)]
    if labels.empty:
        raise ValueError("No reference samples match the requested filtering_ref_types.")
    labels = labels.sort_index()
    split = labels.to_frame()
    split.index.name = "sample_id"
    split["role"] = "fitting"
    rows = []
    for cancer_type in sorted(labels.unique()):
        ids = labels.index[labels == cancer_type].tolist()
        n_heldout = int(np.ceil(fraction * len(ids)))
        if not 0 < n_heldout < len(ids):
            raise ValueError(
                f"Cancer type {cancer_type!r} has an infeasible stratified split: "
                f"{len(ids)} samples, {n_heldout} held-out. Each type needs fitting and held-out samples."
            )
        cancer_seed = int.from_bytes(
            hashlib.sha256(f"{seed}\0{cancer_type}".encode("utf-8")).digest()[:8], "little"
        )
        heldout = np.random.default_rng(cancer_seed).choice(ids, n_heldout, replace=False)
        split.loc[heldout, "role"] = "held_out"
        rows.append({
            "cancer_type": cancer_type, "n_samples": len(ids),
            "n_fitting": len(ids) - n_heldout, "n_held_out": n_heldout,
            "held_out_fraction": n_heldout / len(ids),
        })
    return reference.loc[split.index].copy(), split, pd.DataFrame(rows)


def _persist_table(path, expected, index=True):
    if path.exists():
        existing = pd.read_csv(path, index_col=0 if index else None)
        try:
            pd.testing.assert_frame_equal(
                existing, expected, check_dtype=False, rtol=1e-12, atol=1e-12
            )
        except AssertionError as exc:
            raise ValueError(
                f"Incompatible reference table: {path}. Use a new dataset name/output location."
            ) from exc
    else:
        expected.to_csv(path, index=index, float_format="%.17g")


class LocalTCGAFilter(FixedGeneFilter):
    """Fit on reference samples, calibrate on reserved samples, score candidates."""

    workflow_version = LOCAL_WORKFLOW_VERSION
    supported_methods = ("knn_residual",)
    score_chunk_size = 256

    def __init__(self, genes, reference_linear, reference_file, reference_subset,
                 gep_params, result_dir, generation_n_genes, shared_n_genes,
                 split, split_summary, annotation_file, correlation_file=None):
        validate_local_parameters(gep_params)
        super().__init__(
            genes, reference_linear, reference_file, reference_subset, gep_params,
            result_dir, generation_n_genes, shared_n_genes, correlation_file,
        )
        self.split = split
        self.split_summary = split_summary
        self.all_reference_log = self.reference_log.copy()
        self.fitting_ids = split.index[split.role == "fitting"].tolist()
        self.heldout_ids = split.index[split.role == "held_out"].tolist()
        if len(self.fitting_ids) < 2:
            raise ValueError("knn_residual requires at least two fitting-reference samples for PCA.")
        if len(self.fitting_ids) < self.gep.knn_n_neighbors:
            raise ValueError("Fitting TCGA reference has fewer samples than gep_knn_n_neighbors.")
        self.heldout_linear = self.reference_linear.loc[self.heldout_ids].copy()
        self.heldout_log = self.reference_log.loc[self.heldout_ids].copy()
        self.reference_linear = self.reference_linear.loc[self.fitting_ids].copy()
        self.reference_log = self.reference_log.loc[self.fitting_ids].copy()
        self.pca_identity.update({
            "annotation_file": str(annotation_file),
            "annotation_labels_hash": identity_hash(split.cancer_type.to_dict()),
            "split_version": SPLIT_VERSION,
            "split_unit": "sample",
            "holdout_fraction": self.gep.reference_holdout_fraction,
            "holdout_seed": self.gep.reference_holdout_seed,
            "fitting_sample_ids": self.fitting_ids,
            "heldout_sample_ids": self.heldout_ids,
        })
        self.calibration_identity = {
            **self.pca_identity,
            "knn_n_neighbors": self.gep.knn_n_neighbors,
            "neighbor_norm": self.gep.norm_ord,
            "knn_distance_quantile": self.gep.knn_distance_quantile,
            "pca_residual_quantile": self.gep.pca_residual_quantile,
            "range_enabled": self.gep.filtering_by_gene_range,
            "range_quantiles": self.gep.gene_quantile_range,
            "range_min_fraction": self.gep.min_percentage_within_gene_range,
        }
        self.identity.update(self.calibration_identity)
        self.identity["max_generation_batches"] = self.gep.max_generation_batches
        self.split_path = self.result_dir / "tcga_reference_split.csv"
        self.split_summary_path = self.result_dir / "tcga_reference_split_summary.csv"
        self.calibration_scores_path = self.result_dir / "tcga_heldout_calibration_scores.csv"
        self.neighbors_path = self.result_dir / "gep_filtering_neighbors.joblib"
        self.calibration_metadata_path = self.result_dir / "gep_filtering_calibration.metadata.json"
        self.candidate_scores_path = self.result_dir / "gep_filtering_candidate_scores.csv"
        self.neighbors = None
        self.calibration_scores = None
        self.knn_threshold = self.residual_threshold = None
        self.knn_kept = self.residual_kept = self.residual_after_knn_kept = 0
        self.generation_batches = 0
        self.last_scores = pd.DataFrame()
        self.diagnostic_outputs = {}

    def validate_model(self):
        super().validate_model()
        if self.pca is not None and self.pca.whiten:
            raise ValueError("Local filtering requires unwhitened PCA. Use a new dataset name/output location.")
        paths = (self.neighbors_path, self.calibration_metadata_path, self.calibration_scores_path)
        if any(path.exists() for path in paths):
            metadata = validate_metadata(self.calibration_metadata_path, self.calibration_identity)
            if self.pca is None or not all(path.exists() for path in paths):
                raise ValueError("Calibration artifacts are incomplete. Use a new dataset name/output location.")
            self.neighbors = load(self.neighbors_path)
            if (getattr(self.neighbors, "reference_sample_ids_", None) != self.fitting_ids
                    or self.neighbors.n_features_in_ != self.pca.n_components_):
                raise ValueError("Neighbor reference differs from PCA/split. Use a new output location.")
            self.knn_threshold = metadata["knn_threshold"]
            self.residual_threshold = metadata["residual_threshold"]
            self.calibration_scores = pd.read_csv(self.calibration_scores_path, index_col=0)
            if (self.calibration_scores.index.tolist() != self.heldout_ids
                    or identity_hash(self.calibration_scores.to_json()) != metadata["scores_hash"]):
                raise ValueError("Calibration scores differ from stored metadata. Use a new output location.")
            if not all(np.isfinite(t) and t >= 0 for t in (self.knn_threshold, self.residual_threshold)):
                raise ValueError("Calibration thresholds are invalid. Use a new output location.")

    def prepare_calibration(self):
        if self.pca is None or self.neighbors is None:
            self.validate_model()
        self.result_dir.mkdir(parents=True, exist_ok=True)
        _persist_table(self.split_path, self.split)
        _persist_table(self.split_summary_path, self.split_summary, index=False)
        if self.calibration_scores is not None:
            return
        self._prepare_pca()
        self.neighbors = NearestNeighbors(
            n_neighbors=self.gep.knn_n_neighbors, algorithm="ball_tree",
            metric="minkowski", p=self.gep.norm_ord,
        ).fit(self.reference_coordinates)
        self.neighbors.reference_sample_ids_ = self.fitting_ids
        scores = self.score_log_profiles(self.heldout_log)
        self.knn_threshold = float(scores.knn_distance.quantile(self.gep.knn_distance_quantile))
        self.residual_threshold = float(scores.pca_residual.quantile(self.gep.pca_residual_quantile))
        scores.insert(0, "cancer_type", self.split.loc[self.heldout_ids, "cancer_type"])
        scores["gene_range_pass"] = self._range_mask(self.heldout_linear)
        scores["knn_pass"] = scores.knn_distance <= self.knn_threshold
        scores["residual_pass"] = scores.pca_residual <= self.residual_threshold
        scores["joint_pass"] = scores[["gene_range_pass", "knn_pass", "residual_pass"]].all(axis=1)
        scores.index.name = "sample_id"
        scores.to_csv(self.calibration_scores_path, float_format="%.17g")
        # Hash the serialized representation that the cached path reads.
        self.calibration_scores = pd.read_csv(self.calibration_scores_path, index_col=0)
        dump(self.neighbors, self.neighbors_path)
        write_metadata(
            self.calibration_metadata_path, self.calibration_identity,
            knn_threshold=self.knn_threshold, residual_threshold=self.residual_threshold,
            scores_hash=identity_hash(self.calibration_scores.to_json()),
        )
        logger.info(
            "TCGA calibration: %d fitting, %d held-out; k=%d, neighbor threshold=%g, residual threshold=%g.",
            len(self.fitting_ids), len(self.heldout_ids), self.gep.knn_n_neighbors,
            self.knn_threshold, self.residual_threshold,
        )
        logger.info("gep_filtering_quantile applies only to centroid methods; knn_residual has no lower cutoff.")
        for cancer_type, group in self.calibration_scores.groupby("cancer_type"):
            logger.info("Held-out %s: %d/%d pass all checks.", cancer_type, group.joint_pass.sum(), len(group))

    def _range_mask(self, linear):
        if not self.gep.filtering_by_gene_range:
            return pd.Series(True, index=linear.index)
        if self.range_bounds is None:
            lower, _, upper = self.gep.gene_quantile_range
            self.range_bounds = self.reference_linear.quantile([lower, upper])
        return (
            linear.ge(self.range_bounds.iloc[0], axis=1)
            & linear.le(self.range_bounds.iloc[1], axis=1)
        ).mean(axis=1) >= self.gep.min_percentage_within_gene_range

    def score_log_profiles(self, log_expression):
        if self.pca is None or self.neighbors is None:
            raise RuntimeError("Prepare TCGA calibration before scoring profiles.")
        if log_expression.columns.tolist() != self.genes:
            raise ValueError("Scoring gene order differs from G.")
        parts = []
        for start in range(0, len(log_expression), self.score_chunk_size):
            chunk = log_expression.iloc[start:start + self.score_chunk_size]
            coordinates = safe_pca_transform(self.pca, chunk, label="Local-filter log2(TPM + 1)")
            distances = self.neighbors.kneighbors(coordinates, return_distance=True)[0].mean(axis=1)
            reconstructed = np.einsum(
                "ik,kj->ij", coordinates, self.pca.components_, optimize=False
            ) + self.pca.mean_
            residuals = np.linalg.norm(chunk.to_numpy() - reconstructed, ord=2, axis=1)
            if not np.isfinite(distances).all() or not np.isfinite(residuals).all():
                raise ValueError("Local filtering produced non-finite scores.")
            parts.append(pd.DataFrame(
                {"knn_distance": distances, "pca_residual": residuals}, index=chunk.index
            ))
        return pd.concat(parts) if parts else pd.DataFrame(
            columns=["knn_distance", "pca_residual"], index=log_expression.index, dtype=float
        )

    def check_batch_limit(self, exported_count, generated_count):
        if self.generation_batches >= self.gep.max_generation_batches:
            raise RuntimeError(
                f"knn_residual reached gep_max_generation_batches={self.gep.max_generation_batches}: "
                f"generated={generated_count}, range_surviving={self.distance_input}, "
                f"neighbor_surviving={self.knn_kept}, residual_surviving={self.residual_after_knn_kept}, "
                f"exported={exported_count}. Thresholds were not loosened; intermediate artifacts are preserved."
            )

    def filter(self, simulated_linear):
        normalized = normalize_selected(simulated_linear, self.genes, "Simulated profiles")
        self.last_scores = pd.DataFrame()
        self.generation_batches += 1
        if normalized.empty:
            return normalized
        if self.calibration_scores is None:
            raise RuntimeError("TCGA calibration must complete before target generation.")
        range_pass = self._range_mask(normalized)
        if self.gep.filtering_by_gene_range:
            self.range_input += len(normalized)
            self.range_kept += int(range_pass.sum())
            logger.info("Gene-range acceptance: %d/%d profiles.", range_pass.sum(), len(normalized))
        else:
            logger.info("Gene-range check disabled; %d profiles proceed to local scoring.", len(normalized))
        normalized = normalized.loc[range_pass]
        if normalized.empty:
            return normalized
        scores = self.score_log_profiles(np.log2(normalized + 1))
        scores["knn_pass"] = scores.knn_distance <= self.knn_threshold
        scores["residual_pass"] = scores.pca_residual <= self.residual_threshold
        scores["joint_pass"] = scores.knn_pass & scores.residual_pass
        self.distance_input += len(scores)
        self.knn_kept += int(scores.knn_pass.sum())
        self.residual_kept += int(scores.residual_pass.sum())
        self.distance_kept += int(scores.joint_pass.sum())
        self.residual_after_knn_kept += int(scores.joint_pass.sum())
        self.last_scores = scores
        logger.info("Local acceptance: neighbors=%d, residuals=%d, joint=%d/%d.",
                    scores.knn_pass.sum(), scores.residual_pass.sum(), scores.joint_pass.sum(), len(scores))
        return normalized.loc[scores.joint_pass].copy()

    def save_batch_scores(self, exported_ids):
        if self.last_scores.empty:
            return
        scores = self.last_scores.copy()
        scores["batch_number"] = self.generation_batches
        scores["exported"] = scores.index.isin(exported_ids)
        exists = self.candidate_scores_path.exists()
        scores.to_csv(
            self.candidate_scores_path, mode="a" if exists else "w",
            header=not exists, index_label="sample_id", float_format="%.17g",
        )
        self.last_scores = pd.DataFrame()

    def validate_saved_scores(self, exported_count):
        if not self.candidate_scores_path.exists():
            if self.distance_input or exported_count:
                raise ValueError("Candidate scores are missing. Use a new dataset name/output location.")
            return
        n_scored = n_exported = 0
        for chunk in pd.read_csv(self.candidate_scores_path, index_col=0, chunksize=10000):
            required = {"knn_distance", "pca_residual", "knn_pass", "residual_pass",
                        "joint_pass", "batch_number", "exported"}
            if not required <= set(chunk.columns):
                raise ValueError("Candidate score columns are missing. Use a new dataset name/output location.")
            distances = chunk[["knn_distance", "pca_residual"]].to_numpy(dtype=float)
            flags = ("knn_pass", "residual_pass", "joint_pass", "exported")
            if (not np.isfinite(distances).all() or (distances < 0).any()
                    or not all(pd.api.types.is_bool_dtype(chunk[c]) for c in flags)):
                raise ValueError("Candidate scores/flags are invalid. Use a new dataset name/output location.")
            if ((chunk.joint_pass != (chunk.knn_pass & chunk.residual_pass)).any()
                    or (chunk.exported & ~chunk.joint_pass).any()):
                raise ValueError("Candidate acceptance flags are inconsistent. Use a new output location.")
            if not chunk.batch_number.between(1, self.generation_batches).all():
                raise ValueError("Candidate batch counts are incompatible. Use a new output location.")
            n_scored += len(chunk)
            n_exported += int(chunk.exported.sum())
        if n_scored != self.distance_input or n_exported != exported_count:
            raise ValueError("Candidate score counts differ from checkpoint. Use a new output location.")

    def stats(self):
        return {
            **super().stats(), "knn_input_n_samples": self.distance_input,
            "residual_input_n_samples": self.distance_input,
            "knn_kept_n_samples": self.knn_kept,
            "residual_kept_n_samples": self.residual_kept,
            "residual_after_knn_kept_n_samples": self.residual_after_knn_kept,
            "knn_residual_joint_kept_n_samples": self.distance_kept,
            "generation_batches": self.generation_batches,
        }

    def restore_stats(self, stats):
        required = {
            "gene_range_input_n_samples", "gene_range_kept_n_samples",
            "pc_distance_input_n_samples", "pc_distance_kept_n_samples",
            "knn_kept_n_samples", "residual_kept_n_samples",
            "residual_after_knn_kept_n_samples", "generation_batches",
        }
        if not required <= stats.keys() or any(
            isinstance(stats[key], bool) or not isinstance(stats[key], int) or stats[key] < 0
            for key in required
        ):
            raise ValueError(
                "Local-filter checkpoint statistics are incomplete or invalid. "
                "Use a new dataset name/output location."
            )
        if stats["generation_batches"] > self.gep.max_generation_batches:
            raise ValueError("Checkpoint exceeds the generation batch bound. Use a new output location.")
        super().restore_stats(stats)
        self.knn_kept = stats.get("knn_kept_n_samples", 0)
        self.residual_kept = stats.get("residual_kept_n_samples", 0)
        self.residual_after_knn_kept = stats.get("residual_after_knn_kept_n_samples", 0)
        self.generation_batches = stats.get("generation_batches", 0)

    def summary(self):
        patients = self.split.index.to_series().str.extract(r"^(TCGA-[^-]+-[^-]+)(?:-|$)", expand=False)
        fitting_patients = set(patients.loc[self.fitting_ids].dropna())
        heldout_patients = set(patients.loc[self.heldout_ids].dropna())
        calibration_by_type = {}
        for cancer_type, group in self.calibration_scores.groupby("cancer_type"):
            calibration_by_type[cancer_type] = {
                "n_samples": len(group),
                **{f"{column}_n_samples": int(group[column].sum()) for column in
                   ("gene_range_pass", "knn_pass", "residual_pass", "joint_pass")},
            }
        return {
            "reference_split_unit": "sample", "reference_holdout_fraction": self.gep.reference_holdout_fraction,
            "reference_holdout_seed": self.gep.reference_holdout_seed,
            "reference_fitting_n_samples": len(self.fitting_ids),
            "reference_heldout_n_samples": len(self.heldout_ids),
            "reference_patient_overlap_n": len(fitting_patients & heldout_patients),
            "knn_n_neighbors": self.gep.knn_n_neighbors,
            "knn_distance_threshold": self.knn_threshold,
            "pca_residual_threshold": self.residual_threshold,
            "calibration_by_cancer_type": calibration_by_type,
            "max_generation_batches": self.gep.max_generation_batches,
            "tcga_reference_split_file": str(self.split_path),
            "tcga_reference_split_summary_file": str(self.split_summary_path),
            "tcga_heldout_calibration_scores_file": str(self.calibration_scores_path),
            "gep_filtering_neighbors_file": str(self.neighbors_path),
            "gep_filtering_calibration_metadata_file": str(self.calibration_metadata_path),
            "gep_filtering_candidate_scores_file": str(self.candidate_scores_path),
            **{f"calibration_{column}_n_samples": int(self.calibration_scores[column].sum()) for column in
               ("gene_range_pass", "knn_pass", "residual_pass", "joint_pass")},
            **self.diagnostic_outputs,
        }
