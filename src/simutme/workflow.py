import json
import os
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Literal, Optional

from .utility import do_pca_analysis, plot_pca, safe_pca_transform
from .utility import cal_corr_gene_exp_with_cell_frac
# Assuming these are existing imports in your project
from .utility import ReadH5AD, ReadExp, create_h5ad_dataset
from .utility import check_dir, non_log2cpm, log2_transform, get_gene_list_filtered_by_quantile_range
from .simulation import BulkGEPGenerator, SingleCellTypeGEPGenerator, GEPGenerationConfig
from .simulation.all_cell_neighbors import validate_cell_cap
from .simulation.sampling import (
    estimate_cell_prop_maxima_from_pred_cell_props,
    estimate_dirichlet_alpha_from_pred_cell_props,
    flatten_cell_type2subtype,
    format_segment_fraction_caps,
    get_default_dirichlet_alpha,
    merge_cell_prop_prior_with_maxima,
    restrict_cell_type2subtype_to_output_types,
)
from .sct_imputation import run_sorted_bulk_guided_sct_imputation
from .simulation.sample_diversity import run_sct_sample_diversity_analysis
from .simulation.filtering import (
    FixedGeneFilter, NORMALIZATION_VERSION, identity_hash,
    normalize_selected, select_fixed_genes, reference_variance_mask, validate_metadata, write_metadata,
)
from .simulation.local_filtering import (
    LOCAL_FILTER_METHODS, LocalTCGAFilter, split_tcga_reference, validate_local_parameters,
)


def _build_sorted_bulk_imputed_sct_path(
    sct_dataset_file_path: str | Path,
    output_suffix: str,
) -> Path:
    sct_dataset_file_path = Path(sct_dataset_file_path)
    return sct_dataset_file_path.with_name(
        f"{sct_dataset_file_path.stem}{output_suffix}{sct_dataset_file_path.suffix}"
    )


def _resolve_sct_aux_output_dir(
    sct_dataset_file_path: str | Path,
    dataset_name: str,
) -> Path:
    """Store SCT auxiliary outputs beside the SCT dataset under `<parent>/<dataset_name>/`."""
    sct_dataset_file_path = Path(sct_dataset_file_path)
    return sct_dataset_file_path.parent / dataset_name


def _resolve_mixed_sct_reference_version(version_value: Any, context: str) -> str:
    version = str(version_value).strip()
    allowed_versions = {"original", "sorted_bulk_imputed"}
    if version not in allowed_versions:
        raise ValueError(
            f'Invalid "sct_reference.version" for {context}: '
            f'{version_value!r}. Expected one of {sorted(allowed_versions)}.'
        )
    return version


def _resolve_mixed_sct_dataset_path(
    dataset_name: str,
    params: Dict[str, Any],
    config: "SimuConfig",
) -> Path:
    if "sct_dataset_version" in params:
        raise ValueError(
            f'Mixed dataset "{dataset_name}" uses deprecated "sct_dataset_version". '
            f'Please replace it with the explicit "sct_reference" block.'
        )
    sct_reference = params.get("sct_reference")
    if not isinstance(sct_reference, dict):
        raise ValueError(
            f'Mixed dataset "{dataset_name}" must define an "sct_reference" mapping.'
        )

    source_type = str(sct_reference.get("source_type", "")).strip()
    requested_version: str | None = None

    if source_type == "path":
        file_path_value = sct_reference.get("file_path", None)
        if file_path_value in (None, ""):
            raise ValueError(
                f'Mixed dataset "{dataset_name}" with `sct_reference.source_type: "path"` '
                f'must provide "sct_reference.file_path".'
            )
        if sct_reference.get("dataset_name", None) not in (None, ""):
            raise ValueError(
                f'Mixed dataset "{dataset_name}" with `sct_reference.source_type: "path"` '
                f'must not provide "sct_reference.dataset_name".'
            )
        if sct_reference.get("version", None) not in (None, ""):
            raise ValueError(
                f'Mixed dataset "{dataset_name}" with `sct_reference.source_type: "path"` '
                f'must not provide "sct_reference.version". The given "file_path" is used directly.'
            )
        resolved_sct_path = Path(file_path_value)
    elif source_type == "generated_dataset":
        requested_version = _resolve_mixed_sct_reference_version(
            sct_reference.get("version", None),
            f'Mixed dataset "{dataset_name}"',
        )
        referenced_dataset_name = str(sct_reference.get("dataset_name", "")).strip()
        if referenced_dataset_name == "":
            raise ValueError(
                f'Mixed dataset "{dataset_name}" with `sct_reference.source_type: "generated_dataset"` '
                f'must provide "sct_reference.dataset_name".'
            )
        if sct_reference.get("file_path", None) not in (None, ""):
            raise ValueError(
                f'Mixed dataset "{dataset_name}" with `sct_reference.source_type: "generated_dataset"` '
                f'must not provide "sct_reference.file_path".'
            )
        referenced_params = config._dataset_definitions.get(referenced_dataset_name)
        if referenced_params is None:
            raise ValueError(
                f'Mixed dataset "{dataset_name}" references unknown SCT dataset '
                f'"{referenced_dataset_name}" in "sct_reference.dataset_name".'
            )
        if str(referenced_params.get("dataset_type", "mixed")).lower() != "sct":
            raise ValueError(
                f'Mixed dataset "{dataset_name}" must reference an SCT dataset, but '
                f'"{referenced_dataset_name}" has dataset_type='
                f'{referenced_params.get("dataset_type")!r}.'
            )
        if dataset_name not in config._dataset_order or referenced_dataset_name not in config._dataset_order:
            raise ValueError(
                f'Mixed dataset "{dataset_name}" cannot resolve `sct_reference.dataset_name` '
                f'without dataset-order metadata. Load the workflow via `SimuConfig.from_yaml(...)`.'
            )
        current_dataset_index = config._dataset_order.index(dataset_name)
        referenced_dataset_index = config._dataset_order.index(referenced_dataset_name)
        if referenced_dataset_index >= current_dataset_index:
            raise ValueError(
                f'Mixed dataset "{dataset_name}" references SCT dataset "{referenced_dataset_name}" '
                f'that appears later in the YAML order. Please place the SCT dataset earlier.'
            )
        generated_outputs = config._generated_sct_outputs.get(referenced_dataset_name)
        if generated_outputs is None:
            raise ValueError(
                f'Mixed dataset "{dataset_name}" references SCT dataset "{referenced_dataset_name}" '
                f'but no recorded outputs are available. Run the referenced SCT dataset first.'
            )
        if requested_version == "original":
            resolved_sct_path = generated_outputs["original"]
        else:
            resolved_sct_path = generated_outputs["sorted_bulk_imputed"]
            if resolved_sct_path in (None, ""):
                raise ValueError(
                    f'Mixed dataset "{dataset_name}" requested the "sorted_bulk_imputed" version '
                    f'of SCT dataset "{referenced_dataset_name}", but that SCT dataset does not '
                    f'have an imputed output. Set `apply_sorted_bulk_imputation: true` for the '
                    f'referenced SCT dataset or use `version: "original"`.'
                )
    else:
        raise ValueError(
            f'Invalid "sct_reference.source_type" for Mixed dataset "{dataset_name}": '
            f'{source_type!r}. Expected "path" or "generated_dataset".'
        )

    if not resolved_sct_path.exists():
        if requested_version is None:
            raise FileNotFoundError(
                f'Mixed dataset "{dataset_name}" requested SCT reference file '
                f"{resolved_sct_path}, but it does not exist."
            )
        raise FileNotFoundError(
            f'Mixed dataset "{dataset_name}" requested SCT reference version '
            f'"{requested_version}", but the resolved file does not exist: '
            f"{resolved_sct_path}."
        )

    if requested_version is None:
        logging.info(
            "   > Mixed dataset %s will use sct_reference(source_type=%s): %s",
            dataset_name,
            source_type,
            resolved_sct_path,
        )
    else:
        logging.info(
            "   > Mixed dataset %s will use sct_reference(source_type=%s, version=%s): %s",
            dataset_name,
            source_type,
            requested_version,
            resolved_sct_path,
        )
    return resolved_sct_path


def _resolve_effective_sct_dataset_ids_for_naming(
    params: Dict[str, Any],
    config: "SimuConfig",
) -> List[str]:
    """Resolve dataset IDs used in output-directory naming.

    Naming follows the user's configured SCT subset lists only when both
    `cancer_cell_dataset_ids` and `non_cancer_cell_dataset_ids` are provided.
    In that case we use their unique union (preserving order). Otherwise we
    fall back to `sc_dataset_ids`.
    """
    cancer_ids = params.get("cancer_cell_dataset_ids", None)
    non_cancer_ids = params.get("non_cancer_cell_dataset_ids", None)
    if cancer_ids is not None and non_cancer_ids is not None:
        ordered_unique_ids: list[str] = []
        seen: set[str] = set()
        for dataset_id in list(cancer_ids) + list(non_cancer_ids):
            cleaned_id = str(dataset_id).strip()
            if cleaned_id == "" or cleaned_id in seen:
                continue
            seen.add(cleaned_id)
            ordered_unique_ids.append(cleaned_id)
        if ordered_unique_ids:
            return ordered_unique_ids

    return list(params.get("sc_dataset_ids", config.sc_dataset_ids))


def _resolve_effective_sct_n_base(
    dataset_name: str,
    params: Dict[str, Any],
    config: "SimuConfig",
) -> int:
    """Resolve and validate the effective positive-sample base size."""
    value = params.get("n_base", config.n_base)
    try:
        if isinstance(value, bool):
            raise TypeError
        if isinstance(value, float) and not value.is_integer():
            raise ValueError
        resolved_n_base = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'Invalid "n_base" for SCT dataset "{dataset_name}": '
            f"{value!r}. Expected a positive integer."
        ) from exc
    if resolved_n_base <= 0:
        raise ValueError(
            f'Invalid "n_base" for SCT dataset "{dataset_name}": '
            f"{value!r}. Expected a positive integer."
        )
    return resolved_n_base


def _resolve_output_dir_naming_metadata(
    params: Dict[str, Any],
    config: "SimuConfig",
    dataset_type: str,
    dataset_name: str = "<dataset>",
) -> tuple[int, int]:
    """Return `(n_datasets, n_base)` for output directory naming."""
    if dataset_type == "sct":
        n_base = _resolve_effective_sct_n_base(dataset_name, params, config)
        dataset_ids = _resolve_effective_sct_dataset_ids_for_naming(params, config)
        return len(dataset_ids), n_base

    n_base = int(config.n_base)
    dataset_ids: Optional[List[str]] = None
    sct_reference = params.get("sct_reference")
    if isinstance(sct_reference, dict) and str(sct_reference.get("source_type", "")).strip() == "generated_dataset":
        referenced_dataset_name = str(sct_reference.get("dataset_name", "")).strip()
        referenced_params = config._dataset_definitions.get(referenced_dataset_name)
        if isinstance(referenced_params, dict):
            dataset_ids = _resolve_effective_sct_dataset_ids_for_naming(referenced_params, config)
            n_base = _resolve_effective_sct_n_base(
                referenced_dataset_name,
                referenced_params,
                config,
            )

    if dataset_ids is None:
        dataset_ids = list(params.get("sc_dataset_ids", config.sc_dataset_ids))

    return len(dataset_ids), n_base


def _resolve_sorted_bulk_imputation_bulk_weight(
    bulk_weight_value: Any,
    context: str,
) -> float:
    try:
        bulk_weight = float(bulk_weight_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'Invalid "sorted_bulk_imputation_bulk_weight" for {context}: '
            f"{bulk_weight_value!r}. Expected a numeric value in [0, 1]."
        ) from exc
    if not 0.0 <= bulk_weight <= 1.0:
        raise ValueError(
            f'Invalid "sorted_bulk_imputation_bulk_weight" for {context}: '
            f"{bulk_weight}. Expected a value in [0, 1]."
        )
    return bulk_weight


def _resolve_sample_diversity_gene_mode(gene_mode_value: Any, context: str) -> str:
    gene_mode = str(gene_mode_value).strip().lower()
    allowed = {"all", "hvg", "provided_gene_list"}
    if gene_mode not in allowed:
        raise ValueError(
            f'Invalid "sample_diversity_gene_mode" for {context}: '
            f"{gene_mode_value!r}. Expected one of {sorted(allowed)}."
        )
    return gene_mode


def _resolve_sample_diversity_distance_metric(metric_value: Any, context: str) -> str:
    metric = str(metric_value).strip().lower()
    allowed = {"cosine", "l2"}
    if metric not in allowed:
        raise ValueError(
            f'Invalid "sample_diversity_distance_metric" for {context}: '
            f"{metric_value!r}. Expected one of {sorted(allowed)}."
        )
    return metric


# --- 1. Configuration Management ---
@dataclass
class SimuConfig:
    """Stores all simulation-related paths and global parameters."""
    # Path configuration
    simu_bulk_dir: Path
    tcga2cancer_type_file_path: Path
    tcga_merged_tpm_file_path: Path
    high_corr_gene_file_path: Path | None
    merged_sc_dataset_file_path: Path

    # Core constants
    sc_dataset_ids: List[str]
    cell_type2subtypes: Dict[str, List[str]]
    total_rna_coefficient: Dict[str, float]
    n_base: int
    output_types: List[str]

    # Filtering parameters
    gep_filtering_quantile: Tuple[float, float]
    gene_quantile_range: List[float] = field(default_factory=lambda: [0.005, 0.5, 0.995])
    pca_n_components: float = 0.9
    gene_list_type: str = "high_corr_gene"
    gep_filtering_enabled_default: bool = False
    gep_filtering_reference_file: Path | None = None
    gep_filtering_ref_exp_type: str | None = None
    gep_filtering_method: str = "median_gep"
    gep_n_top: int = 20
    gep_show_filtering_info: bool = False
    gep_filtering_by_gene_range: bool = False
    gep_min_percentage_within_gene_range: float = 0.95
    gep_gene_quantile_range: List[float] | None = None
    gep_filtering_in_pca_space: bool = True
    gep_pca_n_components: int | float | None = None
    gep_norm_ord: int = 1
    gep_reference_holdout_fraction: float = 0.2
    gep_reference_holdout_seed: int = 42
    gep_knn_n_neighbors: int = 10
    gep_knn_distance_quantile: float = 0.95
    gep_pca_residual_quantile: float = 0.95
    gep_max_generation_batches: int = 100
    gene_filtering_enabled_default: bool = False
    gene_filtering_reference_file: Path | None = None
    gene_filtering_type: str | None = None
    gene_filtering_quantile_range: List[float] | None = None
    gene_filtering_q_col_name: List[str] | None = None
    gene_filtering_corr_threshold: float = 0.3
    gene_filtering_reference_variance_threshold: float | None = None
    gene_filtering_n_gene_max: int = 1000
    gene_filtering_save_filtered_h5ad: bool = True
    gene_filtering_filtered_dataset_postfix: str | None = None
    calculate_gene_cell_prop_correlation: bool = False
    gene_filtering_correlation_reference_dataset: str | None = None
    plot_pca_for_comparison: bool | None = None
    gene_filtering_plot_pca: bool | None = None  # Deprecated alias for plot_pca_for_comparison
    gene_filtering_pca_n_components: int = 2
    gene_filtering_pca_figsize: Tuple[float, float] = (5.0, 5.0)
    sorted_bulk_expr_file_path: Path | None = None
    sorted_bulk_annot_file_path: Path | None = None
    sorted_bulk_cell_type_mapping_file_path: Path | None = None
    sorted_bulk_cell_type_col: str = "Tumor_model_annot"
    sorted_bulk_imputation_eps: float = 1e-8
    sorted_bulk_imputation_seed: int = 123
    dirichlet_sampling_seed: int | None = None
    sorted_bulk_imputation_batch_size: int = 128
    sorted_bulk_imputation_boxplot_sample_size: int = 1000
    sorted_bulk_imputation_y_frac_threshold: float = 0.15
    sorted_bulk_imputation_bulk_weight: float = 0.5
    sorted_bulk_imputation_output_suffix: str = "_sorted_bulk_imputed"
    run_sample_diversity_analysis: bool = True
    sample_diversity_gene_mode: Literal["all", "hvg", "provided_gene_list"] = "all"
    sample_diversity_n_top_genes: int = 2000
    sample_diversity_gene_list_path: Path | None = None
    sample_diversity_distance_metric: Literal["cosine", "l2"] = "cosine"
    sample_diversity_min_samples_per_cell_type: int = 5
    stratify_by_dataset: Literal["none", "balanced_quota"] = "none"
    mixed_sct_loading_mode: Literal["auto", "eager", "lazy"] = "auto"
    mixed_sct_auto_eager_max_gb: float = 4.0

    # Cached attributes
    _high_corr_gene_list: List[str] = field(init=False, default=None)
    _dataset_definitions: Dict[str, Dict[str, Any]] = field(init=False, default_factory=dict)
    _dataset_order: List[str] = field(init=False, default_factory=list)
    _generated_sct_outputs: Dict[str, Dict[str, Path | None]] = field(init=False, default_factory=dict)
    _generated_mixed_outputs: Dict[str, Dict[str, Path | None]] = field(init=False, default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        if "sct_dataset_file_path" in data.get("paths", {}):
            raise ValueError(
                'Top-level "paths.sct_dataset_file_path" is no longer supported. '
                'Please define Mixed dataset inputs with the explicit "sct_reference" block.'
            )

        # Flatten the YAML structure to match the Dataclass fields
        # Combining paths, constants, and parameters into one dictionary
        config_data = {
            **data['paths'],
            **data['constants'],
            **data['parameters']
        }
        if "sorted_bulk_imputation_out_dir" in config_data:
            logging.warning(
                'Ignoring deprecated "sorted_bulk_imputation_out_dir"; '
                "auxiliary SCT outputs are now stored automatically beside the SCT dataset."
            )
            config_data.pop("sorted_bulk_imputation_out_dir", None)

        # Convert list to tuple for gep_filtering_quantile if necessary
        if 'gep_filtering_quantile' in config_data:
            config_data['gep_filtering_quantile'] = tuple(config_data['gep_filtering_quantile'])

        config = cls(**config_data)
        config._dataset_definitions = data.get("datasets", {})
        config._dataset_order = list(config._dataset_definitions.keys())
        return config

    def __post_init__(self):
        if self.gep_pca_n_components in (None, ""):
            self.gep_pca_n_components = self.pca_n_components
        if self.plot_pca_for_comparison is None:
            self.plot_pca_for_comparison = (
                True if self.gene_filtering_plot_pca is None else bool(self.gene_filtering_plot_pca)
            )
        if self.gene_filtering_plot_pca is None:
            self.gene_filtering_plot_pca = bool(self.plot_pca_for_comparison)
        if self.gene_filtering_type in (None, ""):
            self.gene_filtering_type = self.gene_list_type
        if self.gene_filtering_quantile_range in (None, []):
            self.gene_filtering_quantile_range = self.gene_quantile_range
        self.sorted_bulk_imputation_bulk_weight = _resolve_sorted_bulk_imputation_bulk_weight(
            self.sorted_bulk_imputation_bulk_weight,
            "global parameters",
        )
        if isinstance(self.gene_filtering_pca_figsize, list):
            self.gene_filtering_pca_figsize = tuple(float(v) for v in self.gene_filtering_pca_figsize)
        if len(self.gene_filtering_pca_figsize) != 2:
            raise ValueError('"gene_filtering_pca_figsize" must contain exactly two values.')
        self.sample_diversity_gene_mode = _resolve_sample_diversity_gene_mode(
            self.sample_diversity_gene_mode,
            "global parameters",
        )
        self.sample_diversity_distance_metric = _resolve_sample_diversity_distance_metric(
            self.sample_diversity_distance_metric,
            "global parameters",
        )
        self.mixed_sct_loading_mode = _resolve_mixed_sct_loading_mode(
            self.mixed_sct_loading_mode,
            "global parameters",
        )
        self.mixed_sct_auto_eager_max_gb = _resolve_positive_float(
            self.mixed_sct_auto_eager_max_gb,
            "mixed_sct_auto_eager_max_gb",
            "global parameters",
        )
        self.sample_diversity_n_top_genes = int(self.sample_diversity_n_top_genes)
        if self.sample_diversity_n_top_genes <= 0:
            raise ValueError('"sample_diversity_n_top_genes" must be positive.')
        self.sample_diversity_min_samples_per_cell_type = int(self.sample_diversity_min_samples_per_cell_type)
        if self.sample_diversity_min_samples_per_cell_type < 2:
            raise ValueError('"sample_diversity_min_samples_per_cell_type" must be at least 2.')
        if self.sample_diversity_gene_list_path not in (None, ""):
            self.sample_diversity_gene_list_path = Path(self.sample_diversity_gene_list_path)
            if not self.sample_diversity_gene_list_path.exists():
                raise FileNotFoundError(
                    f'"sample_diversity_gene_list_path" does not exist: {self.sample_diversity_gene_list_path}'
                )
        else:
            self.sample_diversity_gene_list_path = None

    @property
    def high_corr_genes(self) -> List[str]:
        """Lazy load high correlation genes"""
        if self._high_corr_gene_list is None:
            if self.high_corr_gene_file_path and os.path.exists(self.high_corr_gene_file_path):
                df = pd.read_csv(self.high_corr_gene_file_path)
                self._high_corr_gene_list = df['gene_name'].to_list()
            else:
                logging.warning(f"Warning: High corr gene file not found: {self.high_corr_gene_file_path}")
                self._high_corr_gene_list = []
        return self._high_corr_gene_list


@dataclass(frozen=True)
class ResolvedGEPFilteringParams:
    enabled: bool
    reference_file: str | None
    ref_exp_type: str | None
    filtering_method: str
    filtering_ref_types: List[str] | None
    gep_filtering_quantile: Tuple[float | None, float | None]
    n_top: int
    show_filtering_info: bool
    filtering_by_gene_range: bool
    min_percentage_within_gene_range: float
    gene_quantile_range: List[float] | None
    filtering_in_pca_space: bool
    pca_n_components: int | float
    norm_ord: int
    high_corr_gene_list_file: str | None
    high_corr_gene_list: List[str] | None
    reference_holdout_fraction: float = 0.2
    reference_holdout_seed: int = 42
    knn_n_neighbors: int = 10
    knn_distance_quantile: float = 0.95
    pca_residual_quantile: float = 0.95
    max_generation_batches: int = 100


@dataclass(frozen=True)
class ResolvedGeneFilteringParams:
    enabled: bool
    reference_file: str | None
    filtering_type: str
    quantile_range: List[float] | None
    q_col_name: List[str] | None
    corr_threshold: float
    n_gene_max: int
    high_corr_gene_file: str | None
    save_filtered_h5ad: bool
    filtered_dataset_postfix: str | None
    plot_pca: bool
    pca_n_components: int
    pca_figsize: Tuple[float, float]
    correlation_reference_dataset: str | None = None
    reference_variance_threshold: float | None = None


def _resolve_optional_path_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _resolve_dataset_override(
    params: Dict[str, Any],
    config: SimuConfig,
    field_name: str,
    default: Any = None,
) -> Any:
    if field_name in params:
        return params[field_name]
    if hasattr(config, field_name):
        return getattr(config, field_name)
    return default


def _read_gene_list_file(file_path: str | Path) -> List[str]:
    gene_df = pd.read_csv(file_path)
    if "gene_name" in gene_df.columns:
        return gene_df["gene_name"].dropna().astype(str).tolist()
    if gene_df.shape[1] == 1:
        return gene_df.iloc[:, 0].dropna().astype(str).tolist()
    raise ValueError(f"Gene list file must contain a gene_name column: {file_path}")


def _resolve_reusable_high_corr_gene_file(
    file_path: str | Path | None,
    context: str,
) -> str | None:
    normalized = _resolve_optional_path_value(file_path)
    if normalized is None:
        return None
    if os.path.exists(normalized):
        return normalized
    logging.warning(
        'High-correlation gene file not found for %s: %s. '
        'Falling back to on-the-fly generation when a filtering step needs it.',
        context,
        normalized,
    )
    return None


def _resolve_quantile_triplet(values: Any, field_name: str) -> List[float] | None:
    if values in (None, ""):
        return None
    if isinstance(values, str):
        raise ValueError(f'"{field_name}" must be a 3-value list, not a string.')
    resolved = [float(value) for value in list(values)]
    if len(resolved) != 3:
        raise ValueError(f'"{field_name}" must contain exactly three values.')
    return resolved


def _format_quantile_label(value: float) -> str:
    return f"{int(value * 1000) / 10:g}"


def _resolve_quantile_col_names(q_col_name: Any, quantile_range: List[float] | None, field_name: str) -> List[str] | None:
    if q_col_name in (None, ""):
        if quantile_range is None:
            return None
        return [f"q_{_format_quantile_label(value)}" for value in quantile_range]
    if isinstance(q_col_name, str):
        resolved = [q_col_name]
    else:
        resolved = [str(value) for value in list(q_col_name)]
    if quantile_range is not None and len(resolved) != len(quantile_range):
        raise ValueError(f'"{field_name}" must have the same length as the quantile range.')
    return resolved


def _parse_pca_component_value(value: Any) -> int | float:
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    return float(text) if "." in text else int(text)


def _resolve_mixed_sct_loading_mode(value: Any, context: str) -> Literal["auto", "eager", "lazy"]:
    resolved = str(value).strip().lower()
    allowed = {"auto", "eager", "lazy"}
    if resolved not in allowed:
        raise ValueError(
            f'Invalid "mixed_sct_loading_mode" for {context}: {value!r}. '
            f'Expected one of {sorted(allowed)}.'
        )
    return resolved


def _resolve_positive_float(value: Any, field_name: str, context: str) -> float:
    resolved = float(value)
    if resolved <= 0:
        raise ValueError(f'Invalid "{field_name}" for {context}: {value!r}. Expected a positive number.')
    return resolved


def _resolve_filtering_ref_types(
    ref_types_value: Any,
    config: SimuConfig,
    context: str,
) -> List[str] | None:
    if ref_types_value in (None, ""):
        return None
    if isinstance(ref_types_value, str):
        resolved_input = [ref_types_value]
    else:
        resolved_input = list(ref_types_value)
    resolved = [str(value).strip() for value in resolved_input if str(value).strip()]
    if not resolved:
        return None
    if len(resolved) == 1 and resolved[0].lower() == "all":
        if not config.tcga2cancer_type_file_path:
            raise ValueError(
                f'"filtering_ref_types" requires "tcga2cancer_type_file_path" for {context}.'
            )
        tcga_df = pd.read_csv(config.tcga2cancer_type_file_path, index_col=0)
        if "cancer_type" not in tcga_df.columns:
            raise ValueError(
                f'"tcga2cancer_type_file_path" must contain a "cancer_type" column for {context}.'
            )
        return sorted(tcga_df["cancer_type"].dropna().astype(str).unique().tolist())
    return resolved


def _resolve_gep_filtering_params(
    dataset_name: str,
    params: Dict[str, Any],
    config: SimuConfig,
) -> ResolvedGEPFilteringParams:
    enabled = bool(_resolve_dataset_override(params, config, "gep_filtering", config.gep_filtering_enabled_default))
    reference_file = _resolve_optional_path_value(
        _resolve_dataset_override(params, config, "gep_filtering_reference_file", config.gep_filtering_reference_file)
    ) or str(config.tcga_merged_tpm_file_path)
    ref_exp_type = _resolve_dataset_override(params, config, "gep_filtering_ref_exp_type", config.gep_filtering_ref_exp_type) or "TPM"
    filtering_ref_types = _resolve_filtering_ref_types(
        params.get("filtering_ref_types", None),
        config=config,
        context=f'Mixed dataset "{dataset_name}"',
    )
    quantile = _resolve_dataset_override(params, config, "gep_filtering_quantile", config.gep_filtering_quantile)
    if quantile in (None, ""):
        quantile = (None, 0.95)
    quantile = tuple(
        None if value in (None, "", "null") else float(value)
        for value in list(quantile)
    )
    if len(quantile) != 2:
        raise ValueError('"gep_filtering_quantile" must contain exactly two values.')
    gene_quantile_range = _resolve_quantile_triplet(
        params.get(
            "gep_gene_quantile_range",
            params.get(
                "gene_quantile_range",
                config.gep_gene_quantile_range if config.gep_gene_quantile_range is not None else config.gene_quantile_range,
            ),
        ),
        "gep_gene_quantile_range",
    )
    high_corr_gene_list_file = _resolve_reusable_high_corr_gene_file(
        params.get("gep_high_corr_gene_list_file"),
        f'Mixed dataset "{dataset_name}" GEP filtering override',
    )
    if high_corr_gene_list_file is None and config.high_corr_gene_file_path not in (None, ""):
        high_corr_gene_list_file = _resolve_reusable_high_corr_gene_file(
            config.high_corr_gene_file_path,
            f'Mixed dataset "{dataset_name}" shared GEP filtering fallback',
        )
    high_corr_gene_list = None
    if high_corr_gene_list_file:
        high_corr_gene_list = _read_gene_list_file(high_corr_gene_list_file)
    filtering_method = str(_resolve_dataset_override(
        params, config, "gep_filtering_method", config.gep_filtering_method))
    norm_ord = _resolve_dataset_override(params, config, "gep_norm_ord", config.gep_norm_ord)
    filtering_in_pca_space = _resolve_dataset_override(
        params, config, "gep_filtering_in_pca_space", config.gep_filtering_in_pca_space
    )
    if "gep_filtering_in_pca_space" not in params:
        filtering_in_pca_space = params.get(
            "filtering_in_pca_space", filtering_in_pca_space
        )
    resolved = ResolvedGEPFilteringParams(
        enabled=enabled,
        reference_file=reference_file,
        ref_exp_type=str(ref_exp_type) if ref_exp_type is not None else None,
        filtering_method=filtering_method,
        filtering_ref_types=filtering_ref_types,
        gep_filtering_quantile=quantile,
        n_top=int(_resolve_dataset_override(params, config, "gep_n_top", config.gep_n_top if hasattr(config, "gep_n_top") else 20)),
        show_filtering_info=bool(_resolve_dataset_override(params, config, "gep_show_filtering_info", config.gep_show_filtering_info)),
        filtering_by_gene_range=bool(
            _resolve_dataset_override(params, config, "gep_filtering_by_gene_range", config.gep_filtering_by_gene_range)
        ),
        min_percentage_within_gene_range=float(
            _resolve_dataset_override(
                params,
                config,
                "gep_min_percentage_within_gene_range",
                config.gep_min_percentage_within_gene_range,
            )
        ),
        gene_quantile_range=gene_quantile_range,
        filtering_in_pca_space=bool(filtering_in_pca_space),
        pca_n_components=_parse_pca_component_value(
            _resolve_dataset_override(params, config, "gep_pca_n_components", config.gep_pca_n_components)
        ),
        norm_ord=norm_ord if filtering_method in LOCAL_FILTER_METHODS else int(norm_ord),
        high_corr_gene_list_file=high_corr_gene_list_file,
        high_corr_gene_list=high_corr_gene_list,
        reference_holdout_fraction=float(_resolve_dataset_override(
            params, config, "gep_reference_holdout_fraction", 0.2)),
        reference_holdout_seed=_resolve_dataset_override(
            params, config, "gep_reference_holdout_seed", 42),
        knn_n_neighbors=_resolve_dataset_override(params, config, "gep_knn_n_neighbors", 10),
        knn_distance_quantile=float(_resolve_dataset_override(
            params, config, "gep_knn_distance_quantile", 0.95)),
        pca_residual_quantile=float(_resolve_dataset_override(
            params, config, "gep_pca_residual_quantile", 0.95)),
        max_generation_batches=_resolve_dataset_override(params, config, "gep_max_generation_batches", 100),
    )
    if resolved.enabled:
        if resolved.filtering_method in LOCAL_FILTER_METHODS:
            validate_local_parameters(resolved)
            if not config.tcga2cancer_type_file_path:
                raise ValueError(f"{resolved.filtering_method} requires paths.tcga2cancer_type_file_path.")
        if not resolved.reference_file:
            raise ValueError(
                f'"gep_filtering_reference_file" must be set when gep_filtering is enabled for Mixed dataset "{dataset_name}".'
            )
        if not resolved.ref_exp_type:
            raise ValueError(
                f'"gep_filtering_ref_exp_type" must be set when gep_filtering is enabled for Mixed dataset "{dataset_name}".'
            )
        if resolved.filtering_ref_types and not config.tcga2cancer_type_file_path:
            raise ValueError(
                f'"filtering_ref_types" requires "tcga2cancer_type_file_path" for Mixed dataset "{dataset_name}".'
            )
    return resolved


def _resolve_gene_filtering_params(
    dataset_name: str,
    params: Dict[str, Any],
    config: SimuConfig,
) -> ResolvedGeneFilteringParams:
    enabled = bool(_resolve_dataset_override(params, config, "gene_filtering", config.gene_filtering_enabled_default))
    quantile_range = _resolve_quantile_triplet(
        _resolve_dataset_override(params, config, "gene_filtering_quantile_range", config.gene_filtering_quantile_range),
        "gene_filtering_quantile_range",
    )
    filtering_type = str(
        _resolve_dataset_override(params, config, "gene_filtering_type")
        or config.gene_list_type
    )
    if quantile_range is None and "quantile_range" in filtering_type:
        quantile_range = _resolve_quantile_triplet(config.gene_quantile_range, "gene_quantile_range")
    q_col_name = _resolve_quantile_col_names(
        _resolve_dataset_override(params, config, "gene_filtering_q_col_name", config.gene_filtering_q_col_name),
        quantile_range,
        "gene_filtering_q_col_name",
    )
    high_corr_gene_file = _resolve_reusable_high_corr_gene_file(
        params.get("gene_filtering_high_corr_gene_file"),
        f'Mixed dataset "{dataset_name}" gene filtering override',
    )
    if high_corr_gene_file is None and config.high_corr_gene_file_path not in (None, ""):
        high_corr_gene_file = _resolve_reusable_high_corr_gene_file(
            config.high_corr_gene_file_path,
            f'Mixed dataset "{dataset_name}" shared gene filtering fallback',
        )
    if enabled and high_corr_gene_file is None and filtering_type != "high_corr_gene":
        raise ValueError(
            f'Mixed dataset "{dataset_name}" uses legacy gene_filtering_type={filtering_type!r}. '
            'Migrate to gene_filtering_type: high_corr_gene and remove '
            'gene_filtering_quantile_range and gene_filtering_q_col_name; '
            'use gep_gene_quantile_range for profile acceptance.'
        )
    correlation_reference_dataset = None
    if enabled and high_corr_gene_file is None and "high_corr_gene" in filtering_type:
        if "gene_filtering_correlation_reference_dataset" in params:
            correlation_reference_dataset = params["gene_filtering_correlation_reference_dataset"]
        else:
            correlation_reference_dataset = getattr(
                config,
                "gene_filtering_correlation_reference_dataset",
                None,
            )
    if correlation_reference_dataset not in (None, ""):
        correlation_reference_dataset = str(correlation_reference_dataset).strip() or None
    if correlation_reference_dataset == dataset_name:
        if enabled:
            raise ValueError(
                f'Mixed dataset "{dataset_name}" cannot use itself as a gene-correlation reference.'
            )
        correlation_reference_dataset = None
    if enabled and high_corr_gene_file is None and not correlation_reference_dataset:
        raise ValueError(
            f'Mixed dataset "{dataset_name}" requires '
            '"gene_filtering_correlation_reference_dataset" or a supplied gene list '
            'before generation. No implicit pilot or target-derived correlations are allowed.'
        )
    if correlation_reference_dataset:
        dataset_definitions = getattr(config, "_dataset_definitions", {})
        dataset_order = getattr(config, "_dataset_order", [])
        source_params = dataset_definitions.get(correlation_reference_dataset)
        if source_params is None:
            raise ValueError(
                f'Mixed dataset "{dataset_name}" references unknown '
                f'gene-correlation source dataset "{correlation_reference_dataset}".'
            )
        if dataset_name not in dataset_order or correlation_reference_dataset not in dataset_order:
            raise ValueError(
                f'Mixed dataset "{dataset_name}" and its gene-correlation source must be loaded '
                'through SimuConfig.from_yaml so their YAML order is available.'
            )
        if dataset_order.index(correlation_reference_dataset) >= dataset_order.index(dataset_name):
            raise ValueError(
                f'Gene-correlation source dataset "{correlation_reference_dataset}" must appear '
                f'before Mixed dataset "{dataset_name}" in the configuration.'
            )
        source_type = str(source_params.get("dataset_type", "")).strip().lower()
        if source_type != "mixed":
            raise ValueError(
                f'Gene-correlation source dataset "{correlation_reference_dataset}" must have '
                '"dataset_type: Mixed".'
            )
        source_sampling_method = str(source_params.get("sampling_method_prop", "segment")).strip().lower()
        if source_sampling_method != "segment":
            raise ValueError(
                f'Gene-correlation source dataset "{correlation_reference_dataset}" must use '
                '"sampling_method_prop: segment".'
            )
        source_gep_filtering = bool(
            source_params.get("gep_filtering", config.gep_filtering_enabled_default)
        )
        if source_gep_filtering:
            raise ValueError(
                f'Gene-correlation source dataset "{correlation_reference_dataset}" must set '
                '"gep_filtering: false".'
            )
        source_gene_filtering = bool(
            source_params.get("gene_filtering", config.gene_filtering_enabled_default)
        )
        if source_gene_filtering:
            raise ValueError(
                f'Gene-correlation source dataset "{correlation_reference_dataset}" must set '
                '"gene_filtering: false".'
            )

    plot_pca = params.get("plot_pca_for_comparison")
    if plot_pca is None:
        plot_pca = params.get("gene_filtering_plot_pca")
    if plot_pca is None:
        plot_pca = getattr(config, "plot_pca_for_comparison", None)
    if plot_pca is None:
        plot_pca = getattr(config, "gene_filtering_plot_pca", True)

    variance_threshold = _resolve_dataset_override(
        params, config, "gene_filtering_reference_variance_threshold", None)
    if variance_threshold is not None:
        if (isinstance(variance_threshold, bool)
                or not isinstance(variance_threshold, (int, float))
                or not np.isfinite(variance_threshold) or variance_threshold < 0):
            raise ValueError("gene_filtering_reference_variance_threshold must be finite and >= 0, or null.")
        variance_threshold = float(variance_threshold)
    resolved = ResolvedGeneFilteringParams(
        enabled=enabled,
        reference_file=_resolve_optional_path_value(
            _resolve_dataset_override(params, config, "gene_filtering_reference_file", config.gene_filtering_reference_file)
        ) or str(config.tcga_merged_tpm_file_path),
        filtering_type=filtering_type,
        quantile_range=quantile_range,
        q_col_name=q_col_name,
        corr_threshold=float(_resolve_dataset_override(params, config, "gene_filtering_corr_threshold", config.gene_filtering_corr_threshold)),
        n_gene_max=int(_resolve_dataset_override(params, config, "gene_filtering_n_gene_max", config.gene_filtering_n_gene_max)),
        high_corr_gene_file=high_corr_gene_file,
        save_filtered_h5ad=bool(
            _resolve_dataset_override(params, config, "gene_filtering_save_filtered_h5ad", config.gene_filtering_save_filtered_h5ad)
        ),
        filtered_dataset_postfix=_resolve_dataset_override(
            params,
            config,
            "gene_filtering_filtered_dataset_postfix",
            config.gene_filtering_filtered_dataset_postfix,
        ),
        plot_pca=bool(plot_pca),
        pca_n_components=int(
            _resolve_dataset_override(params, config, "gene_filtering_pca_n_components", config.gene_filtering_pca_n_components)
        ),
        pca_figsize=tuple(
            float(v)
            for v in _resolve_dataset_override(params, config, "gene_filtering_pca_figsize", config.gene_filtering_pca_figsize)
        ),
        correlation_reference_dataset=correlation_reference_dataset or None,
        reference_variance_threshold=variance_threshold,
    )
    if len(resolved.pca_figsize) != 2:
        raise ValueError('"gene_filtering_pca_figsize" must contain exactly two values.')
    if resolved.enabled and not resolved.reference_file:
        raise ValueError(
            f'"gene_filtering_reference_file" must be set when gene filtering is enabled for Mixed dataset "{dataset_name}".'
        )
    if resolved.plot_pca and not resolved.reference_file:
        raise ValueError(
            f'"gene_filtering_reference_file" must be set when PCA comparison is enabled for Mixed dataset "{dataset_name}".'
        )
    if resolved.enabled:
        if "quantile_range" in resolved.filtering_type and resolved.quantile_range is None:
            raise ValueError(
                f'"gene_filtering_quantile_range" must be set when the filtering type uses quantile_range for Mixed dataset "{dataset_name}".'
            )
        if resolved.q_col_name is not None and resolved.quantile_range is not None:
            if len(resolved.q_col_name) != len(resolved.quantile_range):
                raise ValueError(
                    f'"gene_filtering_q_col_name" must match the length of the quantile range for Mixed dataset "{dataset_name}".'
                )
    if resolved.plot_pca and resolved.pca_n_components < 2:
        raise ValueError(
            f'"gene_filtering_pca_n_components" must be >= 2 when plot_pca_for_comparison is enabled for Mixed dataset "{dataset_name}".'
        )
    return resolved


def _resolve_stratify_by_dataset(
    params: Dict[str, Any],
    config: SimuConfig,
) -> str:
    allowed = {"none", "balanced_quota"}
    override = params.get("stratify_by_dataset", None)
    if override is not None:
        value = str(override).strip()
        if value not in allowed:
            raise ValueError(
                f'Invalid params.stratify_by_dataset override: {override!r}. '
                f'Expected one of {sorted(allowed)}.'
            )
        return value
    value = str(getattr(config, "stratify_by_dataset", "none")).strip()
    if value not in allowed:
        raise ValueError(
            f'Invalid parameters.stratify_by_dataset: {value!r}. '
            f'Expected one of {sorted(allowed)}.'
        )
    return value


def _resolve_neighbor_sampling_scheme(params: Dict[str, Any]) -> str:
    allowed = {"single_anchor", "two_half_anchor"}
    value = str(params.get("neighbor_sampling_scheme", "single_anchor")).strip()
    if value not in allowed:
        raise ValueError(
            f'Invalid SCT "neighbor_sampling_scheme": {value!r}. '
            f'Expected one of {sorted(allowed)}.'
        )
    return value


def _resolve_selected_subtypes_by_cell_type(
    params: Dict[str, Any],
    config: SimuConfig,
) -> Dict[str, List[str] | str] | None:
    value = params.get("selected_subtypes_by_cell_type", None)
    if value in (None, ""):
        return None
    if not isinstance(value, dict):
        raise ValueError('"selected_subtypes_by_cell_type" must be a mapping from parent cell type to subtype list.')

    normalized: Dict[str, List[str]] = {}
    for parent_cell_type, subtypes in value.items():
        parent_name = str(parent_cell_type).strip()
        if parent_name == "":
            raise ValueError('Invalid empty parent cell type in "selected_subtypes_by_cell_type".')
        if parent_name not in config.cell_type2subtypes:
            raise ValueError(
                f'Unknown parent cell type in "selected_subtypes_by_cell_type": {parent_name!r}. '
                f'Available parent cell types: {sorted(config.cell_type2subtypes)}'
            )
        if isinstance(subtypes, str) and subtypes.strip().lower() == "auto":
            normalized[parent_name] = "auto"
            continue
        if isinstance(subtypes, str) or subtypes in (None, ""):
            raise ValueError(
                f'"selected_subtypes_by_cell_type[{parent_name}]" must be a non-empty list of subtype names.'
            )
        cleaned_subtypes = [str(subtype).strip() for subtype in list(subtypes)]
        if any(subtype == "" for subtype in cleaned_subtypes):
            raise ValueError(
                f'Invalid empty subtype name found in "selected_subtypes_by_cell_type[{parent_name}]".'
            )
        deduped_subtypes = list(dict.fromkeys(cleaned_subtypes))
        if not deduped_subtypes:
            raise ValueError(
                f'"selected_subtypes_by_cell_type[{parent_name}]" must contain at least one subtype.'
            )
        allowed_subtypes = set(config.cell_type2subtypes[parent_name])
        invalid_subtypes = [subtype for subtype in deduped_subtypes if subtype not in allowed_subtypes]
        if invalid_subtypes:
            raise ValueError(
                f'Invalid subtype(s) for parent cell type {parent_name!r}: {invalid_subtypes}. '
                f'Allowed subtypes: {config.cell_type2subtypes[parent_name]}'
            )
        normalized[parent_name] = deduped_subtypes
    return normalized


# --- 2. Main Workflow Function ---
def run_bulk_simulation(dataset_name: str,
                        params: Dict[str, Any],
                        config: SimuConfig,
                        log_file_path: str = None) -> str:
    """
    Executes the complete simulation workflow for a single dataset: Generation -> Filtering -> PCA Analysis.

    :param dataset_name: Name of the dataset (e.g., 'Mixed_N10K_segment')
    :param params: Dataset-specific parameters (sc_dataset_ids, n_samples, etc.)
    :param config: Global configuration object (paths, constants)
    :param log_file_path: Path to the log file
    :return: The file path of the final generated .h5ad file
    """
    logging.info(f"\n=== Starting Simulation for {dataset_name} ===")
    dataset_type = params.get('dataset_type', 'mixed').lower()
    dirichlet_only_keys = {
        "dirichlet_alpha",
        "dirichlet_sampling_seed",
    }
    if dataset_type != 'sct':
        if 'n_base' in params:
            raise ValueError(
                f'"n_base" is only supported for SCT datasets; '
                f'Mixed dataset "{dataset_name}" must not define it.'
            )
        if 'cancer_cell_dataset_ids' in params or 'non_cancer_cell_dataset_ids' in params:
            raise ValueError(
                '"cancer_cell_dataset_ids" and "non_cancer_cell_dataset_ids" are only supported for SCT datasets.'
            )
        if 'filtering' in params:
            raise ValueError(
                '"filtering" is no longer supported for Mixed datasets. '
                'Please rename it to "gep_filtering".'
            )
        if 'stratify_by_dataset' in params:
            raise ValueError(
                '"stratify_by_dataset" override is only supported for SCT datasets. Use the global parameters.stratify_by_dataset for default behavior; Mixed datasets inherit their pool from an upstream SCT dataset via sct_reference.'
            )
        if 'selected_subtypes_by_cell_type' in params:
            raise ValueError(
                '"selected_subtypes_by_cell_type" is only supported for SCT datasets.'
            )
        if 'neighbor_sampling_scheme' in params:
            raise ValueError(
                '"neighbor_sampling_scheme" is only supported for SCT datasets.'
            )
    else:
        if 'sct_dataset_version' in params:
            raise ValueError(
                '"sct_dataset_version" is no longer supported. Use "sct_reference" for Mixed datasets.'
            )
        if 'sct_reference' in params:
            raise ValueError(
                '"sct_reference" is only supported for Mixed datasets.'
            )
        invalid_keys = [
            key for key in dirichlet_only_keys | {"pred_cell_prop_file_path"}
            if params.get(key, None) not in (None, "")
        ]
        if invalid_keys:
            raise ValueError(
                f'{invalid_keys} are only supported for Mixed datasets with '
                '"sampling_method_prop: dirichlet".'
            )

    # 1. Initialize generator
    simu_bulk_dir = Path(config.simu_bulk_dir)
    n_sc_datasets, n_base = _resolve_output_dir_naming_metadata(
        dataset_name=dataset_name,
        params=params,
        config=config,
        dataset_type=dataset_type,
    )
    resolved_stratify = _resolve_stratify_by_dataset(params=params, config=config)
    if dataset_type == 'sct':
        logging.info(f"   > stratify_by_dataset: {resolved_stratify}")
    if dataset_type == 'sct':  # single-cell-type dataset
        logging.info(f"   > Using Single-Cell-Type GEP Generation for {dataset_name}")
        resolved_selected_subtypes = _resolve_selected_subtypes_by_cell_type(
            params=params,
            config=config,
        )
        if resolved_selected_subtypes:
            logging.info(f"   > selected_subtypes_by_cell_type: {resolved_selected_subtypes}")
        if 'sampling_method_gep' not in params:
            raise ValueError(
                f'Missing "sampling_method_gep" for SCT dataset "{dataset_name}". '
                'Please set params["sampling_method_gep"] explicitly to "random" or "n_neighbors".'
            )
        sampling_method_gep = params.get('sampling_method_gep')
        neighbor_sampling_scheme = _resolve_neighbor_sampling_scheme(params)
        max_cells_per_cell_type = validate_cell_cap(
            params.get("max_cells_per_cell_type", 15000),
            sampling_method_gep, neighbor_sampling_scheme,
        )
        logging.info(f"   > neighbor_sampling_scheme: {neighbor_sampling_scheme}")

        sc_dataset_dir = simu_bulk_dir / f'generated_sct_dataset_{n_sc_datasets}ds_n_base{n_base}_all_subtypes'
        sct_generator = SingleCellTypeGEPGenerator(
            simu_bulk_dir=sc_dataset_dir,
            cell_type2subtype=config.cell_type2subtypes,
            sc_dataset_ids=config.sc_dataset_ids,
            cancer_cell_dataset_ids=params.get('cancer_cell_dataset_ids', None),
            non_cancer_cell_dataset_ids=params.get('non_cancer_cell_dataset_ids', None),
            stratify_by_dataset=resolved_stratify,
            selected_subtypes_by_cell_type=resolved_selected_subtypes,
            stratify_seed=config.sorted_bulk_imputation_seed,
            bulk_dataset_name=dataset_name,
            merged_sc_dataset_file_path=config.merged_sc_dataset_file_path,
            zero_ratio_threshold=params.get('zero_ratio_threshold', 0.97),
            cell_type_col_name=params.get('cell_type_col', 'cell_type'),
            subtype_col_name=params.get('cell_subtype_col', 'cell_subtype'),
            output_types=config.output_types,
            sampling_method_gep=sampling_method_gep,
        )
        generated_sct_fp = sct_generator.generated_bulk_gep_fp
        if not generated_sct_fp.exists():
            sct_generator.generate_samples(n_sample_each_cell_type=params['n_samples'],
                                           sample_type='positive',
                                           simu_method='ave',  # average multiple cells to get one sct GEP
                                           sep_by_patient=False,
                                           n_base_for_positive_samples=n_base,
                                           sampling_method=sampling_method_gep, # sampling method for finding single cells
                                           neighbor_sampling_scheme=neighbor_sampling_scheme,
                                           max_cells_per_cell_type=max_cells_per_cell_type
                                           )
            logging.info(f"   > Single-Cell-Type Dataset Generation Completed for {dataset_name}")
        else:
            logging.info(f"   > Single-Cell-Type GEP file exists: {generated_sct_fp}")
        current_sct_output_path = generated_sct_fp
        imputed_sct_fp = None
        if params.get('apply_sorted_bulk_imputation', False):
            imputed_sct_fp = _run_sorted_bulk_imputation_for_sct_dataset(
                dataset_name=dataset_name,
                generated_sct_fp=generated_sct_fp,
                params=params,
                config=config,
            )
            current_sct_output_path = imputed_sct_fp
            logging.info("   > Recorded imputed SCT dataset for %s: %s", dataset_name, imputed_sct_fp)
        config._generated_sct_outputs[dataset_name] = {
            "original": generated_sct_fp,
            "sorted_bulk_imputed": imputed_sct_fp,
        }
        _run_sample_diversity_for_sct_dataset(
            dataset_name=dataset_name,
            params=params,
            config=config,
            sct_dataset_file_path=current_sct_output_path,
        )
        return str(current_sct_output_path)
    else:  # mixed bulk GEPs
        logging.info(f"   > Using Bulk GEP (with Mixed Cell Type) Generation for Dataset {dataset_name}")
        resolved_sct_dataset_file_path = _resolve_mixed_sct_dataset_path(
            dataset_name=dataset_name,
            params=params,
            config=config,
        )
        resolved_cell_type2subtypes = _resolve_dataset_override(
            params,
            config,
            "cell_type2subtypes",
            config.cell_type2subtypes,
        )
        resolved_total_rna_coefficient = _resolve_dataset_override(
            params,
            config,
            "total_rna_coefficient",
            config.total_rna_coefficient,
        )
        resolved_gep_filtering = _resolve_gep_filtering_params(
            dataset_name=dataset_name,
            params=params,
            config=config,
        )
        resolved_gene_filtering = _resolve_gene_filtering_params(
            dataset_name=dataset_name,
            params=params,
            config=config,
        )
        if resolved_gep_filtering.enabled:
            filtering_str = f'with_filtering_{resolved_gep_filtering.gep_filtering_quantile[1]}'
        else:
            filtering_str = 'no_filtering'
        if not 'sampling_method_prop' in params:
            logging.info("   > Warning: 'sampling_method_prop' not specified in params, defaulting to 'segment'")
            sampling_method_prop = 'segment'
        else:
            sampling_method_prop = params.get('sampling_method_prop')
        calculate_gene_cell_prop_correlation = bool(
            _resolve_dataset_override(
                params,
                config,
                "calculate_gene_cell_prop_correlation",
                getattr(config, "calculate_gene_cell_prop_correlation", False),
            )
        )
        if calculate_gene_cell_prop_correlation:
            if sampling_method_prop != "segment":
                raise ValueError(
                    f'Mixed dataset "{dataset_name}" can calculate a reusable gene/cell-fraction '
                    'correlation file only with "sampling_method_prop: segment".'
                )
            if resolved_gep_filtering.enabled:
                raise ValueError(
                    f'Mixed dataset "{dataset_name}" must set "gep_filtering: false" when '
                    "calculating a reusable gene/cell-fraction correlation file."
                )
            if resolved_gene_filtering.enabled:
                raise ValueError(
                    f'Mixed dataset "{dataset_name}" must set "gene_filtering: false" when '
                    "calculating a reusable gene/cell-fraction correlation file."
                )
        resolved_cell_prop_prior = params.get('cell_prop_prior', None)
        pred_cell_prop_file_path = params.get('pred_cell_prop_file_path', None)
        segment_fraction_max_caps = None
        segment_fraction_cap_metadata = None
        if sampling_method_prop != 'dirichlet':
            invalid_keys = [
                key for key in dirichlet_only_keys
                if params.get(key, None) not in (None, "")
            ]
            if sampling_method_prop != "segment" and pred_cell_prop_file_path not in (None, ""):
                invalid_keys.append("pred_cell_prop_file_path")
            if invalid_keys:
                raise ValueError(
                    f'{invalid_keys} are only supported when "sampling_method_prop" is "dirichlet" '
                    f'for Mixed dataset "{dataset_name}".'
                )
            resolved_dirichlet_alpha = None
            dirichlet_sampling_seed = None
            if sampling_method_prop == "segment" and pred_cell_prop_file_path not in (None, ""):
                if not resolved_gep_filtering.enabled:
                    raise ValueError(
                        f'Mixed dataset "{dataset_name}" can use '
                        '"pred_cell_prop_file_path" with segment sampling only when '
                        '"gep_filtering: true".'
                    )
                output_type_mapping = restrict_cell_type2subtype_to_output_types(
                    resolved_cell_type2subtypes,
                    config.output_types,
                )
                segment_cell_types = flatten_cell_type2subtype(output_type_mapping)
                segment_fraction_max_caps, segment_fraction_cap_metadata = (
                    estimate_cell_prop_maxima_from_pred_cell_props(
                        pred_cell_prop_file_path,
                        segment_cell_types,
                    )
                )
                segment_fraction_cap_metadata["segment_fraction_cap_hash"] = identity_hash(
                    segment_fraction_max_caps
                )
                resolved_cell_prop_prior = merge_cell_prop_prior_with_maxima(
                    segment_cell_types,
                    resolved_cell_prop_prior,
                    segment_fraction_max_caps,
                )
                logging.info(
                    'Segment fraction caps from %s: %s',
                    pred_cell_prop_file_path,
                    format_segment_fraction_caps(segment_fraction_max_caps),
                )
        else:
            output_cell_type2subtypes = restrict_cell_type2subtype_to_output_types(
                resolved_cell_type2subtypes,
                config.output_types,
            )
            resolved_dirichlet_alpha = params.get(
                'dirichlet_alpha',
                get_default_dirichlet_alpha(output_cell_type2subtypes),
            )
            dirichlet_sampling_seed = _resolve_dataset_override(
                params,
                config,
                "dirichlet_sampling_seed",
                config.dirichlet_sampling_seed,
            )
        if 'sampling_method_gep' in params:
            logging.info(
                '   > Mixed dataset %s ignores params["sampling_method_gep"]=%r and uses "random" '
                'because Mixed generation samples one SCT profile per cell type from the referenced SCT dataset.',
                dataset_name,
                params.get('sampling_method_gep'),
            )
        sampling_method_gep = 'random'

        if not resolved_gep_filtering.enabled:
            output_tag = "no_filtering"
        elif resolved_gep_filtering.filtering_method in LOCAL_FILTER_METHODS:
            output_tag = f"with_filtering_{resolved_gep_filtering.filtering_method}"
        elif resolved_gep_filtering.filtering_method == "mgep":
            output_tag = f"{filtering_str}_mgep"
        else:
            output_tag = f"{filtering_str}_median_gep"
        simu_bulk_dir = simu_bulk_dir / f'{sampling_method_prop}_{n_sc_datasets}ds_n_base{n_base}_{output_tag}'
        bulk_generator = BulkGEPGenerator(
            simu_bulk_dir=simu_bulk_dir,
            merged_sc_dataset_file_path=config.merged_sc_dataset_file_path,
            cell_type2subtype=resolved_cell_type2subtypes,
            sc_dataset_ids=config.sc_dataset_ids,
            stratify_by_dataset=resolved_stratify,
            stratify_seed=config.sorted_bulk_imputation_seed,
            bulk_dataset_name=dataset_name,
            sct_dataset_file_path=resolved_sct_dataset_file_path,
            check_basic_info=False,
            tcga2cancer_type_file_path=config.tcga2cancer_type_file_path,
            total_rna_coefficient=resolved_total_rna_coefficient,
            cell_type_col_name='cell_type',
            subtype_col_name='cell_type',
            output_types=config.output_types,
            sampling_method_gep=sampling_method_gep,
            mixed_sct_loading_mode=config.mixed_sct_loading_mode,
            mixed_sct_auto_eager_max_gb=config.mixed_sct_auto_eager_max_gb,
        )

        generated_bulk_gep_fp = bulk_generator.generated_bulk_gep_fp
        filtering_space = None
        if resolved_gene_filtering.enabled or resolved_gep_filtering.enabled:
            filtering_space = _prepare_fixed_gene_filter(
                dataset_name, resolved_sct_dataset_file_path, config.output_types,
                resolved_gep_filtering, resolved_gene_filtering, config,
                _resolve_mixed_dataset_result_dir(generated_bulk_gep_fp, dataset_name),
            )
            if resolved_gene_filtering.enabled and resolved_gene_filtering.save_filtered_h5ad:
                filtering_space.additional_artifacts = [
                    _build_filtered_h5ad_path(
                        generated_bulk_gep_fp, "high_corr_gene",
                        resolved_gene_filtering.filtered_dataset_postfix,
                    )
                ]

        # 2. Generate GEP (if it doesn't exist)
        generated_bulk_gep_was_cached = os.path.exists(generated_bulk_gep_fp)
        if not generated_bulk_gep_was_cached or filtering_space is not None:
            gep_gen_config = GEPGenerationConfig(
                n_samples=params['n_samples'],
                simu_method='mul',  # matrix multiplication by cell proportions and sampled GEPs to get bulk GEPs
                sampling_method_prop=sampling_method_prop,
                reference_file=resolved_gep_filtering.reference_file,
                ref_exp_type=resolved_gep_filtering.ref_exp_type,
                filtering=resolved_gep_filtering.enabled,
                filtering_ref_types=resolved_gep_filtering.filtering_ref_types,
                gep_filtering_quantile=resolved_gep_filtering.gep_filtering_quantile,
                n_threads=5,
                log_file_path=log_file_path,
                n_top=resolved_gep_filtering.n_top,
                show_filtering_info=resolved_gep_filtering.show_filtering_info,
                filtering_method=resolved_gep_filtering.filtering_method,
                cell_prop_prior=resolved_cell_prop_prior,
                dirichlet_alpha=resolved_dirichlet_alpha,
                pred_cell_prop_file_path=pred_cell_prop_file_path,
                dirichlet_sampling_seed=dirichlet_sampling_seed,
                segment_fraction_max_caps=segment_fraction_max_caps,
                segment_fraction_cap_metadata=segment_fraction_cap_metadata,
                high_corr_gene_list=resolved_gep_filtering.high_corr_gene_list,
                filtering_by_gene_range=resolved_gep_filtering.filtering_by_gene_range,
                min_percentage_within_gene_range=resolved_gep_filtering.min_percentage_within_gene_range,
                gene_quantile_range=resolved_gep_filtering.gene_quantile_range,
                filtering_in_pca_space=resolved_gep_filtering.filtering_in_pca_space,
                norm_ord=resolved_gep_filtering.norm_ord,
                pca_n_components=resolved_gep_filtering.pca_n_components,
                filtering_space=filtering_space,
            )
            bulk_generator.generate_gep(config=gep_gen_config)
        else:
            logging.info(f"   > GEP file exists: {generated_bulk_gep_fp}")

        mixed_result_dir = _resolve_mixed_dataset_result_dir(generated_bulk_gep_fp, dataset_name)
        full_gene_corr_file = mixed_result_dir / "full_gene_corr_with_cell_fraction.csv"
        if calculate_gene_cell_prop_correlation:
            _calculate_full_gene_cell_prop_correlation(
                generated_bulk_gep_fp=generated_bulk_gep_fp,
                result_file_path=full_gene_corr_file,
            )
        elif not full_gene_corr_file.exists():
            full_gene_corr_file = None
        if not hasattr(config, "_generated_mixed_outputs"):
            config._generated_mixed_outputs = {}
        config._generated_mixed_outputs[dataset_name] = {
            "generated_bulk_gep_file": Path(generated_bulk_gep_fp),
            "generated_cell_fraction_file": Path(bulk_generator.generated_cell_fraction_fp),
            "result_dir": mixed_result_dir,
            "full_gene_corr_with_cell_fraction_file": full_gene_corr_file,
        }

        _save_cancer_cell_fraction_distribution(
            dataset_name=dataset_name,
            cell_fraction_file_path=bulk_generator.generated_cell_fraction_fp,
            output_dir=Path(simu_bulk_dir) / dataset_name,
            plot_all_cell_types=sampling_method_prop == "dirichlet",
        )
        if sampling_method_prop == "dirichlet" and pred_cell_prop_file_path not in (None, ""):
            _save_estimated_dirichlet_alpha_from_pred_cell_props(
                pred_cell_prop_file_path=pred_cell_prop_file_path,
                cell_types=list(config.output_types),
                output_dir=Path(simu_bulk_dir) / dataset_name,
            )

        postprocess_outputs = _run_gene_filtering_postprocess(
            dataset_name=dataset_name,
            generated_bulk_gep_fp=generated_bulk_gep_fp,
            params=params,
            config=config,
            resolved_params=resolved_gene_filtering,
            filtering_space=filtering_space,
        )
        if full_gene_corr_file is not None:
            postprocess_outputs["full_gene_corr_with_cell_fraction_file"] = str(full_gene_corr_file)
        _build_mixed_dataset_summary(
            dataset_name=dataset_name,
            generated_bulk_gep_fp=generated_bulk_gep_fp,
            generated_cell_fraction_fp=bulk_generator.generated_cell_fraction_fp,
            gep_filtering=resolved_gep_filtering,
            gene_filtering=resolved_gene_filtering,
            postprocess_outputs=postprocess_outputs,
            gep_filtering_input_n_samples=bulk_generator.gep_filtering_input_n_samples,
            gep_filtering_kept_n_samples=bulk_generator.gep_filtering_kept_n_samples,
            gep_filtering_removed_n_samples=bulk_generator.gep_filtering_removed_n_samples,
            gep_filtering_pca_target=bulk_generator.gep_filtering_pca_target,
            gep_filtering_n_pcs=bulk_generator.gep_filtering_n_pcs,
            preserve_existing_gep_pca_metadata=generated_bulk_gep_was_cached,
            filtering_space=filtering_space,
        )

        return str(generated_bulk_gep_fp)


# --- 3. Internal Helper Functions ---
def _safe_filename(text: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(text))
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_") or "unknown"


def _save_estimated_dirichlet_alpha_from_pred_cell_props(
    pred_cell_prop_file_path: str | Path,
    cell_types: List[str],
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_hat = estimate_dirichlet_alpha_from_pred_cell_props(
        pred_cell_prop_file_path=pred_cell_prop_file_path,
        cell_types=cell_types,
    )
    alpha_df = pd.DataFrame(
        [{"cell_type": cell_type, "alpha": float(alpha_value)} for cell_type, alpha_value in alpha_hat.items()]
    )
    output_path = output_dir / "dirichlet_alpha_estimated_from_pred_cell_props.csv"
    alpha_df.to_csv(output_path, index=False, float_format="%.6f")
    return output_path


def _save_cancer_cell_fraction_distribution(
    dataset_name: str,
    cell_fraction_file_path: str | Path,
    output_dir: str | Path,
    plot_all_cell_types: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    cell_fraction_file_path = Path(cell_fraction_file_path)
    if not cell_fraction_file_path.exists():
        raise FileNotFoundError(f"Cell fraction file not found: {cell_fraction_file_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_frac = pd.read_csv(cell_fraction_file_path, index_col=0)
    if "Cancer Cells" not in cell_frac.columns:
        raise ValueError(
            f'Mixed dataset "{dataset_name}" is missing the "Cancer Cells" column in '
            f"{cell_fraction_file_path}."
        )

    cancer_frac = cell_frac.loc[:, "Cancer Cells"].astype(float).copy()
    summary_df = pd.DataFrame(
        [
            {
                "dataset_id": dataset_name,
                "cell_type": "Cancer Cells",
                "n_samples": int(cancer_frac.shape[0]),
                "mean": float(cancer_frac.mean()),
                "median": float(cancer_frac.median()),
                "std": float(cancer_frac.std(ddof=1)) if cancer_frac.shape[0] > 1 else 0.0,
                "min": float(cancer_frac.min()),
                "max": float(cancer_frac.max()),
                "p25": float(cancer_frac.quantile(0.25)),
                "p75": float(cancer_frac.quantile(0.75)),
                "p90": float(cancer_frac.quantile(0.90)),
            }
        ]
    )
    summary_df.to_csv(output_dir / "cancer_cell_fraction_summary.csv", index=False, float_format="%.6f")

    plt.figure(figsize=(6, 4))
    sns.histplot(cancer_frac, bins=min(40, max(10, cancer_frac.shape[0] // 20 or 10)), kde=True, color="#c44e52")
    plt.xlabel("Cancer Cells proportion")
    plt.ylabel("Sample count")
    plt.title(f"{dataset_name}: Cancer Cells proportion")
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig(output_dir / f"cancer_cell_fraction_distribution.{suffix}", dpi=300, bbox_inches="tight")
    plt.close()

    if not plot_all_cell_types:
        return

    summary_rows: list[dict[str, Any]] = []
    for cell_type in cell_frac.columns.astype(str):
        if cell_type == "Cancer Cells":
            continue
        frac = cell_frac.loc[:, cell_type].astype(float).copy()
        summary_rows.append(
            {
                "dataset_id": dataset_name,
                "cell_type": cell_type,
                "n_samples": int(frac.shape[0]),
                "mean": float(frac.mean()),
                "median": float(frac.median()),
                "std": float(frac.std(ddof=1)) if frac.shape[0] > 1 else 0.0,
                "min": float(frac.min()),
                "max": float(frac.max()),
                "p25": float(frac.quantile(0.25)),
                "p75": float(frac.quantile(0.75)),
                "p90": float(frac.quantile(0.90)),
            }
        )

        plt.figure(figsize=(6, 4))
        sns.histplot(frac, bins=min(40, max(10, frac.shape[0] // 20 or 10)), kde=True, color="#4c78a8")
        plt.xlabel(f"{cell_type} proportion")
        plt.ylabel("Sample count")
        plt.title(f"{dataset_name}: {cell_type} proportion")
        plt.tight_layout()
        safe_cell_type = _safe_filename(cell_type)
        for suffix in ("png", "pdf"):
            plt.savefig(
                output_dir / f"{safe_cell_type}_fraction_distribution.{suffix}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close()

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "cell_fraction_summary_all_cell_types.csv",
            index=False,
            float_format="%.6f",
        )


def _resolve_mixed_dataset_result_dir(generated_bulk_gep_fp: str | Path, dataset_name: str) -> Path:
    return Path(generated_bulk_gep_fp).parent / dataset_name


def _calculate_full_gene_cell_prop_correlation(
    generated_bulk_gep_fp: str | Path,
    result_file_path: str | Path,
) -> Path:
    result_file_path = Path(result_file_path)
    if result_file_path.exists():
        return result_file_path
    check_dir(result_file_path.parent)

    bulk_obj = ReadH5AD(generated_bulk_gep_fp)
    gene_exp = bulk_obj.get_df(convert_to_tpm=True)
    cell_frac = bulk_obj.get_cell_fraction()
    if gene_exp.shape[0] < 2 or gene_exp.shape[1] == 0:
        raise ValueError("At least two Mixed samples and one gene are required to calculate correlations.")
    if cell_frac.shape[1] == 0:
        raise ValueError("At least one cell type is required to calculate gene/cell-fraction correlations.")
    if not gene_exp.index.isin(cell_frac.index).all():
        raise ValueError("Mixed expression samples and cell-fraction samples do not match.")
    cell_frac = cell_frac.loc[gene_exp.index]
    cal_corr_gene_exp_with_cell_frac(
        gene_exp=gene_exp,
        cell_frac=cell_frac,
        result_file_path=result_file_path,
    )
    return result_file_path


def _build_filtered_h5ad_path(
    generated_bulk_gep_fp: str | Path,
    filtering_type: str,
    filtered_dataset_postfix: str | None,
) -> Path:
    generated_bulk_gep_fp = Path(generated_bulk_gep_fp)
    postfix = filtered_dataset_postfix
    if postfix in (None, ""):
        postfix = f"filtered_by_{filtering_type}"
    postfix = str(postfix).strip("_")
    return generated_bulk_gep_fp.with_name(f"{generated_bulk_gep_fp.stem}_{postfix}{generated_bulk_gep_fp.suffix}")


def _prepare_fixed_gene_filter(dataset_name, sct_file, cell_types, gep, gene, config, result_dir):
    reference_file = gep.reference_file if gep.enabled else gene.reference_file
    reference_type = gep.ref_exp_type if gep.enabled else "TPM"
    reference = ReadExp(str(reference_file), exp_type=reference_type).exp
    if reference_type == "log_space":
        values = reference.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Reference contains negative or non-finite log expression.")
        reference = pd.DataFrame(np.expm1(values * np.log(2)), index=reference.index,
                                 columns=reference.columns)
    reference_subset = gep.filtering_ref_types if gep.enabled else None
    local_mode = gep.enabled and gep.filtering_method in LOCAL_FILTER_METHODS
    local_kwargs = {}
    if local_mode:
        annotation_file = Path(config.tcga2cancer_type_file_path)
        annotation = pd.read_csv(annotation_file, index_col=0)
        reference, split, split_summary = split_tcga_reference(
            reference, annotation, gep.reference_holdout_fraction,
            gep.reference_holdout_seed, reference_subset,
        )
        local_kwargs = {
            "split": split, "split_summary": split_summary,
            "annotation_file": str(annotation_file.resolve()),
        }
    elif reference_subset:
        mapping = pd.read_csv(config.tcga2cancer_type_file_path, index_col=0)
        sample_ids = mapping.index[mapping["cancer_type"].isin(reference_subset)]
        reference = reference.loc[reference.index.isin(sample_ids)].copy()
        if reference.empty:
            raise ValueError(f'No reference samples match filtering_ref_types for "{dataset_name}".')
    if reference.empty:
        raise ValueError("Filtering reference has no samples.")
    variance_metadata = {}
    variance_table = None
    threshold = gene.reference_variance_threshold
    eligible_reference_genes = reference.columns.tolist()
    if threshold is not None:
        variance_reference = (
            reference.loc[split.index[split.role == "fitting"]] if local_mode else reference
        )
        variances, keep = reference_variance_mask(variance_reference, threshold)
        eligible_reference_genes = variances.index[keep].tolist()
        variance_metadata = {
            "reference_variance_threshold": threshold,
            "reference_variance_space": "linear-TPM",
            "reference_variance_ddof": 0,
            "reference_variance_scope": "fitting-only" if local_mode else "reference",
            "reference_variance_n_samples": len(variance_reference),
            "reference_variance_input_n_genes": len(variances),
            "reference_variance_removed_n_genes": int((~keep).sum()),
        }
        variance_table = pd.DataFrame({"variance_TPM2": variances, "variance_pass": keep})
        logging.info("Reference variance > %.3g TPM^2: kept %d/%d genes (%s).",
                     threshold, int(keep.sum()), len(keep), variance_metadata["reference_variance_scope"])
    sct_reader = ReadH5AD(sct_file, backed="r")
    try:
        sct_genes = sct_reader.get_h5ad().var_names.tolist()
    finally:
        sct_reader.get_h5ad().file.close()
    supplied_genes = None
    correlations = None
    correlation_file = None
    if gene.enabled and gene.high_corr_gene_file:
        supplied_genes = _read_gene_list_file(gene.high_corr_gene_file)
    elif gene.enabled:
        source = config._generated_mixed_outputs.get(gene.correlation_reference_dataset)
        if source is None:
            raise ValueError(
                f'Gene-correlation source "{gene.correlation_reference_dataset}" '
                "has not been generated before its target."
            )
        correlation_file = source.get("full_gene_corr_with_cell_fraction_file")
        if correlation_file is None or not Path(correlation_file).exists():
            raise ValueError(
                f'Gene-correlation source "{gene.correlation_reference_dataset}" has no '
                'full_gene_corr_with_cell_fraction.csv. Set '
                'calculate_gene_cell_prop_correlation: true and run the source first.'
            )
        correlations = pd.read_csv(correlation_file, index_col=0)
    genes, shared_count = select_fixed_genes(
        sct_genes, eligible_reference_genes, cell_types, supplied_genes=supplied_genes,
        correlations=correlations, corr_threshold=gene.corr_threshold, n_gene_max=gene.n_gene_max,
    )
    filter_type = LocalTCGAFilter if local_mode else FixedGeneFilter
    space = filter_type(
        genes, reference, reference_file, reference_subset, gep, result_dir,
        len(sct_genes), shared_count, correlation_file=correlation_file, **local_kwargs,
    )
    space.reference_variance_metadata = variance_metadata
    if variance_metadata:
        space.pca_identity.update(variance_metadata)
        space.identity.update(variance_metadata)
        if local_mode:
            space.calibration_identity.update(variance_metadata)
        # Validate compatibility before writing any auxiliary audit output.
        space.validate_model()
        variance_table["selected_in_G"] = variance_table.index.isin(genes)
        variance_table.index.name = "gene_name"
        space.reference_variance_table = variance_table
    return space


def _postprocess_fixed_gene_filter(dataset_name, generated_file, gene_params, filtering_space):
    """Export/plot the already fixed space without selecting genes again."""
    import anndata as ad
    from scipy.sparse import issparse

    outputs = {
        "filtered_bulk_gep_file": None,
        "gene_list_file": str(filtering_space.gene_list_file),
        "corr_result_file": None,
        "pca_model_file": None,
        "pca_data_file": None,
        "pca_plot_file": None,
        "full_gene_corr_with_cell_fraction_file": filtering_space.correlation_file,
    }
    bulk = ad.read_h5ad(generated_file)
    if bulk.var_names.tolist() != filtering_space.genes:
        raise ValueError("Accepted artifact gene order differs from the fixed gene list G.")
    values = bulk.X.toarray() if issparse(bulk.X) else bulk.X
    bulk_log = pd.DataFrame(values, index=bulk.obs_names, columns=bulk.var_names)
    if gene_params.enabled and gene_params.save_filtered_h5ad:
        output = _build_filtered_h5ad_path(
            generated_file, "high_corr_gene", gene_params.filtered_dataset_postfix
        )
        metadata_path = output.with_suffix(".metadata.json")
        if not output.exists():
            bulk.write_h5ad(output, compression="gzip")
            write_metadata(metadata_path, filtering_space.identity)
        else:
            validate_metadata(metadata_path, filtering_space.identity)
            existing = ad.read_h5ad(output, backed="r")
            try:
                if (existing.var_names.tolist() != filtering_space.genes
                        or existing.obs_names.tolist() != bulk.obs_names.tolist()):
                    raise ValueError("Selected-gene export differs from G or sample IDs. Use a new output location.")
            finally:
                existing.file.close()
        outputs["filtered_bulk_gep_file"] = str(output)
    if gene_params.plot_pca:
        result_dir = filtering_space.result_dir
        postfix = filtering_space.workflow_version
        model = result_dir / f"comparison_pca_{gene_params.pca_n_components}.joblib"
        coordinates = result_dir / f"comparison_pca_{gene_params.pca_n_components}.csv"
        comparison_metadata = model.with_suffix(".metadata.json")
        comparison_identity = {
            **filtering_space.pca_identity,
            "comparison_components": gene_params.pca_n_components,
            "bulk_hash": identity_hash(pd.util.hash_pandas_object(bulk_log, index=True).tolist()),
            "model_scope": "joint-reference-and-simulated-diagnostic",
            "ordered_genes": filtering_space.genes,
        }
        if model.exists() or coordinates.exists() or comparison_metadata.exists():
            validate_metadata(comparison_metadata, comparison_identity)
            if not model.exists() or not coordinates.exists():
                raise ValueError("Comparison PCA artifacts are incomplete. Use a new output location.")
            cached_model = load(model)
            if cached_model.n_features_in_ != len(filtering_space.genes):
                raise ValueError("Comparison PCA gene order differs from G. Use a new output location.")
        filtering_by_gene_list_and_pca_plot(
            bulk_log, getattr(filtering_space, "all_reference_log", filtering_space.reference_log),
            filtering_space.genes,
            str(result_dir), dataset_name, n_components=gene_params.pca_n_components,
            pca_model_name_postfix=postfix, pca_model_file_path=str(model),
            pca_data_file_path=str(coordinates), figsize=gene_params.pca_figsize,
            if_plot_pca=True, normalized_log_inputs=True,
        )
        write_metadata(comparison_metadata, comparison_identity)
        loading_paths = _export_pca_gene_loadings(load(model), filtering_space.genes, model)
        outputs.update(
            pca_model_file=str(model), pca_data_file=str(coordinates),
            pca_plot_file=str(result_dir / f"{dataset_name}_PCA_with_TCGA_{postfix}_PC0_PC1.png"),
            **loading_paths,
        )
        if isinstance(filtering_space, LocalTCGAFilter):
            from .simulation.filtering_diagnostics import plot_local_filter_diagnostics
            filtering_space.diagnostic_outputs = plot_local_filter_diagnostics(
                filtering_space, bulk_log, dataset_name
            )
    return outputs


def _run_gene_filtering_postprocess(
    dataset_name: str,
    generated_bulk_gep_fp: str | Path,
    params: Dict[str, Any],
    config: SimuConfig,
    resolved_params: ResolvedGeneFilteringParams,
    filtering_space: FixedGeneFilter | None = None,
) -> Dict[str, str | None]:
    if filtering_space is not None:
        return _postprocess_fixed_gene_filter(
            dataset_name, generated_bulk_gep_fp, resolved_params, filtering_space
        )
    outputs: Dict[str, str | None] = {
        "filtered_bulk_gep_file": None,
        "gene_list_file": None,
        "corr_result_file": None,
        "pca_model_file": None,
        "pca_data_file": None,
        "pca_plot_file": None,
        "full_gene_corr_with_cell_fraction_file": None,
    }
    current_result_dir = _resolve_mixed_dataset_result_dir(generated_bulk_gep_fp, dataset_name)
    correlation_reference_file = None
    if resolved_params.correlation_reference_dataset:
        source_outputs = getattr(config, "_generated_mixed_outputs", {}).get(
            resolved_params.correlation_reference_dataset
        )
        if source_outputs is None:
            raise ValueError(
                f'Gene-correlation source dataset "{resolved_params.correlation_reference_dataset}" '
                "has not been generated before its target."
            )
        correlation_reference_file = source_outputs.get(
            "full_gene_corr_with_cell_fraction_file"
        )
        if correlation_reference_file is None or not Path(correlation_reference_file).exists():
            raise ValueError(
                f'Gene-correlation source dataset "{resolved_params.correlation_reference_dataset}" '
                'has no "full_gene_corr_with_cell_fraction.csv". Set '
                '"calculate_gene_cell_prop_correlation: true" on the source dataset and run it first.'
            )
        outputs["full_gene_corr_with_cell_fraction_file"] = str(correlation_reference_file)

    if not resolved_params.enabled:
        logging.info("   > Gene-level filtering disabled for %s.", dataset_name)
        if not resolved_params.plot_pca:
            return outputs

        check_dir(current_result_dir)
        bulk_exp = ReadH5AD(generated_bulk_gep_fp).get_df()
        tcga_exp = ReadExp(resolved_params.reference_file, exp_type="TPM").get_exp()
        gene_list = [gene for gene in bulk_exp.columns if gene in tcga_exp.columns]
        if not gene_list:
            raise ValueError(
                f'No genes are shared between Mixed dataset "{dataset_name}" and the PCA reference.'
            )

        gene_list_file_path = current_result_dir / "gene_list_all_genes_for_pca.csv"
        pca_model_fp = current_result_dir / f"both_TCGA_and_simu_{dataset_name}_PCA_all_genes_for_pca.joblib"
        pca_data_fp = current_result_dir / f"{dataset_name}_PCA_with_TCGA_all_genes_for_pca.csv"
        pca_plot_fp = current_result_dir / f"{dataset_name}_PCA_with_TCGA_all_genes_for_pca_PC0_PC1.png"
        pd.DataFrame({"gene_name": gene_list}).to_csv(gene_list_file_path, index=False)
        filtering_by_gene_list_and_pca_plot(
            bulk_exp=bulk_exp,
            tcga_exp=tcga_exp,
            gene_list=gene_list,
            result_dir=str(current_result_dir),
            simu_dataset_name=dataset_name,
            n_components=resolved_params.pca_n_components,
            pca_model_name_postfix="all_genes_for_pca",
            pca_model_file_path=str(pca_model_fp),
            pca_data_file_path=str(pca_data_fp),
            figsize=resolved_params.pca_figsize,
            if_plot_pca=True,
        )
        outputs.update(
            {
                "gene_list_file": str(gene_list_file_path),
                "pca_model_file": str(pca_model_fp),
                "pca_data_file": str(pca_data_fp),
                "pca_plot_file": str(pca_plot_fp),
            }
        )
        return outputs

    logging.info(
        "   > Running post-processing (gene filtering: %s) for %s...",
        resolved_params.filtering_type,
        dataset_name,
    )
    check_dir(current_result_dir)

    gene_list_file_path = current_result_dir / f"gene_list_filtered_by_{resolved_params.filtering_type}.csv"
    corr_result_fp = current_result_dir / "gene_corr_with_cell_fraction.csv"
    full_corr_result_fp = current_result_dir / "full_gene_corr_with_cell_fraction.csv"
    pca_model_fp = current_result_dir / f"both_TCGA_and_simu_{dataset_name}_PCA_{resolved_params.filtering_type}.joblib"
    pca_data_fp = current_result_dir / f"{dataset_name}_PCA_with_TCGA_{resolved_params.filtering_type}.csv"
    filtered_h5ad_file = None
    if resolved_params.save_filtered_h5ad:
        filtered_h5ad_file = _build_filtered_h5ad_path(
            generated_bulk_gep_fp=generated_bulk_gep_fp,
            filtering_type=resolved_params.filtering_type,
            filtered_dataset_postfix=resolved_params.filtered_dataset_postfix,
        )

    gene_list: List[str]
    high_corr_file = resolved_params.high_corr_gene_file
    full_corr_file_for_summary = None
    if high_corr_file is None and "high_corr_gene" in resolved_params.filtering_type:
        if correlation_reference_file is not None:
            full_corr_file_for_summary = str(correlation_reference_file)
        else:
            full_corr_file_for_summary = str(full_corr_result_fp)

    if high_corr_file and "high_corr_gene" in resolved_params.filtering_type:
        high_corr_gene_list = _read_gene_list_file(high_corr_file)
        if resolved_params.filtering_type == "high_corr_gene":
            gene_list = high_corr_gene_list
            pd.DataFrame({"gene_name": gene_list}).to_csv(gene_list_file_path, index=False)
        elif resolved_params.filtering_type == "high_corr_gene_and_quantile_range":
            quantile_gene_list = get_gene_list_for_filtering(
                bulk_exp_file=generated_bulk_gep_fp,
                tcga_file=resolved_params.reference_file,
                result_file_path=gene_list_file_path.with_name(
                    f"{gene_list_file_path.stem}_quantile_range.csv"
                ),
                q_col_name=resolved_params.q_col_name,
                filtering_type="quantile_range",
                corr_threshold=resolved_params.corr_threshold,
                n_gene_max=resolved_params.n_gene_max,
                corr_result_fp=str(corr_result_fp),
                quantile_range=resolved_params.quantile_range,
            )
            quantile_gene_set = set(quantile_gene_list)
            gene_list = [gene for gene in high_corr_gene_list if gene in quantile_gene_set]
            pd.DataFrame({"gene_name": gene_list}).to_csv(gene_list_file_path, index=False)
        else:
            gene_list = get_gene_list_for_filtering(
                bulk_exp_file=generated_bulk_gep_fp,
                tcga_file=resolved_params.reference_file,
                result_file_path=gene_list_file_path,
                q_col_name=resolved_params.q_col_name,
                filtering_type=resolved_params.filtering_type,
                corr_threshold=resolved_params.corr_threshold,
                n_gene_max=resolved_params.n_gene_max,
                corr_result_fp=str(corr_result_fp),
                quantile_range=resolved_params.quantile_range,
            )
    else:
        gene_list = get_gene_list_for_filtering(
            bulk_exp_file=generated_bulk_gep_fp,
            tcga_file=resolved_params.reference_file,
            result_file_path=gene_list_file_path,
            q_col_name=resolved_params.q_col_name,
            filtering_type=resolved_params.filtering_type,
            corr_threshold=resolved_params.corr_threshold,
            n_gene_max=resolved_params.n_gene_max,
            corr_result_fp=str(corr_result_fp),
            quantile_range=resolved_params.quantile_range,
            correlation_reference_file=correlation_reference_file,
            full_corr_result_fp=(
                str(full_corr_result_fp) if correlation_reference_file is None else None
            ),
            force_recompute="high_corr_gene" in resolved_params.filtering_type,
        )

    bulk_exp_obj = ReadH5AD(generated_bulk_gep_fp)
    bulk_exp = bulk_exp_obj.get_df()
    bulk_exp_cell_frac = bulk_exp_obj.get_cell_fraction()
    tcga_exp = ReadExp(resolved_params.reference_file, exp_type="TPM").get_exp()
    filtering_by_gene_list_and_pca_plot(
        bulk_exp=bulk_exp,
        tcga_exp=tcga_exp,
        gene_list=gene_list,
        result_dir=str(current_result_dir),
        simu_dataset_name=dataset_name,
        n_components=resolved_params.pca_n_components,
        pca_model_name_postfix=resolved_params.filtering_type,
        pca_model_file_path=str(pca_model_fp),
        pca_data_file_path=str(pca_data_fp),
        h5ad_file_path=str(filtered_h5ad_file) if filtered_h5ad_file is not None else None,
        cell_frac_file=bulk_exp_cell_frac,
        figsize=resolved_params.pca_figsize,
        if_plot_pca=resolved_params.plot_pca,
    )
    outputs.update(
        {
            "filtered_bulk_gep_file": str(filtered_h5ad_file) if filtered_h5ad_file is not None else None,
            "gene_list_file": str(gene_list_file_path),
            "corr_result_file": str(corr_result_fp),
            "pca_model_file": str(pca_model_fp) if resolved_params.plot_pca else None,
            "pca_data_file": str(pca_data_fp) if resolved_params.plot_pca else None,
            "pca_plot_file": (
                str(current_result_dir / f"{dataset_name}_PCA_with_TCGA_{resolved_params.filtering_type}_PC0_PC1.png")
                if resolved_params.plot_pca
                else None
            ),
            "full_gene_corr_with_cell_fraction_file": full_corr_file_for_summary,
        }
    )
    return outputs


def _get_h5ad_shape(file_path: str | Path | None) -> tuple[int, int] | None:
    if file_path in (None, ""):
        return None
    file_path = Path(file_path)
    if not file_path.exists():
        return None
    h5ad_obj = ReadH5AD(str(file_path)).get_h5ad()
    return int(h5ad_obj.shape[0]), int(h5ad_obj.shape[1])


def _load_gep_filtering_pca_metadata(
    reference_file: str | Path | None,
    pca_target: int | float | None,
) -> tuple[int | float | None, int | None]:
    if reference_file in (None, "") or pca_target is None:
        return None, None
    pca_model_path = (
        Path(reference_file).parent.parent
        / f"pca_model_{pca_target}"
        / "tcga_pca_model_for_gep_filtering.pkl"
    )
    if not pca_model_path.exists():
        return None, None
    try:
        pca_model = load(pca_model_path)
        return pca_model.n_components, int(pca_model.n_components_)
    except Exception as exc:
        logging.warning("Could not read GEP-filtering PCA metadata from %s: %s", pca_model_path, exc)
        return None, None


def _build_mixed_dataset_summary(
    dataset_name: str,
    generated_bulk_gep_fp: str | Path,
    generated_cell_fraction_fp: str | Path,
    gep_filtering: ResolvedGEPFilteringParams,
    gene_filtering: ResolvedGeneFilteringParams,
    postprocess_outputs: Dict[str, str | None],
    gep_filtering_input_n_samples: int | None = None,
    gep_filtering_kept_n_samples: int | None = None,
    gep_filtering_removed_n_samples: int | None = None,
    gep_filtering_pca_target: int | float | None = None,
    gep_filtering_n_pcs: int | None = None,
    preserve_existing_gep_pca_metadata: bool = False,
    filtering_space: FixedGeneFilter | None = None,
) -> Path:
    summary_dir = _resolve_mixed_dataset_result_dir(generated_bulk_gep_fp, dataset_name)
    check_dir(summary_dir)
    summary_path = summary_dir / "mixed_dataset_summary.json"
    if preserve_existing_gep_pca_metadata and summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            existing_summary = json.load(f)
        if existing_summary.get("generated_bulk_gep_file") == str(generated_bulk_gep_fp):
            if gep_filtering_pca_target is None:
                gep_filtering_pca_target = existing_summary.get("gep_pca_n_components")
            if gep_filtering_n_pcs is None:
                gep_filtering_n_pcs = existing_summary.get("gep_filtering_n_pcs")
    if (
        preserve_existing_gep_pca_metadata
        and filtering_space is None
        and gep_filtering.enabled
        and gep_filtering.filtering_in_pca_space
        and (gep_filtering_pca_target is None or gep_filtering_n_pcs is None)
    ):
        cache_target = (
            gep_filtering_pca_target
            if gep_filtering_pca_target is not None
            else gep_filtering.pca_n_components
        )
        cached_target, cached_n_pcs = _load_gep_filtering_pca_metadata(
            gep_filtering.reference_file,
            cache_target,
        )
        if gep_filtering_pca_target is None:
            gep_filtering_pca_target = cached_target
        if gep_filtering_n_pcs is None:
            gep_filtering_n_pcs = cached_n_pcs
    generated_shape = _get_h5ad_shape(generated_bulk_gep_fp)
    filtered_shape = _get_h5ad_shape(postprocess_outputs.get("filtered_bulk_gep_file"))
    gene_list_file = postprocess_outputs.get("gene_list_file")
    filtered_gene_count = None
    if gene_filtering.enabled and gene_list_file and Path(gene_list_file).exists():
        filtered_gene_count = len(_read_gene_list_file(gene_list_file))
    filtered_n_genes = filtered_shape[1] if filtered_shape else None
    removed_n_genes = None
    if gene_filtering.enabled and generated_shape:
        filtered_gene_count_for_summary = (
            filtered_n_genes if filtered_n_genes is not None else filtered_gene_count
        )
        if filtered_gene_count_for_summary is not None:
            removed_n_genes = generated_shape[1] - filtered_gene_count_for_summary
    summary = {
        "dataset_name": dataset_name,
        "dataset_type": "Mixed",
        "generated_bulk_gep_file": str(generated_bulk_gep_fp),
        "generated_cell_fraction_file": str(generated_cell_fraction_fp),
        "generated_n_samples": generated_shape[0] if generated_shape else None,
        "generated_n_genes": generated_shape[1] if generated_shape else None,
        "gep_filtering_enabled": gep_filtering.enabled,
        "gep_filtering_method": gep_filtering.filtering_method if gep_filtering.enabled else None,
        "gep_pca_n_components": gep_filtering_pca_target,
        "gep_filtering_n_pcs": gep_filtering_n_pcs,
        "gep_filtering_input_n_samples": gep_filtering_input_n_samples,
        "gep_filtering_kept_n_samples": gep_filtering_kept_n_samples,
        "gep_filtering_removed_n_samples": gep_filtering_removed_n_samples,
        "gene_filtering_enabled": gene_filtering.enabled,
        "gene_filtering_type": gene_filtering.filtering_type if gene_filtering.enabled else None,
        "gene_filtering_correlation_reference_dataset": gene_filtering.correlation_reference_dataset,
        "filtered_bulk_gep_file": postprocess_outputs.get("filtered_bulk_gep_file"),
        "filtered_n_samples": filtered_shape[0] if filtered_shape else None,
        "filtered_n_genes": filtered_n_genes,
        "removed_n_genes": removed_n_genes,
        "filtered_gene_count": filtered_gene_count,
        "gene_list_file": gene_list_file,
        "full_gene_corr_with_cell_fraction_file": postprocess_outputs.get(
            "full_gene_corr_with_cell_fraction_file"
        ),
        "pca_model_file": postprocess_outputs.get("pca_model_file"),
        "pca_data_file": postprocess_outputs.get("pca_data_file"),
        "pca_plot_file": postprocess_outputs.get("pca_plot_file"),
        "pca_gene_loadings_PC1_file": postprocess_outputs.get("pca_gene_loadings_PC1_file"),
    }
    if filtering_space is not None:
        summary.update(getattr(filtering_space, "reference_variance_metadata", {}))
        generation_identity = filtering_space.identity.get("generation", {})
        segment_cap_metadata = generation_identity.get("segment_fraction_cap_metadata")
        if segment_cap_metadata:
            summary["segment_fraction_cap_metadata"] = segment_cap_metadata
        summary.update({
            "filtering_workflow_version": filtering_space.workflow_version,
            "normalization_version": NORMALIZATION_VERSION,
            "filtering_identity": filtering_space.identity,
            "filtering_cache_identity": identity_hash(filtering_space.identity),
            "fixed_gene_list_hash": filtering_space.gene_hash,
            "pre_selection_n_genes": filtering_space.generation_n_genes,
            "shared_n_genes": filtering_space.shared_n_genes,
            "selected_n_genes": len(filtering_space.genes),
            "exported_n_genes": generated_shape[1] if generated_shape else None,
            "removed_n_genes": filtering_space.generation_n_genes - len(filtering_space.genes),
            "gep_filtering_pca_model_file": (
                str(filtering_space.model_path) if filtering_space.pca is not None else None
            ),
            **filtering_space.stats(),
        })
        if isinstance(filtering_space, LocalTCGAFilter):
            summary.update(filtering_space.summary())
            summary["exported_n_samples"] = generated_shape[0] if generated_shape else None
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def _run_sample_diversity_for_sct_dataset(
    dataset_name: str,
    params: Dict[str, Any],
    config: SimuConfig,
    sct_dataset_file_path: str | Path,
) -> None:
    run_analysis = params.get("run_sample_diversity_analysis", config.run_sample_diversity_analysis)
    if not run_analysis:
        logging.info("   > Sample diversity analysis disabled for SCT dataset %s.", dataset_name)
        return

    gene_mode = _resolve_sample_diversity_gene_mode(
        params.get("sample_diversity_gene_mode", config.sample_diversity_gene_mode),
        f'SCT dataset "{dataset_name}"',
    )
    distance_metric = _resolve_sample_diversity_distance_metric(
        params.get("sample_diversity_distance_metric", config.sample_diversity_distance_metric),
        f'SCT dataset "{dataset_name}"',
    )
    n_top_genes = int(params.get("sample_diversity_n_top_genes", config.sample_diversity_n_top_genes))
    if n_top_genes <= 0:
        raise ValueError(
            f'"sample_diversity_n_top_genes" must be positive for SCT dataset "{dataset_name}".'
        )
    min_samples = int(
        params.get(
            "sample_diversity_min_samples_per_cell_type",
            config.sample_diversity_min_samples_per_cell_type,
        )
    )
    if min_samples < 2:
        raise ValueError(
            f'"sample_diversity_min_samples_per_cell_type" must be at least 2 for SCT dataset "{dataset_name}".'
        )

    gene_list_path_value = params.get("sample_diversity_gene_list_path", config.sample_diversity_gene_list_path)
    gene_list_path = None
    if gene_list_path_value not in (None, ""):
        gene_list_path = Path(gene_list_path_value)
        if not gene_list_path.exists():
            raise FileNotFoundError(
                f'"sample_diversity_gene_list_path" does not exist for SCT dataset "{dataset_name}": '
                f"{gene_list_path}"
            )

    merged_sc_dataset_file_path = None
    if gene_mode == "hvg":
        merged_sc_dataset_file_path = Path(config.merged_sc_dataset_file_path)
        if not merged_sc_dataset_file_path.exists():
            raise FileNotFoundError(f"merged_sc_dataset_file_path does not exist: {merged_sc_dataset_file_path}")

    dataset_result_dir = _resolve_sct_aux_output_dir(
        sct_dataset_file_path=sct_dataset_file_path,
        dataset_name=dataset_name,
    )

    logging.info("   > Running sample diversity analysis for SCT dataset %s...", dataset_name)
    run_sct_sample_diversity_analysis(
        sct_dataset_file_path=sct_dataset_file_path,
        dataset_name=dataset_name,
        output_dir=dataset_result_dir / "sample_diversity",
        gene_mode=gene_mode,
        n_top_genes=n_top_genes,
        gene_list_path=gene_list_path,
        merged_sc_dataset_file_path=merged_sc_dataset_file_path,
        distance_metric=distance_metric,
        min_samples_per_cell_type=min_samples,
    )


def get_gene_list_for_filtering(bulk_exp_file: str | Path, tcga_file: str | Path, result_file_path: str | Path, q_col_name: list = None,
                                filtering_type: str = 'quantile_range',
                                corr_threshold: float = 0.3, n_gene_max: int = 1000,
                                corr_result_fp: str = None, quantile_range: list = None,
                                correlation_reference_file: str | Path | None = None,
                                full_corr_result_fp: str | Path | None = None,
                                force_recompute: bool = False) -> list:
    """
    Perform gene-level filtering based on the specific filtering type.

    :param bulk_exp_file: str, Path to the simulated bulk expression file in log2(TPM + 1) format to be filtered.
    :param tcga_file: str, Path to the TCGA reference file in TPM format
    :param result_file_path: str, Path where the filtered gene list will be saved.
    :param q_col_name: list, Optional. Column names for quantile ranges.
    :param filtering_type: str, Specify the method of filtering. Options include: 'high_corr_gene', 'quantile_range', 'all_genes', 'high_corr_gene_and_quantile_range'
      - 'high_corr_gene': Select genes that their expression values have high correlation with the cell proportions of any cell type
      - 'quantile_range': Select genes with their median of expression values within the [q_5, q_95] quantile range of their counterparts in TCGA (default)
    :param corr_result_fp: str, path to the result file after 'high_corr_gene' filtering
    :param correlation_reference_file: optional full gene/cell-type correlation matrix to reuse
    :param full_corr_result_fp: optional path for the unfiltered correlation matrix
    :param force_recompute: recompute the selected gene list using current thresholds
    :param quantile_range: tuple,  (lower_quantile, median_quantile, upper_quantile). Genes are removed if their median expression in simulated bulk GEPs is outside this range in TCGA data
        of the corresponding gene in TCGA dataset will be removed


    :param corr_threshold: float, correlation threshold for gene filtering
    :param n_gene_max: int, maximum number of genes to select for each cell type during 'high_corr_gene' filtering
    :return: list, filtered gene list
    """

    assert filtering_type in ['high_corr_gene', 'quantile_range', 'all_genes', 'high_corr_gene_and_quantile_range'], \
        f'filtering_type: {filtering_type} is not supported, only support high_corr_gene, quantile_range, all_genes, ' \
        f'high_corr_gene_and_quantile_range'
    if force_recompute or not os.path.exists(result_file_path):
        bulk_exp = ReadH5AD(bulk_exp_file).get_df(convert_to_tpm=True)
        tcga_obj = ReadExp(tcga_file, exp_type='TPM')
        tcga_obj.align_with_gene_list(gene_list=bulk_exp.columns.to_list())  # common genes will be kept
        tcga_obj.to_log2cpm1p()
        tcga = tcga_obj.get_exp()
        bulk_exp = bulk_exp.loc[:, tcga.columns].copy()
        bulk_exp = non_log2cpm(bulk_exp)
        bulk_exp = log2_transform(bulk_exp)
        assert (len(tcga.columns) == len(bulk_exp.columns)) and np.all(tcga.columns == bulk_exp.columns), \
            f'Columns of bulk_exp and tcga are not the same'
        gene_list = []
        if 'high_corr_gene' in filtering_type:
            h5_obj = ReadH5AD(bulk_exp_file)
            gene_exp = h5_obj.get_df(convert_to_tpm=True)
            cell_frac = h5_obj.get_cell_fraction()
            correlation_df = None
            if correlation_reference_file:
                correlation_df = pd.read_csv(correlation_reference_file, index_col=0)
                missing_cell_types = [
                    cell_type
                    for cell_type in cell_frac.columns
                    if cell_type not in correlation_df.columns
                ]
                if missing_cell_types:
                    raise ValueError(
                        f'Correlation reference "{correlation_reference_file}" is missing target cell types: '
                        f'{missing_cell_types}.'
                    )
                correlation_df = correlation_df.loc[
                    correlation_df.index.isin(bulk_exp.columns),
                    cell_frac.columns,
                ]
                if correlation_df.empty:
                    raise ValueError(
                        f'Correlation reference "{correlation_reference_file}" has no genes shared '
                        "with the target Mixed dataset and TCGA reference."
                    )
            corr_df = cal_corr_gene_exp_with_cell_frac(gene_exp=gene_exp, cell_frac=cell_frac,
                                                       result_file_path=corr_result_fp, filtered_by_corr=corr_threshold,
                                                       filter_by_num=n_gene_max,
                                                       full_correlation_file_path=(
                                                           full_corr_result_fp
                                                           if correlation_df is None
                                                           else None
                                                       ),
                                                       correlation_df=correlation_df)
            del gene_exp
            gene_list = corr_df.index.to_list()
            logging.info(f'{len(gene_list)} genes are selected by high correlation')
        if 'quantile_range' in filtering_type:
            # both bulk_exp and tcga are in log-space
            gene_list_qr = get_gene_list_filtered_by_quantile_range(bulk_exp=bulk_exp, tcga_exp=tcga,
                                                                    quantile_range=quantile_range,
                                                                    q_col_name=q_col_name)
            logging.info(f'{len(gene_list_qr)} genes are selected by quantile range')
            if len(gene_list) > 0:  # if high-correlation genes exist, then filter by high-correlation genes
                gene_list = [gene for gene in gene_list if gene in gene_list_qr]
                logging.info(f'{len(gene_list)} genes are selected by both high correlation and quantile range')
            else:
                gene_list = gene_list_qr
        if filtering_type == 'all_genes':
            gene_list = bulk_exp.columns.to_list()
            logging.info(f'All {len(gene_list)} genes will be used')

        gene_list_df = pd.DataFrame(columns=['gene_name'])
        gene_list_df['gene_name'] = gene_list
        gene_list_df.to_csv(result_file_path, index=False)
    else:
        logging.info(f'Loading the gene list from file: {result_file_path}')
        gene_list_df = pd.read_csv(result_file_path)
        gene_list = gene_list_df['gene_name'].to_list()
    return gene_list


def _export_pca_gene_loadings(model, genes, model_path):
    """Export one signed PC1/PC2 loading table in descending PC1 order."""
    components = np.asarray(model.components_)
    if components.shape[0] < 2 or components.shape[1] != len(genes):
        raise ValueError("PCA loading shape does not match PC1/PC2 and ordered genes.")
    if not np.isfinite(components[:2]).all():
        raise ValueError("PCA loadings contain non-finite values.")
    table = pd.DataFrame({
        "gene_name": genes, "PC1_loading": components[0], "PC2_loading": components[1],
    })
    model_path = Path(model_path)
    path = model_path.with_name(f"{model_path.stem}_gene_loadings_sorted_PC1.csv")
    table.sort_values(
        ["PC1_loading", "gene_name"], ascending=[False, True], kind="stable",
    ).to_csv(path, index=False, float_format="%.17g")
    return {"pca_gene_loadings_PC1_file": str(path)}


def filtering_by_gene_list_and_pca_plot(bulk_exp: pd.DataFrame, tcga_exp: pd.DataFrame, gene_list: list,
                                        result_dir: str, simu_dataset_name: str,
                                        n_components=5, pca_model_name_postfix='', bulk_exp_type='log_space',
                                        tcga_exp_type='TPM', pca_model_file_path=None, pca_data_file_path=None,
                                        h5ad_file_path=None, cell_frac_file: pd.DataFrame = None, figsize=(5, 5),
                                        if_plot_pca: bool = False, normalized_log_inputs: bool = False):
    """
    Applying gene-level filtering based on a specific gene list to simulated bulk GEPs.
    After filtering, the bulk GEPs are rescaled to log2(TPM+1).
    And perform PCA and plot the results for both filtered bulk GEPs and TCGA samples.

    :param bulk_exp: pd.DataFrame, Simulated bulk expression data in log2cpm1p format.
    :param tcga_exp: pd.DataFrame, TCGA expression data in TPM format.
    :param gene_list: list, List of gene names used to filter the bulk GEPs.
    :param result_dir: str, Directory where results will be saved.
    :param simu_dataset_name: str, Name of the simulated dataset
    :param n_components: int, Number of PCA components to compute. Must be >= 2.
    :param pca_model_name_postfix: str, Postfix for naming the PCA model file.
    :param bulk_exp_type: str, Type of bulk GEPs, either 'TPM' or 'log_space'.
    :param tcga_exp_type: str, Type of TCGA samples, either 'TPM' or 'log_space'.
    :param pca_model_file_path: str, Path to save the fitted PCA model.
    :param pca_data_file_path: str, Path to save the PCA-transformed data.
    :param h5ad_file_path: str, Path to save the filtered bulk GEPs as a .h5ad file, if not None.
    :param cell_frac_file: pd.DataFrame, Cell proportion matrix used to generate simulated bulk GEPs. Required if h5ad_file_path is specified.
    :param figsize: tuple, Size of the figure for PCA plotting (width, height).
    :param if_plot_pca: boolean, if True plot PCA components using both simulated bulk GEPs and TCGA samples.
    :return: None
    """
    if normalized_log_inputs:
        bulk_exp = bulk_exp.loc[:, gene_list].copy()
        tcga_exp = tcga_exp.loc[:, gene_list].copy()
    else:
        bulk_obj = ReadExp(bulk_exp, exp_type=bulk_exp_type)
        bulk_obj.align_with_gene_list(gene_list=gene_list)
        bulk_exp = bulk_obj.get_exp()

    # PCA and plot
    if if_plot_pca:
        if not normalized_log_inputs:
            bulk_linear = np.expm1(bulk_exp * np.log(2)) if bulk_exp_type == "log_space" else bulk_exp
            bulk_exp = np.log2(normalize_selected(bulk_linear, gene_list, "Comparison bulk") + 1)
            tcga_linear = np.expm1(tcga_exp * np.log(2)) if tcga_exp_type == "log_space" else tcga_exp
            tcga_exp = np.log2(normalize_selected(tcga_linear, gene_list, "Comparison reference") + 1)

        check_dir(result_dir)
        assert n_components >= 2, 'n_components must be >= 2'
        reuse_existing_pca = (
            pca_data_file_path is not None and
            pca_model_file_path is not None and
            os.path.exists(pca_data_file_path) and
            os.path.exists(pca_model_file_path)
        )
        if not reuse_existing_pca:
            if pca_data_file_path is not None and os.path.exists(pca_data_file_path):
                logging.info(
                    'PCA data exists without a reusable PCA model; recomputing PCA artifacts for %s.',
                    simu_dataset_name,
                )
            # combine both simulated bulk cell GEPs and TCGA dataset together
            simu_bulk_with_tcga = pd.concat([bulk_exp, tcga_exp])
            pca_model = do_pca_analysis(exp_df=simu_bulk_with_tcga, n_components=n_components,
                                        pca_result_fp=pca_model_file_path,
                                        save_model=pca_model_file_path is not None)
            pcs = safe_pca_transform(
                pca_model,
                simu_bulk_with_tcga,
                label='Combined simulated bulk and TCGA expression',
            )
            if n_components >= 3:
                pca_df = pd.DataFrame(pcs[:, range(3)], index=simu_bulk_with_tcga.index, columns=['PC1', 'PC2', 'PC3'])
            else:
                pca_df = pd.DataFrame(pcs[:, range(2)], index=simu_bulk_with_tcga.index, columns=['PC1', 'PC2'])
            if pca_data_file_path is not None:
                pca_df.to_csv(pca_data_file_path)
        else:
            logging.info(f'{pca_data_file_path} already exists, skip PCA analysis')
            pca_df = pd.read_csv(pca_data_file_path, index_col=0)
            pca_model = load(pca_model_file_path)

        if pca_model_file_path is not None:
            _export_pca_gene_loadings(pca_model, gene_list, pca_model_file_path)
        # plot
        title = f'{simu_dataset_name}_PCA_with_TCGA_{pca_model_name_postfix}'
        color_code = np.array([simu_dataset_name] * bulk_exp.shape[0] + ['TCGA'] * tcga_exp.shape[0])
        # cumsum = np.cumsum(pca_model.explained_variance_ratio_)
        # pca_df['class'] = color_code
        plot_pca(data=pca_df, figsize=figsize,
                 result_fp=os.path.join(result_dir, title + '.png'),
                 color_code=color_code, explained_variance_ratio=pca_model.explained_variance_ratio_)

    if h5ad_file_path is not None:
        assert cell_frac_file is not None, 'cell_frac_file should be provided if h5ad_file_path is not None'
        logging.info(f'Saving filtered bulk exp to file: {h5ad_file_path}')
        if not os.path.exists(h5ad_file_path):
            logging.info('Saving as .h5ad file...')
            dataset_info = f'Some genes were removed from this dataset for increasing the similarity ' \
                           f'between TCGA and simulated bulk cell GEPs. This similarity was evaluated by PCA analysis.'
            create_h5ad_dataset(simulated_bulk_exp_file_path=bulk_exp,
                                cell_fraction_file_path=cell_frac_file, dataset_info=dataset_info,
                                result_file_path=h5ad_file_path)


def _run_sorted_bulk_imputation_for_sct_dataset(
    dataset_name: str,
    generated_sct_fp: str | Path,
    params: Dict[str, Any],
    config: SimuConfig,
) -> Path:
    bulk_expr_value = params.get("sorted_bulk_expr_file_path", config.sorted_bulk_expr_file_path)
    bulk_annot_value = params.get("sorted_bulk_annot_file_path", config.sorted_bulk_annot_file_path)
    mapping_value = params.get(
        "sorted_bulk_cell_type_mapping_file_path",
        config.sorted_bulk_cell_type_mapping_file_path,
    )
    missing_inputs = [
        name
        for name, value in [
            ("sorted_bulk_expr_file_path", bulk_expr_value),
            ("sorted_bulk_annot_file_path", bulk_annot_value),
            ("sorted_bulk_cell_type_mapping_file_path", mapping_value),
        ]
        if value in (None, "")
    ]
    if missing_inputs:
        raise ValueError(
            f"Missing required sorted-bulk imputation config for {dataset_name}: {missing_inputs}"
        )

    generated_sct_fp = Path(generated_sct_fp)
    output_suffix = params.get("sorted_bulk_imputation_output_suffix", config.sorted_bulk_imputation_output_suffix)
    output_h5ad = _build_sorted_bulk_imputed_sct_path(generated_sct_fp, output_suffix)
    if output_h5ad.exists():
        logging.info(
            "   > Skipping optional sorted-bulk-guided SCT imputation for %s because output already exists: %s",
            dataset_name,
            output_h5ad,
        )
        return output_h5ad
    out_dir = _resolve_sct_aux_output_dir(
        sct_dataset_file_path=generated_sct_fp,
        dataset_name=dataset_name,
    )
    bulk_weight = _resolve_sorted_bulk_imputation_bulk_weight(
        params.get(
            "sorted_bulk_imputation_bulk_weight",
            config.sorted_bulk_imputation_bulk_weight,
        ),
        f'SCT dataset "{dataset_name}"',
    )

    logging.info("   > Running optional sorted-bulk-guided SCT imputation for %s", dataset_name)
    return run_sorted_bulk_guided_sct_imputation(
        input_h5ad=generated_sct_fp,
        output_h5ad=output_h5ad,
        bulk_expr=Path(bulk_expr_value),
        bulk_annot=Path(bulk_annot_value),
        cell_type_mapping=Path(mapping_value),
        out_dir=out_dir,
        output_prefix=dataset_name,
        bulk_cell_type_col=params.get("sorted_bulk_cell_type_col", config.sorted_bulk_cell_type_col),
        eps=float(params.get("sorted_bulk_imputation_eps", config.sorted_bulk_imputation_eps)),
        seed=int(params.get("sorted_bulk_imputation_seed", config.sorted_bulk_imputation_seed)),
        batch_size=int(params.get("sorted_bulk_imputation_batch_size", config.sorted_bulk_imputation_batch_size)),
        sct_boxplot_sample_size=int(
            params.get(
                "sorted_bulk_imputation_boxplot_sample_size",
                config.sorted_bulk_imputation_boxplot_sample_size,
            )
        ),
        bulk_weight=bulk_weight,
        y_frac_threshold=float(
            params.get(
                "sorted_bulk_imputation_y_frac_threshold",
                config.sorted_bulk_imputation_y_frac_threshold,
            )
        ),
    )
