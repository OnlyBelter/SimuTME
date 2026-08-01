import os
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Literal

from .utility import do_pca_analysis, plot_pca
from .utility import cal_corr_gene_exp_with_cell_frac
# Assuming these are existing imports in your project
from .utility import ReadH5AD, ReadExp, create_h5ad_dataset
from .utility import check_dir, non_log2cpm, log2_transform, get_gene_list_filtered_by_quantile_range
from .simulation import BulkGEPGenerator, SingleCellTypeGEPGenerator, GEPGenerationConfig
from .simulation.sampling import estimate_dirichlet_alpha_from_pred_cell_props, get_default_dirichlet_alpha
from .sct_imputation import run_sorted_bulk_guided_sct_imputation
from .simulation.sample_diversity import run_sct_sample_diversity_analysis


def _infer_sampling_method_gep_from_sct_path(sct_dataset_file_path: Any) -> str | None:
    if not sct_dataset_file_path:
        return None
    try:
        p = Path(sct_dataset_file_path).name.lower()
    except TypeError:
        p = str(sct_dataset_file_path).lower()
    if "n_neighbors" in p:
        return "n_neighbors"
    if "random" in p:
        return "random"
    return None


def _build_sorted_bulk_imputed_sct_path(
    sct_dataset_file_path: str | Path,
    output_suffix: str,
) -> Path:
    sct_dataset_file_path = Path(sct_dataset_file_path)
    return sct_dataset_file_path.with_name(
        f"{sct_dataset_file_path.stem}{output_suffix}{sct_dataset_file_path.suffix}"
    )


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

    output_suffix = str(
        params.get(
            "sorted_bulk_imputation_output_suffix",
            config.sorted_bulk_imputation_output_suffix,
        )
    ).strip()
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
    high_corr_gene_file_path: Path
    merged_sc_dataset_file_path: Path

    # Core constants
    sc_dataset_ids: List[str]
    cell_type2subtypes: Dict[str, List[str]]
    total_rna_coefficient: Dict[str, float]
    n_base: int
    output_types: List[str]

    # Filtering parameters
    gene_quantile_range: List[float]
    gep_filtering_quantile: Tuple[float, float]
    pca_n_components: float
    gene_list_type: str
    sorted_bulk_expr_file_path: Path | None = None
    sorted_bulk_annot_file_path: Path | None = None
    sorted_bulk_cell_type_mapping_file_path: Path | None = None
    sorted_bulk_cell_type_col: str = "Tumor_model_annot"
    sorted_bulk_imputation_eps: float = 1e-8
    sorted_bulk_imputation_seed: int = 123
    sorted_bulk_imputation_batch_size: int = 128
    sorted_bulk_imputation_boxplot_sample_size: int = 1000
    sorted_bulk_imputation_y_frac_threshold: float = 0.15
    sorted_bulk_imputation_bulk_weight: float = 0.5
    sorted_bulk_imputation_output_suffix: str = "_sorted_bulk_imputed"
    sorted_bulk_imputation_out_dir: Path | None = None
    run_sample_diversity_analysis: bool = True
    sample_diversity_gene_mode: Literal["all", "hvg", "provided_gene_list"] = "all"
    sample_diversity_n_top_genes: int = 2000
    sample_diversity_gene_list_path: Path | None = None
    sample_diversity_distance_metric: Literal["cosine", "l2"] = "cosine"
    sample_diversity_min_samples_per_cell_type: int = 5

    # Cached attributes
    _high_corr_gene_list: List[str] = field(init=False, default=None)
    _dataset_definitions: Dict[str, Dict[str, Any]] = field(init=False, default_factory=dict)
    _dataset_order: List[str] = field(init=False, default_factory=list)
    _generated_sct_outputs: Dict[str, Dict[str, Path | None]] = field(init=False, default_factory=dict)

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

        # Convert list to tuple for gep_filtering_quantile if necessary
        if 'gep_filtering_quantile' in config_data:
            config_data['gep_filtering_quantile'] = tuple(config_data['gep_filtering_quantile'])

        config = cls(**config_data)
        config._dataset_definitions = data.get("datasets", {})
        config._dataset_order = list(config._dataset_definitions.keys())
        return config

    def __post_init__(self):
        self.sorted_bulk_imputation_bulk_weight = _resolve_sorted_bulk_imputation_bulk_weight(
            self.sorted_bulk_imputation_bulk_weight,
            "global parameters",
        )
        self.sample_diversity_gene_mode = _resolve_sample_diversity_gene_mode(
            self.sample_diversity_gene_mode,
            "global parameters",
        )
        self.sample_diversity_distance_metric = _resolve_sample_diversity_distance_metric(
            self.sample_diversity_distance_metric,
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
            if os.path.exists(self.high_corr_gene_file_path):
                df = pd.read_csv(self.high_corr_gene_file_path)
                self._high_corr_gene_list = df['gene_name'].to_list()
            else:
                logging.warning(f"Warning: High corr gene file not found: {self.high_corr_gene_file_path}")
                self._high_corr_gene_list = []
        return self._high_corr_gene_list


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
    dirichlet_only_keys = {"dirichlet_alpha", "pred_cell_prop_file_path"}
    if dataset_type != 'sct':
        if 'cancer_cell_dataset_ids' in params or 'non_cancer_cell_dataset_ids' in params:
            raise ValueError(
                '"cancer_cell_dataset_ids" and "non_cancer_cell_dataset_ids" are only supported for SCT datasets.'
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
            key for key in dirichlet_only_keys
            if params.get(key, None) not in (None, "")
        ]
        if invalid_keys:
            raise ValueError(
                f'{invalid_keys} are only supported for Mixed datasets with '
                '"sampling_method_prop: dirichlet".'
            )

    # 1. Initialize generator
    simu_bulk_dir = Path(config.simu_bulk_dir)
    n_sc_datasets = len(params.get('sc_dataset_ids', config.sc_dataset_ids))
    n_base = config.n_base
    if dataset_type == 'sct':  # single-cell-type dataset
        logging.info(f"   > Using Single-Cell-Type GEP Generation for {dataset_name}")
        if 'sampling_method_gep' not in params:
            logging.info("   > Warning: 'sampling_method_gep' not specified in params, defaulting to 'random'")
            sampling_method_gep = 'random'
        else:
            sampling_method_gep = params.get('sampling_method_gep')

        sc_dataset_dir = simu_bulk_dir / f'generated_sct_dataset_{n_sc_datasets}ds_n_base{n_base}_all_subtypes'
        sct_generator = SingleCellTypeGEPGenerator(
            simu_bulk_dir=sc_dataset_dir,
            cell_type2subtype=config.cell_type2subtypes,
            sc_dataset_ids=config.sc_dataset_ids,
            cancer_cell_dataset_ids=params.get('cancer_cell_dataset_ids', None),
            non_cancer_cell_dataset_ids=params.get('non_cancer_cell_dataset_ids', None),
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
                                           max_cells_per_cell_type=params.get('max_cells_per_cell_type', 15000)
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
        if params.get('filtering', False):
            filtering_str = f'with_filtering_{config.gep_filtering_quantile[1]}'
        else:
            filtering_str = 'no_filtering'
        if not 'sampling_method_prop' in params:
            logging.info("   > Warning: 'sampling_method_prop' not specified in params, defaulting to 'segment'")
            sampling_method_prop = 'segment'
        else:
            sampling_method_prop = params.get('sampling_method_prop')
        if sampling_method_prop != 'dirichlet':
            invalid_keys = [
                key for key in dirichlet_only_keys
                if params.get(key, None) not in (None, "")
            ]
            if invalid_keys:
                raise ValueError(
                    f'{invalid_keys} are only supported when "sampling_method_prop" is "dirichlet" '
                    f'for Mixed dataset "{dataset_name}".'
                )
            resolved_dirichlet_alpha = None
            pred_cell_prop_file_path = None
        else:
            resolved_dirichlet_alpha = params.get(
                'dirichlet_alpha',
                get_default_dirichlet_alpha(params.get('cell_type2subtypes', config.cell_type2subtypes)),
            )
            pred_cell_prop_file_path = params.get('pred_cell_prop_file_path', None)
        if not 'sampling_method_gep' in params:
            inferred = _infer_sampling_method_gep_from_sct_path(resolved_sct_dataset_file_path)
            if inferred:
                logging.info(
                    "   > Warning: 'sampling_method_gep' not specified in params, inferred from sct_dataset_file_path:"
                    f" {inferred}"
                )
                sampling_method_gep = inferred
            else:
                raise ValueError(
                    f'Missing "sampling_method_gep" for Mixed dataset "{dataset_name}". '
                    f'Provide params["sampling_method_gep"] explicitly, or include a marker in '
                    f'the resolved SCT reference filename (e.g., "random" or "n_neighbors"). '
                    f"Current sct_dataset_file_path: {resolved_sct_dataset_file_path}"
                )
        else:
            sampling_method_gep = params.get('sampling_method_gep')

        simu_bulk_dir = simu_bulk_dir / f'{sampling_method_prop}_{n_sc_datasets}ds_n_base{n_base}_{filtering_str}_median_gep'
        bulk_generator = BulkGEPGenerator(
            simu_bulk_dir=simu_bulk_dir,
            merged_sc_dataset_file_path=config.merged_sc_dataset_file_path,
            cell_type2subtype=params.get('cell_type2subtypes', config.cell_type2subtypes),
            sc_dataset_ids=config.sc_dataset_ids,
            bulk_dataset_name=dataset_name,
            sct_dataset_file_path=resolved_sct_dataset_file_path,
            check_basic_info=False,
            tcga2cancer_type_file_path=config.tcga2cancer_type_file_path,
            total_rna_coefficient=config.total_rna_coefficient,
            cell_type_col_name='cell_type',
            subtype_col_name='cell_type',
            sampling_method_gep=sampling_method_gep,
        )

        generated_bulk_gep_fp = bulk_generator.generated_bulk_gep_fp

        # 2. Generate GEP (if it doesn't exist)
        if not os.path.exists(generated_bulk_gep_fp):
            gep_gen_config = GEPGenerationConfig(
                n_samples=params['n_samples'],
                simu_method='mul',  # matrix multiplication by cell proportions and sampled GEPs to get bulk GEPs
                sampling_method_prop=sampling_method_prop,
                reference_file=config.tcga_merged_tpm_file_path,
                ref_exp_type='TPM',
                filtering=params.get('filtering', False),
                filtering_ref_types=params.get('filtering_ref_types', None),
                gep_filtering_quantile=config.gep_filtering_quantile,
                n_threads=5,
                log_file_path=log_file_path,
                show_filtering_info=False,
                filtering_method='median_gep',
                cell_prop_prior=params.get('cell_prop_prior', None),
                dirichlet_alpha=resolved_dirichlet_alpha,
                pred_cell_prop_file_path=pred_cell_prop_file_path,
                filtering_in_pca_space=params.get('filtering_in_pca_space', True),
                norm_ord=1,
                pca_n_components=config.pca_n_components
            )
            bulk_generator.generate_gep(config=gep_gen_config)
        else:
            logging.info(f"   > GEP file exists: {generated_bulk_gep_fp}")

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

        # 3. Post-processing: Gene-level filtering and PCA (if needed)
        if params.get('filtering', False):
            _run_post_processing(dataset_name, generated_bulk_gep_fp, config)

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


def _run_post_processing(dataset_name: str,
                         generated_bulk_gep_fp: str,
                         config: SimuConfig):
    """Handles complex gene filtering, intersection logic, and PCA plotting."""

    # Define output filenames
    replace_by = f'_filtered_by_{config.gene_list_type}.h5ad'
    filtered_file_path = generated_bulk_gep_fp.replace('.h5ad', replace_by)

    if os.path.exists(filtered_file_path):
        logging.info(f"   > Filtered file exists: {filtered_file_path}")
        return

    logging.info(f"   > Running post-processing (Gene Filtering & PCA) for {dataset_name}...")
    current_result_dir = os.path.join(config.simu_bulk_dir, dataset_name)
    check_dir(current_result_dir)

    # A. Get/Generate Quantile Gene List
    gene_list = config.high_corr_genes.copy()  # Start with high corr genes

    if 'quantile_range' in config.gene_list_type:
        q_names = [str(x) for x in config.gene_quantile_range]  # Simplified naming logic
        # Note: Filename generation logic is simplified here; you may need to adjust q_names based on original code.
        csv_name = f'gene_list_filtered_by_{config.gene_list_type}_{q_names[0]}_{q_names[-1]}.csv'
        gene_list_file_path = os.path.join(current_result_dir, csv_name)

        if not os.path.exists(gene_list_file_path):
            logging.info(f'      Generating gene list: {gene_list_file_path}')
            quantile_gene_list = get_gene_list_for_filtering(
                bulk_exp_file=generated_bulk_gep_fp,
                filtering_type='quantile_range',
                tcga_file=config.tcga_merged_tpm_file_path,
                quantile_range=config.gene_quantile_range,
                result_file_path=gene_list_file_path,
                q_col_name=q_names
            )
        else:
            logging.info(f'      Loading existing gene list: {gene_list_file_path}')
            quantile_gene_list = pd.read_csv(gene_list_file_path)['gene_name'].to_list()

        # Take intersection
        gene_list = [g for g in gene_list if g in quantile_gene_list]

    # B. Prepare data for PCA
    bulk_exp_obj = ReadH5AD(generated_bulk_gep_fp)
    bulk_exp = bulk_exp_obj.get_df()
    bulk_exp_cell_frac = bulk_exp_obj.get_cell_fraction()

    # Note: Reading large TCGA files is time-consuming.
    # If calling in a loop, consider passing pre-loaded data or using system caching.
    tcga_exp = ReadExp(config.tcga_merged_tpm_file_path, exp_type='TPM').get_exp()

    pca_model_fp = os.path.join(current_result_dir,
                                f'both_TCGA_and_simu_{dataset_name}_PCA_{config.gene_list_type}.joblib')
    pca_data_fp = os.path.join(current_result_dir, f'{dataset_name}_PCA_with_TCGA_{config.gene_list_type}.csv')

    # C. Execute filtering and plotting
    filtering_by_gene_list_and_pca_plot(
        bulk_exp=bulk_exp,
        tcga_exp=tcga_exp,
        gene_list=gene_list,
        result_dir=current_result_dir,
        n_components=2,
        simu_dataset_name=dataset_name,
        pca_model_name_postfix=config.gene_list_type,
        pca_model_file_path=pca_model_fp,
        pca_data_file_path=pca_data_fp,
        h5ad_file_path=filtered_file_path,
        cell_frac_file=bulk_exp_cell_frac,
        figsize=(5, 5)
    )


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

    dataset_result_root_value = params.get("sorted_bulk_imputation_out_dir", config.sorted_bulk_imputation_out_dir)
    if dataset_result_root_value in (None, ""):
        dataset_result_root = Path(config.simu_bulk_dir)
    else:
        dataset_result_root = Path(dataset_result_root_value)
    dataset_result_dir = dataset_result_root / dataset_name

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
                                corr_result_fp: str = None, quantile_range: list = None) -> list:
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
    :param quantile_range: tuple,  (lower_quantile, median_quantile, upper_quantile). Genes are removed if their median expression in simulated bulk GEPs is outside this range in TCGA data
        of the corresponding gene in TCGA dataset will be removed


    :param corr_threshold: float, correlation threshold for gene filtering
    :param n_gene_max: int, maximum number of genes to select for each cell type during 'high_corr_gene' filtering
    :return: list, filtered gene list
    """

    assert filtering_type in ['high_corr_gene', 'quantile_range', 'all_genes', 'high_corr_gene_and_quantile_range'], \
        f'filtering_type: {filtering_type} is not supported, only support high_corr_gene, quantile_range, all_genes, ' \
        f'high_corr_gene_and_quantile_range'
    if not os.path.exists(result_file_path):
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
            corr_df = cal_corr_gene_exp_with_cell_frac(gene_exp=gene_exp, cell_frac=cell_frac,
                                                       result_file_path=corr_result_fp, filtered_by_corr=corr_threshold,
                                                       filter_by_num=n_gene_max)
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


def filtering_by_gene_list_and_pca_plot(bulk_exp: pd.DataFrame, tcga_exp: pd.DataFrame, gene_list: list,
                                        result_dir: str, simu_dataset_name: str,
                                        n_components=5, pca_model_name_postfix='', bulk_exp_type='log_space',
                                        tcga_exp_type='TPM', pca_model_file_path=None, pca_data_file_path=None,
                                        h5ad_file_path=None, cell_frac_file: pd.DataFrame = None, figsize=(5, 5),
                                        if_plot_pca: bool = False):
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
    bulk_obj = ReadExp(bulk_exp, exp_type=bulk_exp_type)
    bulk_obj.align_with_gene_list(gene_list=gene_list)
    bulk_exp = bulk_obj.get_exp()

    # PCA and plot
    if if_plot_pca:
        tcga_obj = ReadExp(tcga_exp, exp_type=tcga_exp_type)
        tcga_obj.align_with_gene_list(gene_list=gene_list)
        tcga_obj.to_log2cpm1p()
        tcga_exp = tcga_obj.get_exp()

        check_dir(result_dir)
        assert n_components >= 2, 'n_components must be >= 2'
        if not os.path.exists(pca_data_file_path):
            # combine both simulated bulk cell GEPs and TCGA dataset together
            simu_bulk_with_tcga = pd.concat([bulk_exp, tcga_exp])
            pca_model = do_pca_analysis(exp_df=simu_bulk_with_tcga, n_components=n_components,
                                        pca_result_fp=pca_model_file_path)
            pcs = pca_model.transform(simu_bulk_with_tcga)
            if n_components >= 3:
                pca_df = pd.DataFrame(pcs[:, range(3)], index=simu_bulk_with_tcga.index, columns=['PC1', 'PC2', 'PC3'])
            else:
                pca_df = pd.DataFrame(pcs[:, range(2)], index=simu_bulk_with_tcga.index, columns=['PC1', 'PC2'])
            pca_df.to_csv(pca_data_file_path)
        else:
            logging.info(f'{pca_data_file_path} already exists, skip PCA analysis')
            pca_df = pd.read_csv(pca_data_file_path, index_col=0)
            pca_model = load(pca_model_file_path)

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
    out_dir_parent_value = params.get("sorted_bulk_imputation_out_dir", config.sorted_bulk_imputation_out_dir)
    out_dir_parent = Path(out_dir_parent_value) if out_dir_parent_value not in (None, "") else generated_sct_fp.parent
    out_dir = out_dir_parent / dataset_name
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
