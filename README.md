# SimuTME: Bulk RNA-seq Simulator for Tumor Microenvironment

SimuTME is a Python package for simulating bulk RNA-seq gene expression profiles (GEPs) for tumor microenvironment (TME) mixtures. It helps benchmark and train deconvolution models by generating synthetic bulk datasets with known ground-truth cell proportions.

## Key Features

- Configuration-driven workflow (YAML) for generating multiple datasets in one run
- Mixed (multi-cell-type) bulk simulation and SCT (single-cell-type) profile generation
- Optional sorted-bulk-guided post-imputation for SCT datasets, with diagnostics and handoff into later Mixed simulations
- Multiple proportion-sampling strategies to mimic tumor purity and immune sparsity
- Optional TCGA-guided filtering and PCA-based validation plots

## Installation

Requirements:

- Python = 3.12

From source (recommended during development):

```bash
conda create -n simutme python=3.12
conda activate simutme
pip install -e .
```

With Poetry:

```bash
poetry install
```

## Quickstart

1) Copy the example configuration and fill in your real file paths:

- Example YAML: [config_example.yaml](file:///Users/belter/github/SimuTME/src/config_example.yaml)

2) Run the workflow in Python:

```python
import yaml
from simutme import SimuConfig, run_bulk_simulation

config = SimuConfig.from_yaml("config.yaml")

with open("config.yaml", "r") as f:
    datasets_to_run = yaml.safe_load(f).get("datasets", {})

for dataset_name, params in datasets_to_run.items():
    run_bulk_simulation(dataset_name, params, config)
```

## Configuration Reference

The YAML file has four top-level sections:

- `paths`: filesystem locations (input/output files)
- `constants`: biological constants and taxonomy (cell types/subtypes)
- `parameters`: global knobs for filtering and PCA
- `datasets`: one or more dataset definitions to generate

### Required Inputs

At minimum, you need:

- `paths.merged_sc_dataset_file_path`: a merged single-cell RNA-seq (scRNA-seq) dataset (`.h5ad`)
- `constants.cell_type2subtypes`: mapping from major cell types to subtypes used for simulation

For filtering / validation you typically also provide:

- `paths.tcga_merged_tpm_file_path`
- `paths.high_corr_gene_file_path`
- `paths.tcga2cancer_type_file_path`

For the optional sorted-bulk-guided SCT imputation workflow, also provide:

- `paths.sorted_bulk_expr_file_path`: sorted bulk reference expression matrix
- `paths.sorted_bulk_annot_file_path`: annotation table for the sorted bulk samples
- `paths.sorted_bulk_cell_type_mapping_file_path`: mapping from SCT cell types to sorted bulk reference cell types
- `paths.sorted_bulk_imputation_out_dir`: parent output directory for the imputation diagnostics; SimuTME will save results under `<this_dir>/<dataset_name>/`

### Dataset Types

Each item under `datasets:` is one dataset run. Supported dataset types:

#### Mixed (multi-cell-type bulk)

Bulk GEPs with mixed cell types.

Required keys:

- `dataset_type: "Mixed"` (case-insensitive)
- `n_samples`: number of bulk samples to generate
- `sampling_method_prop`: proportion sampling strategy (see below)
- `filtering`: `true|false`

Optional keys (common):

- `cell_prop_prior`: prior constraints for some cell types (used by `segment`)
- `filtering_ref_types`: restrict filtering reference cancer types
- `filtering_in_pca_space`: `true|false`

Example:

```yaml
datasets:
  Mixed_N100_segment:
    dataset_type: "Mixed"
    n_samples: 100
    sampling_method_prop: "segment"
    filtering: true
```

#### SCT (single-cell-type profiles)

Required keys:

- `dataset_type: "SCT"` (case-insensitive)
- `n_samples`: number of SCT samples per cell type

Optional keys (common):

- `sampling_method_gep`: `random` or `n_neighbors`, defining how to sampling cells from the merged scRNA-seq dataset.
- `zero_ratio_threshold`: remove cells with high zero ratio
- `max_cells_per_cell_type`: cap for distance computation when using `n_neighbors`
- `cell_type_col`, `cell_subtype_col`: column names in `adata.obs`
- `apply_sorted_bulk_imputation`: run sorted-bulk-guided zero imputation after SCT generation

If `apply_sorted_bulk_imputation: true`, SimuTME will:

1. generate the SCT `.h5ad` dataset
2. run the sorted-bulk-guided imputation workflow on that SCT dataset
3. save the imputed SCT dataset beside the original SCT output using `parameters.sorted_bulk_imputation_output_suffix`
4. write diagnostics such as pre/post SCT sparsity plots and `y_fraction_of_total` summaries under `paths.sorted_bulk_imputation_out_dir/<dataset_name>/`
5. update `config.sct_dataset_file_path` to the imputed SCT dataset so later Mixed simulations in the same run use the imputed SCT file

When you want later Mixed datasets to consume the imputed SCT dataset, place the SCT dataset earlier than those Mixed datasets in the `datasets:` section, because the workflow runs datasets in YAML order.

Example:

```yaml
datasets:
  SCT_N10:
    dataset_type: "SCT"
    n_samples: 10
    sampling_method_gep: "n_neighbors"
    apply_sorted_bulk_imputation: true
    filtering: false
```

### Optional Sorted-Bulk-Guided SCT Imputation Parameters

The following global parameters control the optional SCT post-imputation workflow:

- `parameters.sorted_bulk_cell_type_col`: cell-type column in the sorted bulk annotation table
- `parameters.sorted_bulk_imputation_eps`: zero threshold used for sparsity and imputation logic
- `parameters.sorted_bulk_imputation_seed`: random seed for bulk-reference sampling
- `parameters.sorted_bulk_imputation_batch_size`: batch size used during SCT imputation
- `parameters.sorted_bulk_imputation_boxplot_sample_size`: number of SCT cells sampled for boxplot rendering
- `parameters.sorted_bulk_imputation_y_frac_threshold`: cap for the fraction of newly added expression per SCT sample
- `parameters.sorted_bulk_imputation_output_suffix`: suffix appended to the imputed SCT `.h5ad`

This workflow is only applied to datasets whose dataset section sets:

```yaml
dataset_type: "SCT"
apply_sorted_bulk_imputation: true
```

## Sampling Strategies

### Proportion sampling (`sampling_method_prop`)

Valid values:

- `segment`: stick-breaking style, creates high-variance compositions
- `seg_random`: one dominant fraction + remaining random background
- `sparse`: leave-out procedure, many cell types absent (zeros)
- `random`: dense random compositions (uniform then normalized)
- `dirichlet`: Dirichlet distribution (standard compositional model)
- `mixed`: evenly mixes all strategies above

Implementation: [sampling.py](file:///Users/belter/github/SimuTME/src/simutme/simulation/sampling.py)

### SCT cell sampling (`sampling_method_gep`)

Valid values:

- `random`: random selection of `n_base` cells per SCT profile
- `n_neighbors`: pick 1 random cell and average its nearest neighbors within the same type

## Outputs

- Mixed and SCT workflows write `.h5ad` datasets under `paths.simu_bulk_dir` (default GEPs are saved as $log_2 (TPM+1)$).
- When `filtering: true`, the workflow may also generate filtered versions and PCA artifacts in the dataset output directory.
- When sorted-bulk-guided SCT imputation is enabled, SimuTME additionally writes:
  - an imputed SCT `.h5ad`
  - SCT sparsity summaries before and after imputation
  - `y_fraction_of_total` summaries before and after threshold adjustment
  - sorted bulk reference sparsity summaries used by the workflow
  - all diagnostic CSVs and figures under `paths.sorted_bulk_imputation_out_dir/<dataset_name>/`

Note for mixed bulk runs: if `sampling_method_gep` is omitted in a Mixed dataset config, it is inferred from `paths.sct_dataset_file_path` when the filename includes `gepsamp-random` or `gepsamp-n_neighbors`.
If a preceding SCT dataset enabled sorted-bulk-guided imputation, then `paths.sct_dataset_file_path` is updated in memory to the imputed SCT dataset and that file is used by later Mixed runs in the same workflow execution.

## Troubleshooting

- `ModuleNotFoundError: No module named 'yaml'`: install dependencies (or use Poetry); YAML loading requires PyYAML.
- If mixed runs fail with missing `sampling_method_prop`: ensure your dataset section uses `sampling_method_prop` (not `sampling_method`).
- If memory is high with `n_neighbors` sampling: reduce `max_cells_per_cell_type` and/or the number of generated samples.

## Development

```bash
python -m compileall -q src
```

## References

The sampling methodologies are inspired by:

- Xiong, X., Liu, Y., et al. DeSide: A unified deep learning approach for cellular deconvolution of tumor microenvironment. Proc. Natl. Acad. Sci. U.S.A. 121, e2407096121 (2024).
- Menden, K., Marouf, A., Oller, S. et al. Deep learning-based cell composition analysis from tissue expression profiles. Sci Adv 6, eaba2619 (2020).
