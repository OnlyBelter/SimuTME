# Command-line usage

Use the command-line runner to execute the datasets defined in a YAML
configuration. The example configuration is a template: replace its input
paths and adjust its optional workflows to match the files available to you.

## Run a configuration

Run these commands from the SimuTME repository root in an environment where
SimuTME and its dependencies are installed:

```bash
cp src/config_example.yaml config.yaml
```

Edit `config.yaml`, then start the simulation:

```bash
python src/main_simu_bulk.py --config config.yaml
```

The runner executes every entry under `datasets` in YAML order. To run only
some datasets, keep only those entries in your working configuration. Paths in
the YAML are resolved from the current working directory, so relative paths
such as `./datasets/...` are relative to the repository root when you run the
command above. Use absolute paths when your data lives elsewhere.

To see the available command-line options, run:

```bash
python src/main_simu_bulk.py --help
```

The runner checks that the configuration file exists and that it contains a
nonempty `datasets` mapping. It reports invalid configuration structure before
starting the simulation.

## Set input paths

The `src/config_example.yaml` file lists the supported paths under `paths`.
SimuTME does not download or include the biological input datasets; point each
entry to files prepared for your analysis. In particular, build the merged
single-cell file from the datasets named in `constants.sc_dataset_ids`, and
use TCGA expression and annotations from the same reference-data preparation
workflow so their sample IDs and labels stay aligned.

| Configuration field | What it points to | Where to get or prepare it |
| --- | --- | --- |
| `simu_bulk_dir` | Writable directory for generated datasets and diagnostics. | Choose a project output directory. SimuTME creates output subdirectories under it. |
| `merged_sc_dataset_file_path` | Merged single-cell AnnData file (`.h5ad`) used to generate SCT profiles. | Use the merged and annotated single-cell dataset produced by your preprocessing workflow. Confirm that its dataset IDs and cell-type labels match `constants.sc_dataset_ids`, `constants.cell_type2subtypes`, and the configured `cell_type_col` and `cell_subtype_col`. |
| `tcga_merged_tpm_file_path` | TCGA reference expression matrix used by Mixed filtering and PCA comparison in the example configuration. | Use the processed, linear-TPM TCGA expression matrix from your analysis workflow. It must have samples as rows and genes as columns, with sample IDs matching the annotation file when both are used. |
| `tcga2cancer_type_file_path` | TCGA sample-to-cancer-type annotation. | Use the annotation table corresponding to the expression matrix. SimuTME reads sample IDs from the first column as the index and expects a `cancer_type` column. It is needed by cancer-type selection and local KNN filtering workflows. |
| `high_corr_gene_file_path` | Optional shared high-correlation gene-list file. | Use a previously generated or prepared gene list with a `gene_name` column or one column of gene names. Set it to `null` when you do not have a reusable list; filtering workflows that need a list must then configure a correlation source as described in the [configuration reference](configuration_io.md). |
| `sorted_bulk_expr_file_path` | Sorted-bulk reference expression matrix for optional SCT imputation. | Use the linear-TPM expression data for the sorted-bulk reference cohort selected for your analysis. The file must contain gene names in the first column and sample IDs in the header; values must be numeric. |
| `sorted_bulk_annot_file_path` | Sample annotations for the sorted-bulk expression matrix. | Use the matching annotation table. Its sample IDs must match the expression-matrix header, and it must include the cell-type column named by `parameters.sorted_bulk_cell_type_col` (default: `Tumor_model_annot`). |
| `sorted_bulk_cell_type_mapping_file_path` | Mapping between generated SCT labels and sorted-bulk cell-type labels. | Prepare a CSV with `cell_type_SCT_N6K_13Jul`, `cell_type_Kassandra_SortedBulk`, and `included` columns. Set `included` to `1` or `0` for each mapping row. |

Keep the five core path keys shown in the example because `SimuConfig` expects
them. Set an unused reference, such as `high_corr_gene_file_path` or a TCGA
path that no enabled workflow needs, to `null` instead of deleting its key.
Provide real files whenever an enabled filtering or comparison workflow uses
those references.

Keep each expression matrix aligned with its own sample annotations. TCGA and
sorted-bulk references serve different workflows, so do not substitute one
reference for the other. Keep source and preprocessing details with your
analysis so the simulation can be reproduced.

## Match paths to enabled workflows

The example YAML enables several optional operations, so it needs more input
files than a basic SCT simulation:

- Mixed GEP filtering and PCA comparison use the TCGA expression reference.
- Local KNN filtering or cancer-type-specific reference selection also needs
  the TCGA annotation file.
- `apply_sorted_bulk_imputation: true` on the example SCT dataset requires all
  three sorted-bulk paths. If you do not have a matched sorted-bulk reference,
  set `apply_sorted_bulk_imputation: false` and change any Mixed reference from
  `version: "sorted_bulk_imputed"` to `version: "original"`.
- A Mixed dataset with `sct_reference.source_type: "path"` needs an existing
  SCT `.h5ad` at `sct_reference.file_path`. Alternatively, use
  `source_type: "generated_dataset"` to refer to an SCT dataset earlier in the
  same configuration.
- When Mixed gene filtering uses `high_corr_gene`, provide a reusable gene
  list or configure an earlier unfiltered Mixed dataset as its correlation
  source. See the [Mixed workflow guide](workflow_mixed.md) and
  [configuration reference](configuration_io.md) for the selection rules.

If you do not have an optional reference, disable the workflow that consumes
it or change the relevant dataset settings. Do not leave example values such
as `path/to/merged_tpm.csv` in a configuration you run.

## Example path configuration

Replace these example locations with files available on your system. The
absolute paths below are illustrative:

```yaml
paths:
  simu_bulk_dir: "./results/simutme"
  tcga2cancer_type_file_path: "/data/reference/tcga_sample_to_cancer_type.csv"
  tcga_merged_tpm_file_path: "/data/reference/tcga_tpm_samples_by_genes.csv"
  high_corr_gene_file_path: null
  merged_sc_dataset_file_path: "/data/single_cell/merged_sc_dataset.h5ad"
  sorted_bulk_expr_file_path: "/data/sorted_bulk/expression.csv"
  sorted_bulk_annot_file_path: "/data/sorted_bulk/annotation.tsv"
  sorted_bulk_cell_type_mapping_file_path: "/data/sorted_bulk/cell_type_mapping.csv"
```

For detailed input formats, configuration precedence, and output descriptions,
see the [configuration and I/O reference](configuration_io.md), the
[SCT workflow guide](workflow_sct.md), and the [Mixed workflow guide](workflow_mixed.md).
