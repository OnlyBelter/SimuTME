# CLI usage and configuration path guidance

## Goal

Make the command-line runner straightforward to use with a specific YAML file,
remove its email-notification feature, and explain how to populate the example
configuration with local input data.

## Design

Keep `src/main_simu_bulk.py` as the supported command-line entry point. It
accepts one required `--config PATH` argument, validates the selected YAML,
then runs every dataset in its `datasets` mapping in YAML order. Remove the
email configuration argument, environment-variable lookup, SMTP function, and
email-only imports. Keep the existing YAML and dataset validation and
non-interactive behavior.

Add a concise CLI usage page linked from the README, documentation index, and
installation guide. Show how to copy `src/config_example.yaml` to a working
configuration, edit it, and run it from the repository root:

```bash
cp src/config_example.yaml config.yaml
python src/main_simu_bulk.py --config config.yaml
```

Explain that every configured dataset runs, relative paths in the YAML resolve
from the current working directory, and placeholder paths must be replaced
before running. Include `--help` and invalid-path behavior.

The guide will map each top-level `paths` field to its role and preparation
source:

- `simu_bulk_dir` is a writable output directory the user chooses.
- `merged_sc_dataset_file_path` is the user's merged and annotated single-cell
  `.h5ad`; the configured `sc_dataset_ids` and cell-type columns must match it.
- `tcga2cancer_type_file_path` and `tcga_merged_tpm_file_path` point to the
  user's prepared TCGA sample annotations and bulk expression matrix when
  enabled Mixed filtering or comparison requires them.
- `high_corr_gene_file_path` is an optional shared gene-list fallback. Explain
  when a supplied list is needed and when a correlation-source dataset can be
  used instead.
- The three `sorted_bulk_*` paths are needed only when sorted-bulk imputation
  is enabled. Explain that users must supply a matched expression matrix,
  sample annotation, and cell-type mapping with the required `included`
  column, or disable imputation.

State that SimuTME does not download or include these biological datasets.
Users should use their existing preprocessing outputs and follow the detailed
file-format requirements in `configuration_io.md`, `workflow_sct.md`, and
`workflow_mixed.md`. Clarify that Mixed `sct_reference` may require a separate
existing SCT `.h5ad` when configured with `source_type: "path"`.

Retain the Python API quickstart. Do not add a packaged console-script entry
point or change the example YAML schema.

## Verification

- `python src/main_simu_bulk.py --help` displays the required config option and
  no email options without importing the scientific stack.
- A nonexistent config path exits with an argparse error.
- `python -m compileall -q src` succeeds.
- Documentation links resolve, and the CLI/path instructions match the
  example YAML and configuration reference.
