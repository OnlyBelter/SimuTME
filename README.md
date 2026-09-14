# SimuTME: bulk RNA-seq simulator for tumor microenvironment reconstruction

SimuTME is a Python package for simulating bulk RNA-seq gene expression
profiles (GEPs) for tumor microenvironment mixtures. It helps you generate
synthetic datasets with known cell fractions for benchmarking, method
development, and training.

## Overview

SimuTME provides two linked workflows:

- `SCT`: generate single-cell-type reference profiles from a merged scRNA-seq
  dataset
- `Mixed`: generate mixed bulk RNA-seq profiles from those references using
  configurable composition sampling

For Mixed datasets, the global parameter `parameters.mixed_sct_loading_mode`
controls how the SCT reference `.h5ad` is loaded:

- `auto` uses eager in-memory loading for smaller SCT references and lazy
  backed loading for larger ones, using
  `parameters.mixed_sct_auto_eager_max_gb` as the `auto` threshold
- `eager` maximizes throughput when RAM is sufficient
- `lazy` minimizes memory usage for very large SCT references

The package is YAML-driven. One config file defines file paths, biological
constants, global parameters, and the datasets to run.

## Key capabilities

This package lets you:

- run multiple SCT and Mixed datasets from one YAML file
- choose different proportion-sampling strategies for Mixed simulation
- reference either an external SCT file or an earlier SCT dataset in the same
  run
- apply optional sorted-bulk-guided post-imputation to SCT datasets
- apply optional GEP-level and gene-level filtering to Mixed datasets

## Installation

SimuTME requires Python 3.11 or later.

Its dependency stack is aligned with
[VAEDecon](https://github.com/OnlyBelter/VAEDecon), so SimuTME can be
installed in the same environment as VAEDecon.

From source:

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

Start from the example config at [src/config_example.yaml](src/config_example.yaml),
replace the placeholder paths, and run the workflow:

```python
import yaml
from simutme import SimuConfig, run_bulk_simulation

config_path = "config.yaml"
config = SimuConfig.from_yaml(config_path)

with open(config_path, "r", encoding="utf-8") as f:
    datasets_to_run = yaml.safe_load(f).get("datasets", {})

for dataset_name, params in datasets_to_run.items():
    run_bulk_simulation(dataset_name, params, config)
```

## Documentation

Use the docs for workflow and parameter details:

- [Documentation overview](docs/index.md)
- [Installation](docs/installation.md)
- [Configuration and I/O reference](docs/configuration_io.md)
- [SCT workflow](docs/workflow_sct.md)
- [Mixed workflow](docs/workflow_mixed.md)
- [Changelog](docs/changelog.md)

## Development

Run a quick syntax check during development:

```bash
python -m compileall -q src
```

## References

SimuTME builds on ideas from:

- Xiong, X., Liu, Y., et al. DeSide: A unified deep learning approach for
  cellular deconvolution of tumor microenvironment. Proc. Natl. Acad. Sci.
  U.S.A. 121, e2407096121 (2024).
- Menden, K., Marouf, A., Oller, S. et al. Deep learning-based cell
  composition analysis from tissue expression profiles. Sci Adv 6, eaba2619
  (2020).
