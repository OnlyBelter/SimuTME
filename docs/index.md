# SimuTME Documentation

## Overview

SimuTME is a Python package for simulating bulk RNA-seq gene expression profiles
(GEPs) for tumor microenvironment studies. It supports two closely related
workflows:

1. `SCT` dataset simulation:
   generate single-cell-type reference profiles by sampling and averaging cells
   from a merged scRNA-seq dataset.
2. `Mixed` bulk RNA-seq simulation:
   generate mixed-cell-type bulk profiles by sampling cell-type proportions and
   composing bulk GEPs from reference profiles.

The package is configuration-driven. A single YAML file defines:

- file paths
- biological constants
- global simulation parameters
- one or more datasets to generate

SimuTME can also apply an optional sorted-bulk-guided post-imputation workflow
to SCT datasets. When enabled, the imputed SCT dataset becomes the in-memory
reference for later Mixed simulations in the same run.

## Documentation Map

- [Command-line usage and input paths](usage.md)
- [Installation](installation.md)
- [Changelog](changelog.md)
- [SCT Workflow](workflow_sct.md)
- [Mixed Workflow](workflow_mixed.md)
- [Configuration and I/O Reference](configuration_io.md)

## Package Scope

SimuTME is designed for simulation and benchmarking workflows where the user
needs:

- SCT reference datasets with controlled sampling behavior
- mixed bulk datasets with known ground-truth cell proportions
- optional downstream filtering and PCA-based validation
- optional SCT refinement using sorted bulk reference datasets

## Workflow Figures

### SCT Simulation

![SCT workflow](figures/workflow_sct.svg)

### Mixed Bulk Simulation

![Mixed workflow](figures/workflow_bulk.svg)

## Typical Usage Pattern

For the command-line runner, copy `src/config_example.yaml`, update its input
paths, and run `python src/main_simu_bulk.py --config config.yaml` from the
repository root. The [CLI usage guide](usage.md) explains how to prepare each
input and which files depend on optional workflows.

When you use the Python API directly:

1. Prepare a YAML configuration file.
2. Load the configuration with `SimuConfig.from_yaml(...)`.
3. Read the `datasets` section from the same YAML file.
4. Run each dataset in order with `run_bulk_simulation(...)`.

Example:

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

## Important Ordering Rule

If a Mixed dataset should use an imputed SCT dataset generated earlier in the
same run, place the SCT dataset earlier than the Mixed dataset in the YAML
`datasets:` section. SimuTME processes datasets in YAML order.
