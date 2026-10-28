# Installation

## Requirements

- Python `>= 3.12`

## Install From Source

This is the recommended method during development.

```bash
conda create -n simutme python=3.12
conda activate simutme
pip install -e /path/to/SimuTME
```

## Install With Poetry

```bash
cd /path/to/SimuTME
poetry install
```

## Main Dependencies

The package depends on:

- `pandas`
- `pyyaml`
- `joblib`
- `scienceplots`
- `scanpy`
- `tqdm`

These are defined in `pyproject.toml`.

## Verify The Installation

You can do a quick import check:

```bash
python - <<'PY'
from simutme import SimuConfig, run_bulk_simulation
print("SimuTME import OK")
PY
```

You can also run a lightweight syntax check on the source tree:

```bash
python -m compileall -q src
```

## Package Entry Points

Run configured datasets from the repository root with the command-line script:

```bash
python src/main_simu_bulk.py --config config.yaml
```

Prepare `config.yaml` from [the example configuration](../src/config_example.yaml)
and replace its placeholder paths before running. See the
[CLI usage guide](usage.md) for input-file requirements and workflow-specific
dependencies. The script runs every dataset in the YAML file in order.

You can also use the package from Python by importing:

- `SimuConfig`
- `run_bulk_simulation`

Minimal example:

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

## Recommended Project Layout

At minimum, keep:

- one YAML configuration file
- one merged single-cell `.h5ad` input
- one output directory for generated datasets

If you enable the optional SCT post-imputation workflow, you will also need:

- a sorted bulk expression matrix
- a sorted bulk annotation table
- a cell-type mapping file between SCT and sorted bulk cell types
