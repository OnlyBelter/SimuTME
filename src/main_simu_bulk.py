"""Run SimuTME datasets from YAML.

Relative paths inside the configuration resolve from the current working
directory.
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
import warnings


logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config", required=True, type=Path,
        help="YAML configuration file; all datasets run in their configured order.",
    )
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Configuration file does not exist or is not a file: {config_path}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    warnings.filterwarnings(
        "ignore", category=FutureWarning,
        message=".*dtype argument is deprecated.*",
    )

    # Parse CLI arguments before importing the scientific stack so --help
    # and invalid paths work even without the simulation dependencies.
    import yaml
    from simutme import SimuConfig, run_bulk_simulation
    from simutme.workflow import (
        _register_statistics_inputs,
        preflight_segment_cell_prop_priors,
    )

    with config_path.open(encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    if not isinstance(yaml_data, dict):
        parser.error("Configuration must contain a YAML mapping.")
    datasets_to_run = yaml_data.get("datasets", {})
    if not isinstance(datasets_to_run, dict) or not datasets_to_run:
        parser.error("Configuration must contain a nonempty 'datasets' mapping.")
    for dataset_name, params in datasets_to_run.items():
        if not isinstance(params, dict):
            parser.error(f"Dataset '{dataset_name}' must contain a parameter mapping.")

    config = SimuConfig.from_yaml(str(config_path))
    preflight_segment_cell_prop_priors(config, datasets_to_run)
    statistics_inputs = set()
    for name, params in datasets_to_run.items():
        if (str(params.get("dataset_type", "mixed")).lower() != "sct"
                and params.get("simu_mode", getattr(config, "simu_mode", "full_simulation"))
                == "simulation_for_stat"):
            _register_statistics_inputs(name, params, config)
            reference = params.get("sct_reference", {})
            if isinstance(reference, dict) and reference.get("source_type") == "generated_dataset":
                statistics_inputs.add(reference.get("dataset_name"))
            correlation = params.get(
                "gene_filtering_correlation_reference_dataset",
                getattr(config, "gene_filtering_correlation_reference_dataset", None),
            )
            if correlation:
                statistics_inputs.add(correlation)
    try:
        mp.set_start_method("spawn")
    except (RuntimeError, NotImplementedError):
        logger.warning("Could not set multiprocessing start method to spawn.")

    logger.info("Using configuration: %s", config_path)
    for dataset_name, params in datasets_to_run.items():
        if dataset_name in statistics_inputs:
            logger.info("Using existing input for statistics: %s", dataset_name)
            continue
        if (getattr(config, "simu_mode", "full_simulation") == "simulation_for_stat"
                and str(params.get("dataset_type", "mixed")).lower() == "sct"
                and "simu_mode" not in params):
            logger.info("Skipping SCT generation in statistics mode: %s", dataset_name)
            continue
        logger.info("Running dataset: %s", dataset_name)
        run_bulk_simulation(dataset_name, params, config)

    logger.info("All simulations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
