import yaml
from simutme import run_bulk_simulation, SimuConfig


if __name__ == '__main__':
    # 1. Load Configuration
    config_path = 'config_example.yaml'
    config = SimuConfig.from_yaml(config_path)

    # 2. Load Dataset Definitions from YAML
    with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
        datasets_to_run = yaml_data.get('datasets', {})

    # 3. Run Loop
    for dataset_name, params in datasets_to_run.items():
        run_bulk_simulation(dataset_name, params, config)
