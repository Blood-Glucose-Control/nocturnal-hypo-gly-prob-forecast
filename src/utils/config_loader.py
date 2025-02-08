import yaml


def load_yaml_config(file_path):
    """Load YAML configuration file from relative path"""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)
