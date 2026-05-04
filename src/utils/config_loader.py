"""
Configuration loading utilities for the Nocturnal Hypoglycemia Probability Forecasting project.

This module provides functions for loading configuration data from YAML files,
ensuring consistent handling of configuration settings throughout the project.

Functions:
    load_yaml_config: Loads and parses YAML configuration files with UTF-8 encoding.
"""

import yaml


def load_yaml_config(file_path):
    """Load YAML configuration file from relative path"""
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
