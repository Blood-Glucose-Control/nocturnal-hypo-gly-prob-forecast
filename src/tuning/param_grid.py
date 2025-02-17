import numpy as np
from src.tuning.load_estimators import get_estimator, load_all_forecasters
from src.utils.config_loader import load_yaml_config


def generate_param_grid(model_name, config):
    """Generate a parameter grid for sktime's ForecastingGridSearchCV.

    Args:
        model_name (str): Name of the forecasting model (e.g., "AutoARIMA") as listed in config.
        config (dict): Model-specific hyperparameter settings loaded from YAML.

    Returns:
        dict: A parameter grid where keys are parameter names and values are lists of possible values.

    Raises:
        ValueError: If the model is not found in the config or if a parameter type is unsupported.
    """
    param_grid = {}

    if model_name not in config:
        raise ValueError(f"Model '{model_name}' not found in YAML config.")

    model_params = config[model_name]

    for param, details in model_params.items():
        param_type = details["type"]

        if param_type == "list":
            param_grid[param] = details["values"]
        elif param_type == "int":
            start, end = details["range"]
            param_grid[param] = list(range(start, end + 1))
        elif param_type == "float":
            start, end, step = details["range"]
            param_grid[param] = np.arange(start, end + step, step).tolist()
        elif param_type == "bool":
            param_grid[param] = [True, False]
        elif param_type == "estimator":
            estimators = details["estimators"]
            estimator_kwargs = details["estimator_kwargs"]
            forecasters = load_all_forecasters()
            param_grid[param] = [
                get_estimator(forecasters, estimator)(**kwargs)
                for estimator, kwargs in zip(estimators, estimator_kwargs)
            ]
        else:
            raise ValueError(f"Unsupported param type '{param_type}' for {param}")

    return param_grid


if __name__ == "__main__":
    # Example Usage
    config = load_yaml_config("./src/tuning/configs/modset1.yaml")
    param_grid = generate_param_grid("AutoARIMA", config)
    print(param_grid)
