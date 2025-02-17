from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.registry import all_estimators
import yaml
import pandas as pd


def quantile_forecast_model_loop(
    y_train,
    quantiles=[0.05, 0.95],
    model_hyperparameters={},
    fit_hyperparameters={},
    skip_models=[],
):
    """
    Loops through all models that support interval prediction (specifically quantile prediction),
    trains the model on the given data, and returns the trained model objects.

    Args:
        y_train (pd.DataFrame): The features to train on.
        quantiles (list): The quantiles to predict (eg: [0.05, 0.95] predicts for the quantiles 0.05 and 0.95).
        model_hyperparameters (dict): A dictionary of model hyperparameters to pass to the model. The keys should be the model names.
            And the values should be a list of dictionaries, where each dictionary represents a set of hyperparameters. Eg:
            model_hyperparameters = {
                "NaiveForecaster": [
                    {
                        "strategy": "last",
                        "sp": 12,
                        ... [other hyperparameters]
                    },
                    {
                        "strategy": "last",
                        "sp": 24,
                        ... Another set of parameters
                    }
                ]
                "Model2": [
                    ...
                ]
            }
        fit_hyperparameters: a dictionary from model name to hyperparameters used while calling fit()
        skip_models (list): a list of model names you want to skip training

    Returns:
        models (dict): A dictionary of trained model objects. The keys will be the model names,
            and the values will be a list of trained model objects corresponding to the different hyperparameter sets.
    """
    # TODO: skip model / add defaults if required hyperparameters exist

    # loads models that support interval prediction
    all_models = all_estimators(
        "forecaster", filter_tags={"capability:pred_int": True}, as_dataframe=True
    )

    models = {}
    # loop through each row, and import from all_models["Object"]
    for index, row in all_models.iterrows():
        if row["name"] in skip_models:
            print("Skipping model " + row["name"])
            continue
        print("Training model " + row["name"] + "...")
        model_name = row["name"]
        cls = row["object"]

        # get the list of hyperparameter sets for this model, if available
        param_sets = model_hyperparameters.get(
            model_name, [{}]
        )  # default to [{}] if no hyperparameters are provided

        # loop over each parameter set and train a separate model
        model_instances = []
        fit_hypers = fit_hyperparameters.get(model_name, {})
        for param_set in param_sets:
            print("Training on hyperparameters: ", param_set)
            try:
                instance = cls(**param_set)
                instance.fit(y_train, **fit_hypers)
                model_instances.append(instance)
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                continue

        # store the list of trained models
        models[model_name] = model_instances

    print("TOTAL MODELS")
    print(len(models))
    return models


def get_quantile_forecasts(
    models, y_test, forecast_horizon=None, quantiles=[0.05, 0.95]
):
    """
    Loops through the trained models, and returns the quantile forecasts for each model.

    Args:
        models (dict): A dictionary of trained model objects. The keys will be the model names,
            and the values will be the list of trained model objects (list because you may have used different hyperparameters!).
        y_test (pd.DataFrame): The features to predict on.
        forecast_horizon (list[int]): The forecast horizon for each model. If None, the forecast horizon will be the same as the training data.
        quantiles (list): The quantiles to predict (eg: [0.05, 0.95] predicts for the quantiles 0.05 and 0.95).

    Returns:
        quantile_forecasts (dict): A dictionary of quantile forecasts. The keys will be the model names,
            and the values will be a list of the quantile forecasts for each model.
    """
    quantile_forecasts = {}
    for model_name, models in models.items():
        print("Forecasting for model ", model_name)
        quantile_forecasts[model_name] = []
        for model in models:
            predictions = model.predict_quantiles(
                X=y_test, alpha=quantiles, fh=forecast_horizon
            )
            # Ensure predictions are handled correctly (e.g., as a DataFrame or array)
            quantile_forecasts[model_name].append(
                pd.DataFrame(predictions, columns=[f"quantile_{q}" for q in quantiles])
            )

    return quantile_forecasts


if __name__ == "__main__":
    # load hyperparameters from YAML file
    yaml_file_path = "./quantile_forecast_hyperparameters.yaml"
    with open(yaml_file_path, "r") as file:
        config = yaml.safe_load(file)
    # load model hyperparameters
    model_hyperparameters = config.get("model_hyperparameters", {})

    # TODO: load data (split into train and test)
    sample_y = load_airline()
    y_train, y_test = temporal_train_test_split(sample_y, test_size=36)

    # train models
    models = quantile_forecast_model_loop(
        y_train, model_hyperparameters=model_hyperparameters
    )
    # get quantile forecasts
    quantile_forecasts = get_quantile_forecasts(models, y_test)
    # TODO: plot forecasts and store in "results" folder
