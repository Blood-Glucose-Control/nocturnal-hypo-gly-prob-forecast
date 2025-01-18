from sktime.registry import all_estimators
import importlib

def quantile_forecast_model_loop(y_train, quantiles=[0.05, 0.95], model_hyperparameters={}):
    '''
    Loops through all models that support interval prediction (specifically quantile prediction),
    trains the model on the given data, and returns the trained model objects.

    Args:
        y_train (pd.DataFrame): The features to train on.
        quantiles (list): The quantiles to predict (eg: [0.05, 0.95] predicts for the quantiles 0.05 and 0.95).
        model_hyperparameters (dict): A dictionary of model hyperparameters to pass to the model. The keys should be the model names.
            And the values should be a dictionary of hyperparameters for that model.

    Returns:
        models (dict): A dictionary of trained model objects. The keys will be the model names, 
            and the values will be the trained model objects.
    '''
    # loads models that support interval prediction
    all_models = all_estimators(
        "forecaster", filter_tags={"capability:pred_int": True}, as_dataframe=True
    )

    models = {}
    # loop through each row, and import from all_models["Object"]
    for row in all_models.iterrows():
        model_name = row["name"]
        model_class = row["Object"]
        # split the fully qualified name into module and class
        module_name, class_name = model_class.rsplit(".", 1)
        
        # import the module
        module = importlib.import_module(module_name)
        
        # get the class from the module
        cls = getattr(module, class_name)
        instance = cls(**model_hyperparameters.get(model_name, {}))
        # fit the model
        instance.fit(y_train)

        # store the model
        models[model_name] = instance
    
    return models 

def get_quantile_forecasts(models, y_test, forecast_horizon=None, quantiles=[0.05, 0.95]):
    '''
    Loops through the trained models, and returns the quantile forecasts for each model.

    Args:
        models (dict): A dictionary of trained model objects. The keys will be the model names, 
            and the values will be the trained model objects.
        y_test (pd.DataFrame): The features to predict on.
        forecast_horizon (list[int]): The forecast horizon for each model. If None, the forecast horizon will be the same as the training data.
        quantiles (list): The quantiles to predict (eg: [0.05, 0.95] predicts for the quantiles 0.05 and 0.95).

    Returns:
        quantile_forecasts (dict): A dictionary of quantile forecasts. The keys will be the model names, 
            and the values will be the quantile forecasts for each model.
    '''
    quantile_forecasts = {}
    for model_name, model in models.items():
        quantile_forecasts[model_name] = model.predict_quantiles(X=y_test, alpha=quantiles, fh=forecast_horizon)
    
    return quantile_forecasts