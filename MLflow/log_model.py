import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from src.data_helpers.data_utils import DataHandler
from src.model_helpers.model_utils import ModelHandler
from sktime.forecasting.naive import NaiveForecaster
import numpy as np
import pandas as pd
from typing import Optional


class ModelLogger:
    """
    A class to handle logging and deployment of models using MLflow.
    """

    def __init__(self, experiment_name: str = "My Experiment"):
        """
        Initializes the ModelLogger with a default experiment name.

        Args:
            experiment_name (str): The name of the MLflow experiment. Defaults to "My Experiment".
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def log_experiment(
        self,
        model,
        X_train: pd.DataFrame,
        model_save_path: str,
        params: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        """
        Logs hyperparameters, metrics, and artifacts along with the model to MLflow.

        Args:
            model: The trained model to log.
            X_train: Training data used for logging metadata.
            params (dict, optional): Hyperparameters to log. Defaults to None.
            metrics (dict, optional): Metrics to log. Defaults to None.

        Returns:
            None
        """
        input_example = X_train[:2]  # Used 2 arbitrarily here
        predictions = model.predict(
            input_example
        )  # Replace with specific model prediction logic (e.g., predict_var)
        signature = infer_signature(input_example, predictions)

        with mlflow.start_run():
            # Log hyperparameters
            if params:
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            # Log metrics
            if metrics:
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

            # Log the model with metadata
            mlflow.pyfunc.log_model(
                artifact_path=model_save_path,
                python_model=model,
                input_example=input_example,
                signature=signature,
            )

            print("Experiment and model logged successfully!")

    def deploy_model(
        self,
        X_test: pd.DataFrame,
        run_id: str,
        deployment_type: str = "local",
        port: int = 5000,
    ) -> None:
        """
        Deploys the logged model for inference. Supports local or REST API deployment.

        Args:
            run_id (str): The MLflow run ID of the logged model.
            deployment_type (str): Deployment type, either "local" or "rest". Defaults to "local".
            port (int): Port to serve the REST API on if deployment_type is "rest". Defaults to 5000.

        Returns:
            None
        """
        model_uri = f"runs:/{run_id}/model"

        if deployment_type == "local":
            # Load the model locally for inference
            model = mlflow.pyfunc.load_model(model_uri)
            predictions = model.predict(X_test)
            print(f"Predictions: {predictions}")

        elif deployment_type == "rest":
            # Serve the model as a REST API
            import subprocess

            subprocess.run(
                ["mlflow", "models", "serve", "-m", model_uri, "-p", str(port)]
            )
            print(f"Model served at http://localhost:{port}")

        else:
            raise ValueError("Invalid deployment type. Choose 'local' or 'rest'.")


if __name__ == "__main__":
    RANDOM_SEED = 101
    test_size = 0.3
    data_path = "./data/example.csv"
    model_save_path = "./models/example.pkl"
    fh = np.arange(1, 6)

    # Step 1: Train
    data_handler = DataHandler(test_size=test_size, random_seed=RANDOM_SEED)
    data = data_handler.load_data(data_path)
    cleaned_data = data_handler.clean_data(data)
    X_train, X_test, y_train, y_test = data_handler.split_data(
        cleaned_data, target_column="target"
    )

    forecaster = NaiveForecaster(strategy="last")
    handler = ModelHandler(forecaster)
    trained_model = handler.fit_variance_forecaster(y=y_train, X=X_train, fh=fh)

    model_logger = ModelLogger()

    # Step 2: Log the experiment and model
    model_logger.log_experiment(trained_model, X_train, model_save_path=model_save_path)

    # Step 3: Deploy the model
    run_id = "12345"
    model_logger.deploy_model(X_test, run_id)
