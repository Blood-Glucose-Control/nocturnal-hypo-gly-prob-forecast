import mlflow
import pandas as pd

class ExperimentAnalyzer:
    """
    A class to handle analysis of MLflow experiments, including finding best models,
    comparing hyperparameters, evaluating change point detection, and generating reports.
    """

    def __init__(self, experiment_name: str):
        """
        Initializes the ExperimentAnalyzer with a specified experiment name.

        Args:
            experiment_name (str): The name of the MLflow experiment to analyze.
        """
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' does not exist.")
        self.experiment_id = self.experiment.experiment_id

    def find_best_models(self, metric_name: str = "MAE", threshold: float = 0.1) -> pd.DataFrame:
        """
        Find the best-performing models based on a metric threshold.

        Args:
            metric_name (str): Metric to filter on (e.g., MAE). Defaults to "MAE".
            threshold (float): Threshold value for the metric. Defaults to 0.1.

        Returns:
            pd.DataFrame: DataFrame of filtered runs with metrics below the threshold.
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"metrics.{metric_name} < {threshold}",
            order_by=[f"metrics.{metric_name} ASC"]
        )
        print(f"Found {len(runs)} runs with {metric_name} < {threshold}.")
        return runs

    def compare_hyperparameters(self, metric_name: str = "MAE") -> pd.DataFrame:
        """
        Compare models across different hyperparameter configurations.

        Args:
            metric_name (str): Metric to compare (e.g., MAE). Defaults to "MAE".

        Returns:
            pd.DataFrame: DataFrame grouped by hyperparameters with average metric values.
        """
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        hyperparameter_cols = [col for col in runs.columns if col.startswith("params.")]
        df = runs[hyperparameter_cols + [f"metrics.{metric_name}"]]
        summary = df.groupby(hyperparameter_cols).mean().reset_index()
        print(summary)
        return summary

    def evaluate_change_point_detection(self) -> pd.DataFrame:
        """
        Evaluate change point detection algorithms for sensitivity and precision.

        Returns:
            pd.DataFrame: DataFrame with sensitivity, precision, and F1-score metrics.
        """
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        metrics = runs[["metrics.sensitivity", "metrics.precision", "metrics.F1_score"]]
        print("Change Point Detection Evaluation Summary:")
        print(metrics.describe())
        return metrics

    def generate_report(
        self,
        best_models: pd.DataFrame,
        hyperparameter_comparison: pd.DataFrame,
        change_point_eval: pd.DataFrame,
        output_file: str = "experiment_analysis.xlsx"
    ) -> None:
        """
        Generate a summary report from the queried data.

        Args:
            best_models (pd.DataFrame): DataFrame of the best-performing models.
            hyperparameter_comparison (pd.DataFrame): DataFrame of hyperparameter comparison results.
            change_point_eval (pd.DataFrame): DataFrame of change point evaluation metrics.
            output_file (str): Path to save the report as an Excel file. Defaults to "experiment_analysis.xlsx".

        Returns:
            None
        """
        with pd.ExcelWriter(output_file) as writer:
            best_models.to_excel(writer, sheet_name="Best Models", index=False)
            hyperparameter_comparison.to_excel(writer, sheet_name="Hyperparameter Comparison", index=False)
            change_point_eval.to_excel(writer, sheet_name="Change Point Evaluation", index=False)

        print(f"Report generated and saved to {output_file}.")


# Example Usage
if __name__ == "__main__":

    experiment_name = "My Experiment"

    analyzer = ExperimentAnalyzer(experiment_name=experiment_name)

    # Find best-performing models
    best_models = analyzer.find_best_models(metric_name="MAE", threshold=0.1)

    # Compare models across different hyperparameters
    hyperparameter_comparison = analyzer.compare_hyperparameters(metric_name="MAE")

    # Evaluate change point detection algorithms
    change_point_eval = analyzer.evaluate_change_point_detection()

    # Generate report
    analyzer.generate_report(
        best_models, hyperparameter_comparison, change_point_eval, output_file="experiment_analysis.xlsx"
    )
