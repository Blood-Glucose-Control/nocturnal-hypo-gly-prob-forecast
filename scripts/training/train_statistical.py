"""Script for training statistical models."""

import argparse
from pathlib import Path

from src.training import TrainingPipeline


def main():
    """Train a statistical model."""
    parser = argparse.ArgumentParser(description="Train a statistical model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--patients",
        type=str,
        nargs="*",
        help="Optional list of patient IDs to train on",
    )
    args = parser.parse_args()

    # Load config and create pipeline
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file {config_path} does not exist")
        return 1

    try:
        pipeline = TrainingPipeline.from_config_file(str(config_path))
        print(
            f"Training {pipeline.config.model_name} model on {pipeline.config.dataset_name} dataset..."
        )

        # Train model
        results = pipeline.train(patient_ids=args.patients)

        # Save trained model
        if pipeline.trainer.model is not None:
            model_path = pipeline.save_model(pipeline.trainer.model, results)
            print(f"Model saved to: {model_path}")

        # Save results
        print(f"Training complete. Results: {results}")
        return 0

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
