"""Model module exports."""

from src.models.factory import create_model_and_config

# Trigger @ModelRegistry.register() for all model classes.
# Missing optional deps (autogluon, tsfm_public, etc.) are silently skipped.
for _mod in [
    "src.models.ttm",
    "src.models.chronos2",
    "src.models.tide",
    "src.models.sundial",
    "src.models.tsmixer",
    "src.models.timesfm",
    "src.models.timegrad",
    "src.models.moirai",
    "src.models.moment",
]:
    try:
        __import__(_mod)
    except ModuleNotFoundError:
        pass

__all__ = ["create_model_and_config"]
