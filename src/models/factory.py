"""Model-family registry constants for compatibility checks.

Runtime constructor routing moved to
`src.workflows.forecasting.modeling.create_model_and_config`.
"""

SUPPORTED_MODEL_TYPES = (
    "sundial",
    "ttm",
    "chronos2",
    "moment",
    "toto",
    "moirai",
    "timegrad",
    "timesfm",
    "timesfm3",
    "tide",
    "gbm",
    "naive_baseline",
    "statistical",
    "deepar",
    "patchtst",
    "tft",
    "tsmixer",
)
