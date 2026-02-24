# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
TiDE model configuration and best hyperparameters.

TiDE (Time-series Dense Encoder) is a pure MLP model wrapped in AutoGluon's
TimeSeriesPredictor. Key advantage: AutoGluon's MeanScaler prevents prediction
discontinuities at window boundaries (see docs-internal/tide_validation_findings.md).

Best config found via Bayesian HPO (45 trials, Brown 2019 dataset):
  - Temporal holdout RMSE: 1.876 mM (trial 3456bc05)
  - Patient-level holdout RMSE: 2.547 mM (8 unseen patients)
  - Discontinuity: 0.160-0.170 mM (PASS < 0.2 threshold)
"""

# Best hyperparameters from full HPO search (trial 3456bc05)
# Validated on Brown 2019 with midnight-anchored nocturnal evaluation
BEST_HYPERPARAMETERS = {
    "TiDE": {
        "context_length": 512,
        "encoder_hidden_dim": 256,
        "decoder_hidden_dim": 256,
        "temporal_hidden_dim": 256,
        "num_layers_encoder": 2,
        "num_layers_decoder": 2,
        "distr_hidden_dim": 8,
        "dropout": 0.1,
        "lr": 0.000931,
        "num_batches_per_epoch": 300,
        "batch_size": 256,
        "scaling": "mean",
        "trainer_kwargs": {
            "gradient_clip_val": 1.0,
            "precision": "16-mixed",
        },
    }
}

# Forecast configuration
FORECAST_HORIZON = 72  # 6 hours at 5-min intervals
CONTEXT_LENGTH = 512  # ~42.7 hours at 5-min intervals
INTERVAL_MINS = 5

# Discontinuity validation threshold (mM)
# Predictions must have avg |last_context - first_forecast| < this value
DISCONTINUITY_THRESHOLD = 0.2

# Key findings:
# - encoder_hidden_dim MUST equal decoder_hidden_dim (hard architectural constraint)
# - MeanScaler (scaling="mean") is CRITICAL for low discontinuity
#   - MeanScaler: x / mean(|x|), multiplicative only, no zero-centering
#   - StdScaler (TTM): (x - mean) / std, zero-centers -> regression-to-mean -> discontinuity
# - IOB as known covariate provides ~22% RMSE improvement
# - NormalOutput() distribution head recommended over StudentT (better calibrated)
# - distr_hidden_dim=8 slightly better than 16 (less overfitting)
