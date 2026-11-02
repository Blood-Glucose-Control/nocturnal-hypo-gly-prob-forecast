# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Optuna hyperparameter-search proof of concept for Chronos-2.

This package wires AutoGluon's Chronos-2 fine-tuning into an Optuna study that
optimises for the MetaboNet Glucose Prediction Challenge submission shape:

    context_length in [24, 288] steps  ->  forecast_length = 24 steps

At 5-minute sampling the forecast covers +30/+60/+90/+120 minutes (steps
6/12/18/24). The primary objective is RMSE (mg/dL) at the +120-minute horizon;
RMSE/MARD/DTS at +30/+60/+90/+120 are also reported per trial.

The study uses a shared SQLite backend so multiple GPU-pinned workers (launched
in tmux) can contribute trials to one search space concurrently.
"""

from .metrics import HORIZON_MINUTES, HORIZON_STEPS, evaluate_horizons

__all__ = ["HORIZON_MINUTES", "HORIZON_STEPS", "evaluate_horizons"]
