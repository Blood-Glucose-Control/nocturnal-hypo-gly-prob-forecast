# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""TimesFM 3.0 (LoRA fine-tune) x Optuna sweep on MetaboNet.

Reuses the Chronos-2 sweep data/metrics/feature pipeline and the generic Optuna
study infrastructure; swaps the model to a per-trial LoRA fine-tune of Google's
TimesFM 3.0 that consumes selected insulin channels as **known-future**
covariates spanning the context + forecast horizon (the competition permits
future insulin), optimizing ``mard_avg`` on an internal temporal validation
split.

The worker (``study``/``objective``) must run in the ``.venvs/timesfm3``
interpreter (``timesfm3`` + ``peft`` + ``torch``); ``search_space`` is pure and
importable anywhere Optuna is available.
"""
