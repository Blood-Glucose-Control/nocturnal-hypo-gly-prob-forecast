# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Moirai-1.x (native fine-tune) x Optuna sweep on MetaboNet.

Reuses the Chronos-2 sweep data/metrics/feature pipeline and the generic Optuna
study infrastructure; swaps the model to a per-trial native fine-tune of
Salesforce's Moirai-1.x (via uni2ts ``MoiraiFinetune``) that consumes insulin as
**known-future** covariates spanning the context + forecast horizon (the
competition permits future insulin), optimizing ``mard_avg`` on an internal
temporal validation split.

Only Moirai-1.x exposes a fine-tune module in uni2ts 2.0.0 (Moirai-2.0/MoE are
inference-only), so this sweep is restricted to the 1.0-R / 1.1-R checkpoints.

The worker (``study``/``objective``) must run in the ``.venvs/moirai``
interpreter (``uni2ts`` + ``gluonts`` + ``torch``); ``search_space`` is pure and
importable anywhere Optuna is available.
"""
