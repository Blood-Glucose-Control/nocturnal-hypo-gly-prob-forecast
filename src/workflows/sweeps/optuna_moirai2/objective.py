# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Optuna objective for the Moirai-2.0 / Moirai-MoE PEFT-LoRA sweep.

Reuses :class:`~..optuna_moirai.objective.MoiraiObjective` wholesale (data
windowing, known-future covariate packing, validation scoring, pruning hook) and
swaps only:

* the model wrapper -> :class:`~src.models.moirai2.model.Moirai2LoraForecaster`
  (+ :class:`~src.models.moirai2.model.Moirai2Config`), and
* the sampled search point -> :func:`..search_space.suggest_moirai2_point`.
"""

from __future__ import annotations

import optuna

from ..optuna_moirai.objective import MoiraiObjective
from ..optuna_moirai.search_space import MoiraiSearchPoint
from .search_space import Moirai2SearchSpaceBounds, suggest_moirai2_point


class Moirai2Objective(MoiraiObjective):
    """Moirai-2/MoE LoRA objective (see module docstring)."""

    @staticmethod
    def _import_wrapper():
        from src.models.moirai2.model import (  # pyright: ignore[reportMissingImports]
            Moirai2Config,
            Moirai2LoraForecaster,
        )

        return Moirai2Config, Moirai2LoraForecaster

    def _suggest(self, trial: optuna.Trial) -> MoiraiSearchPoint:
        bounds = self.bounds
        assert isinstance(bounds, Moirai2SearchSpaceBounds)
        return suggest_moirai2_point(trial, bounds)
