#!/usr/bin/env python3
"""Chronos-2 profile wrapper over the generic sweep training orchestrator."""

from __future__ import annotations

from typing import Sequence

from src.workflows.sweeps.train import main as generic_sweep_main


DEFAULT_SWEEP_SPEC = (
    "configs/experiments/nocturnal_forecast/chronos2_forecasting_train_sweep.yaml"
)


def main(argv: Sequence[str] | None = None) -> int:
    return generic_sweep_main(
        argv,
        default_model_type="chronos2",
        default_sweep_spec=DEFAULT_SWEEP_SPEC,
    )


if __name__ == "__main__":
    raise SystemExit(main())
