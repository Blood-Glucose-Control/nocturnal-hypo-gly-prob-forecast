#!/usr/bin/env python3
"""Chronos-2 profile wrapper over the generic sweep evaluation orchestrator."""

from __future__ import annotations

from typing import Sequence

from src.workflows.sweeps.eval import main as generic_sweep_eval_main


DEFAULT_SWEEP_SPEC = (
    "configs/experiments/nocturnal_forecast/chronos2_forecasting_eval_sweep.yaml"
)


def main(argv: Sequence[str] | None = None) -> int:
    return generic_sweep_eval_main(
        argv,
        default_model_type="chronos2",
        default_sweep_spec=DEFAULT_SWEEP_SPEC,
        default_python_executable=".venvs/autogluon/bin/python",
    )


if __name__ == "__main__":
    raise SystemExit(main())
