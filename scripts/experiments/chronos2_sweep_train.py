#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""CLI entrypoint for Chronos-2 sweep training orchestrator."""

from src.workflows.forecasting.orchestrators.chronos2_train_sweep_profile import main


if __name__ == "__main__":
    raise SystemExit(main())
