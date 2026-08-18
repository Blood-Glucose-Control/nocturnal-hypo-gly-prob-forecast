#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Canonical CLI for the generic forecasting training/evaluation workflow."""

from src.workflows.forecasting.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
