#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Example surface for the generic forecasting workflow.

This thin wrapper keeps the public onboarding entrypoint in scripts/examples
while delegating implementation to the maintained orchestrator core.
"""

from src.workflows.forecasting.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
