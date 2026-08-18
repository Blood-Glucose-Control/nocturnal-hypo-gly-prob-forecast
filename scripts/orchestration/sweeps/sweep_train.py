#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Generic sweep training CLI entrypoint in canonical orchestration namespace."""

from src.workflows.sweeps.train import main


if __name__ == "__main__":
    raise SystemExit(main())
