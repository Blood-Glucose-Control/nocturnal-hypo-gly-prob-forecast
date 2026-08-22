#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Generic sweep evaluation CLI entrypoint in canonical orchestration namespace."""

from src.workflows.sweeps.eval import main

if __name__ == "__main__":
    raise SystemExit(main())
