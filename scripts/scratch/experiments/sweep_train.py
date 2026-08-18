#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Compatibility shim for legacy experiments path."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = (
        Path(__file__).resolve().parents[2]
        / "orchestration"
        / "sweeps"
        / "sweep_train.py"
    )
    runpy.run_path(str(target), run_name="__main__")
