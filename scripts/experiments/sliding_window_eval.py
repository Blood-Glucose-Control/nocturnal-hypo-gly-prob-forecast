#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Compatibility shim: canonical path is scripts/evaluation/sliding_window_eval.py."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = (
        Path(__file__).resolve().parents[1] / "evaluation" / "sliding_window_eval.py"
    )
    runpy.run_path(str(target), run_name="__main__")
