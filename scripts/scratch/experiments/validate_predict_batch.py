#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Compatibility shim: canonical path is scripts/evaluation/validate_predict_batch.py."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = (
        Path(__file__).resolve().parents[2] / "evaluation" / "validate_predict_batch.py"
    )
    runpy.run_path(str(target), run_name="__main__")
