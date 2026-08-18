#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Compatibility shim: canonical path is scripts/workflows/personalization/per_patient_finetune.py."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = (
        Path(__file__).resolve().parents[2]
        / "workflows"
        / "personalization"
        / "per_patient_finetune.py"
    )
    runpy.run_path(str(target), run_name="__main__")
