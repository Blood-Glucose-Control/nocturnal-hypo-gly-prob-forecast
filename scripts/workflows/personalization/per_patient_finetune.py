#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Canonical per-patient fine-tuning CLI entrypoint."""

from src.workflows.personalization.per_patient_finetune import main


if __name__ == "__main__":
    main()
