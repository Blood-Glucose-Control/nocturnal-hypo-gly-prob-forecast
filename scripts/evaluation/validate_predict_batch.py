#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Canonical predict_batch validation CLI entrypoint."""

from src.workflows.evaluation.validate_predict_batch import main


if __name__ == "__main__":
    main()
