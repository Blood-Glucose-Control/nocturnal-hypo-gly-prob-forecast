#!/usr/bin/env python3
"""Task/experiment-aware sweep training dispatcher."""

from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, Sequence, Tuple

from src.workflows.sweeps.tasks.forecasting.train import (
    main as forecasting_train_main,
)

TrainAdapter = Callable[[Sequence[str] | None], int]


def _build_adapter_registry() -> Dict[Tuple[str, str], TrainAdapter]:
    return {
        ("forecasting", "nocturnal_forecast"): forecasting_train_main,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dispatch sweep training orchestration by task family and experiment type."
        ),
        add_help=False,
    )
    parser.add_argument(
        "--task-family",
        type=str,
        default=None,
        help="Task family (default: TASK_FAMILY env or forecasting).",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        default=None,
        help="Experiment type/profile family (default: EXPERIMENT_TYPE env or nocturnal_forecast).",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="show_help",
        help="Show dispatcher help and selected adapter help.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    parsed, remaining = parser.parse_known_args(argv)

    task_family = (
        parsed.task_family or os.environ.get("TASK_FAMILY", "forecasting")
    ).strip()
    experiment_type = (
        parsed.experiment_type
        or os.environ.get("EXPERIMENT_TYPE", "nocturnal_forecast")
    ).strip()

    adapters = _build_adapter_registry()
    key = (task_family, experiment_type)
    adapter = adapters.get(key)
    if adapter is None:
        supported = ", ".join(f"{task}/{exp}" for task, exp in sorted(adapters.keys()))
        raise ValueError(
            "Unsupported train sweep adapter "
            f"task_family='{task_family}' experiment_type='{experiment_type}'. "
            f"Supported adapters: {supported}"
        )

    if parsed.show_help:
        print("Sweep training dispatcher")
        print(f"  task_family={task_family}")
        print(f"  experiment_type={experiment_type}")
        print("")
        return adapter(["--help"])

    return adapter(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
