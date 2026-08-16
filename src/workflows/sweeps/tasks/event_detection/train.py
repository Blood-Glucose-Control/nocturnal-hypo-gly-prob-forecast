#!/usr/bin/env python3
"""Event-detection sweep training adapter scaffold."""

from __future__ import annotations

import argparse
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Event-detection sweep training adapter scaffold."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Model type identifier for event-detection training sweeps.",
    )
    parser.add_argument(
        "--sweep-spec",
        type=str,
        default=None,
        help=(
            "Event-detection sweep spec path "
            "(for example: configs/experiments/nocturnal_events/<profile>.yaml)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Reserved for future event-detection adapter implementation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    raise NotImplementedError(
        "Sweep adapter scaffold is registered but not implemented for "
        "task_family='event_detection', experiment_type='nocturnal_events'. "
        f"Received model_type={args.model_type!r}, sweep_spec={args.sweep_spec!r}. "
        "Implement orchestration in "
        "src/workflows/sweeps/tasks/event_detection/train.py."
    )


if __name__ == "__main__":
    raise SystemExit(main())
