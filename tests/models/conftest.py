"""Model test environment guard.

Tests under tests/models/ require model-specific virtual environments because
different models depend on the same packages with conflicting version requirements.

If you run pytest from the main venv (.noctprob-venv) these tests will be
collected but skipped with a clear message pointing to the correct venv.

To run model tests use the Makefile targets:
    make test-ttm
    make test-sundial
    make test-timesfm

Or invoke directly:
    .venvs/ttm/bin/python -m pytest tests/models/ -v
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Map: substring matched against test filename -> (model_name, venv_relative_path)
_MODEL_VENV_MAP: dict[str, tuple[str, str]] = {
    "ttm": ("ttm", ".venvs/ttm"),
    "sundial": ("sundial", ".venvs/sundial"),
    "timesfm": ("timesfm", ".venvs/timesfm"),
    "chronos2": ("chronos2", ".venvs/chronos2"),
}


def _is_inside_venv(venv_rel_path: str) -> bool:
    """Return True if the current Python interpreter lives inside the given venv."""
    venv_abs = os.path.realpath(os.path.join(_REPO_ROOT, venv_rel_path))
    current = os.path.realpath(sys.executable)
    return current.startswith(venv_abs + os.sep) or current == venv_abs


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip model tests that require a different venv than the one currently active."""
    cached_markers: dict[str, pytest.MarkDecorator] = {}

    for item in items:
        path_obj = getattr(item, "path", None)
        if path_obj is not None:
            test_filename = path_obj.name.lower()
        else:
            test_filename = os.path.basename(str(item.fspath)).lower()

        for keyword, (model_name, venv_rel) in _MODEL_VENV_MAP.items():
            if keyword not in test_filename:
                continue

            if _is_inside_venv(venv_rel):
                break  # correct venv — no skip needed

            if model_name not in cached_markers:
                venv_python = os.path.join(_REPO_ROOT, venv_rel, "bin", "python")
                cached_markers[model_name] = pytest.mark.skip(
                    reason=(
                        f"Requires the '{model_name}' virtual environment "
                        f"(current: {sys.executable}).\n"
                        f"  Run with:  {venv_python} -m pytest tests/models/ -v\n"
                        f"  Or use:    make test-{model_name}"
                    )
                )
            item.add_marker(cached_markers[model_name])
            break
