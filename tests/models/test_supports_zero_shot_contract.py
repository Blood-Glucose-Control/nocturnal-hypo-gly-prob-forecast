"""Contract tests for supports_zero_shot model API shape.

The runtime checks this as an attribute (`self.supports_zero_shot`), so model
implementations must expose it as a property, not a plain method.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILES = [
    "src/models/autogluon_base.py",
    "src/models/base/base_model.py",
    "src/models/chronos2/model.py",
    "src/models/deepar/model.py",
    "src/models/moment/model.py",
    "src/models/moirai/model.py",
    "src/models/naive_baseline/model.py",
    "src/models/patchtst/model.py",
    "src/models/statistical/model.py",
    "src/models/sundial/model.py",
    "src/models/tft/model.py",
    "src/models/tide/model.py",
    "src/models/timegrad/model.py",
    "src/models/timesfm/model.py",
    "src/models/toto/model.py",
    "src/models/tsmixer/model.py",
    "src/models/ttm/model.py",
]


def _decorator_names(fn: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Name):
            names.add(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.add(dec.attr)
    return names


def _find_supports_zero_shot(file_path: Path) -> ast.FunctionDef:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "supports_zero_shot":
            return node
    raise AssertionError(f"{file_path} does not define supports_zero_shot")


def test_supports_zero_shot_is_property_on_all_models():
    for rel_path in MODEL_FILES:
        file_path = REPO_ROOT / rel_path
        fn = _find_supports_zero_shot(file_path)
        decorators = _decorator_names(fn)
        assert "property" in decorators, (
            f"{rel_path}: supports_zero_shot must be a @property; "
            f"found decorators={sorted(decorators)}"
        )


def test_base_model_supports_zero_shot_is_abstract_property():
    file_path = REPO_ROOT / "src/models/base/base_model.py"
    fn = _find_supports_zero_shot(file_path)
    decorators = _decorator_names(fn)
    assert "property" in decorators
    assert "abstractmethod" in decorators
