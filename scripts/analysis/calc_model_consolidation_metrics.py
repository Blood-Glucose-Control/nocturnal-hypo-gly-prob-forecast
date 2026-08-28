#!/usr/bin/env python3
"""Compute consistent per-model method and LOC metrics for consolidation tracking."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

MODEL_PATHS = {
    "TTM": "src/models/ttm/model.py",
    "TimesFM": "src/models/timesfm/model.py",
    "Moirai": "src/models/moirai/model.py",
    "Moment": "src/models/moment/model.py",
    "Chronos2": "src/models/chronos2/model.py",
    "Toto": "src/models/toto/model.py",
    "Tide": "src/models/tide/model.py",
    "TimeGrad": "src/models/timegrad/model.py",
    "PatchTST": "src/models/patchtst/model.py",
    "TSMixer": "src/models/tsmixer/model.py",
    "DeepAR": "src/models/deepar/model.py",
    "TFT": "src/models/tft/model.py",
    "Statistical": "src/models/statistical/model.py",
    "NaiveBaseline": "src/models/naive_baseline/model.py",
    "Sundial": "src/models/sundial/model.py",
}


@dataclass(frozen=True)
class ModelMetrics:
    model: str
    path: str
    method_count: int
    method_loc: int
    file_loc: int
    file_nonempty_loc: int


def _class_name_is_forecaster(node: ast.ClassDef) -> bool:
    return node.name.endswith("Forecaster")


def _extract_forecaster_methods(
    tree: ast.Module,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _class_name_is_forecaster(node):
            return [
                member
                for member in node.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
    raise ValueError("No *Forecaster class found in module.")


def _compute_metrics(repo_root: Path, model: str, rel_path: str) -> ModelMetrics:
    path = repo_root / rel_path
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    methods = _extract_forecaster_methods(tree)
    method_loc = sum(
        int(member.end_lineno or member.lineno) - int(member.lineno) + 1
        for member in methods
    )
    file_nonempty_loc = sum(1 for line in lines if line.strip())
    return ModelMetrics(
        model=model,
        path=rel_path,
        method_count=len(methods),
        method_loc=method_loc,
        file_loc=len(lines),
        file_nonempty_loc=file_nonempty_loc,
    )


def _render_markdown(metrics: list[ModelMetrics]) -> str:
    rows = [
        "| Model | Method count | Method LOC | File LOC | File non-empty LOC | Path |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in metrics:
        rows.append(
            f"| {item.model} | {item.method_count} | {item.method_loc} | "
            f"{item.file_loc} | {item.file_nonempty_loc} | `{item.path}` |"
        )
    return "\n".join(rows)


def _render_csv(metrics: list[ModelMetrics]) -> str:
    rows = ["model,path,method_count,method_loc,file_loc,file_nonempty_loc"]
    for item in metrics:
        rows.append(
            f"{item.model},{item.path},{item.method_count},{item.method_loc},"
            f"{item.file_loc},{item.file_nonempty_loc}"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-model method and LOC metrics for consolidation tracking."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root path (default: inferred from script location).",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output format.",
    )
    args = parser.parse_args()

    metrics = [
        _compute_metrics(args.repo_root, model, rel_path)
        for model, rel_path in MODEL_PATHS.items()
    ]

    if args.format == "csv":
        print(_render_csv(metrics))
    else:
        print(_render_markdown(metrics))


if __name__ == "__main__":
    main()
