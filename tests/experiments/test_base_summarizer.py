# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Tests for the ExperimentSummarizer base class.

Uses a minimal concrete subclass (``_DummySummarizer``) that returns a
pre-set dict so the base-class logic can be tested without needing real
JSON files on disk.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.experiments.base.experiment import ExperimentSummarizer, _validate_metric


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

RUN_DIR_RE_NAME = "2026-02-16_1808_dataset_a_mode"


def _make_run_tree(root: Path, runs: list[dict]) -> None:
    """Create a synthetic experiment directory tree under *root*.

    Each entry in *runs* is a dict with keys:
      ctx_fh, model, run_name, results (dict written as JSON), config (optional dict).
    """
    for run in runs:
        run_dir = root / run["ctx_fh"] / run["model"] / run["run_name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        if "results" in run:
            (run_dir / "results.json").write_text(json.dumps(run["results"]))
        if "config" in run:
            (run_dir / "experiment_configs.json").write_text(json.dumps(run["config"]))


class _DummySummarizer(ExperimentSummarizer):
    """Minimal concrete subclass for testing base-class behaviour."""

    def __init__(self, root: Path, rows: list[dict[str, Any]]) -> None:
        super().__init__(root, "dummy_experiment")
        self._rows = iter(rows)

    def _parse_run_dir(self, ctx_fh, model, run_dir):
        try:
            return next(self._rows)
        except StopIteration:
            return None


# ---------------------------------------------------------------------------
# Tests: _validate_metric
# ---------------------------------------------------------------------------


class TestValidateMetric:
    def test_valid_metrics_pass(self):
        for m in ("rmse", "mae", "mape", "mse"):
            _validate_metric(m)  # should not raise

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            _validate_metric("accuracy")


# ---------------------------------------------------------------------------
# Tests: _iter_run_dirs
# ---------------------------------------------------------------------------


class TestIterRunDirs:
    def test_yields_matching_run_dirs(self, tmp_path):
        runs = [
            {
                "ctx_fh": "512ctx_96fh",
                "model": "ttm",
                "run_name": "2026-02-16_1808_aleppo_2017_finetuned",
            },
            {
                "ctx_fh": "512ctx_96fh",
                "model": "sundial",
                "run_name": "2026-02-17_0100_brown_2019_zeroshot",
            },
        ]
        exp_dir = tmp_path / "dummy_experiment"
        _make_run_tree(exp_dir, runs)

        summarizer = _DummySummarizer(tmp_path, [])
        found = list(summarizer._iter_run_dirs())

        assert len(found) == 2
        ctx_fh_values = {f[0] for f in found}
        models = {f[1] for f in found}
        assert ctx_fh_values == {"512ctx_96fh"}
        assert models == {"ttm", "sundial"}

    def test_ignores_non_run_directories(self, tmp_path):
        """Directories not matching the YYYY-MM-DD_HHMM_* pattern are skipped."""
        exp_dir = tmp_path / "dummy_experiment" / "512ctx_96fh" / "ttm"
        (exp_dir / "not_a_run_dir").mkdir(parents=True)
        (exp_dir / "2026-02-16_1808_aleppo_finetuned").mkdir()

        summarizer = _DummySummarizer(tmp_path, [])
        found = list(summarizer._iter_run_dirs())
        assert len(found) == 1

    def test_missing_experiment_dir_returns_empty(self, tmp_path):
        summarizer = _DummySummarizer(tmp_path / "nonexistent", [])
        assert list(summarizer._iter_run_dirs()) == []


# ---------------------------------------------------------------------------
# Tests: summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def _make_tree_with_run(self, tmp_path):
        exp_dir = tmp_path / "dummy_experiment"
        (
            exp_dir / "512ctx_96fh" / "ttm" / "2026-02-16_1808_aleppo_2017_finetuned"
        ).mkdir(parents=True)
        return tmp_path

    def test_returns_dataframe(self, tmp_path):
        root = self._make_tree_with_run(tmp_path)
        row = {"model": "ttm", "dataset": "aleppo", "rmse": 3.5, "mae": 2.7}
        summarizer = _DummySummarizer(root, [row])
        df = summarizer.summarize()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["rmse"] == 3.5

    def test_writes_summary_csv(self, tmp_path):
        root = self._make_tree_with_run(tmp_path)
        summarizer = _DummySummarizer(
            root, [{"model": "ttm", "dataset": "d", "rmse": 1.0}]
        )
        summarizer.summarize()
        assert (tmp_path / "dummy_experiment" / "summary.csv").exists()

    def test_skips_none_rows(self, tmp_path):
        root = self._make_tree_with_run(tmp_path)
        # _DummySummarizer returns None when exhausted
        summarizer = _DummySummarizer(root, [])  # no rows → None for only run
        df = summarizer.summarize()
        assert df.empty

    def test_invalid_metric_raises(self, tmp_path):
        summarizer = _DummySummarizer(tmp_path, [])
        with pytest.raises(ValueError):
            summarizer.summarize(metric="f1_score")

    def test_custom_output_path(self, tmp_path):
        root = self._make_tree_with_run(tmp_path)
        out = tmp_path / "custom_out.csv"
        summarizer = _DummySummarizer(
            root, [{"model": "ttm", "dataset": "d", "rmse": 1.0}]
        )
        summarizer.summarize(output_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: best_runs
# ---------------------------------------------------------------------------


class TestBestRuns:
    def _make_multi_run_tree(self, tmp_path):
        exp_dir = tmp_path / "dummy_experiment"
        for run_name in [
            "2026-02-16_1808_aleppo_finetuned",
            "2026-02-16_1900_aleppo_zeroshot",
            "2026-02-16_2000_brown_zeroshot",
        ]:
            (exp_dir / "512ctx_96fh" / "ttm" / run_name).mkdir(parents=True)
        return tmp_path

    def test_by_model_dataset_picks_lowest_metric(self, tmp_path):
        root = self._make_multi_run_tree(tmp_path)
        rows = [
            {"model": "ttm", "dataset": "aleppo", "rmse": 4.0, "mae": 3.0},
            {"model": "ttm", "dataset": "aleppo", "rmse": 2.0, "mae": 1.5},  # best
            {"model": "ttm", "dataset": "brown", "rmse": 3.5, "mae": 2.0},
        ]
        result = _DummySummarizer(root, rows).best_runs(metric="rmse")
        by_md = result["by_model_dataset"]
        assert len(by_md) == 2  # one per (model × dataset)
        aleppo_row = by_md[by_md["dataset"] == "aleppo"].iloc[0]
        assert aleppo_row["rmse"] == 2.0

    def test_by_model_picks_global_lowest(self, tmp_path):
        root = self._make_multi_run_tree(tmp_path)
        rows = [
            {"model": "ttm", "dataset": "aleppo", "rmse": 4.0, "mae": 3.0},
            {"model": "ttm", "dataset": "aleppo", "rmse": 2.0, "mae": 1.5},
            {"model": "ttm", "dataset": "brown", "rmse": 3.5, "mae": 2.0},
        ]
        result = _DummySummarizer(root, rows).best_runs(metric="rmse")
        by_m = result["by_model"]
        assert len(by_m) == 1
        assert by_m.iloc[0]["rmse"] == 2.0

    def test_writes_csv_files(self, tmp_path):
        root = self._make_multi_run_tree(tmp_path)
        rows = [{"model": "ttm", "dataset": "aleppo", "rmse": 3.0, "mae": 2.0}]
        _DummySummarizer(root, rows).best_runs()
        out_dir = tmp_path / "dummy_experiment"
        assert (out_dir / "best_by_model_dataset.csv").exists()
        assert (out_dir / "best_by_model.csv").exists()

    def test_returns_empty_dicts_on_no_runs(self, tmp_path):
        result = _DummySummarizer(tmp_path, []).best_runs()
        assert result["by_model_dataset"].empty
        assert result["by_model"].empty
