# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Tests for StandardForecastingSummarizer.

Uses a temporary directory tree with realistic synthetic ``results.json``
and ``experiment_configs.json`` files.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.experiments.standard_forecasting.summarize import StandardForecastingSummarizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESULTS_TEMPLATE = {
    "model": "ttm",
    "mode": "fine-tuned",
    "checkpoint": "trained_models/artifacts/some_run",
    "dataset": "aleppo_2017",
    "timestamp": "2026-02-16T18:51:20.393668",
    "config": {"context_length": 512, "forecast_length": 96},
    "overall": {"rmse": 3.48, "mae": 2.70, "mape": 36.5, "mse": 12.1},
    "per_patient": [
        {
            "patient": "ale_102",
            "episodes": 6,
            "mse": 13.1,
            "rmse": 3.6,
            "mae": 3.0,
            "mape": 37.7,
        },
        {
            "patient": "ale_45",
            "episodes": 1,
            "mse": 11.9,
            "rmse": 3.4,
            "mae": 2.9,
            "mape": 47.8,
        },
    ],
}

_CONFIG_TEMPLATE = {
    "cli_args": {"model": "ttm", "dataset": "aleppo_2017"},
    "model_config": {"context_length": 512, "forecast_length": 96},
    "environment": {"git_commit": "abc1234", "python_version": "3.12.3"},
}


def _make_run(
    root: Path,
    ctx_fh: str,
    model: str,
    run_name: str,
    results: dict,
    config: dict | None = None,
) -> Path:
    run_dir = root / "standard_forecasting" / ctx_fh / model / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(results))
    if config:
        (run_dir / "experiment_configs.json").write_text(json.dumps(config))
    return run_dir


# ---------------------------------------------------------------------------
# Tests: _parse_run_dir
# ---------------------------------------------------------------------------


class TestParseRunDir:
    def test_parses_valid_results_json(self, tmp_path):
        run_dir = _make_run(
            tmp_path,
            "512ctx_96fh",
            "ttm",
            "2026-02-16_1808_aleppo_2017_finetuned",
            _RESULTS_TEMPLATE,
            _CONFIG_TEMPLATE,
        )
        s = StandardForecastingSummarizer(tmp_path)
        row = s._parse_run_dir("512ctx_96fh", "ttm", run_dir)

        assert row is not None
        assert row["model"] == "ttm"
        assert row["dataset"] == "aleppo_2017"
        assert row["mode"] == "finetuned"  # hyphen stripped
        assert row["rmse"] == pytest.approx(3.48)
        assert row["mae"] == pytest.approx(2.70)
        assert row["mape"] == pytest.approx(36.5)
        assert row["mse"] == pytest.approx(12.1)
        assert row["context_length"] == 512
        assert row["forecast_length"] == 96
        assert row["total_episodes"] == 7  # 6 + 1
        assert row["git_commit"] == "abc1234"
        assert row["experiment_type"] == "standard_forecasting"
        assert row["ctx_fh"] == "512ctx_96fh"

    def test_returns_none_when_results_missing(self, tmp_path):
        run_dir = (
            tmp_path
            / "standard_forecasting"
            / "512ctx_96fh"
            / "ttm"
            / "2026-02-16_1808_aleppo_zeroshot"
        )
        run_dir.mkdir(parents=True)
        s = StandardForecastingSummarizer(tmp_path)
        assert s._parse_run_dir("512ctx_96fh", "ttm", run_dir) is None

    def test_returns_none_for_malformed_json(self, tmp_path):
        run_dir = (
            tmp_path
            / "standard_forecasting"
            / "512ctx_96fh"
            / "ttm"
            / "2026-02-16_1808_d_m"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "results.json").write_text("NOT { valid json }")
        s = StandardForecastingSummarizer(tmp_path)
        assert s._parse_run_dir("512ctx_96fh", "ttm", run_dir) is None

    def test_returns_none_when_overall_block_empty(self, tmp_path):
        bad = dict(_RESULTS_TEMPLATE, overall={})
        run_dir = _make_run(tmp_path, "512ctx_96fh", "ttm", "2026-02-16_1808_d_m", bad)
        s = StandardForecastingSummarizer(tmp_path)
        assert s._parse_run_dir("512ctx_96fh", "ttm", run_dir) is None

    def test_git_commit_none_when_no_config(self, tmp_path):
        run_dir = _make_run(
            tmp_path, "512ctx_96fh", "ttm", "2026-02-16_1808_d_m", _RESULTS_TEMPLATE
        )
        s = StandardForecastingSummarizer(tmp_path)
        row = s._parse_run_dir("512ctx_96fh", "ttm", run_dir)
        assert row is not None
        assert row["git_commit"] is None

    def test_mode_normalised_for_zeroshot(self, tmp_path):
        results = dict(_RESULTS_TEMPLATE, mode="zero-shot")
        run_dir = _make_run(
            tmp_path, "512ctx_96fh", "ttm", "2026-02-16_1808_d_m", results
        )
        s = StandardForecastingSummarizer(tmp_path)
        row = s._parse_run_dir("512ctx_96fh", "ttm", run_dir)
        assert row["mode"] == "zeroshot"


# ---------------------------------------------------------------------------
# Tests: summarize
# ---------------------------------------------------------------------------


class TestStandardSummarize:
    def test_produces_one_row_per_run(self, tmp_path):
        for i, dataset in enumerate(["aleppo_2017", "brown_2019", "lynch_2022"]):
            results = dict(
                _RESULTS_TEMPLATE,
                dataset=dataset,
                overall={"rmse": 3.0 + i, "mae": 2.0, "mape": 30.0, "mse": 9.0},
            )
            _make_run(tmp_path, "512ctx_96fh", "ttm", f"2026-02-16_180{i}_d_m", results)

        s = StandardForecastingSummarizer(tmp_path)
        df = s.summarize()
        assert len(df) == 3
        assert set(df["dataset"]) == {"aleppo_2017", "brown_2019", "lynch_2022"}

    def test_skips_incomplete_runs(self, tmp_path):
        # One complete, one missing results.json
        _make_run(
            tmp_path, "512ctx_96fh", "ttm", "2026-02-16_1808_d_m", _RESULTS_TEMPLATE
        )
        incomplete = (
            tmp_path
            / "standard_forecasting"
            / "512ctx_96fh"
            / "ttm"
            / "2026-02-16_1900_d_m"
        )
        incomplete.mkdir(parents=True)

        s = StandardForecastingSummarizer(tmp_path)
        df = s.summarize()
        assert len(df) == 1

    def test_handles_multiple_models(self, tmp_path):
        for model in ["ttm", "sundial"]:
            results = dict(_RESULTS_TEMPLATE, model=model)
            _make_run(
                tmp_path,
                "512ctx_96fh",
                model,
                "2026-02-16_1808_aleppo_zeroshot",
                results,
            )

        s = StandardForecastingSummarizer(tmp_path)
        df = s.summarize()
        assert set(df["model"]) == {"ttm", "sundial"}

    def test_summary_csv_has_correct_columns(self, tmp_path):
        _make_run(
            tmp_path, "512ctx_96fh", "ttm", "2026-02-16_1808_d_m", _RESULTS_TEMPLATE
        )
        s = StandardForecastingSummarizer(tmp_path)
        s.summarize()
        df = pd.read_csv(tmp_path / "standard_forecasting" / "summary.csv")
        for col in [
            "run_id",
            "model",
            "dataset",
            "mode",
            "rmse",
            "mae",
            "mape",
            "mse",
            "total_episodes",
        ]:
            assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Tests: best_runs
# ---------------------------------------------------------------------------


class TestStandardBestRuns:
    def _populate(self, tmp_path):
        runs = [
            ("aleppo_2017", "finetuned", 3.5),
            ("aleppo_2017", "zeroshot", 2.9),  # best aleppo
            ("brown_2019", "finetuned", 4.1),
            ("brown_2019", "zeroshot", 3.8),  # best brown
        ]
        for i, (dataset, mode, rmse) in enumerate(runs):
            results = dict(
                _RESULTS_TEMPLATE,
                dataset=dataset,
                mode=mode,
                overall={"rmse": rmse, "mae": 2.0, "mape": 30.0, "mse": 9.0},
            )
            _make_run(
                tmp_path,
                "512ctx_96fh",
                "ttm",
                f"2026-02-16_180{i}_{dataset}_{mode}",
                results,
            )

    def test_best_by_model_dataset_rows(self, tmp_path):
        self._populate(tmp_path)
        result = StandardForecastingSummarizer(tmp_path).best_runs()
        by_md = result["by_model_dataset"]
        assert len(by_md) == 2
        aleppo = by_md[by_md["dataset"] == "aleppo_2017"].iloc[0]
        assert aleppo["rmse"] == pytest.approx(2.9)

    def test_best_by_model_is_global_minimum(self, tmp_path):
        self._populate(tmp_path)
        result = StandardForecastingSummarizer(tmp_path).best_runs()
        by_m = result["by_model"]
        assert len(by_m) == 1
        assert by_m.iloc[0]["rmse"] == pytest.approx(2.9)

    def test_ranking_with_mae(self, tmp_path):
        runs = [
            ("aleppo_2017", 3.5, 2.0),  # rmse=3.5, mae=2.0
            ("aleppo_2017", 2.9, 3.5),  # rmse=2.9, mae=3.5 — best by rmse, worst by mae
        ]
        for i, (dataset, rmse, mae) in enumerate(runs):
            results = dict(
                _RESULTS_TEMPLATE,
                dataset=dataset,
                overall={"rmse": rmse, "mae": mae, "mape": 30.0, "mse": 9.0},
            )
            _make_run(tmp_path, "512ctx_96fh", "ttm", f"2026-02-16_180{i}_d_m", results)

        result = StandardForecastingSummarizer(tmp_path).best_runs(metric="mae")
        by_md = result["by_model_dataset"]
        # best by mae should be the run with mae=2.0
        assert by_md.iloc[0]["mae"] == pytest.approx(2.0)
