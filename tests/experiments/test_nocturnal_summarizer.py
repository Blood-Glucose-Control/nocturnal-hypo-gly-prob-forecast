# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Tests for NocturnalSummarizer.

Uses a temporary directory tree with synthetic ``nocturnal_results.json``
and ``experiment_config.json`` files matching the structure produced by
``scripts/experiments/nocturnal_hypo_eval.py``.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.experiments.nocturnal.summarize import NocturnalSummarizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESULTS_TEMPLATE = {
    "evaluation_type": "nocturnal_hypoglycemia",
    "model": "ttm",
    "mode": "zero-shot",
    "checkpoint": None,
    "dataset": "brown_2019",
    "timestamp": "2026-02-21T22:43:03.959874",
    "config": {"context_length": 512, "forecast_length": 72},
    "overall_rmse": 3.53,
    "total_episodes": 1830,
    "per_patient": [
        {
            "patient_id": "bro_75",
            "episodes": 6,
            "mse": 4.9,
            "rmse": 2.2,
            "mae": 1.6,
            "mape": 22.2,
        },
        {
            "patient_id": "bro_93",
            "episodes": 5,
            "mse": 6.2,
            "rmse": 2.5,
            "mae": 2.1,
            "mape": 28.6,
        },
    ],
}

_CONFIG_TEMPLATE = {
    "cli_args": {"model": "ttm", "dataset": "brown_2019"},
    "environment": {"git_commit": "9ee5b38", "python_version": "3.12.3"},
}


def _make_run(
    root: Path,
    ctx_fh: str,
    model: str,
    run_name: str,
    results: dict,
    config: dict | None = None,
    config_filename: str = "experiment_config.json",
) -> Path:
    run_dir = root / "nocturnal_forecasting" / ctx_fh / model / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "nocturnal_results.json").write_text(json.dumps(results))
    if config:
        (run_dir / config_filename).write_text(json.dumps(config))
    return run_dir


# ---------------------------------------------------------------------------
# Tests: _parse_run_dir
# ---------------------------------------------------------------------------


class TestNocturnalParseRunDir:
    def test_parses_valid_results_json(self, tmp_path):
        run_dir = _make_run(
            tmp_path,
            "512ctx_72fh",
            "ttm",
            "2026-02-21_2239_brown_2019_zeroshot",
            _RESULTS_TEMPLATE,
            _CONFIG_TEMPLATE,
        )
        s = NocturnalSummarizer(tmp_path)
        row = s._parse_run_dir("512ctx_72fh", "ttm", run_dir)

        assert row is not None
        assert row["model"] == "ttm"
        assert row["dataset"] == "brown_2019"
        assert row["mode"] == "zeroshot"  # hyphen stripped
        assert row["rmse"] == pytest.approx(3.53)
        assert row["total_episodes"] == 1830
        assert row["context_length"] == 512
        assert row["forecast_length"] == 72
        assert row["git_commit"] == "9ee5b38"
        assert row["experiment_type"] == "nocturnal_forecasting"
        assert row["ctx_fh"] == "512ctx_72fh"

    def test_mae_mape_mse_are_nan(self, tmp_path):
        """Overall MAE/MAPE/MSE are not in nocturnal outputs — should be NaN."""
        run_dir = _make_run(
            tmp_path, "512ctx_72fh", "ttm", "2026-02-21_2239_d_m", _RESULTS_TEMPLATE
        )
        s = NocturnalSummarizer(tmp_path)
        row = s._parse_run_dir("512ctx_72fh", "ttm", run_dir)
        assert row is not None
        assert math.isnan(row["mae"])
        assert math.isnan(row["mape"])
        assert math.isnan(row["mse"])

    def test_returns_none_when_results_missing(self, tmp_path):
        run_dir = (
            tmp_path
            / "nocturnal_forecasting"
            / "512ctx_72fh"
            / "ttm"
            / "2026-02-21_2000_d_m"
        )
        run_dir.mkdir(parents=True)
        s = NocturnalSummarizer(tmp_path)
        assert s._parse_run_dir("512ctx_72fh", "ttm", run_dir) is None

    def test_returns_none_for_malformed_json(self, tmp_path):
        run_dir = (
            tmp_path
            / "nocturnal_forecasting"
            / "512ctx_72fh"
            / "ttm"
            / "2026-02-21_2000_d_m"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "nocturnal_results.json").write_text("INVALID")
        s = NocturnalSummarizer(tmp_path)
        assert s._parse_run_dir("512ctx_72fh", "ttm", run_dir) is None

    def test_returns_none_when_overall_rmse_missing(self, tmp_path):
        bad = {k: v for k, v in _RESULTS_TEMPLATE.items() if k != "overall_rmse"}
        run_dir = _make_run(tmp_path, "512ctx_72fh", "ttm", "2026-02-21_2000_d_m", bad)
        s = NocturnalSummarizer(tmp_path)
        assert s._parse_run_dir("512ctx_72fh", "ttm", run_dir) is None

    def test_fallback_episode_count_from_per_patient(self, tmp_path):
        """If total_episodes is absent, sum from per_patient list."""
        results = {k: v for k, v in _RESULTS_TEMPLATE.items() if k != "total_episodes"}
        run_dir = _make_run(
            tmp_path, "512ctx_72fh", "ttm", "2026-02-21_2000_d_m", results
        )
        s = NocturnalSummarizer(tmp_path)
        row = s._parse_run_dir("512ctx_72fh", "ttm", run_dir)
        assert row is not None
        assert row["total_episodes"] == 11  # 6 + 5 from per_patient fixture

    def test_reads_alternate_config_filename(self, tmp_path):
        """experiment_configs.json (plural) is a valid fallback."""
        run_dir = _make_run(
            tmp_path,
            "512ctx_72fh",
            "ttm",
            "2026-02-21_2000_d_m",
            _RESULTS_TEMPLATE,
            _CONFIG_TEMPLATE,
            config_filename="experiment_configs.json",
        )
        s = NocturnalSummarizer(tmp_path)
        row = s._parse_run_dir("512ctx_72fh", "ttm", run_dir)
        assert row is not None
        assert row["git_commit"] == "9ee5b38"


# ---------------------------------------------------------------------------
# Tests: summarize
# ---------------------------------------------------------------------------


class TestNocturnalSummarize:
    def test_produces_one_row_per_run(self, tmp_path):
        for i, dataset in enumerate(["brown_2019", "aleppo_2017"]):
            results = dict(_RESULTS_TEMPLATE, dataset=dataset, overall_rmse=3.0 + i)
            _make_run(
                tmp_path,
                "512ctx_72fh",
                "ttm",
                f"2026-02-21_200{i}_{dataset}_zeroshot",
                results,
            )

        df = NocturnalSummarizer(tmp_path).summarize()
        assert len(df) == 2
        assert set(df["dataset"]) == {"brown_2019", "aleppo_2017"}

    def test_summary_csv_columns(self, tmp_path):
        _make_run(
            tmp_path, "512ctx_72fh", "ttm", "2026-02-21_2239_d_m", _RESULTS_TEMPLATE
        )
        NocturnalSummarizer(tmp_path).summarize()
        df = pd.read_csv(tmp_path / "nocturnal_forecasting" / "summary.csv")
        for col in ["run_id", "model", "dataset", "mode", "rmse", "total_episodes"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_skips_runs_without_results_file(self, tmp_path):
        _make_run(
            tmp_path, "512ctx_72fh", "ttm", "2026-02-21_2239_d_m", _RESULTS_TEMPLATE
        )
        incomplete = (
            tmp_path
            / "nocturnal_forecasting"
            / "512ctx_72fh"
            / "ttm"
            / "2026-02-21_2300_d_m"
        )
        incomplete.mkdir(parents=True)

        df = NocturnalSummarizer(tmp_path).summarize()
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Tests: best_runs
# ---------------------------------------------------------------------------


class TestNocturnalBestRuns:
    def _populate(self, tmp_path):
        runs = [
            ("brown_2019", 3.53),
            ("brown_2019", 3.20),  # best brown
            ("aleppo_2017", 2.85),  # best aleppo (and globally)
        ]
        for i, (dataset, rmse) in enumerate(runs):
            results = dict(_RESULTS_TEMPLATE, dataset=dataset, overall_rmse=rmse)
            _make_run(
                tmp_path,
                "512ctx_72fh",
                "ttm",
                f"2026-02-21_200{i}_{dataset}_zeroshot",
                results,
            )

    def test_best_by_model_dataset(self, tmp_path):
        self._populate(tmp_path)
        result = NocturnalSummarizer(tmp_path).best_runs()
        by_md = result["by_model_dataset"]
        assert len(by_md) == 2
        brown = by_md[by_md["dataset"] == "brown_2019"].iloc[0]
        assert brown["rmse"] == pytest.approx(3.20)

    def test_best_by_model(self, tmp_path):
        self._populate(tmp_path)
        result = NocturnalSummarizer(tmp_path).best_runs()
        by_m = result["by_model"]
        assert len(by_m) == 1
        assert by_m.iloc[0]["rmse"] == pytest.approx(2.85)
