"""Tests for sweep task config loading via schema-backed YAML parsing."""
# pyright: reportMissingImports=false

from pathlib import Path

from src.workflows.sweeps.tasks.forecasting.eval import _load_eval_configs
from src.workflows.sweeps.tasks.forecasting.train import _load_sweep_configs


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_load_sweep_configs_reads_schema_validated_jobs(tmp_path: Path) -> None:
    spec = _write_yaml(
        tmp_path / "train_sweep.yaml",
        """
jobs:
  - model_config: configs/models/ttm/fine_tune.yaml
    datasets: [aleppo_2017, lynch_2022]
        """,
    )

    configs = _load_sweep_configs(
        sweep_spec=spec,
        model_config_dir=None,
        model_config_glob="*.yaml",
        datasets=None,
    )

    assert len(configs) == 1
    assert configs[0].model_config_path == "configs/models/ttm/fine_tune.yaml"
    assert configs[0].datasets == ("aleppo_2017", "lynch_2022")


def test_load_eval_configs_keeps_default_and_override_precedence(
    tmp_path: Path,
) -> None:
    spec = _write_yaml(
        tmp_path / "eval_sweep.yaml",
        """
probabilistic: false
no_dilate: true
forecast_length: 120
output_dir_template: runs/{dataset}
jobs:
  - model_config: configs/models/ttm/fine_tune.yaml
    context_length: 512
    finetuned_datasets: [aleppo_2017]
  - model_config: configs/models/ttm/zero_shot.yaml
    context_length: 256
    zeroshot_datasets: [lynch_2022]
    forecast_length: 96
    probabilistic: true
    no_dilate: false
        """,
    )

    configs = _load_eval_configs(
        sweep_spec=spec,
        probabilistic_override=None,
        no_dilate_override=None,
        forecast_length_override=None,
    )

    assert len(configs) == 2
    assert configs[0].forecast_length == 120
    assert configs[0].probabilistic is False
    assert configs[0].no_dilate is True
    assert configs[0].output_dir_template == "runs/{dataset}"
    assert configs[1].forecast_length == 96
    assert configs[1].probabilistic is True
    assert configs[1].no_dilate is False

    overridden = _load_eval_configs(
        sweep_spec=spec,
        probabilistic_override=True,
        no_dilate_override=False,
        forecast_length_override=72,
    )
    assert all(item.probabilistic is True for item in overridden)
    assert all(item.no_dilate is False for item in overridden)
    assert all(item.forecast_length == 72 for item in overridden)
