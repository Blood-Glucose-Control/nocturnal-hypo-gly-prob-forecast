"""Static wiring tests for Darts-backed TSMixer integration surfaces."""

from pathlib import Path


def test_model_factory_has_tsmixer_branch() -> None:
    factory_text = Path("src/models/factory.py").read_text(encoding="utf-8")
    assert 'elif model_type == "tsmixer":' in factory_text
    assert "TSMixerForecaster" in factory_text
    assert "TSMixerConfig" in factory_text


def test_workflow_modeling_has_tsmixer_create_and_load_paths() -> None:
    modeling_text = Path("src/workflows/forecasting/modeling.py").read_text(
        encoding="utf-8"
    )
    assert 'if model_type == "tsmixer":' in modeling_text
    assert 'if model_type_lower == "tsmixer":' in modeling_text
    assert '"tsmixer"' in modeling_text


def test_tsmixer_covariate_capability_and_smoke_profile() -> None:
    grand_summary_text = Path("src/experiments/nocturnal/grand_summary.py").read_text(
        encoding="utf-8"
    )
    assert '"tsmixer": {' in grand_summary_text
    assert '"supports_past_covariates": True' in grand_summary_text

    smoke_cfg = Path("configs/models/tsmixer/00_iob_cob_smoke.yaml").read_text(
        encoding="utf-8"
    )
    assert 'covariate_cols: ["iob", "cob"]' in smoke_cfg
