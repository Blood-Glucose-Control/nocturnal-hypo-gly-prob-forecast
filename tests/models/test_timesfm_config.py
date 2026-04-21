"""Tests for TimesFM configuration validation."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="requires torch",
)


@pytest.fixture
def timesfm_config_cls():
    from src.models.timesfm.config import TimesFMConfig

    return TimesFMConfig


class TestTimesFMConfigValidation:
    def test_interval_mins_default_is_valid(self, timesfm_config_cls):
        cfg = timesfm_config_cls()
        assert cfg.interval_mins == 5

    @pytest.mark.parametrize("interval_mins", [0, -5])
    def test_interval_mins_must_be_positive(self, timesfm_config_cls, interval_mins):
        with pytest.raises(ValueError, match="interval_mins must be positive"):
            timesfm_config_cls(interval_mins=interval_mins)
