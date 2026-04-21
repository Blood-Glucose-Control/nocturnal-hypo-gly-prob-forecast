"""Tests for TimesFM configuration validation."""

import pytest

pytest.importorskip("torch")

from src.models.timesfm.config import TimesFMConfig


class TestTimesFMConfigValidation:
    def test_interval_mins_default_is_valid(self):
        cfg = TimesFMConfig()
        assert cfg.interval_mins == 5

    @pytest.mark.parametrize("interval_mins", [0, -5])
    def test_interval_mins_must_be_positive(self, interval_mins):
        with pytest.raises(ValueError, match="interval_mins must be positive"):
            TimesFMConfig(interval_mins=interval_mins)
