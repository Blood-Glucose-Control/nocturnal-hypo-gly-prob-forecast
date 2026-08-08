from src.models.chronos2.config import Chronos2Config


class TestChronos2ConfigKnownCovariates:
    """Tests for known_covariate_cols in Chronos2Config."""

    def test_default_empty_and_independent_of_past_covariates(self):
        """Default is empty; past and known covariates are independent fields."""
        config = Chronos2Config()
        assert config.known_covariate_cols == []

        config = Chronos2Config(
            covariate_cols=["iob"],
            known_covariate_cols=["hour_sin", "hour_cos"],
        )
        assert config.covariate_cols == ["iob"]
        assert config.known_covariate_cols == ["hour_sin", "hour_cos"]

    def test_backward_compat_missing_field(self):
        """Config from dict without known_covariate_cols gets default []."""
        config = Chronos2Config(**{"model_type": "chronos2", "forecast_length": 72})
        assert config.known_covariate_cols == []
