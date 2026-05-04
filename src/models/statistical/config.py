"""
Statistical forecasting model configuration.

Wraps AutoGluon's AutoARIMA, Theta, and NPTS models as classical baselines
for the nocturnal CGM forecasting task.

Design decisions:
  - All three models are non-seasonal (no daily/weekly period present in
    overnight 5-min CGM data). Seasonal components are explicitly disabled.
  - AutoARIMA is the only model that uses covariate columns as exogenous
    regressors; Theta and NPTS are purely univariate.
  - A per-job time_limit (default 7200 s) caps AutoARIMA's per-series fit
    time. When the budget is hit, AutoGluon substitutes Naive forecasts for
    unfit series and saves what it has — no work is lost.
  - Quantile synthesis is performed by AutoGluon from training residuals for
    all three models. The resulting PIT histograms are expected to be
    miscalibrated (reported in the paper).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.models.base import ModelConfig, TrainingBackend

_VALID_MODEL_NAMES = frozenset({"AutoARIMA", "Theta", "NPTS"})


@dataclass
class StatisticalConfig(ModelConfig):
    """Configuration for AutoARIMA / Theta / NPTS via AutoGluon.

    Attributes:
        model_name: "AutoARIMA", "Theta", or "NPTS".

    AutoARIMA notes:
        - seasonal=False: disables SARIMA search (no daily period in overnight CGM).
        - seasonal_period=1: defensive fallback so AutoGluon doesn't try to infer.
        - d=None, D=None: let AutoARIMA choose differencing order automatically.
        - max_p, max_q: search bounds for non-seasonal AR/MA terms.

    Theta notes:
        - decomposition_type="multiplicative" disabled for glucose (can go near zero).
        - season_length=1: skip deseasonalization step.

    NPTS notes:
        - Non-parametric; no seasonal or model hyperparameters to set.
        - Natively probabilistic (kernel-based quantile forecasts).
    """

    model_type: str = "statistical"
    training_mode: str = "from_scratch"
    training_backend: TrainingBackend = TrainingBackend.CUSTOM

    # Which AutoGluon statistical model to use
    model_name: str = "AutoARIMA"  # or "Theta" or "NPTS"

    # AutoARIMA-specific hyperparameters
    autoarima_max_p: int = 3
    autoarima_max_q: int = 3
    autoarima_d: Optional[int] = None  # auto-select differencing order
    autoarima_D: Optional[int] = None  # no seasonal differencing

    # Theta-specific hyperparameters
    theta_decomposition_type: str = "additive"  # additive safe for CGM near-zero values

    # Minimal segment length to keep after gap handling
    min_segment_length: Optional[int] = None

    # Gap handling
    imputation_threshold_mins: int = 45

    # Covariate config
    # AutoARIMA: covariate_cols are passed as exogenous regressors.
    # Theta/NPTS: covariate_cols ignored by the model but kept for pipeline compat.
    covariate_cols: List[str] = field(default_factory=list)
    target_col: str = "bg_mM"
    patient_col: str = "p_num"
    time_col: str = "datetime"
    interval_mins: int = 5

    # AutoGluon settings
    eval_metric: str = "WQL"
    enable_ensemble: bool = False
    # 2-hour cap per (model × dataset) job; AutoGluon substitutes Naive for
    # unfit series on timeout — completed fits are never discarded.
    time_limit: int = 7200

    # Quantile levels for probabilistic output
    quantile_levels: Optional[List[float]] = None

    def __post_init__(self):
        if self.model_name not in _VALID_MODEL_NAMES:
            raise ValueError(
                f"model_name must be one of {sorted(_VALID_MODEL_NAMES)}, "
                f"got '{self.model_name}'."
            )
        if self.min_segment_length is None:
            # Statistical models need at least context_length points to fit
            self.min_segment_length = self.context_length + self.forecast_length

    def get_autogluon_hyperparameters(self) -> Dict:
        """Build hyperparameters dict for TimeSeriesPredictor.fit().

        Returns:
            Dict with the appropriate AutoGluon model key and its hyperparameters.
        """
        if self.model_name == "AutoARIMA":
            hp: Dict = {
                "seasonal": False,
                "seasonal_period": 1,
                "max_p": self.autoarima_max_p,
                "max_q": self.autoarima_max_q,
            }
            if self.autoarima_d is not None:
                hp["d"] = self.autoarima_d
            if self.autoarima_D is not None:
                hp["D"] = self.autoarima_D
            return {"AutoARIMA": hp}

        elif self.model_name == "Theta":
            return {
                "Theta": {
                    "season_length": 1,  # no deseasonalization
                    "decomposition_type": self.theta_decomposition_type,
                }
            }

        else:  # NPTS
            return {"NPTS": {}}
