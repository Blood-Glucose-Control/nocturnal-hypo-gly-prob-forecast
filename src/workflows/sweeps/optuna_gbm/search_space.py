# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Search space for the GBM (gradient-boosted tree) x Optuna sweep.

Tuned for the **annual competition (IOBP2 extension) holdout** reality:

- AutoGluon's tabular reductions (``RecursiveTabular`` / ``DirectTabular``)
  support known-future covariates and static features, but not past covariates,
  so selected insulin/meal channels are routed as known-future features (mirrors
  the TiDE routing).
- Only CGM + insulin (+ optional meal timing) are available at holdout forecast
  time, so the known covariate candidates are insulin-only plus optional
  deterministic hour features and (when enabled) meal-timing signals.
- The gradient-boosting engine (LightGBM / CatBoost / XGBoost) and reduction
  strategy (recursive / direct) are searchable knobs so the study can pick the
  strongest combination.
- ``eval_metric`` is fixed to MAPE because it is low-impact for this externally
  scored objective and MAPE is equivalent to MARD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import optuna

from ....models.gbm.config import _BACKEND_TO_MODEL_NAME, _STRATEGY_TO_AG_KEY
from ..optuna_chronos2.data import (
    PIECEWISE_STATIC_CANDIDATES,  # noqa: F401  (kept for reference/YAML overrides)
    STATIC_COVARIATE_CANDIDATES,  # noqa: F401
)
from ..optuna_chronos2.features import (
    INSULIN_OBSERVED_COL,
    KNOWN_COVARIATE_CANDIDATES,
    MEAL_EVENT_COL,
    MINS_TO_MEAL_COL,
    PAST_COVARIATE_CANDIDATES,
    PAST_COVARIATE_MASKS,
)

# Insulin-derived known-future covariates only (carb-derived features are
# unavailable in the holdout and are excluded). Preserves
# PAST_COVARIATE_CANDIDATES order.
INSULIN_KNOWN_CANDIDATES: tuple[str, ...] = tuple(
    c
    for c in PAST_COVARIATE_CANDIDATES
    if PAST_COVARIATE_MASKS.get(c) == INSULIN_OBSERVED_COL
)

# Static covariates that vary within IOBP2 AND are populated in the annual
# holdout (mirrors the TiDE static candidate set).
IOBP2_STATIC_CANDIDATES: tuple[str, ...] = (
    "age",
    "gender",
    "height",
    "weight",
    "insulin_type_basal",
    "insulin_type_bolus",
)

# Named lag sets over 5-minute CGM cadence (12 steps/hour). Recursive/direct
# tabular reductions autoregress over these lagged targets.
LAG_SETS: Dict[str, List[int]] = {
    "hourly": [1, 2, 3, 6, 9, 12],
    "recent": [1, 2, 3, 4, 5, 6, 9, 12, 18, 24],
    "multiscale": [1, 2, 3, 6, 12, 18, 24, 36, 48, 72],
}


@dataclass
class GBMSearchPoint:
    """A fully-sampled GBM trial configuration."""

    model_key: str  # AutoGluon TimeSeries model key (RecursiveTabular/DirectTabular)
    hyperparameters: Dict[str, Any]  # inner params for the model_key
    strategy: str
    backend: str
    forecast_length: int
    interval_mins: int
    eval_metric: str
    known_cols: List[str]
    static_cols: List[str]


@dataclass
class GBMSearchSpaceBounds:
    """Tunable bounds for the GBM search space (overridable from YAML/config)."""

    forecast_length: int = 24
    interval_mins: int = 5
    eval_metric: str = "MAPE"

    strategy_choices: List[str] = field(default_factory=lambda: ["recursive", "direct"])
    backend_choices: List[str] = field(
        default_factory=lambda: ["lightgbm", "catboost", "xgboost"]
    )
    lag_set_choices: List[str] = field(
        default_factory=lambda: ["hourly", "recent", "multiscale"]
    )
    target_scaler_choices: List[Optional[str]] = field(
        default_factory=lambda: [None, "standard", "robust", "mean_abs"]
    )
    use_first_difference_choices: List[bool] = field(
        default_factory=lambda: [False, True]
    )

    # Tree count is NOT searched: AutoGluon's tabular models fit with adaptive
    # early stopping on an internal validation split (mlforecast passes X_val/
    # y_val), so we set the round cap high and let early stopping pick the
    # effective count -- matching AutoGluon's own design (DEFAULT_NUM_BOOST_ROUND
    # = 10000). Applies to num_boost_round (LGB) / iterations (CAT) / n_estimators
    # (XGB).
    num_boost_round: int = 10000
    # Shared learning-rate range (AutoGluon HPO caps at 0.2 for regression).
    learning_rate_min: float = 5e-3
    learning_rate_max: float = 2e-1

    # LightGBM-specific (ranges anchored to AutoGluon's regression HPO space,
    # widened slightly for our large multi-covariate panel; L1/L2 + row bagging
    # added for regularization).
    num_leaves_min: int = 16
    num_leaves_max: int = 96
    feature_fraction_min: float = 0.6
    feature_fraction_max: float = 1.0
    min_data_in_leaf_min: int = 5
    min_data_in_leaf_max: int = 120
    bagging_fraction_min: float = 0.6
    bagging_fraction_max: float = 1.0
    lgb_lambda_l1_min: float = 1e-8
    lgb_lambda_l1_max: float = 10.0
    lgb_lambda_l2_min: float = 1e-8
    lgb_lambda_l2_max: float = 10.0

    # CatBoost-specific
    depth_min: int = 4
    depth_max: int = 9
    l2_leaf_reg_min: float = 1.0
    l2_leaf_reg_max: float = 10.0
    random_strength_min: float = 1e-3
    random_strength_max: float = 10.0
    bagging_temperature_min: float = 0.0
    bagging_temperature_max: float = 1.0

    # XGBoost-specific
    max_depth_min: int = 3
    max_depth_max: int = 10
    subsample_min: float = 0.6
    subsample_max: float = 1.0
    colsample_bytree_min: float = 0.6
    colsample_bytree_max: float = 1.0
    min_child_weight_min: int = 1
    min_child_weight_max: int = 10
    xgb_reg_alpha_min: float = 1e-8
    xgb_reg_alpha_max: float = 10.0
    xgb_reg_lambda_min: float = 1e-8
    xgb_reg_lambda_max: float = 10.0
    xgb_gamma_min: float = 1e-8
    xgb_gamma_max: float = 5.0

    # Pass-through mlforecast sampling limits (keep trials tractable).
    max_num_items: Optional[int] = 20000
    max_num_samples: Optional[int] = 1000000

    # Covariate candidates
    known_candidates: List[str] = field(
        default_factory=lambda: list(INSULIN_KNOWN_CANDIDATES)
    )
    static_candidates: List[str] = field(
        default_factory=lambda: list(IOBP2_STATIC_CANDIDATES)
    )
    insulin_as_known: bool = True
    include_meal: bool = False
    meal_known_cols: List[str] = field(
        default_factory=lambda: [MEAL_EVENT_COL, MINS_TO_MEAL_COL]
    )


def _suggest_model_hyperparameters(
    trial: optuna.Trial, bounds: GBMSearchSpaceBounds, backend: str
) -> Dict[str, Any]:
    """Sample the underlying tabular model's native hyperparameters.

    The tree count is fixed high (``bounds.num_boost_round``); AutoGluon's
    adaptive early stopping on the internal validation split picks the effective
    number of trees, so it is intentionally not searched.
    """
    learning_rate = trial.suggest_float(
        "learning_rate",
        bounds.learning_rate_min,
        bounds.learning_rate_max,
        log=True,
    )
    mhp: Dict[str, Any] = {"learning_rate": learning_rate}
    if backend == "lightgbm":
        mhp["num_boost_round"] = bounds.num_boost_round
        mhp["num_leaves"] = trial.suggest_int(
            "num_leaves", bounds.num_leaves_min, bounds.num_leaves_max
        )
        mhp["feature_fraction"] = trial.suggest_float(
            "feature_fraction",
            bounds.feature_fraction_min,
            bounds.feature_fraction_max,
        )
        mhp["min_data_in_leaf"] = trial.suggest_int(
            "min_data_in_leaf",
            bounds.min_data_in_leaf_min,
            bounds.min_data_in_leaf_max,
        )
        mhp["bagging_fraction"] = trial.suggest_float(
            "bagging_fraction",
            bounds.bagging_fraction_min,
            bounds.bagging_fraction_max,
        )
        mhp["bagging_freq"] = 1
        mhp["lambda_l1"] = trial.suggest_float(
            "lambda_l1", bounds.lgb_lambda_l1_min, bounds.lgb_lambda_l1_max, log=True
        )
        mhp["lambda_l2"] = trial.suggest_float(
            "lambda_l2", bounds.lgb_lambda_l2_min, bounds.lgb_lambda_l2_max, log=True
        )
    elif backend == "catboost":
        mhp["iterations"] = bounds.num_boost_round
        mhp["depth"] = trial.suggest_int("depth", bounds.depth_min, bounds.depth_max)
        mhp["l2_leaf_reg"] = trial.suggest_float(
            "l2_leaf_reg",
            bounds.l2_leaf_reg_min,
            bounds.l2_leaf_reg_max,
            log=True,
        )
        mhp["random_strength"] = trial.suggest_float(
            "random_strength",
            bounds.random_strength_min,
            bounds.random_strength_max,
            log=True,
        )
        mhp["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature",
            bounds.bagging_temperature_min,
            bounds.bagging_temperature_max,
        )
    elif backend == "xgboost":
        mhp["n_estimators"] = bounds.num_boost_round
        mhp["max_depth"] = trial.suggest_int(
            "max_depth", bounds.max_depth_min, bounds.max_depth_max
        )
        mhp["min_child_weight"] = trial.suggest_int(
            "min_child_weight",
            bounds.min_child_weight_min,
            bounds.min_child_weight_max,
        )
        mhp["subsample"] = trial.suggest_float(
            "subsample", bounds.subsample_min, bounds.subsample_max
        )
        mhp["colsample_bytree"] = trial.suggest_float(
            "colsample_bytree",
            bounds.colsample_bytree_min,
            bounds.colsample_bytree_max,
        )
        mhp["reg_alpha"] = trial.suggest_float(
            "reg_alpha", bounds.xgb_reg_alpha_min, bounds.xgb_reg_alpha_max, log=True
        )
        mhp["reg_lambda"] = trial.suggest_float(
            "reg_lambda", bounds.xgb_reg_lambda_min, bounds.xgb_reg_lambda_max, log=True
        )
        mhp["gamma"] = trial.suggest_float(
            "gamma", bounds.xgb_gamma_min, bounds.xgb_gamma_max, log=True
        )
    else:  # pragma: no cover - guarded by categorical choices
        raise ValueError(f"Unknown GBM backend {backend!r}")
    return mhp


def _suggest_covariates(
    trial: optuna.Trial, bounds: GBMSearchSpaceBounds
) -> tuple[List[str], List[str]]:
    """Sample known-future + static covariate selections (mirrors TiDE/Chronos-2)."""
    use_known_time = trial.suggest_categorical("use_known_time", [False, True])
    known_cols = list(KNOWN_COVARIATE_CANDIDATES) if use_known_time else []
    if bounds.insulin_as_known:
        insulin_known = [
            col
            for col in bounds.known_candidates
            if trial.suggest_categorical(f"use_known__{col}", [False, True])
        ]
        known_cols = list(dict.fromkeys([*known_cols, *insulin_known]))
    if bounds.include_meal:
        meal_known = [
            col
            for col in bounds.meal_known_cols
            if trial.suggest_categorical(f"use_known__{col}", [False, True])
        ]
        known_cols = list(dict.fromkeys([*known_cols, *meal_known]))

    static_cols = [
        col
        for col in bounds.static_candidates
        if trial.suggest_categorical(f"use_static__{col}", [False, True])
    ]
    return known_cols, static_cols


def suggest_gbm_point(
    trial: optuna.Trial, bounds: GBMSearchSpaceBounds
) -> GBMSearchPoint:
    """Sample one GBM configuration + covariate selection for a trial."""
    strategy = trial.suggest_categorical("strategy", bounds.strategy_choices)
    backend = trial.suggest_categorical("backend", bounds.backend_choices)
    lag_set = trial.suggest_categorical("lag_set", bounds.lag_set_choices)
    target_scaler = trial.suggest_categorical(
        "target_scaler", bounds.target_scaler_choices
    )
    use_first_difference = trial.suggest_categorical(
        "use_first_difference", bounds.use_first_difference_choices
    )

    model_hyperparameters = _suggest_model_hyperparameters(trial, bounds, backend)
    known_cols, static_cols = _suggest_covariates(trial, bounds)

    hyperparameters: Dict[str, Any] = {
        "model_name": _BACKEND_TO_MODEL_NAME[backend],
        "model_hyperparameters": model_hyperparameters,
        "lags": list(LAG_SETS[lag_set]),
    }
    if target_scaler is not None:
        hyperparameters["target_scaler"] = target_scaler
    if use_first_difference:
        hyperparameters["differences"] = [1]
    if bounds.max_num_items is not None:
        hyperparameters["max_num_items"] = bounds.max_num_items
    if bounds.max_num_samples is not None:
        hyperparameters["max_num_samples"] = bounds.max_num_samples

    return GBMSearchPoint(
        model_key=_STRATEGY_TO_AG_KEY[strategy],
        hyperparameters=hyperparameters,
        strategy=strategy,
        backend=backend,
        forecast_length=bounds.forecast_length,
        interval_mins=bounds.interval_mins,
        eval_metric=bounds.eval_metric,
        known_cols=known_cols,
        static_cols=static_cols,
    )
