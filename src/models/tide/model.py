# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
TiDE model class — STUB for future implementation.

STATUS: Experiment scripts validated. Model class not yet implemented.
See scripts/tide_registry_experiment.py for the working experiment pipeline.

TODO (future PR):
  1. Implement TiDEForecaster(Model) following the Chronos-2 pattern:
     - fit(): DatasetRegistry -> gap handling -> AutoGluon TimeSeriesPredictor
     - predict(): midnight episodes -> AutoGluon predict -> DataFrame
  2. Wire into model factory (src/models/factory.py)
  3. Add to holdout_eval.py for standardized evaluation

Architecture notes:
  - TiDE is wrapped in AutoGluon's TimeSeriesPredictor (not used directly)
  - AutoGluon handles: MultiWindowBacktesting, MeanScaler, GlobalCovariateScaler
  - Model class should delegate to AutoGluon, similar to Chronos2Forecaster

Reference implementation: scripts/tide_registry_experiment.py
  - Data loading via DatasetRegistry
  - Gap handling via segment_all_patients()
  - Training via TimeSeriesPredictor.fit()
  - Evaluation via build_midnight_episodes()
"""
