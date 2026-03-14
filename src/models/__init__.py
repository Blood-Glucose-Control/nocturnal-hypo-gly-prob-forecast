"""Model module exports."""

import logging as _logging

from src.models.factory import create_model_and_config

# Trigger @ModelRegistry.register() side-effects for all model classes.
# Each import below ensures the decorator has run and the class is present
# in ModelRegistry._registry.  Imports are wrapped in try/except because
# many models depend on optional heavy packages (tsfm_public, autogluon,
# gluonts/pts, moirai, momentfm) that may not be installed.

_logger = _logging.getLogger(__name__)

try:
    from src.models.ttm import TTMForecaster  # noqa: F401
except ImportError:
    pass
except Exception:
    _logger.debug("Failed to import TTMForecaster for registry", exc_info=True)

try:
    from src.models.chronos2 import Chronos2Forecaster  # noqa: F401
except ImportError:
    pass
except Exception:
    _logger.debug("Failed to import Chronos2Forecaster for registry", exc_info=True)

try:
    from src.models.tide import TiDEForecaster  # noqa: F401
except ImportError:
    pass
except Exception:
    _logger.debug("Failed to import TiDEForecaster for registry", exc_info=True)

try:
    from src.models.sundial import SundialForecaster  # noqa: F401
except ImportError:
    pass
except Exception:
    _logger.debug("Failed to import SundialForecaster for registry", exc_info=True)

try:
    from src.models.tsmixer import TSMixerForecaster  # noqa: F401
except ImportError:
    pass
except Exception:
    _logger.debug("Failed to import TSMixerForecaster for registry", exc_info=True)

try:
    from src.models.timesfm import TimesFMForecaster  # noqa: F401
except ImportError:
    pass
except Exception:
    _logger.debug("Failed to import TimesFMForecaster for registry", exc_info=True)

try:
    from src.models.timegrad import TimeGradForecaster  # noqa: F401
except ImportError:
    pass
except Exception:
    _logger.debug("Failed to import TimeGradForecaster for registry", exc_info=True)

try:
    from src.models.moirai import MoiraiForecaster  # noqa: F401
except ImportError:
    pass
except Exception:
    _logger.debug("Failed to import MoiraiForecaster for registry", exc_info=True)

try:
    from src.models.moment import MomentForecaster  # noqa: F401
except ImportError:
    pass
except Exception:
    _logger.debug("Failed to import MomentForecaster for registry", exc_info=True)

__all__ = ["create_model_and_config"]
