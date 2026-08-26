"""Compatibility shims for pytorchts + modern GluonTS imports."""

# pyright: reportMissingImports=false

import inspect
import sys


def _install_distribution_output_alias() -> None:
    try:
        import gluonts.torch.modules.distribution_output  # noqa: F401
    except ModuleNotFoundError:
        from gluonts.torch.distributions import distribution_output

        sys.modules["gluonts.torch.modules.distribution_output"] = distribution_output


def _install_predictor_freq_kwarg_compat() -> None:
    from gluonts.torch.model.predictor import PyTorchPredictor

    init_signature = inspect.signature(PyTorchPredictor.__init__)
    if "freq" in init_signature.parameters:
        return

    original_init = PyTorchPredictor.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.pop("freq", None)
        original_init(self, *args, **kwargs)

    PyTorchPredictor.__init__ = _patched_init


_install_distribution_output_alias()
_install_predictor_freq_kwarg_compat()
