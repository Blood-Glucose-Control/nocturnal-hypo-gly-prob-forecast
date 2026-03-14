"""Tests for ModelRegistry."""

import pytest

from src.models.base.registry import ModelRegistry


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Clear registry before each test, restore after."""
    saved = dict(ModelRegistry._registry)
    ModelRegistry._registry.clear()
    yield
    ModelRegistry._registry.clear()
    ModelRegistry._registry.update(saved)


def test_register_and_get():
    @ModelRegistry.register("_test")
    class _M:
        pass

    assert ModelRegistry.get("_test") is _M


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="_nope"):
        ModelRegistry.get("_nope")


def test_conflict_raises():
    @ModelRegistry.register("_dup")
    class _A:
        pass

    with pytest.raises(ValueError, match="_dup"):
        ModelRegistry.register("_dup")(type("_B", (), {}))


def test_lazy_import():
    """get() lazily imports the model module on first access."""
    available = ModelRegistry.available_models()
    if not available:
        pytest.skip("No model deps installed.")

    from src.models.base import BaseTimeSeriesFoundationModel

    # Pick first available model — get() should lazily import it
    name = available[0]
    cls = ModelRegistry.get(name)
    assert issubclass(cls, BaseTimeSeriesFoundationModel)
