"""Tests for ModelRegistry."""

import pytest

from src.models.base.registry import ModelRegistry


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Save and restore registry around each test."""
    saved = dict(ModelRegistry._registry)
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


def test_real_models_registered():
    """At least some models register when src.models is imported."""
    import src.models  # noqa: F401

    registered = ModelRegistry.list_models()
    if not registered:
        pytest.skip("No model deps installed.")

    from src.models.base import BaseTimeSeriesFoundationModel

    # Verify each registered class is actually a model, not a stray class
    for name in registered:
        cls = ModelRegistry.get(name)
        assert issubclass(cls, BaseTimeSeriesFoundationModel)
