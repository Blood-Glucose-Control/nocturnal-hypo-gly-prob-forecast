"""Tests for ModelRegistry."""

import pytest

from src.models.base.registry import ModelRegistry


@pytest.fixture()
def clean_registry():
    """Clear registry before test, restore after."""
    saved = dict(ModelRegistry._registry)
    ModelRegistry._registry.clear()
    yield
    ModelRegistry._registry.clear()
    ModelRegistry._registry.update(saved)


def test_register_and_get(clean_registry):
    @ModelRegistry.register("_test")
    class _M:
        pass

    assert ModelRegistry.get("_test") is _M


def test_get_unknown_raises(clean_registry):
    with pytest.raises(KeyError, match="_nope"):
        ModelRegistry.get("_nope")


def test_conflict_raises(clean_registry):
    @ModelRegistry.register("_dup")
    class _A:
        pass

    with pytest.raises(ValueError, match="_dup"):
        ModelRegistry.register("_dup")(type("_B", (), {}))


def test_get_missing_dep_raises_key_error(clean_registry, monkeypatch):
    """get() wraps ModuleNotFoundError into a helpful KeyError."""
    from src.models.base import registry as reg_mod

    # Inject a fake model module that will fail to import
    monkeypatch.setitem(reg_mod._MODEL_MODULES, "_fake", "no.such.module")

    with pytest.raises(KeyError, match="failed to import"):
        ModelRegistry.get("_fake")


def test_real_models_available():
    """At least some models are importable and register correctly."""
    available = ModelRegistry.available_models()
    if not available:
        pytest.skip("No model deps installed.")

    from src.models.base import BaseTimeSeriesFoundationModel

    for name in available:
        cls = ModelRegistry.get(name)
        assert issubclass(cls, BaseTimeSeriesFoundationModel)
