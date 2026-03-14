# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Tests for ModelRegistry.

These tests cover:
- Registry is importable with no heavy deps
- @ModelRegistry.register() decorator works
- get() returns the registered class
- get() raises KeyError for unknown names
- list_models() returns sorted names
- Re-registration of the same class is a no-op
- Conflicting registration raises ValueError
- load_from_checkpoint() resolves by registry key in metadata.json
- load_from_checkpoint() resolves by class name (legacy fallback)
- load_from_checkpoint() raises FileNotFoundError when metadata.json absent
- load_from_checkpoint() raises KeyError when model_type unrecognised
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.models.base.registry import ModelRegistry, _CLASS_NAME_TO_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RegistryIsolation:
    """Context manager that saves and restores ModelRegistry._registry."""

    def __enter__(self):
        self._saved = dict(ModelRegistry._registry)
        return self

    def __exit__(self, *args):
        ModelRegistry._registry.clear()
        ModelRegistry._registry.update(self._saved)


# ---------------------------------------------------------------------------
# Basic registration
# ---------------------------------------------------------------------------


def test_register_and_get():
    with _RegistryIsolation():

        @ModelRegistry.register("_test_model")
        class _DummyModel:
            pass

        assert ModelRegistry.get("_test_model") is _DummyModel


def test_get_raises_for_unknown():
    with _RegistryIsolation():
        with pytest.raises(KeyError, match="_no_such_model"):
            ModelRegistry.get("_no_such_model")


def test_list_models_is_sorted():
    with _RegistryIsolation():

        @ModelRegistry.register("_z_model")
        class _Z:
            pass

        @ModelRegistry.register("_a_model")
        class _A:
            pass

        names = ModelRegistry.list_models()
        assert names == sorted(names)
        assert "_z_model" in names
        assert "_a_model" in names


def test_reregistration_same_class_is_noop():
    with _RegistryIsolation():

        @ModelRegistry.register("_noop_model")
        class _Noop:
            pass

        # Re-registering the identical class should not raise
        ModelRegistry.register("_noop_model")(_Noop)
        assert ModelRegistry.get("_noop_model") is _Noop


def test_conflicting_registration_raises():
    with _RegistryIsolation():

        @ModelRegistry.register("_conflict_model")
        class _First:
            pass

        class _Second:
            pass

        with pytest.raises(ValueError, match="_conflict_model"):
            ModelRegistry.register("_conflict_model")(_Second)


# ---------------------------------------------------------------------------
# load_from_checkpoint
# ---------------------------------------------------------------------------


def test_load_from_checkpoint_by_registry_key():
    """load_from_checkpoint resolves model type from metadata.json registry key."""
    with _RegistryIsolation():
        loaded = []

        @ModelRegistry.register("_cp_model")
        class _CPModel:
            @classmethod
            def load(cls, path):
                loaded.append(path)
                return cls()

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = {"model_type": "_cp_model", "is_fitted": True}
            (Path(tmpdir) / "metadata.json").write_text(json.dumps(metadata))

            instance = ModelRegistry.load_from_checkpoint(tmpdir)

        assert len(loaded) == 1
        assert isinstance(instance, _CPModel)


def test_load_from_checkpoint_by_class_name_fallback():
    """load_from_checkpoint falls back to _CLASS_NAME_TO_KEY mapping."""
    with _RegistryIsolation():
        loaded = []

        @ModelRegistry.register("_legacy_key")
        class _LegacyModel:
            @classmethod
            def load(cls, path):
                loaded.append(path)
                return cls()

        # Temporarily patch _CLASS_NAME_TO_KEY
        original = dict(_CLASS_NAME_TO_KEY)
        _CLASS_NAME_TO_KEY["_OldClassName"] = "_legacy_key"

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                metadata = {"model_type": "_OldClassName", "is_fitted": True}
                (Path(tmpdir) / "metadata.json").write_text(json.dumps(metadata))

                instance = ModelRegistry.load_from_checkpoint(tmpdir)

            assert isinstance(instance, _LegacyModel)
        finally:
            _CLASS_NAME_TO_KEY.clear()
            _CLASS_NAME_TO_KEY.update(original)


def test_load_from_checkpoint_missing_metadata():
    with _RegistryIsolation():
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="metadata.json"):
                ModelRegistry.load_from_checkpoint(tmpdir)


def test_load_from_checkpoint_unknown_model_type():
    with _RegistryIsolation():
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = {"model_type": "_totally_unknown_xyz"}
            (Path(tmpdir) / "metadata.json").write_text(json.dumps(metadata))

            with pytest.raises(KeyError, match="_totally_unknown_xyz"):
                ModelRegistry.load_from_checkpoint(tmpdir)


# ---------------------------------------------------------------------------
# Real-world models are registered (when imports succeed)
# ---------------------------------------------------------------------------


def test_real_models_registered_when_imported():
    """Importing src.models triggers registrations; check at least one succeeds.

    This test is not parameterised per-model because each model has optional
    heavy dependencies.  We simply assert that after importing src.models at
    least *some* models are in the registry — if ALL imports failed (no deps
    at all) we skip with an informative message rather than failing.
    """
    import src.models  # noqa: F401 — side-effect import

    registered = ModelRegistry.list_models()
    if not registered:
        pytest.skip(
            "No models registered after importing src.models. "
            "This environment is missing all optional model dependencies."
        )

    # Verify each registered model has at least a `load` classmethod
    for name in registered:
        cls = ModelRegistry.get(name)
        assert hasattr(
            cls, "load"
        ), f"Registered model '{name}' ({cls.__name__}) lacks a .load() classmethod"


def test_class_name_to_key_covers_all_registered():
    """Verify _CLASS_NAME_TO_KEY stays in sync with registered models.

    Every registered model's class name must appear in _CLASS_NAME_TO_KEY
    so that load_from_checkpoint() works on old checkpoints that store
    the class name instead of the registry key.
    """
    import src.models  # noqa: F401

    registered = ModelRegistry.list_models()
    if not registered:
        pytest.skip("No models registered in this environment.")

    for name in registered:
        cls = ModelRegistry.get(name)
        assert cls.__name__ in _CLASS_NAME_TO_KEY, (
            f"Registered model '{name}' ({cls.__name__}) is missing from "
            f"_CLASS_NAME_TO_KEY — old checkpoints won't load via "
            f"load_from_checkpoint()"
        )
        assert _CLASS_NAME_TO_KEY[cls.__name__] == name, (
            f"_CLASS_NAME_TO_KEY['{cls.__name__}'] = "
            f"'{_CLASS_NAME_TO_KEY[cls.__name__]}' but model is registered "
            f"as '{name}'"
        )
