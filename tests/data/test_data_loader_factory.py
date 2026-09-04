import pytest

from src.data.diabetes_datasets import data_loader as data_loader_module


def test_get_loader_forwards_use_cached_to_gluroo(monkeypatch: pytest.MonkeyPatch):
    captured_kwargs: dict[str, object] = {}

    class DummyGlurooLoader:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(data_loader_module, "GlurooDataLoader", DummyGlurooLoader)

    loader = data_loader_module.get_loader(
        data_source_name="gluroo",
        keep_columns=["bg_mM"],
        use_cached=False,
        max_workers=3,
        load_all=True,
    )

    assert isinstance(loader, DummyGlurooLoader)
    assert captured_kwargs["keep_columns"] == ["bg_mM"]
    assert captured_kwargs["use_cached"] is False
    assert captured_kwargs["max_workers"] == 3
    assert captured_kwargs["load_all"] is True


def test_get_loader_returns_metabonet_loader(monkeypatch: pytest.MonkeyPatch):
    captured_kwargs: dict[str, object] = {}

    class DummyMetabonetLoader:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(
        data_loader_module,
        "MetabonetDataLoader",
        DummyMetabonetLoader,
    )

    loader = data_loader_module.get_loader(
        data_source_name="metabonet",
        keep_columns=["bg_mM"],
        use_cached=True,
        parallel=False,
        max_workers=1,
        load_all=True,
    )

    assert isinstance(loader, DummyMetabonetLoader)
    assert captured_kwargs["keep_columns"] == ["bg_mM"]
    assert captured_kwargs["use_cached"] is True
    assert captured_kwargs["parallel"] is False
    assert captured_kwargs["max_workers"] == 1
    assert captured_kwargs["load_all"] is True
