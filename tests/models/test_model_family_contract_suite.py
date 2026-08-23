"""Cross-family static contract checks for model/config implementations."""

from __future__ import annotations

import ast
from pathlib import Path

from src.config.schemas import get_registered_model_config_types

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "src" / "models"
FACTORY_PATH = MODELS_ROOT / "factory.py"
REGISTRY_PATH = MODELS_ROOT / "base" / "registry.py"

MODEL_FAMILY_CLASS_CONTRACT: dict[str, dict[str, object]] = {
    "sundial": {
        "config_class": "SundialConfig",
        "model_class": "SundialForecaster",
        "allowed_model_bases": {"BaseTimeSeriesFoundationModel"},
    },
    "ttm": {
        "config_class": "TTMConfig",
        "model_class": "TTMForecaster",
        "allowed_model_bases": {"BaseTimeSeriesFoundationModel"},
    },
    "chronos2": {
        "config_class": "Chronos2Config",
        "model_class": "Chronos2Forecaster",
        "allowed_model_bases": {"BaseTimeSeriesFoundationModel"},
    },
    "moment": {
        "config_class": "MomentConfig",
        "model_class": "MomentForecaster",
        "allowed_model_bases": {"BaseTimeSeriesFoundationModel"},
    },
    "toto": {
        "config_class": "TotoConfig",
        "model_class": "TotoForecaster",
        "allowed_model_bases": {"BaseTimeSeriesFoundationModel"},
    },
    "moirai": {
        "config_class": "MoiraiConfig",
        "model_class": "MoiraiForecaster",
        "allowed_model_bases": {"BaseTimeSeriesFoundationModel"},
    },
    "timegrad": {
        "config_class": "TimeGradConfig",
        "model_class": "TimeGradForecaster",
        "allowed_model_bases": {"BaseTimeSeriesFoundationModel"},
    },
    "timesfm": {
        "config_class": "TimesFMConfig",
        "model_class": "TimesFMForecaster",
        "allowed_model_bases": {"BaseTimeSeriesFoundationModel"},
    },
    "tide": {
        "config_class": "TiDEConfig",
        "model_class": "TiDEForecaster",
        "allowed_model_bases": {"BaseTimeSeriesFoundationModel"},
    },
    "naive_baseline": {
        "config_class": "NaiveBaselineConfig",
        "model_class": "NaiveBaselineForecaster",
        "allowed_model_bases": {"AutoGluonBaseModel"},
    },
    "statistical": {
        "config_class": "StatisticalConfig",
        "model_class": "StatisticalForecaster",
        "allowed_model_bases": {"AutoGluonBaseModel"},
    },
    "deepar": {
        "config_class": "DeepARConfig",
        "model_class": "DeepARForecaster",
        "allowed_model_bases": {"AutoGluonBaseModel"},
    },
    "patchtst": {
        "config_class": "PatchTSTConfig",
        "model_class": "PatchTSTForecaster",
        "allowed_model_bases": {"AutoGluonBaseModel"},
    },
    "tft": {
        "config_class": "TFTConfig",
        "model_class": "TFTForecaster",
        "allowed_model_bases": {"AutoGluonBaseModel"},
    },
    "tsmixer": {
        "config_class": "TSMixerConfig",
        "model_class": "TSMixerForecaster",
        "allowed_model_bases": {"DartsGlobalModelBase"},
    },
}

SCHEMA_ROUTED_FAMILIES = set(get_registered_model_config_types())
# These families predate the canonical method ordering convention; keep them
# exempt until their implementation-order cleanup pass lands.
SCHEMA_ROUTED_METHOD_ORDER_EXEMPT_FAMILIES = {"moment", "timegrad", "timesfm", "toto"}
AUTOGLUON_THIN_WRAPPER_FAMILIES = {
    "deepar",
    "naive_baseline",
    "patchtst",
    "statistical",
    "tft",
}
FACTORY_NOT_IMPLEMENTED_FAMILIES: set[str] = set()
ORDERED_CORE_METHODS = [
    "__init__",
    "training_backend",
    "supports_zero_shot",
    "supports_probabilistic_forecast",
    "_initialize_model",
    "_prepare_training_data",
    "_train_model",
    "_predict",
    "_predict_batch",
    "_save_checkpoint",
    "_load_checkpoint",
]


def _parse_python(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_by_name(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} not found in module")


def _base_name(base_node: ast.expr) -> str:
    if isinstance(base_node, ast.Name):
        return base_node.id
    if isinstance(base_node, ast.Attribute):
        parts: list[str] = []
        current: ast.AST = base_node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


def _method_names(class_node: ast.ClassDef) -> list[str]:
    return [
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _decorator_names(fn_node: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for decorator in fn_node.decorator_list:
        if isinstance(decorator, ast.Name):
            names.add(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            names.add(decorator.attr)
    return names


def _property_method(
    class_node: ast.ClassDef, method_name: str
) -> ast.FunctionDef | None:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            if "property" in _decorator_names(node):
                return node
    return None


def _config_class_binding(class_node: ast.ClassDef) -> str | None:
    for stmt in class_node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "config_class":
                    if isinstance(stmt.value, ast.Name):
                        return stmt.value.id
        if isinstance(stmt, ast.AnnAssign):
            if (
                isinstance(stmt.target, ast.Name)
                and stmt.target.id == "config_class"
                and isinstance(stmt.value, ast.Name)
            ):
                return stmt.value.id
    return None


def _factory_supported_model_types() -> set[str]:
    tree = _parse_python(FACTORY_PATH)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "SUPPORTED_MODEL_TYPES"
                ):
                    if isinstance(node.value, ast.Tuple):
                        values = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(
                                elt.value, str
                            ):
                                values.append(elt.value)
                        return set(values)
    raise AssertionError("SUPPORTED_MODEL_TYPES tuple not found in factory.py")


def _registry_model_types() -> set[str]:
    tree = _parse_python(REGISTRY_PATH)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_MODEL_MODULES":
                    if isinstance(node.value, ast.Dict):
                        keys: set[str] = set()
                        for key in node.value.keys:
                            if isinstance(key, ast.Constant) and isinstance(
                                key.value, str
                            ):
                                keys.add(key.value)
                        return keys
        if isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "_MODEL_MODULES"
                and isinstance(node.value, ast.Dict)
            ):
                keys: set[str] = set()
                for key in node.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
                return keys
    raise AssertionError("_MODEL_MODULES dict not found in registry.py")


def test_contract_map_matches_registry_models() -> None:
    assert set(MODEL_FAMILY_CLASS_CONTRACT) == _registry_model_types()


def test_factory_supported_models_only_add_explicit_unimplemented_families() -> None:
    factory_supported = _factory_supported_model_types()
    expected_supported = (
        set(MODEL_FAMILY_CLASS_CONTRACT) | FACTORY_NOT_IMPLEMENTED_FAMILIES
    )
    assert factory_supported == expected_supported


def test_model_family_modules_define_expected_classes_and_docstrings() -> None:
    for model_type, contract in MODEL_FAMILY_CLASS_CONTRACT.items():
        config_path = MODELS_ROOT / model_type / "config.py"
        model_path = MODELS_ROOT / model_type / "model.py"

        assert config_path.exists(), f"{model_type}: missing config.py"
        assert model_path.exists(), f"{model_type}: missing model.py"

        config_module = _parse_python(config_path)
        model_module = _parse_python(model_path)
        assert ast.get_docstring(config_module), (
            f"{config_path} missing module docstring"
        )
        assert ast.get_docstring(model_module), f"{model_path} missing module docstring"

        config_class_name = str(contract["config_class"])
        model_class_name = str(contract["model_class"])
        config_class = _class_by_name(config_module, config_class_name)
        model_class = _class_by_name(model_module, model_class_name)

        assert ast.get_docstring(config_class), (
            f"{config_path}:{config_class_name} missing class docstring"
        )
        assert ast.get_docstring(model_class), (
            f"{model_path}:{model_class_name} missing class docstring"
        )

        config_bases = {_base_name(base) for base in config_class.bases}
        assert "ModelConfig" in config_bases, (
            f"{model_type}: {config_class_name} must inherit from ModelConfig"
        )

        allowed_bases = set(contract["allowed_model_bases"])  # type: ignore[arg-type]
        model_bases = {_base_name(base) for base in model_class.bases}
        assert model_bases & allowed_bases, (
            f"{model_type}: {model_class_name} must inherit from one of {sorted(allowed_bases)}; "
            f"found {sorted(model_bases)}"
        )

        supports_zero_shot = _property_method(model_class, "supports_zero_shot")
        assert supports_zero_shot is not None, (
            f"{model_type}: {model_class_name}.supports_zero_shot must be a @property"
        )


def test_schema_routed_models_bind_config_class_and_keep_core_method_order() -> None:
    for model_type in sorted(SCHEMA_ROUTED_FAMILIES):
        contract = MODEL_FAMILY_CLASS_CONTRACT[model_type]
        model_class_name = str(contract["model_class"])
        config_class_name = str(contract["config_class"])
        model_path = MODELS_ROOT / model_type / "model.py"
        model_module = _parse_python(model_path)
        model_class = _class_by_name(model_module, model_class_name)

        assert _config_class_binding(model_class) == config_class_name, (
            f"{model_type}: config_class should bind to {config_class_name}"
        )

        if model_type in SCHEMA_ROUTED_METHOD_ORDER_EXEMPT_FAMILIES:
            continue

        method_names = _method_names(model_class)
        previous = -1
        for method_name in ORDERED_CORE_METHODS:
            if method_name in method_names:
                current = method_names.index(method_name)
                assert current >= previous, (
                    f"{model_type}: method order should follow canonical sequence; "
                    f"{method_name} appears out of order in {method_names}"
                )
                previous = current


def test_autogluon_thin_wrappers_keep_shared_base_responsibilities() -> None:
    forbidden_method_redefinitions = {
        "_prepare_training_data",
        "_train_model",
        "_predict",
        "_predict_batch",
        "_save_checkpoint",
        "_load_checkpoint",
    }
    for model_type in sorted(AUTOGLUON_THIN_WRAPPER_FAMILIES):
        contract = MODEL_FAMILY_CLASS_CONTRACT[model_type]
        model_class_name = str(contract["model_class"])
        model_path = MODELS_ROOT / model_type / "model.py"
        model_module = _parse_python(model_path)
        model_class = _class_by_name(model_module, model_class_name)
        method_names = _method_names(model_class)

        predictor_binding = None
        for stmt in model_class.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "_PREDICTOR_JSON_NAME"
                    ):
                        predictor_binding = stmt
        assert predictor_binding is not None, (
            f"{model_type}: {model_class_name} must define _PREDICTOR_JSON_NAME"
        )

        assert "_train_model_info_log" in method_names, (
            f"{model_type}: thin wrapper should provide model-specific training banner"
        )
        assert "supports_zero_shot" in method_names, (
            f"{model_type}: thin wrapper should declare supports_zero_shot"
        )
        assert method_names.index("supports_zero_shot") < method_names.index(
            "_train_model_info_log"
        ), (
            f"{model_type}: supports_zero_shot should be defined before _train_model_info_log"
        )
        assert not (set(method_names) & forbidden_method_redefinitions), (
            f"{model_type}: thin wrapper should not redefine shared AutoGluon base hooks; "
            f"found {sorted(set(method_names) & forbidden_method_redefinitions)}"
        )
