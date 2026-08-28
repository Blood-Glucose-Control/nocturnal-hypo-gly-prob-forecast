import json
import os
import pickle
from types import SimpleNamespace

import pytest

from src.models.base.checkpoint_helpers import CHECKPOINT_PATH_KEY
from src.models.chronos2.config import (  # pyright: ignore[reportMissingImports]
    Chronos2Config,
)
from src.models.chronos2.model import (  # pyright: ignore[reportMissingImports]
    Chronos2Forecaster,
)


def test_materialize_intermediate_checkpoints_creates_snapshot_tree(tmp_path):
    config = Chronos2Config(training_mode="fine_tune", checkpoint_save_steps=1000)
    model = Chronos2Forecaster(config)

    output_dir = tmp_path / "chronos2_run"
    w0_dir = output_dir / "models" / "Chronos2" / "W0"
    fine_tuned_ckpt = w0_dir / "fine-tuned-ckpt"
    checkpoint_dir = w0_dir / "checkpoint-1000"
    fine_tuned_ckpt.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text("{}")
    (output_dir / "models" / "OtherModel").mkdir(parents=True, exist_ok=True)
    (output_dir / "models" / "OtherModel" / "meta.txt").write_text("ok")
    (output_dir / "models" / "Chronos2" / "config.json").write_text("{}")
    (w0_dir / "extra.bin").write_text("x")

    with open(w0_dir / "model.pkl", "wb") as model_file:
        pickle.dump(SimpleNamespace(path="orig"), model_file)

    (fine_tuned_ckpt / "adapter_model.safetensors").write_text("final-adapter")
    (fine_tuned_ckpt / "tokenizer.json").write_text("tokenizer")
    (checkpoint_dir / "adapter_model.safetensors").write_text("step-1000-adapter")

    (output_dir / "model.pt").mkdir(parents=True, exist_ok=True)
    (output_dir / "model.pt" / "metadata.json").write_text("{}")

    model._materialize_intermediate_checkpoints(str(output_dir))

    snapshot_dir = output_dir / "snapshots" / "step_1000"
    snapshot_model_pt = snapshot_dir / "model.pt"
    shadow_predictor = snapshot_dir / "predictor"
    shadow_w0 = shadow_predictor / "models" / "Chronos2" / "W0"
    shadow_ft_ckpt = shadow_w0 / "fine-tuned-ckpt"

    assert snapshot_model_pt.exists()
    assert json.loads((snapshot_model_pt / "chronos2_predictor.json").read_text()) == {
        CHECKPOINT_PATH_KEY: "../predictor"
    }
    assert (snapshot_model_pt / "config.json").exists()
    assert (snapshot_model_pt / "metadata.json").exists()
    assert os.path.islink(shadow_predictor / "metrics.json")

    assert (shadow_ft_ckpt / "adapter_model.safetensors").read_text() == (
        "step-1000-adapter"
    )
    assert os.path.islink(shadow_ft_ckpt / "tokenizer.json")

    with open(shadow_w0 / "model.pkl", "rb") as model_file:
        patched_model = pickle.load(model_file)
    assert patched_model.path == os.path.abspath(shadow_w0)


def test_validate_registered_quantile_levels_enforces_training_registration():
    model = Chronos2Forecaster(
        Chronos2Config(quantile_levels=[0.1, 0.5, 0.9], training_mode="fine_tune")
    )

    assert model._validate_registered_quantile_levels([0.1, 0.9]) == ["0.1", "0.9"]
    with pytest.raises(ValueError, match="not registered at training time"):
        model._validate_registered_quantile_levels([0.2])
