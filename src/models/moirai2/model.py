# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""Moirai-2.0 / Moirai-MoE PEFT-LoRA fine-tune forecaster.

Neither Moirai-2.0 nor Moirai-MoE ships a native uni2ts fine-tune module (only
Moirai-1.x has ``finetune.py``). Both are inference-only wrappers whose modules
*do* consume known-future covariates via the any-variate packing. This class
adds a PEFT-LoRA fine-tune path by **reusing** the Moirai-1.x forecaster's
packing (``_prepare_training_data`` → ``model._convert`` → ``_MoiraiPatchedDataset``;
the ``_convert`` signatures are identical across the three wrappers) and swapping:

* the model wrapper (``Moirai2Forecast`` / ``MoiraiMoEForecast``),
* the trainable module (LoRA-wrapped via ``peft``),
* the training forward + loss:
    - Moirai-MoE: distribution output → ``PackedNLLLoss`` (identical to 1.x).
    - Moirai-2.0: quantile output → ``PackedQuantileMAELoss`` on the immediate
      next-patch head (predict-token 0). Moirai-2.0 natively predicts
      ``num_predict_token`` patches ahead; the 1-step head is a correct,
      well-formed training signal for LoRA adaptation. Full multi-token
      alignment is a documented refinement (see the scoping run-book).

The extra config attributes (``variant`` + LoRA knobs) are read via ``getattr``
with defaults, so a plain ``MoiraiConfig`` works (its ``__init__`` silently drops
unknown keys); the sweep objective sets them on the config instance.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

import torch

from src.models.moirai.config import MoiraiConfig
from src.models.moirai.model import MoiraiForecaster, _MoiraiPatchedDataset
from src.utils.logging_helper import info_print  # type: ignore

_DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")


class Moirai2Config(MoiraiConfig):
    """``MoiraiConfig`` that also retains the Moirai-2/MoE LoRA sweep knobs.

    ``MoiraiConfig.__init__`` silently drops unknown kwargs, so the LoRA/variant
    keys are re-attached here after the base init.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.variant: str = kwargs.get("variant", "moirai2")
        self.lora_r: int = int(kwargs.get("lora_r", 16))
        self.lora_alpha: int = int(kwargs.get("lora_alpha", 32))
        self.lora_dropout: float = float(kwargs.get("lora_dropout", 0.05))
        self.lora_target_modules: list[str] = list(
            kwargs.get("lora_target_modules", _DEFAULT_LORA_TARGETS)
        )


class Moirai2LoraForecaster(MoiraiForecaster):
    """LoRA fine-tune forecaster for Moirai-2.0 and Moirai-MoE.

    Reads the following attributes from ``config`` (set by the sweep objective):

    * ``variant``: ``"moirai2"`` or ``"moirai_moe"`` (default ``"moirai2"``).
    * ``lora_r`` / ``lora_alpha`` / ``lora_dropout`` / ``lora_target_modules``.
    """

    @property
    def variant(self) -> str:
        return str(getattr(self.config, "variant", "moirai2"))

    def _initialize_model(self) -> Any:
        if not self.config.model_path:
            raise ValueError("model_path must be set")
        variant = self.variant
        info_print(f"Loading pretrained {variant}: {self.config.model_path}")
        info_print(f"  past_feat_dynamic_real_dim = {self.config.past_covariate_dim}")
        info_print(f"  feat_dynamic_real_dim = {self.config.known_covariate_dim}")

        if variant == "moirai_moe":
            from uni2ts.model.moirai_moe import (  # pyright: ignore[reportMissingImports]
                MoiraiMoEForecast,
                MoiraiMoEModule,
            )

            module = MoiraiMoEModule.from_pretrained(self.config.model_path)
            patch = self.config.patch_size
            self.model = MoiraiMoEForecast(
                module=module,
                prediction_length=self.config.forecast_length,
                context_length=self.config.context_length,
                target_dim=1,
                feat_dynamic_real_dim=self.config.known_covariate_dim,
                past_feat_dynamic_real_dim=self.config.past_covariate_dim,
                patch_size=int(patch) if isinstance(patch, int) else 16,
                num_samples=self.config.num_samples,
            )
        else:  # moirai2
            from uni2ts.model.moirai2 import (  # pyright: ignore[reportMissingImports]
                Moirai2Forecast,
                Moirai2Module,
            )

            module = Moirai2Module.from_pretrained(self.config.model_path)
            self.model = Moirai2Forecast(
                module=module,
                prediction_length=self.config.forecast_length,
                context_length=self.config.context_length,
                target_dim=1,
                feat_dynamic_real_dim=self.config.known_covariate_dim,
                past_feat_dynamic_real_dim=self.config.past_covariate_dim,
            )

        info_print(f"  {variant} loaded successfully")
        return self.model

    def _select_patch_size(self) -> int:
        """Both Moirai-2.0 and Moirai-MoE-R-small use a fixed patch size of 16."""
        patch = self.config.patch_size
        return int(patch) if isinstance(patch, int) else 16

    def _lora_wrap(self, module: Any) -> Any:
        from peft import (  # pyright: ignore[reportMissingImports]
            LoraConfig,
            get_peft_model,
        )

        targets = list(
            getattr(self.config, "lora_target_modules", _DEFAULT_LORA_TARGETS)
        )
        # PEFT suffix-matches target names; the top-level ``in_proj`` / ``out_proj``
        # ResidualBlocks collide with the attention ``self_attn.out_proj`` Linear,
        # so exclude the container blocks by exact name.
        exclude = [
            n for n, m in module.named_modules() if type(m).__name__ == "ResidualBlock"
        ]
        cfg = LoraConfig(
            r=int(getattr(self.config, "lora_r", 16)),
            lora_alpha=int(getattr(self.config, "lora_alpha", 32)),
            lora_dropout=float(getattr(self.config, "lora_dropout", 0.05)),
            target_modules=targets,
            exclude_modules=exclude or None,
            bias="none",
        )
        return get_peft_model(module, cfg)

    def _train_model(
        self,
        train_data: Any,
        output_dir: str,
        epoch_callback: Optional[Callable[[int, float], None]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        del kwargs
        variant = self.variant
        info_print(f"👉 Starting {variant} LoRA fine-tuning")

        loader, _, _ = self._prepare_training_data(train_data)
        n_samples = len(cast(_MoiraiPatchedDataset, loader.dataset))
        if self.model is None:
            self.model = self._initialize_model()
        model = cast(Any, self.model)

        device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.config.use_cpu else "cpu"
        )

        # LoRA-wrap the trainable module.
        peft_module = self._lora_wrap(model.module)
        peft_module.to(device)
        peft_module.train()
        trainable = sum(p.numel() for p in peft_module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_module.parameters())
        info_print(f"  LoRA trainable: {trainable / 1e6:.3f}M / {total / 1e6:.1f}M")

        if variant == "moirai_moe":
            from uni2ts.loss.packed import (  # pyright: ignore[reportMissingImports]
                PackedNLLLoss,
            )

            loss_func: Any = PackedNLLLoss()
            num_q, patch_sz = 0, 0
        else:
            from uni2ts.loss.packed import (  # pyright: ignore[reportMissingImports]
                PackedQuantileMAELoss,
            )

            loss_func = PackedQuantileMAELoss()
            num_q = len(model.module.quantile_levels)
            patch_sz = int(model.module.patch_size)

        optimizer = torch.optim.AdamW(
            (p for p in peft_module.parameters() if p.requires_grad),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
        )

        num_epochs = self.config.num_epochs
        best_loss = float("inf")
        for epoch in range(num_epochs):
            epoch_loss, n_batches = 0.0, 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()

                if variant == "moirai_moe":
                    distr = peft_module(
                        target=batch["target"],
                        observed_mask=batch["observed_mask"],
                        sample_id=batch["sample_id"],
                        time_id=batch["time_id"],
                        variate_id=batch["variate_id"],
                        prediction_mask=batch["prediction_mask"],
                        patch_size=batch["patch_size"],
                    )
                    loss = loss_func(
                        pred=distr,
                        target=batch["target"],
                        prediction_mask=batch["prediction_mask"],
                        observed_mask=batch["observed_mask"],
                        sample_id=batch["sample_id"],
                        variate_id=batch["variate_id"],
                    )
                else:  # moirai2: quantile head, predict-token 0 (1-step) loss
                    preds, scaled_target = peft_module(
                        target=batch["target"],
                        observed_mask=batch["observed_mask"],
                        sample_id=batch["sample_id"],
                        time_id=batch["time_id"],
                        variate_id=batch["variate_id"],
                        prediction_mask=batch["prediction_mask"],
                        training_mode=True,
                    )
                    # preds: (..., num_predict_token * num_quantiles * patch).
                    # Keep predict-token 0 → (..., num_quantiles * patch).
                    preds = preds[..., : num_q * patch_sz]
                    loss = loss_func(
                        pred=preds,
                        target=scaled_target,
                        prediction_mask=batch["prediction_mask"],
                        observed_mask=batch["observed_mask"],
                        sample_id=batch["sample_id"],
                        variate_id=batch["variate_id"],
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in peft_module.parameters() if p.requires_grad),
                    self.config.gradient_clip_val,
                )
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            info_print(f"   Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.6f}")
            best_loss = min(best_loss, avg_loss)

            if epoch_callback is not None:
                epoch_callback(epoch, avg_loss)

        # Merge LoRA deltas into the base weights so inference uses the fine-tuned
        # module, then hand it back to the forecast wrapper.
        peft_module.eval()
        merged = peft_module.merge_and_unload()
        model.module = merged
        self.predictor = None  # force predictor rebuild on the merged module

        return {
            "status": "success",
            "samples": int(n_samples),
            "best_loss": best_loss,
            "epochs": num_epochs,
        }
