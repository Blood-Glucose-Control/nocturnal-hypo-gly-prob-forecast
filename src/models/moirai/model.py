"""
Moirai model implementation using the base TSFM framework.

This module provides a concrete implementation of Moirai that inherits from
the base TSFM framework, integrating Salesforce's uni2ts library.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from torch.utils.data import DataLoader, Dataset

# Local imports
from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.models.base.registry import ModelRegistry
from src.models.moirai.config import MoiraiConfig
from src.utils.logging_helper import info_print
from uni2ts.model.moirai import MoiraiForecast, MoiraiFinetune, MoiraiModule


class _MoiraiPatchedDataset(Dataset):
    """Pre-converted training samples in the patched format MoiraiFinetune expects."""

    def __init__(
        self,
        target: torch.Tensor,
        observed_mask: torch.Tensor,
        sample_id: torch.Tensor,
        time_id: torch.Tensor,
        variate_id: torch.Tensor,
        prediction_mask: torch.Tensor,
        patch_size_val: int,
    ):
        self.target = target
        self.observed_mask = observed_mask
        self.sample_id = sample_id
        self.time_id = time_id
        self.variate_id = variate_id
        self.prediction_mask = prediction_mask
        self.patch_size_val = patch_size_val

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "target": self.target[idx],
            "observed_mask": self.observed_mask[idx],
            "sample_id": self.sample_id[idx],
            "time_id": self.time_id[idx],
            "variate_id": self.variate_id[idx],
            "prediction_mask": self.prediction_mask[idx],
            "patch_size": torch.full_like(self.time_id[idx], self.patch_size_val),
        }


@ModelRegistry.register("moirai")
class MoiraiForecaster(BaseTimeSeriesFoundationModel):
    """Moirai forecaster implementation using the base TSFM framework.

    Moirai is a universal time series foundation model from Salesforce that uses
    a transformer-based architecture with patching and optional past-covariate
    support. This class wraps the uni2ts ``MoiraiForecast`` / ``MoiraiFinetune``
    APIs inside the project's unified ``BaseTimeSeriesFoundationModel`` interface.

    Three usage modes are supported:

    1. **Zero-shot inference** — pass a ``MoiraiConfig`` with just ``model_path``
       and optionally ``past_covariate_dim``.  The pretrained HuggingFace weights
       are loaded automatically.

    2. **In-class fine-tuning** — call ``_train_model()`` (via the base class
       training interface) to fine-tune using ``MoiraiFinetune``.  Training
       data is converted to the patched tensor format, a training loop runs
       with the optimizer and LR schedule from ``MoiraiFinetune``, and the
       best weights are saved via ``_save_checkpoint()`` / loaded via
       ``_load_checkpoint()``.

    3. **External checkpoint loading** — set ``config.checkpoint_path`` to a
       ``.ckpt`` file produced by the ``uni2ts`` CLI or any other external
       trainer.  The fine-tuned module weights are extracted and used for
       inference.

    Attributes:
        config: Moirai-specific configuration (``MoiraiConfig`` instance).
        predictor: Lazily created GluonTS predictor; ``None`` until the first
            ``predict()`` / ``predict_episodes()`` call.

    Example:
        >>> # Zero-shot, BG only
        >>> config = MoiraiConfig(model_path="Salesforce/moirai-1.0-R-base")
        >>> model = MoiraiForecaster(config)
        >>> preds = model.predict_episodes(val_episodes, target_col="bg_mM")

        >>> # Fine-tuned, with IOB/COB covariates
        >>> config = MoiraiConfig(
        ...     model_path="Salesforce/moirai-1.0-R-small",
        ...     checkpoint_path="models/moirai_finetuned/v3.ckpt",
        ...     past_covariate_dim=2,
        ...     covariate_cols=["iob", "cob"],
        ... )
        >>> model = MoiraiForecaster(config)
    """

    config_class = MoiraiConfig

    def __init__(self, config: MoiraiConfig, lora_config=None, distributed_config=None):
        """Initialize the Moirai forecaster.

        Args:
            config: Moirai configuration object. If a non-``MoiraiConfig`` is
                passed the essential parameters are extracted and a fresh
                ``MoiraiConfig`` is constructed.
            lora_config: LoRA configuration (reserved for future use).
            distributed_config: Distributed training configuration (reserved).
        """
        if not isinstance(config, MoiraiConfig):
            config = MoiraiConfig(
                model_path=getattr(
                    config, "model_path", "Salesforce/moirai-1.0-R-small"
                ),
                context_length=getattr(config, "context_length", 512),
                forecast_length=getattr(config, "forecast_length", 96),
                learning_rate=getattr(config, "learning_rate", 1e-4),
                batch_size=getattr(config, "batch_size", 32),
                num_epochs=getattr(config, "num_epochs", 10),
            )

        super().__init__(config, lora_config, distributed_config)
        self.config: MoiraiConfig = self.config

        # Lazily initialised GluonTS predictor
        self.predictor: Optional[Any] = None
        self._predictor_batch_size: Optional[int] = None

    @property
    def training_backend(self) -> TrainingBackend:
        """Moirai inference runs through GluonTS / uni2ts, not a HF Trainer."""
        return TrainingBackend.CUSTOM

    @property
    def supports_lora(self) -> bool:
        """Moirai is transformer-based and supports LoRA."""
        return True

    @property
    def supports_zero_shot(self) -> bool:
        """Moirai ships pretrained weights and forecasts out of the box."""
        return True

    def _initialize_model(self) -> MoiraiForecast:
        """Load the MoiraiForecast wrapper.

        Loads a fine-tuned ``.ckpt`` checkpoint when ``config.checkpoint_path``
        is set; otherwise downloads / loads pretrained weights from HuggingFace
        via ``MoiraiModule.from_pretrained()``.

        Returns:
            ``MoiraiForecast`` instance ready for ``create_predictor()``.

        Raises:
            ValueError: If ``config.model_path`` is empty.
            FileNotFoundError: If ``config.checkpoint_path`` is set but the
                file does not exist.
        """
        if not self.config.model_path:
            raise ValueError("MoiraiConfig.model_path must be set")

        checkpoint_path = self.config.checkpoint_path

        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Moirai checkpoint not found: {checkpoint_path}"
                )
            info_print(f"Loading fine-tuned Moirai checkpoint: {checkpoint_path}")
            module = MoiraiFinetune.load_from_checkpoint(
                checkpoint_path, map_location="cpu"
            ).module
        else:
            info_print(f"Loading pretrained Moirai: {self.config.model_path}")
            module = MoiraiModule.from_pretrained(self.config.model_path)

        info_print(f"  past_feat_dynamic_real_dim = {self.config.past_covariate_dim}")

        self.model = MoiraiForecast(
            module=module,
            prediction_length=self.config.forecast_length,
            context_length=self.config.context_length,
            patch_size=self.config.patch_size,
            num_samples=self.config.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=self.config.past_covariate_dim,
        )

        info_print("  Moirai loaded successfully")
        return self.model

    def _predict(
        self,
        data: Any,
        quantile_levels: Optional[List[float]] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Run inference and return mean forecasts.

        Accepts either a pre-built GluonTS ``ListDataset`` (for full control) or
        a list of episode dicts (for convenience — uses ``config.target_col`` and
        ``config.covariate_cols`` to build the dataset automatically).

        The ``predict()`` entry-point in the base class handles the single-episode
        DataFrame path; this method handles the batch/dataset path.

        Args:
            data: One of:

                * ``ListDataset`` — already formatted for GluonTS / Moirai.
                * ``list[dict]`` — episode dicts with ``context_df`` and
                  ``target_bg`` keys (midnight-anchored format).
                * ``pd.DataFrame`` — single-episode DataFrame with a
                  ``config.target_col`` column (called via the base ``predict()``).

            quantile_levels: Ignored for now (Moirai produces sample-based
                probabilistic forecasts; quantile extraction is a future TODO).
            batch_size: Overrides ``config.batch_size`` for the GluonTS predictor.
            **kwargs: Unused; accepted for forward-compatibility.

        Returns:
            Array of shape ``(N, forecast_length)`` with mean predictions, or
            shape ``(forecast_length,)`` when a single DataFrame is passed.
        """
        if self.model is None:
            self.model = self._initialize_model()

        bs = batch_size or self.config.batch_size

        # (Re)build the predictor if needed
        if self.predictor is None or bs != self._predictor_batch_size:
            self.predictor = self.model.create_predictor(batch_size=bs)  # type: ignore MoiraiForecast is not a torch.nn.Module. Harmless because at runtime self.model is actually a MoiraiForecast instance.
            self._predictor_batch_size = bs

        if isinstance(data, pd.DataFrame):
            # Single-episode DataFrame path (called from base class predict())
            dataset = self._dataframe_to_gluonts(data)
            single = True
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # Convenience episode-list path
            dataset = self._episodes_to_gluonts(data)
            single = False
        else:
            # Already a ListDataset (or compatible iterable)
            dataset = data
            single = False

        forecasts = list(self.predictor.predict(dataset))
        means = np.stack([f.mean for f in forecasts], axis=0)  # (N, horizon)

        return means[0] if single else means

    def _prepare_training_data(
        self, data: Any, split: Optional[str] = None
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Prepare training data for Moirai (episodes → DataLoader format, not used for CLI).

        Args:
            data: Dict of episodes {patient_id: [episode_list]} or DataFrame
            split: Optional, not used (data splitting handled by uni2ts)

        Returns:
            Tuple of (DataLoader, None, None). Only train_loader is populated;
            the CLI handles all data splitting internally.

        Note:
            This method is for compatibility with the base class interface but is
            NOT used for actual CLI training. The _train_model method calls
            _export_training_data separately to prepare CSV for the CLI.
        """
        # This is a compatibility stub. Actual CLI training uses _export_training_data.
        # We return a dummy DataLoader here.
        dataset = ListDataset(
            [{"target": np.array([0.0])} for _ in range(10)],
            freq=f"{self.config.interval_mins}min",
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size)
        return loader, None, None

    def _export_training_data(
        self,
        train_data: Any,
        output_dir: str,
        target_col: str = "bg_mM",
        context_len: int = 512,
        horizon: int = 72,
    ) -> str:
        """Export episodes to wide-format CSV for uni2ts CLI training.

        Args:
            train_data: Dict {patient_id: [episode_list]} where each episode contains:
                - "context_df": DataFrame with time-indexed context window
                - "target_bg": np.ndarray of ground truth BG values (horizon)
            output_dir: Directory to save training CSV
            target_col: Name of the BG column in context_df
            context_len: Number of context steps (should match config.context_length)
            horizon: Number of forecast steps (should match config.forecast_length)

        Returns:
            Path to the exported wide-format CSV file

        Raises:
            ValueError: If train_data format is invalid
            FileNotFoundError: If output_dir cannot be created
        """
        if not isinstance(train_data, dict):
            raise ValueError(
                f"train_data must be a dict of episodes {{patient_id: [episodes]}}, "
                f"got {type(train_data)}"
            )

        os.makedirs(output_dir, exist_ok=True)
        total_len = context_len + horizon
        episode_data = {}

        # Flatten all episodes and assign column names
        for patient_id, episodes in train_data.items():
            if not isinstance(episodes, list):
                raise ValueError(
                    f"train_data[{patient_id}] must be a list of episodes, "
                    f"got {type(episodes)}"
                )

            for ep_idx, ep in enumerate(episodes):
                col_name = f"{patient_id}_{ep_idx:03d}"

                # Extract context and target BG
                if (
                    not isinstance(ep, dict)
                    or "context_df" not in ep
                    or "target_bg" not in ep
                ):
                    raise ValueError(
                        f"Each episode must be a dict with 'context_df' and 'target_bg' keys. "
                        f"Got episode at {col_name}: {list(ep.keys()) if isinstance(ep, dict) else type(ep)}"
                    )

                context_df = ep["context_df"]
                target_bg = ep["target_bg"]

                # Extract context BG values
                if target_col not in context_df.columns:
                    raise ValueError(
                        f"Column '{target_col}' not found in context_df for {col_name}. "
                        f"Available columns: {list(context_df.columns)}"
                    )

                context_bg = context_df[target_col].values
                full_series = np.concatenate([context_bg, target_bg])

                if len(full_series) != total_len:
                    info_print(
                        f"Warning: {col_name} has {len(full_series)} steps "
                        f"(expected {total_len}), skipping"
                    )
                    continue

                episode_data[col_name] = full_series

        if not episode_data:
            raise ValueError("No valid episodes found in train_data")

        # Create DataFrame with synthetic aligned timestamps
        synthetic_index = pd.date_range(
            "2024-01-01 00:00:00",
            periods=total_len,
            freq=f"{self.config.interval_mins}min",
        )

        df = pd.DataFrame(episode_data, index=synthetic_index)
        df.index.name = "datetime"

        # Save to CSV
        csv_path = os.path.join(output_dir, "train_wide.csv")
        df.to_csv(csv_path)
        info_print(f"Exported {len(df.columns)} episodes to {csv_path}")

        return csv_path

    def _train_model(
        self, train_data: Any, output_dir: str, **kwargs
    ) -> Dict[str, Any]:
        """Fine-tune Moirai using its native ``MoiraiFinetune`` Lightning module.

        Converts training data to the patched tensor format via
        ``MoiraiForecast._convert()``, then runs a training loop with the
        optimizer and LR schedule defined by ``MoiraiFinetune.configure_optimizers()``.

        The best model weights (by training loss) are saved and loaded back
        into ``self.model`` so subsequent ``predict*`` calls use the
        fine-tuned weights.

        Args:
            train_data: One of:

                * ``pd.DataFrame`` — multi-patient DataFrame from the holdout
                  workflow (columns: ``p_num``, ``datetime``, ``bg_mM``, …).
                * ``dict`` — ``{patient_id: [episode_list]}`` with episode dicts
                  containing ``context_df`` and ``target_bg``.
                * ``list[dict]`` — flat list of episode dicts.

            output_dir: Directory for training outputs and checkpoints.
            **kwargs: Unused; accepted for forward-compatibility.

        Returns:
            Dict with ``status``, ``samples``, ``best_loss``, ``epochs``.
        """
        info_print("👉 Starting Moirai fine-tuning")
        info_print(f"   Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if self.model is None:
            self.model = self._initialize_model()

        device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.config.use_cpu else "cpu"
        )

        # ------------------------------------------------------------------
        # Step 1: Convert training data → (context, target) tensor pairs
        # ------------------------------------------------------------------
        info_print("Step 1: Preparing training tensors...")
        tensors = self._prepare_training_tensors(train_data)
        (
            past_target,
            future_target,
            past_observed,
            future_observed,
            past_is_pad,
            future_is_pad,
            past_covariates,
            past_observed_covariates,
        ) = tensors
        N = len(past_target)
        info_print(f"   Prepared {N} training samples")

        if N == 0:
            raise ValueError("No valid training samples could be extracted")

        # ------------------------------------------------------------------
        # Step 2: Convert to the patched format MoiraiFinetune expects
        # ------------------------------------------------------------------
        info_print("Step 2: Converting to patched training format...")
        patch_size = self._select_patch_size()
        self.config.patch_size = patch_size  # persist resolved value for rebuild & save
        info_print(f"   Using patch_size={patch_size}")

        all_tgt, all_obs, all_sid, all_tid, all_vid, all_pmask = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        chunk = min(self.config.batch_size, N)
        for i in range(0, N, chunk):
            sl = slice(i, min(i + chunk, N))
            cov = past_covariates[sl] if past_covariates is not None else None
            cov_obs = (
                past_observed_covariates[sl]
                if past_observed_covariates is not None
                else None
            )
            tgt, obs, sid, tid, vid, pmask = self.model._convert(
                patch_size,
                past_target=past_target[sl],
                past_observed_target=past_observed[sl],
                past_is_pad=past_is_pad[sl],
                future_target=future_target[sl],
                future_observed_target=future_observed[sl],
                future_is_pad=future_is_pad[sl],
                past_feat_dynamic_real=cov,
                past_observed_feat_dynamic_real=cov_obs,
            )  # type: ignore MoiraiForecast is not a torch.nn.Module. Harmless because at runtime self.model is actually a MoiraiForecast instance.
            all_tgt.append(tgt)
            all_obs.append(obs)
            all_sid.append(sid)
            all_tid.append(tid)
            all_vid.append(vid)
            all_pmask.append(pmask)

        dataset = _MoiraiPatchedDataset(
            target=torch.cat(all_tgt),
            observed_mask=torch.cat(all_obs),
            sample_id=torch.cat(all_sid),
            time_id=torch.cat(all_tid),
            variate_id=torch.cat(all_vid),
            prediction_mask=torch.cat(all_pmask),
            patch_size_val=patch_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # ------------------------------------------------------------------
        # Step 3: Build MoiraiFinetune module and optimizer
        # ------------------------------------------------------------------
        num_epochs = self.config.num_epochs
        steps_per_epoch = max(1, len(loader))
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = min(self.config.warmup_steps, max(1, total_steps // 5))

        finetune_module = MoiraiFinetune(
            module=self.model.module,
            min_patches=2,
            min_mask_ratio=0.15,
            max_mask_ratio=0.5,
            max_dim=128,
            num_training_steps=total_steps,
            num_warmup_steps=warmup_steps,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            finetune_pattern=self.config.finetune_pattern,
            context_length=self.config.context_length,
            prediction_length=self.config.forecast_length,
            patch_size=patch_size,
        )
        finetune_module.to(device)

        opt_config = finetune_module.configure_optimizers()
        optimizer = opt_config["optimizer"]
        scheduler = opt_config["lr_scheduler"]["scheduler"]

        # ------------------------------------------------------------------
        # Step 4: Training loop
        # ------------------------------------------------------------------
        info_print(
            f"Step 3: Training for {num_epochs} epoch(s) "
            f"({total_steps} steps, lr={self.config.learning_rate})..."
        )
        best_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None

        finetune_module.train()
        global_step = 0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad()

                distr = finetune_module(
                    **{
                        field: batch[field]
                        for field in list(finetune_module.seq_fields) + ["sample_id"]
                    }
                )
                loss = finetune_module.hparams.loss_func(
                    pred=distr,
                    target=batch["target"],
                    prediction_mask=batch["prediction_mask"],
                    observed_mask=batch["observed_mask"],
                    sample_id=batch["sample_id"],
                    variate_id=batch["variate_id"],
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    finetune_module.parameters(),
                    self.config.gradient_clip_val,
                )
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            info_print(f"   Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in finetune_module.module.state_dict().items()
                }

        # ------------------------------------------------------------------
        # Step 5: Reload best weights and rebuild MoiraiForecast
        # ------------------------------------------------------------------
        if best_state is not None:
            finetune_module.module.load_state_dict(best_state)

        self.model = MoiraiForecast(
            module=finetune_module.module,
            prediction_length=self.config.forecast_length,
            context_length=self.config.context_length,
            patch_size=self.config.patch_size,
            num_samples=self.config.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=self.config.past_covariate_dim,
        )
        self.predictor = None  # Force predictor rebuild
        self.is_fitted = True

        info_print(f"✅ Fine-tuning complete. Best epoch loss: {best_loss:.6f}")
        return {
            "status": "fitted",
            "samples": N,
            "best_loss": best_loss,
            "epochs": num_epochs,
        }

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save the fine-tuned MoiraiModule weights to ``output_dir``.

        If the model has been fine-tuned (``is_fitted`` is True and
        ``self.model.module`` exists), the full ``state_dict`` is persisted
        as ``moirai_finetuned.pt``.  The base class has already written
        ``config.json`` and ``metadata.json`` before this method is called.
        """
        if (
            not self.is_fitted
            or self.model is None
            or not hasattr(self.model, "module")
        ):
            info_print("No fine-tuned weights to save (zero-shot mode)")
            return

        weights_path = os.path.join(output_dir, "moirai_finetuned.pt")
        torch.save(self.model.module.state_dict(), weights_path)
        info_print(f"Saved fine-tuned Moirai weights: {weights_path}")

    def _load_checkpoint(self, model_dir: str) -> None:
        """Reload the model from ``model_dir``.

        Loading priority:

        1. ``moirai_finetuned.pt`` — fine-tuned weights saved by
           ``_save_checkpoint()``.
        2. ``config.checkpoint_path`` — external ``.ckpt`` Lightning
           checkpoint (e.g. from the ``uni2ts`` CLI).
        3. Pretrained HuggingFace weights (zero-shot fallback).

        Args:
            model_dir: Directory containing ``config.json`` (and optionally
                ``moirai_finetuned.pt``).
        """
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json found in {model_dir}")

        with open(config_path) as f:
            config_dict = json.load(f)

        self.config = MoiraiConfig(**config_dict)

        weights_path = os.path.join(model_dir, "moirai_finetuned.pt")
        if os.path.exists(weights_path):
            info_print(f"Loading fine-tuned Moirai weights: {weights_path}")
            # Start from pretrained architecture, override with saved weights
            module = MoiraiModule.from_pretrained(self.config.model_path)
            state_dict = torch.load(weights_path, map_location="cpu")
            module.load_state_dict(state_dict)

            self.model = MoiraiForecast(
                module=module,
                prediction_length=self.config.forecast_length,
                context_length=self.config.context_length,
                patch_size=self.config.patch_size,
                num_samples=self.config.num_samples,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=self.config.past_covariate_dim,
            )
        else:
            # Fall back to _initialize_model (handles .ckpt and pretrained)
            self.model = self._initialize_model()

        self.predictor = None
        self.is_fitted = True

    def build_gluonts_dataset(
        self,
        episodes: list,
        target_col: str,
        covariate_cols: Optional[List[str]] = None,
    ) -> ListDataset:
        """Build a GluonTS ``ListDataset`` from a list of episode dicts.

        This is the primary bridge between the project's midnight-anchored
        episode format and the GluonTS API that Moirai consumes.

        Args:
            episodes: List of dicts, each containing:

                * ``context_df`` — DataFrame indexed by timestamp.
                * ``target_bg`` — np.ndarray of ground-truth BG (horizon).

            target_col: Name of the BG column in ``context_df``.
            covariate_cols: Optional list of past-covariate column names
                (e.g. ``["iob", "cob"]``).  Length must equal
                ``config.past_covariate_dim`` when provided.

        Returns:
            GluonTS ``ListDataset`` ready for ``predictor.predict()``.

        Example:
            >>> ds = model.build_gluonts_dataset(
            ...     episodes=all_val_episodes,
            ...     target_col="bg_mM",
            ...     covariate_cols=["iob", "cob"],
            ... )
            >>> preds = model._predict(ds)
        """
        freq = f"{self.config.interval_mins}min"

        if covariate_cols and len(covariate_cols) != self.config.past_covariate_dim:
            raise ValueError(
                f"len(covariate_cols)={len(covariate_cols)} does not match "
                f"config.past_covariate_dim={self.config.past_covariate_dim}"
            )

        entries = []

        for ep in episodes:
            ctx = ep["context_df"]
            entry: Dict[str, Any] = {
                "start": ctx.index[0],
                "target": ctx[target_col].to_numpy(dtype=np.float32),
            }
            if covariate_cols:
                missing = [c for c in covariate_cols if c not in ctx.columns]
                if missing:
                    raise ValueError(
                        f"Covariate columns {missing} not found in context_df. "
                        f"Available: {list(ctx.columns)}"
                    )
                # shape: (n_covariates, context_length)
                entry["past_feat_dynamic_real"] = (
                    ctx[covariate_cols].to_numpy(dtype=np.float32).T
                )
            entries.append(entry)

        return ListDataset(entries, freq=freq)

    def predict_episodes(
        self,
        episodes: list,
        target_col: Optional[str] = None,
        covariate_cols: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Evaluate Moirai on a list of episodes and return per-episode metrics.

        Convenience wrapper that builds the GluonTS dataset, runs inference,
        and computes RMSE and MAE for each episode — mirroring the evaluation
        pattern from the notebook.

        Args:
            episodes: List of episode dicts (``context_df`` + ``target_bg``).
            target_col: BG column name; falls back to ``config.target_col``.
            covariate_cols: Past-covariate columns; falls back to
                ``config.covariate_cols`` (empty list = BG-only).
            batch_size: Overrides ``config.batch_size``.

        Returns:
            DataFrame with one row per episode and columns:
            ``rmse``, ``mae``, ``y_pred`` (np.ndarray), ``y_true`` (np.ndarray).

        Example:
            >>> results = model.predict_episodes(all_val_episodes)
            >>> print(f"RMSE: {results['rmse'].mean():.3f} +/- {results['rmse'].std():.3f}")
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        t_col = target_col or self.config.target_col
        cov_cols = (
            covariate_cols if covariate_cols is not None else self.config.covariate_cols
        )

        dataset = self.build_gluonts_dataset(episodes, t_col, cov_cols or None)
        mean_preds = self._predict(dataset, batch_size=batch_size)  # (N, horizon)

        records = []
        for ep, y_pred in zip(episodes, mean_preds):
            y_true = ep["target_bg"]
            records.append(
                {
                    "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                    "y_pred": y_pred,
                    "y_true": y_true,
                }
            )

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def evaluate_probabilistic(
        self,
        episodes: list,
        target_col: Optional[str] = None,
        covariate_cols: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        hypo_threshold: float = 3.9,
    ) -> pd.DataFrame:
        """Evaluate Moirai with full probabilistic outputs.

        Runs inference using all ``config.num_samples`` Monte Carlo samples and
        computes point metrics, prediction-interval calibration, and per-timestep
        hypoglycemia probability for each episode.

        Args:
            episodes: List of episode dicts (``context_df`` + ``target_bg``).
            target_col: BG column name; falls back to ``config.target_col``.
            covariate_cols: Past-covariate columns; falls back to
                ``config.covariate_cols``.
            batch_size: Overrides ``config.batch_size``.
            hypo_threshold: BG threshold (mmol/L) for hypoglycemia.
                Default 3.9 mmol/L (clinical standard).

        Returns:
            DataFrame with one row per episode and columns:

            * ``rmse``, ``mae`` — point forecast metrics
            * ``y_true``, ``y_pred`` — ground truth and mean forecast arrays
            * ``samples`` — raw sample array of shape ``(num_samples, horizon)``
            * ``q10``, ``q25``, ``q75``, ``q90`` — quantile arrays (horizon,)
            * ``p_hypo`` — per-timestep P(BG < threshold) array (horizon,)
            * ``max_p_hypo`` — scalar max P(hypo) across the forecast window
            * ``actual_hypo`` — bool, whether hypo actually occurred
            * ``calibration_90``, ``calibration_50`` — fraction of true values
              within the 90% / 50% prediction intervals

        Example:
            >>> prob = model.evaluate_probabilistic(all_val_episodes)
            >>> print(f"RMSE: {prob['rmse'].mean():.3f}")
            >>> print(f"Calibration 90%: {prob['calibration_90'].mean()*100:.1f}%")
            >>> print(f"Hypo episodes: {prob['actual_hypo'].sum()}/{len(prob)}")
            >>> print(f"ROC AUC input: prob['max_p_hypo'], prob['actual_hypo']")
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        if self.model is None:
            raise ValueError("Model must be initialized before making predictions")

        t_col = target_col or self.config.target_col
        cov_cols = (
            covariate_cols if covariate_cols is not None else self.config.covariate_cols
        )

        dataset = self.build_gluonts_dataset(episodes, t_col, cov_cols or None)

        bs = batch_size or self.config.batch_size
        if self.predictor is None:
            self.predictor = self.model.create_predictor(batch_size=bs)  # type: ignore MoiraiForecast is not a torch.nn.Module. Harmless because at runtime self.model is actually a MoiraiForecast instance.

        forecasts = list(self.predictor.predict(dataset))

        records = []
        for ep, fc in zip(episodes, forecasts):
            y_true = ep["target_bg"]

            # samples shape: (num_samples, horizon)
            samples = fc.samples
            y_pred_mean = fc.mean

            # Quantiles for prediction intervals
            q10 = np.percentile(samples, 10, axis=0)
            q25 = np.percentile(samples, 25, axis=0)
            q75 = np.percentile(samples, 75, axis=0)
            q90 = np.percentile(samples, 90, axis=0)

            # Per-timestep P(hypo)
            p_hypo = (samples < hypo_threshold).mean(axis=0)

            records.append(
                {
                    "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_mean))),
                    "mae": float(mean_absolute_error(y_true, y_pred_mean)),
                    "y_true": y_true,
                    "y_pred": y_pred_mean,
                    "samples": samples,
                    "q10": q10,
                    "q25": q25,
                    "q75": q75,
                    "q90": q90,
                    "p_hypo": p_hypo,
                    "max_p_hypo": float(p_hypo.max()),
                    "actual_hypo": bool((y_true < hypo_threshold).any()),
                    "calibration_90": float(((y_true >= q10) & (y_true <= q90)).mean()),
                    "calibration_50": float(((y_true >= q25) & (y_true <= q75)).mean()),
                }
            )

        df = pd.DataFrame(records)

        info_print(f"RMSE: {df['rmse'].mean():.3f} +/- {df['rmse'].std():.3f}")
        info_print(
            f"Calibration 90% PI: {df['calibration_90'].mean()*100:.1f}% (target: 90%)"
        )
        info_print(
            f"Calibration 50% PI: {df['calibration_50'].mean()*100:.1f}% (target: 50%)"
        )
        info_print(f"Episodes with actual hypo: {df['actual_hypo'].sum()}/{len(df)}")

        return df

    def _dataframe_to_gluonts(self, df: pd.DataFrame) -> ListDataset:
        """Convert a single-episode DataFrame to a one-entry GluonTS dataset.

        Called by ``_predict()`` when the base class ``predict()`` method
        passes a DataFrame.  The DataFrame may have timestamps as the index
        or as a ``datetime`` column (the generic workflow resets the index).

        Args:
            df: Single-episode context DataFrame.

        Returns:
            One-entry ``ListDataset``.
        """
        # Ensure we have a datetime index; the generic workflow may pass
        # datetime as a regular column after reset_index().
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df = df.set_index("datetime")
            else:
                raise ValueError(
                    "DataFrame must have a DatetimeIndex or a 'datetime' column"
                )

        freq = f"{self.config.interval_mins}min"
        entry: Dict[str, Any] = {
            "start": df.index[0],
            "target": df[self.config.target_col].to_numpy(dtype=np.float32),
        }
        if self.config.covariate_cols:
            missing = [c for c in self.config.covariate_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Covariate columns {missing} not found in DataFrame. "
                    f"Available: {list(df.columns)}"
                )
            entry["past_feat_dynamic_real"] = (
                df[self.config.covariate_cols].to_numpy(dtype=np.float32).T
            )
        return ListDataset([entry], freq=freq)

    def _episodes_to_gluonts(self, episodes: list) -> ListDataset:
        """Convenience wrapper that uses config defaults."""
        return self.build_gluonts_dataset(
            episodes,
            self.config.target_col,
            self.config.covariate_cols or None,
        )

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _select_patch_size(self) -> int:
        """Choose a fixed patch size for training.

        If ``config.patch_size`` is an explicit integer it is returned as-is.
        Otherwise the largest available patch size that yields at least 4
        context patches is selected from the ``MoiraiModule.patch_sizes``.
        """
        if isinstance(self.config.patch_size, int):
            return self.config.patch_size
        ctx = self.config.context_length
        available = sorted(self.model.module.patch_sizes, reverse=True)
        for ps in available:
            if ctx // ps >= 4:
                return ps
        return available[-1]

    def _prepare_training_tensors(
        self, train_data: Any
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Convert training data to aligned tensor batches on CPU.

        Returns:
            Tuple of ``(past_target, future_target, past_observed,
            future_observed, past_is_pad, future_is_pad, past_covariates,
            past_observed_covariates)``.
            Each target/observed tensor has shape ``(N, length, dim)``;
            ``past_covariates`` and ``past_observed_covariates`` are ``None``
            when no covariates are configured.
        """
        ctx_len = self.config.context_length
        fh_len = self.config.forecast_length
        total_len = ctx_len + fh_len
        target_col = self.config.target_col
        cov_cols = self.config.covariate_cols or []

        contexts: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        covariates: List[np.ndarray] = []

        if isinstance(train_data, pd.DataFrame):
            available_covs = [c for c in cov_cols if c in train_data.columns]
            for _, pat_df in train_data.groupby("p_num"):
                if "datetime" in pat_df.columns:
                    pat_df = pat_df.set_index("datetime").sort_index()
                elif not isinstance(pat_df.index, pd.DatetimeIndex):
                    continue
                pat_df = pat_df.dropna(subset=[target_col])
                bg = pat_df[target_col].values
                cov = pat_df[available_covs].values if available_covs else None
                for s in range(0, len(bg) - total_len + 1, total_len):
                    contexts.append(bg[s : s + ctx_len])
                    targets.append(bg[s + ctx_len : s + total_len])
                    if cov is not None:
                        covariates.append(cov[s : s + ctx_len])

        elif isinstance(train_data, dict):
            for _pid, episodes in train_data.items():
                if not isinstance(episodes, list):
                    episodes = [episodes]
                for ep in episodes:
                    ctx_df = ep["context_df"]
                    tgt_bg = ep["target_bg"]
                    ctx_bg = ctx_df[target_col].values[-ctx_len:]
                    tgt_bg = tgt_bg[:fh_len]
                    if len(ctx_bg) != ctx_len or len(tgt_bg) != fh_len:
                        continue
                    contexts.append(ctx_bg.astype(np.float32))
                    targets.append(tgt_bg.astype(np.float32))
                    if cov_cols:
                        avail = [c for c in cov_cols if c in ctx_df.columns]
                        if avail:
                            covariates.append(
                                ctx_df[avail].values[-ctx_len:].astype(np.float32)
                            )

        elif isinstance(train_data, list) and train_data:
            for ep in train_data:
                ctx_df = ep["context_df"]
                tgt_bg = ep["target_bg"]
                ctx_bg = ctx_df[target_col].values[-ctx_len:]
                tgt_bg = tgt_bg[:fh_len]
                if len(ctx_bg) != ctx_len or len(tgt_bg) != fh_len:
                    continue
                contexts.append(ctx_bg.astype(np.float32))
                targets.append(tgt_bg.astype(np.float32))
                if cov_cols:
                    avail = [c for c in cov_cols if c in ctx_df.columns]
                    if avail:
                        covariates.append(
                            ctx_df[avail].values[-ctx_len:].astype(np.float32)
                        )

        if not contexts:
            return (
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                None,
                None,
            )

        # Shape: (N, length, 1) for univariate BG
        past_target = torch.tensor(np.stack(contexts), dtype=torch.float32).unsqueeze(
            -1
        )
        future_target = torch.tensor(np.stack(targets), dtype=torch.float32).unsqueeze(
            -1
        )

        past_observed = ~torch.isnan(past_target)
        future_observed = ~torch.isnan(future_target)

        # Impute NaN with 0 (matches uni2ts DummyValueImputation)
        past_target = torch.nan_to_num(past_target, nan=0.0)
        future_target = torch.nan_to_num(future_target, nan=0.0)

        past_is_pad = torch.zeros(len(contexts), ctx_len, dtype=torch.long)
        future_is_pad = torch.zeros(len(targets), fh_len, dtype=torch.long)

        past_covariates: Optional[torch.Tensor] = None
        past_observed_covariates: Optional[torch.Tensor] = None
        if covariates and len(covariates) == len(contexts):
            past_covariates = torch.tensor(np.stack(covariates), dtype=torch.float32)
            past_observed_covariates = ~torch.isnan(past_covariates)
            past_covariates = torch.nan_to_num(past_covariates, nan=0.0)

        return (
            past_target,
            future_target,
            past_observed,
            future_observed,
            past_is_pad,
            future_is_pad,
            past_covariates,
            past_observed_covariates,
        )

    def _dataframe_to_training_dataset(self, df: pd.DataFrame) -> ListDataset:
        """Window a raw multi-patient DataFrame into GluonTS training entries.

        The generic holdout workflow passes a single large DataFrame with all
        patients concatenated.  This method groups by patient, sets a datetime
        index, and slices non-overlapping windows of ``context_length`` steps,
        producing one GluonTS entry per window.

        Args:
            df: Combined training DataFrame with ``p_num``, ``datetime``, and
                at least ``config.target_col``.

        Returns:
            GluonTS ``ListDataset`` of windowed entries.
        """
        freq = f"{self.config.interval_mins}min"
        target_col = self.config.target_col
        cov_cols = self.config.covariate_cols or []
        available_covs = [c for c in cov_cols if c in df.columns]
        window = self.config.context_length
        patient_col = "p_num"

        entries: List[Dict[str, Any]] = []

        for _, pat_df in df.groupby(patient_col):
            # Ensure datetime index
            if "datetime" in pat_df.columns:
                pat_df = pat_df.set_index("datetime").sort_index()
            elif not isinstance(pat_df.index, pd.DatetimeIndex):
                continue

            # Drop rows where target is NaN (can't train on gaps)
            pat_df = pat_df.dropna(subset=[target_col])
            if len(pat_df) < window:
                continue

            # Non-overlapping windows
            for start_idx in range(0, len(pat_df) - window + 1, window):
                chunk = pat_df.iloc[start_idx : start_idx + window]
                entry: Dict[str, Any] = {
                    "start": chunk.index[0],
                    "target": chunk[target_col].to_numpy(dtype=np.float32),
                }
                if available_covs:
                    entry["past_feat_dynamic_real"] = (
                        chunk[available_covs].to_numpy(dtype=np.float32).T
                    )
                entries.append(entry)

        info_print(f"   Windowed {len(entries)} training segments from DataFrame")
        return ListDataset(entries, freq=freq)


def create_moirai_model(
    model_path: str = "Salesforce/moirai-1.0-R-base",
    context_length: int = 512,
    forecast_length: int = 72,
    past_covariate_dim: int = 0,
    covariate_cols: Optional[List[str]] = None,
    checkpoint_path: Optional[str] = None,
    num_samples: int = 100,
    patch_size: str = "auto",
    interval_mins: int = 5,
    target_col: str = "bg_mM",
    **kwargs,
) -> MoiraiForecaster:
    """Factory function to create a ``MoiraiForecaster`` with sensible defaults.

    Args:
        model_path: HuggingFace model ID. Common options:

            * ``"Salesforce/moirai-1.0-R-small"`` — 14 M params, fastest
            * ``"Salesforce/moirai-1.0-R-base"`` — 91 M params, recommended
            * ``"Salesforce/moirai-1.0-R-large"`` — 311 M params
            * ``"Salesforce/moirai-1.1-R-base"`` — improved version
            * ``"Salesforce/moirai-moe-1.0-R-base"`` — MoE variant (avoid
              for zero-shot; it had RMSE ~365 in the notebook)

        context_length: Historical steps (~42 hrs at 5-min intervals = 512).
        forecast_length: Horizon steps (6 hrs at 5-min intervals = 72).
        past_covariate_dim: Number of past covariates (0 = BG-only, 2 = IOB+COB).
        covariate_cols: Column names matching ``past_covariate_dim``.
        checkpoint_path: Path to a ``.ckpt`` fine-tuned checkpoint, or ``None``
            for zero-shot inference.
        num_samples: Monte Carlo samples for probabilistic output (default 100).
        patch_size: Patch size for Moirai; ``"auto"`` is recommended.
        interval_mins: CGM sampling interval in minutes.
        target_col: Name of the target BG column.
        **kwargs: Extra parameters forwarded to ``MoiraiConfig``.

    Returns:
        Initialised ``MoiraiForecaster``.

    Example:
        >>> # Zero-shot, BG only
        >>> model = create_moirai_model()

        >>> # Zero-shot with IOB/COB covariates
        >>> model = create_moirai_model(
        ...     past_covariate_dim=2,
        ...     covariate_cols=["iob", "cob"],
        ... )

        >>> # Fine-tuned small model
        >>> model = create_moirai_model(
        ...     model_path="Salesforce/moirai-1.0-R-small",
        ...     checkpoint_path="models/moirai_finetuned/v3.ckpt",
        ... )
    """
    config = MoiraiConfig(
        model_path=model_path,
        context_length=context_length,
        forecast_length=forecast_length,
        past_covariate_dim=past_covariate_dim,
        covariate_cols=covariate_cols or [],
        checkpoint_path=checkpoint_path,
        num_samples=num_samples,
        patch_size=patch_size,
        interval_mins=interval_mins,
        target_col=target_col,
        **kwargs,
    )
    return MoiraiForecaster(config)
