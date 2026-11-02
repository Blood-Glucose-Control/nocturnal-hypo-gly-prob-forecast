# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Optuna objective for the TimesFM 3.0 (LoRA fine-tune) sweep.

Each trial is a **full LoRA fine-tune** of Google's TimesFM 3.0 (expensive:
minutes on GPU) on IOBP2 training windows, then a score on an internal temporal
validation split. Unlike the committed ``src/models/timesfm3`` adapter (which is
past-only), this sweep feeds selected insulin channels as KNOWN-FUTURE
covariates spanning the context **and** the forecast horizon, using TimesFM 3.0's
grad-enabled ``decode`` path (``decode.__wrapped__``) with a pinball loss.

Pipeline per trial:

1. Build IOBP2 train + temporal-val windows from the shared
   :class:`~..optuna_chronos2.data.MetabonetData` (engineered per-patient frames
   via ``engineer_patient_frame``): context CGM ``(ctx,)`` + known-future
   covariates ``(C, ctx + H)`` + horizon truth ``(H,)``.
2. Load TimesFM 3.0, inject LoRA, freeze non-adapter params.
3. Fine-tune the adapter with a pinball loss (covariate jitter + non-finite
   loss/grad guards + grad clipping) and validation-based early stopping on a
   held-out slice of the training windows.
4. Predict the temporal-val windows and score with
   :func:`~..optuna_chronos2.metrics.evaluate_horizons` -> ``mard_avg``.
5. Free GPU memory before the next trial.

Fit failures degrade gracefully to a large finite ``FAILURE_PENALTY`` so TPE
steers away from bad regions instead of aborting the worker.
"""

from __future__ import annotations

import gc
import os

# Best-effort: disable torch's native (Triton-compiled) op overrides so the
# TimesFM 3.0 covariate ``bmm`` path falls back to standard aten kernels on hosts
# without CUDA/Python build headers. Must be set before torch is imported to take
# effect; harmless when torch is already initialized.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402

from ..optuna_chronos2.data import MetabonetData  # noqa: E402
from ..optuna_chronos2.features import TARGET_COL  # noqa: E402
from ..optuna_chronos2.metrics import HORIZON_STEPS, evaluate_horizons  # noqa: E402
from .search_space import (  # noqa: E402
    TimesFM3SearchPoint,
    TimesFM3SearchSpaceBounds,
    suggest_timesfm3_point,
)

PRIMARY_METRIC = "mard_avg"

# Objective value assigned to a trial whose fit fails. A large finite penalty
# makes TPE steer away from bad regions instead of ignoring hard failures.
FAILURE_PENALTY = 100.0

# Full clinical horizon length (steps): 24 -> +120 min at 5-minute sampling.
_HORIZON_LEN = max(HORIZON_STEPS.values())

_MODEL_ID = "google/timesfm-3.0-pytorch"


def _import_forecaster():
    """Import the native TimesFM 3.0 forecaster class (``.venvs/timesfm3`` only)."""
    from timesfm3.torch.timesfm3_forecaster import (  # pyright: ignore[reportMissingImports]
        TimesFM3Forecaster,
    )

    return TimesFM3Forecaster


def _import_peft():
    """Import the ``peft`` helpers for LoRA injection + adapter state I/O."""
    from peft import (  # pyright: ignore[reportMissingImports]
        LoraConfig,
        get_peft_model_state_dict,
        inject_adapter_in_model,
        set_peft_model_state_dict,
    )

    return (
        LoraConfig,
        inject_adapter_in_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )


class TimesFM3Objective:
    """Callable Optuna objective that LoRA-fine-tunes + scores TimesFM 3.0."""

    def __init__(
        self,
        data: MetabonetData,
        output_root: str | Path,
        *,
        bounds: Optional[TimesFM3SearchSpaceBounds] = None,
        time_limit: Optional[int] = None,
        keep_models: bool = False,
        gpu_label: str = "",
        primary_metric: str = PRIMARY_METRIC,
        objective_metrics: Optional[Sequence[str]] = None,
        report_intermediate: bool = False,
        model_path: str = _MODEL_ID,
        device: str = "cuda",
        train_windows_per_patient: int = 12,
        max_train_windows: int = 3000,
        val_windows_per_patient: int = 4,
        max_val_windows_total: int = 2000,
        val_stride_steps: int = 6,
        es_val_fraction: float = 0.2,
        es_patience: int = 2,
        covariate_jitter: float = 1e-3,
        grad_clip_norm: float = 1.0,
        seed: int = 42,
        adapters_dir: Optional[str | Path] = None,
        keep_top_k: int = 0,
    ) -> None:
        self.data = data
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.bounds = bounds or TimesFM3SearchSpaceBounds()
        self.time_limit = time_limit
        self.keep_models = keep_models
        self.gpu_label = gpu_label
        self.primary_metric = primary_metric
        self.objective_metrics = list(objective_metrics) if objective_metrics else None
        self.is_multi = bool(self.objective_metrics and len(self.objective_metrics) > 1)
        self.report_intermediate = bool(report_intermediate) and not self.is_multi
        self.forecast_length = int(self.bounds.forecast_length)

        # Fine-tune / windowing knobs.
        self.model_path = model_path
        self.device = device
        self.train_windows_per_patient = int(train_windows_per_patient)
        self.max_train_windows = int(max_train_windows)
        self.val_windows_per_patient = int(val_windows_per_patient)
        self.max_val_windows_total = int(max_val_windows_total)
        self.val_stride_steps = max(1, int(val_stride_steps))
        self.es_val_fraction = float(es_val_fraction)
        self.es_patience = int(es_patience)
        self.covariate_jitter = float(covariate_jitter)
        self.grad_clip_norm = float(grad_clip_norm)
        self.seed = int(seed)
        # Top-K adapter cache: save each trial's best LoRA adapter when it is among
        # the study's K best (by primary metric), pruning ones that drop out, so
        # the best configs can be leak-scored later via load+inference (seconds)
        # instead of re-training. keep_top_k=0 disables caching.
        self.adapters_dir = Path(adapters_dir) if adapters_dir else None
        self.keep_top_k = int(keep_top_k)
        if self.adapters_dir and self.keep_top_k > 0:
            self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self._last_adapter: Optional[Dict[str, Any]] = None

        # Covariate-agnostic per-patient cache (interpolated target + split idx),
        # built once and reused across trials.
        self._target_cache: Optional[List[Tuple[Any, np.ndarray, int]]] = None

    # ------------------------------------------------------------------
    # Window building (known-future covariates over context + horizon)
    # ------------------------------------------------------------------
    def _ensure_target_cache(self) -> List[Tuple[Any, np.ndarray, int]]:
        """Cache each patient's interpolated CGM target and train/val split index.

        Mirrors ``prep_train_windows.py``: the CGM target is masked to positive
        values then linearly interpolated over interior gaps so full windows are
        finite; the split index is the count of steps in the training region
        (timestamps <= ``split_ts``).
        """
        if self._target_cache is not None:
            return self._target_cache
        cache: List[Tuple[Any, np.ndarray, int]] = []
        for patient in self.data.patients:
            frame = patient.frame
            target = frame[TARGET_COL].where(frame[TARGET_COL] > 0)
            target = target.interpolate(limit_area="inside")
            tvals = target.to_numpy(dtype=np.float32)
            split_idx = int(frame.index.searchsorted(patient.split_ts, side="right"))
            cache.append((patient, tvals, split_idx))
        self._target_cache = cache
        return cache

    def _covariate_matrix(self, patient: Any, known_cols: Sequence[str]) -> np.ndarray:
        """Stack selected known-future covariate channels as ``(C, T)``.

        Covariate future values are read directly from the engineered frame
        (insulin is known ahead of time in the competition); interior gaps are
        forward-filled then zero-filled so channels are always finite.
        """
        frame = patient.frame
        return np.vstack(
            [
                frame[col].ffill().fillna(0.0).to_numpy(dtype=np.float32)
                for col in known_cols
            ]
        )

    def _build_trial_windows(
        self, known_cols: Sequence[str], context_length: int
    ) -> Dict[str, Any]:
        """Build IOBP2 train + temporal-val windows for one covariate selection.

        Train windows lie entirely within each patient's training region; val
        windows anchor their horizon in the held-out region (context may span the
        boundary, matching real serving). Returns numpy train tensors plus the
        per-window val context/covariate lists and the val truth matrix.
        """
        horizon = self.forecast_length
        need = context_length + horizon
        use_cov = len(known_cols) > 0
        rng = np.random.default_rng(self.seed)

        train_cgm: List[np.ndarray] = []
        train_cov: List[np.ndarray] = []
        train_true: List[np.ndarray] = []
        val_context: List[np.ndarray] = []
        val_cov: List[np.ndarray] = []
        val_true: List[np.ndarray] = []

        for patient, tvals, split_idx in self._ensure_target_cache():
            total = len(tvals)
            if total < need:
                continue
            covmat = self._covariate_matrix(patient, known_cols) if use_cov else None

            # --- Train windows: full window within the train region [0, split_idx).
            if len(train_cgm) < self.max_train_windows:
                max_start = split_idx - need
                if max_start >= 0:
                    starts = rng.integers(
                        0, max_start + 1, size=self.train_windows_per_patient * 3
                    )
                    taken = 0
                    for start in starts:
                        if (
                            taken >= self.train_windows_per_patient
                            or len(train_cgm) >= self.max_train_windows
                        ):
                            break
                        window = tvals[start : start + need]
                        if np.isnan(window).any():
                            continue
                        train_cgm.append(window[:context_length])
                        train_true.append(window[context_length:])
                        if covmat is not None:
                            train_cov.append(covmat[:, start : start + need])
                        taken += 1

            # --- Val windows: horizon anchored in the held-out region.
            if len(val_context) < self.max_val_windows_total:
                anchor_start = max(split_idx, context_length)
                anchor_end = total - horizon
                taken_v = 0
                for anchor in range(
                    anchor_start, anchor_end + 1, self.val_stride_steps
                ):
                    if (
                        taken_v >= self.val_windows_per_patient
                        or len(val_context) >= self.max_val_windows_total
                    ):
                        break
                    context = tvals[anchor - context_length : anchor]
                    if np.isnan(context).any():
                        continue
                    future = tvals[anchor : anchor + horizon]
                    if not np.isfinite(future).any():
                        continue
                    val_context.append(context)
                    val_true.append(future)
                    if covmat is not None:
                        val_cov.append(
                            covmat[:, anchor - context_length : anchor + horizon]
                        )
                    taken_v += 1

        if not train_cgm:
            raise ValueError(
                f"No IOBP2 train windows built for context_length={context_length}."
            )
        if not val_context:
            raise ValueError(
                f"No IOBP2 val windows built for context_length={context_length}."
            )

        return {
            "train_cgm": np.vstack(train_cgm),
            "train_cov": np.stack(train_cov) if use_cov else None,
            "train_true": np.vstack(train_true),
            "val_context": val_context,
            "val_cov": val_cov if use_cov else None,
            "val_true": np.vstack(val_true),
        }

    # ------------------------------------------------------------------
    # Fine-tune + score
    # ------------------------------------------------------------------
    @staticmethod
    def _pinball(logits_q: Any, target: Any, quantiles: Sequence[float]) -> Any:
        """Mean pinball (quantile) loss over the model's quantile grid.

        Args:
            logits_q: Predicted quantiles of shape ``(b, H, Q)``.
            target: Ground-truth horizon of shape ``(b, H)``.
            quantiles: The model's quantile levels.
        """
        import torch

        q = torch.tensor(quantiles, device=logits_q.device, dtype=logits_q.dtype).view(
            1, 1, -1
        )
        errors = target.unsqueeze(-1) - logits_q
        return torch.maximum(q * errors, (q - 1) * errors).mean()

    def _run_epoch(
        self,
        model: Any,
        decode_fn: Any,
        quantiles: Sequence[float],
        cgm: Any,
        cov: Any,
        true: Any,
        device: Any,
        optimizer: Any,
        trainable: List[Any],
        batch_size: int,
        train: bool,
        rng: np.random.Generator,
        scheduler: Any = None,
    ) -> float:
        """Run one train or validation epoch; return the mean pinball loss.

        Ports the proven loop: covariate jitter (train only), non-finite loss +
        grad guards, and max-norm gradient clipping. ``logits`` has shape
        ``(b, num_variates, H, Q)`` and the target variate is index 0.
        """
        import torch

        horizon = self.forecast_length
        n_windows = cgm.shape[0]
        batch_size = max(1, int(batch_size))
        order = rng.permutation(n_windows) if train else np.arange(n_windows)

        total_loss, num_batches = 0.0, 0
        for start in range(0, n_windows, batch_size):
            idx = order[start : start + batch_size]
            target = cgm[idx].unsqueeze(1).to(device)  # (b, 1, ctx)
            y_true = true[idx].to(device)  # (b, H)
            pfc: Any = None
            if cov is not None:
                pfc = cov[idx].to(device)  # (b, C, ctx + H)
                if train and self.covariate_jitter > 0.0:
                    pfc = pfc + self.covariate_jitter * torch.randn_like(pfc)

            if train:
                logits = decode_fn(
                    model,
                    target=target,
                    past_future_covariates=pfc,
                    horizon=horizon,
                )
            else:
                with torch.no_grad():
                    logits = decode_fn(
                        model,
                        target=target,
                        past_future_covariates=pfc,
                        horizon=horizon,
                    )
            loss = self._pinball(logits[:, 0, :horizon, :], y_true, quantiles)

            if train:
                optimizer.zero_grad()
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                grads_finite = all(
                    p.grad is None or torch.isfinite(p.grad).all() for p in trainable
                )
                if not grads_finite:
                    optimizer.zero_grad()
                    continue
                if self.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / num_batches if num_batches else float("nan")

    def _fit_and_score(
        self,
        point: TimesFM3SearchPoint,
        known_cols: Sequence[str],
        windows: Dict[str, Any],
        trial: optuna.Trial,
    ) -> Dict[str, float]:
        """Load TimesFM 3.0, LoRA-fine-tune, and score the temporal-val windows."""
        import torch

        forecaster_cls = _import_forecaster()
        lora_config_cls, inject_adapter, get_peft_sd, set_peft_sd = _import_peft()
        hp = point.hyperparameters
        use_cov = len(known_cols) > 0

        fc: Any = None
        model: Any = None
        try:
            fc = forecaster_cls.from_pretrained(self.model_path, device=self.device)
            model = fc.model
            dev = next(model.parameters()).device
            quantiles = list(model.quantiles)

            tm = str(hp.get("lora_target_modules", "all-linear"))
            if tm == "attention":
                target_modules: Any = [
                    "query_proj",
                    "key_proj",
                    "value_proj",
                    "out_proj",
                ]
            elif tm == "attn-qv":
                target_modules = ["query_proj", "value_proj"]
            else:
                target_modules = "all-linear"
            inject_adapter(
                lora_config_cls(
                    r=int(hp["lora_r"]),
                    lora_alpha=int(hp["lora_alpha"]),
                    lora_dropout=float(hp["lora_dropout"]),
                    target_modules=target_modules,
                    use_dora=bool(hp.get("use_dora", False)),
                    use_rslora=bool(hp.get("use_rslora", False)),
                    bias="none",
                ),
                model,
            )
            for name, param in model.named_parameters():
                param.requires_grad_("lora_" in name)

            decode_fn = getattr(type(model).decode, "__wrapped__", type(model).decode)

            # Split training windows into a fit slice + an early-stopping slice.
            n_train = windows["train_cgm"].shape[0]
            split_rng = np.random.default_rng(self.seed)
            if self.es_val_fraction > 0.0 and n_train >= 4:
                order = split_rng.permutation(n_train)
                n_es = max(
                    1, min(int(round(n_train * self.es_val_fraction)), n_train - 1)
                )
                es_idx = order[:n_es]
                fit_idx = order[n_es:]
            else:
                fit_idx = np.arange(n_train)
                es_idx = None

            def _tensor(arr: np.ndarray) -> Any:
                return torch.tensor(arr, dtype=torch.float32)

            fit_cgm = _tensor(windows["train_cgm"][fit_idx])
            fit_true = _tensor(windows["train_true"][fit_idx])
            fit_cov = _tensor(windows["train_cov"][fit_idx]) if use_cov else None
            es_cgm = es_true = es_cov = None
            if es_idx is not None:
                es_cgm = _tensor(windows["train_cgm"][es_idx])
                es_true = _tensor(windows["train_true"][es_idx])
                es_cov = _tensor(windows["train_cov"][es_idx]) if use_cov else None

            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable, lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"])
            )
            epoch_rng = np.random.default_rng(self.seed)
            batch_size = int(hp["batch_size"])
            max_epochs = int(hp["max_epochs"])

            # Optional LR schedule: linear warmup -> cosine decay (per-batch step).
            import math

            lr_schedule = str(hp.get("lr_schedule", "constant"))
            scheduler = None
            if lr_schedule == "cosine":
                steps_per_epoch = max(
                    1, math.ceil(fit_cgm.shape[0] / max(1, batch_size))
                )
                total_steps = max(1, steps_per_epoch * max_epochs)
                warmup_steps = int(float(hp.get("warmup_ratio", 0.0)) * total_steps)

                def _lr_lambda(
                    step: int, _w: int = warmup_steps, _t: int = total_steps
                ):
                    if _w > 0 and step < _w:
                        return (step + 1) / _w
                    progress = (step - _w) / max(1, _t - _w)
                    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

            best_monitor = float("inf")
            best_state: Optional[Dict[str, Any]] = None
            epochs_without_improve = 0
            for epoch in range(max_epochs):
                model.train()
                train_loss = self._run_epoch(
                    model,
                    decode_fn,
                    quantiles,
                    fit_cgm,
                    fit_cov,
                    fit_true,
                    dev,
                    optimizer,
                    trainable,
                    batch_size,
                    train=True,
                    rng=epoch_rng,
                    scheduler=scheduler,
                )
                val_loss: Optional[float] = None
                if es_cgm is not None:
                    model.eval()
                    val_loss = self._run_epoch(
                        model,
                        decode_fn,
                        quantiles,
                        es_cgm,
                        es_cov,
                        es_true,
                        dev,
                        optimizer,
                        trainable,
                        batch_size,
                        train=False,
                        rng=epoch_rng,
                    )
                monitor = val_loss if val_loss is not None else train_loss

                if self.report_intermediate:
                    trial.report(float(monitor), step=epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned(
                            f"Pruned at epoch {epoch} (monitor={monitor:.4f})."
                        )

                if np.isfinite(monitor) and monitor < best_monitor - 1e-6:
                    best_monitor = monitor
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in get_peft_sd(model).items()
                    }
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                    if self.es_patience and epochs_without_improve >= self.es_patience:
                        break

            if best_state is not None:
                set_peft_sd(model, best_state)
            # Stash the best adapter so __call__ can cache it if the trial ranks
            # among the study's top-K (adapters are tiny: LoRA state only).
            self._last_adapter = best_state
            model.eval()

            # --- Score the temporal-val windows with the fine-tuned forecaster.
            contexts = [np.asarray(c, dtype=np.float32) for c in windows["val_context"]]
            pfc_list = None
            if use_cov:
                pfc_list = [np.asarray(c, dtype=np.float32) for c in windows["val_cov"]]
            outputs = list(
                fc.predict_batch(
                    contexts,
                    horizon=self.forecast_length,
                    past_future_covariates=pfc_list,
                    return_quantiles=True,
                    make_positive=True,
                    use_znorm=bool(hp.get("use_znorm", False)),
                )
            )
            rows: List[np.ndarray] = []
            for output in outputs:
                forecast = np.asarray(output.forecast, dtype=np.float64).ravel()
                forecast = forecast[: self.forecast_length]
                if forecast.size < self.forecast_length:
                    forecast = np.pad(
                        forecast,
                        (0, self.forecast_length - forecast.size),
                        constant_values=np.nan,
                    )
                rows.append(forecast)
            pred_matrix = np.vstack(rows)

            metrics = evaluate_horizons(pred_matrix, windows["val_true"])
            metrics["best_val_pinball"] = (
                float(best_monitor) if np.isfinite(best_monitor) else float("nan")
            )
            return metrics
        finally:
            del model
            del fc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Optuna entrypoint
    # ------------------------------------------------------------------
    def __call__(self, trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
        point = suggest_timesfm3_point(trial, self.bounds)
        known_cols = point.known_cols
        trial.set_user_attr("known_cols", list(known_cols))
        trial.set_user_attr("eval_metric", point.eval_metric)
        trial.set_user_attr("context_length", point.context_length)
        trial.set_user_attr("gpu", self.gpu_label)

        fit_failed = False
        fail_reason = ""
        fit_seconds = 0.0
        metrics: Dict[str, float] = {}
        n_train = 0
        n_val = 0
        try:
            windows = self._build_trial_windows(known_cols, point.context_length)
            n_train = int(windows["train_cgm"].shape[0])
            n_val = int(windows["val_true"].shape[0])
            fit_start = time.time()
            metrics = self._fit_and_score(point, known_cols, windows, trial)
            fit_seconds = time.time() - fit_start
        except optuna.TrialPruned:
            raise
        except Exception as exc:  # noqa: BLE001 - degrade gracefully; penalize trial
            fit_failed = True
            fail_reason = f"{type(exc).__name__}: {exc}"[:300]

        if fit_failed:
            trial.set_user_attr("fit_failed", True)
            trial.set_user_attr("fail_reason", fail_reason)
            penalty = float(FAILURE_PENALTY)
            if self.is_multi:
                assert self.objective_metrics is not None
                return tuple(penalty for _ in self.objective_metrics)
            return penalty

        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("n_train_windows", n_train)
        trial.set_user_attr("n_val_windows", n_val)
        trial.set_user_attr("fit_seconds", round(fit_seconds, 1))

        if self.is_multi:
            assert self.objective_metrics is not None
            values = [metrics.get(m, float("nan")) for m in self.objective_metrics]
            if not all(np.isfinite(v) for v in values):
                raise optuna.TrialPruned(
                    f"A multi-objective metric is not finite: "
                    f"{dict(zip(self.objective_metrics, values))}."
                )
            return tuple(float(v) for v in values)

        primary = metrics.get(self.primary_metric, float("nan"))
        if not np.isfinite(primary):
            raise optuna.TrialPruned(
                f"Primary metric {self.primary_metric} is not finite."
            )
        self._maybe_cache_adapter(trial, float(primary))
        return float(primary)

    def _maybe_cache_adapter(self, trial: optuna.Trial, primary: float) -> None:
        """Save this trial's best LoRA adapter iff it ranks in the study top-K.

        Prunes adapters that drop out of the top-K so disk stays bounded. Keyed by
        trial number (no cross-worker collision); all failures are swallowed so
        caching never breaks a trial.
        """
        if not (self.adapters_dir and self.keep_top_k > 0 and self._last_adapter):
            return
        import torch

        try:
            completed = [
                t
                for t in trial.study.get_trials(deepcopy=False)
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
            ]
            ranked = [(t.number, float(t.value)) for t in completed]
            ranked.append((trial.number, primary))
            ranked.sort(key=lambda x: x[1])
            top_nums = {num for num, _ in ranked[: self.keep_top_k]}
            if trial.number in top_nums:
                torch.save(
                    {
                        "adapter": self._last_adapter,
                        "params": dict(trial.params),
                        "known_cols": list(trial.user_attrs.get("known_cols", [])),
                        "mard_avg": primary,
                    },
                    self.adapters_dir / f"trial_{trial.number:05d}.pt",
                )
                trial.set_user_attr("adapter_cached", True)
            for f in self.adapters_dir.glob("trial_*.pt"):
                try:
                    if int(f.stem.split("_")[1]) not in top_nums:
                        f.unlink()
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001 - caching must never break a trial
            pass
