# AutoGluon Snapshot Evaluation: Shared Cache Gotcha

**Discovered:** 2026-04-25
**Affects:** Any code that loads multiple AutoGluon snapshot checkpoints (e.g. `step_2000`, `step_4000`, ...) and calls `predictor.predict()` on them **within the same evaluation run** or **sequentially against the same data**.

---

## Is This a Bug in Our Code or AutoGluon's?

**It is a trap in AutoGluon's design, not a bug in AutoGluon itself.** AutoGluon snapshots were designed for resuming *training*, not for *multi-checkpoint evaluation*. Using them as independent evaluation targets exposes a non-obvious shared-state problem. Our fix was defensive and correct, but the root cause is in how AutoGluon structures its snapshot directories.

---

## What Happened

When evaluating Chronos-2 IOB configs (04/10/11/12) at training step checkpoints (2000, 4000, 6000, 8000, 10000), all five steps produced **decimal-for-decimal identical results** across all metrics (RMSE, WQL, Brier, etc.) despite the adapter weights being verified as numerically distinct.

---

## Root Cause

### AutoGluon snapshot directory structure

When AutoGluon saves a training snapshot at step N, it creates a **shallow** snapshot directory:

```
{ARTDIR}/snapshots/step_{N}/predictor/
    learner.pkl         → symlink → ../../../learner.pkl       (shared!)
    predictor.pkl       → symlink → ../../../predictor.pkl     (shared!)
    model_config.yaml   → symlink → ../../../model_config.yaml (shared!)
    utils/              → symlink → ../../../utils/            (shared!)
    models/
        trainer.pkl     → symlink → ../../../../models/trainer.pkl  (shared!)
        Chronos2/
            model.pkl   → symlink → ../../../../../../models/Chronos2/model.pkl  (shared!)
            W0/
                fine-tuned-ckpt/
                    adapter_model.safetensors   ← REAL file, unique per step
                    adapter_config.json         → symlink (shared)
                    README.md                   → symlink (shared)
```

The actual LoRA weights (`adapter_model.safetensors`) **are** stored independently per snapshot. But everything else — `learner.pkl`, `trainer.pkl`, `model.pkl` — is a symlink back to the **main ARTDIR**.

### `reset_paths=True` correctly redirects `model_path`

When `TimeSeriesPredictor.load(snapshot_path)` is called with `reset_paths=True` (the default), AutoGluon calls `set_contexts(snapshot_path)` on the loaded objects, which updates `trainer.path` and `model.model_path` to point to the snapshot-specific weights directory. This part works correctly — each snapshot does load its own adapter on the first `predict()` call.

### The prediction cache is the problem

AutoGluon's `TimeSeriesTrainer` writes a prediction cache at:

```
{trainer.path}/cached_predictions.pkl
```

`trainer.pkl` is a **symlink** → the main ARTDIR's `models/trainer.pkl`. So `trainer.path` resolves to:

```
{ARTDIR}/models/
```

**This is the same physical path for ALL snapshots.** The first snapshot to run writes its predictions there. Every subsequent snapshot call to `predictor.predict(same_data, use_cache=True)` finds the cache, sees a hash match, and returns the first snapshot's predictions verbatim — regardless of which adapter was loaded.

This is exacerbated by the fact that we evaluate all snapshots against **the same holdout dataset** (identical `use_cache=True` key), so the cache hit is guaranteed.

### Verification

```python
# All five snapshots resolve to the same trainer.path:
pred = TimeSeriesPredictor.load(f"{ARTDIR}/snapshots/step_2000/predictor")
trainer = pred._learner.load_trainer()
# trainer.path == ARTDIR/models  ← identical for all steps

# Weights ARE distinct (different inodes, different mtimes, different values):
# step_2000: inode=2553738, mtime=1777028525
# step_4000: inode=2553784, mtime=1777030326
# step_8000: inode=2553876, mtime=1777033972
# step_10000: inode=2553922, mtime=1777035767
```

The stale cache file was at:
```
{ARTDIR}/models/cached_predictions.pkl
```

---

## Fix

### In `src/models/chronos2/model.py` — `_predict_batch()`

Pass `use_cache=False` whenever calling `predictor.predict()` for checkpoint evaluation:

```python
# BEFORE (incorrect for snapshot eval):
ag_predictions = self.predictor.predict(ts_data)

# AFTER (correct):
# use_cache=False is critical here: AutoGluon snapshot predictors share a
# symlinked trainer that saves cached_predictions.pkl to the SAME parent
# directory for ALL checkpoints. With the default use_cache=True, the first
# checkpoint to run writes a cache that all subsequent checkpoints silently
# reuse, producing identical results.
ag_predictions = self.predictor.predict(ts_data, use_cache=False)
```

This fix applies to both the fine-tuned path in `_predict_batch()` and any other places in the codebase that call `self.predictor.predict()` in a snapshot-evaluation context.

---

## Impact Assessment for Other AutoGluon Models

This gotcha applies to **any AutoGluon model** loaded from a snapshot predictor path, not just Chronos-2. If you add support for other AutoGluon-backed models:

| Use case | Safe? |
|---|---|
| Load single predictor, call predict once | ✅ Safe (`use_cache=True` is fine) |
| Load single predictor, call predict multiple times, **same data** | ✅ Cache speeds things up correctly |
| Load **multiple snapshot predictors**, call predict with same data | ❌ **Always pass `use_cache=False`** |
| Load final predictor and call predict | ✅ Safe |

**Rule of thumb:** Any time you loop over checkpoints/steps and call `predictor.predict()` on the same holdout data, use `use_cache=False`.

---

## Detection Heuristic

If evaluation results across different checkpoints are **identical to 6+ decimal places across all metrics**, suspect prediction caching before debugging model weights. A quick check:

```bash
# Look for a recently written cache file in the ARTDIR
find {ARTDIR} -name "cached_predictions.pkl" -newer {ARTDIR}/model.pt -maxdepth 3
```

If this returns a file, the cache is the likely culprit.

---

## Cleanup After Discovery

1. Delete stale `cached_predictions.pkl` files for all affected training runs
2. Delete all step-sweep experiment outputs (`experiments/nocturnal_forecasting_step_sweep/`)
3. Clear any done log (`logs/chronos2_step_sweep_done.log`)
4. Re-run evaluations with the patched code

Note: The main sweep and ctx-ablation evaluations are **not affected** — those evaluate each config's **final checkpoint only** (no snapshot sharing), so the cache is always valid for that single checkpoint.
