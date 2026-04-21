# MOMENT Sweep Configs

These configs are intended for quick, comparable MOMENT fine-tuning runs.
They focus on obvious high-impact knobs:

- `forecast_length` defaults to `96` (8 hours at 5-minute sampling)
- `learning_rate`
- `batch_size`
- `context_length`
- `training_config.freeze_backbone`
- `use_wrapper_normalization` (default `false` to rely on MOMENT internals)
- `covariate_cols` (explicit input covariate list; `[]` means BG-only)
- `window_stride` and `max_train_windows` (controls number of training windows)

Windowing notes:

- `window_stride`: how far each rolling training window advances. Smaller
  values produce more overlapping windows and more combinations; larger values
  reduce overlap, reduce compute, and reduce correlation between adjacent
  samples.
- `max_train_windows`: hard cap on the number of windows used for training.
  This keeps sweep runs bounded and comparable across datasets so one long
  dataset does not dominate runtime.

Ablation note:

- `06_wrapper_norm_on_ablation.yaml` intentionally enables wrapper-side
  normalization (`use_wrapper_normalization: true`) to compare against the
  default internal-normalization path.
- `07_stride_48.yaml` and `08_stride_24.yaml` keep the baseline setup but use
  denser rolling windows to increase training-window combinations.
- `09_bg_iob_insulin_availability.yaml` is the best-guess insulin-informed
  config (BG + IOB + insulin_availability).
- `10`-`13` provide data ablations for single-covariate variants (`iob`,
  `insulin_availability`, `cob`, `carb_availability`).
- `14_bg_full_covariates.yaml` includes all four covariates together.

Usage example:

```bash
python scripts/examples/example_holdout_generic_workflow.py \
  --model-type moment \
  --model-config configs/models/moment/00_baseline.yaml \
  --datasets aleppo_2017 \
  --config-dir configs/data/holdout_10pct \
  --skip-steps 4 6 7
```
