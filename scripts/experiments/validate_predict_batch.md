# Validate _predict_batch()

Note: Activate correct .venvs/{model}/bin/activate prior to running.
Suggestion: Use tmux to run in parallel.

```bash
# TTM fine-tuned (expect max_diff == 0.0)
python scripts/experiments/validate_predict_batch.py \
    --model ttm --dataset aleppo_2017 \
    --checkpoint trained_models/artifacts/ttm/2026-02-27_03:53_RID20260227_035316_193673_holdout_workflow/model.pt \
    --context-length 512 --forecast-length 96 \
    --max-episodes 0 

# Chronos2 fine-tuned (expect max_diff == 0.0)
python scripts/experiments/validate_predict_batch.py \
    --model chronos2 --dataset aleppo_2017 \
    --checkpoint trained_models/artifacts/chronos2/2026-02-28_05:54_RID20260228_055400_391511_holdout_workflow/resumed_training/model.pt \
    --context-length 512 --forecast-length 96 \
    --max-episodes 0 

# TiDE fine-tuned (expect max_diff == 0.0)
python scripts/experiments/validate_predict_batch.py \
    --model tide --dataset aleppo_2017 \
    --checkpoint trained_models/artifacts/tide/2026-02-28_21:28_RID20260228_212852_496983_holdout_workflow/model.pt \
    --context-length 512 --forecast-length 96 \
    --max-episodes 0 

# TTM zero-shot (float noise ok, use looser tolerance)
python scripts/experiments/validate_predict_batch.py \
    --model ttm --dataset aleppo_2017 \
    --context-length 512 --forecast-length 96 --tolerance 1e-4 \
    --max-episodes 0 
```
## Expected Results:

| Model	| _predict_batch() path  |	Expected diff |
| - | - | - |
| TTM fine-tuned |	super()._predict_batch() → same sequential loop	| 0.0 exactly |
| Chronos2 fine-tuned | super()._predict_batch() → same sequential loop	| 0.0 exactly |
| TiDE fine-tuned |	super()._predict_batch() → same sequential loop | 0.0 exactly |
| TTM zero-shot	| batched TimeSeriesForecastingPipeline | ~1e-6 float noise |