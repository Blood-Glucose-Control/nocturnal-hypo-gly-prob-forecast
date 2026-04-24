# Forecast Comparison Script

`scripts/analysis/compare_forecasts.py` generates publication-quality grid figures
comparing nocturnal BG forecasts across any combination of models (zero-shot or
fine-tuned) on midnight-anchored holdout episodes.

**Output grid:** one row per patient, one column per RMSE percentile (default:
P90 / P70 / P30 / P10 — hardest to easiest within each patient, ranked by the
first listed model). Default 5 patients × 4 columns = 20 subplots.

---

## Why two stages?

Different models require incompatible Python environments (e.g. TTM and Chronos2
pin different versions of `transformers`). The script splits work into:

- **Stage 1 — Inference:** run once per model in its own env; results cached to
  `results/forecast_comparisons/<hash>.json`
- **Stage 2 — Plot:** run in any env; reads cached JSONs and renders the figure

The cache is keyed by a content hash of all run parameters. Re-running Stage 1
with identical arguments is a no-op — it hits the cache and skips inference.

---

## Quickstart

### Stage 1 — Run inference per model

```bash
# Chronos2 (in its own env)
source scripts/setup_model_env.sh chronos2
python scripts/analysis/compare_forecasts.py \
    --model chronos2::Chronos2 \
    --no-plot
# → Results cached at: results/forecast_comparisons/a3f82c1d.json

# TTM (in its own env)
source scripts/setup_model_env.sh ttm
python scripts/analysis/compare_forecasts.py \
    --model ttm::TTM \
    --no-plot
# → Results cached at: results/forecast_comparisons/9b14e702.json
```

> Both runs **must use the same `--dataset`, `--seed`, `--n-patients` (or
> `--patients`), and `--forecast-length`** so the patient lists match.
> The defaults are identical, so if you don't override anything you're safe.

### Stage 2 — Plot

```bash
# Any env (e.g. the base .noctprob-venv)
source .noctprob-venv/bin/activate
python scripts/analysis/compare_forecasts.py \
    --from-results \
        results/forecast_comparisons/12496a49.json \
        results/forecast_comparisons/d8453cc4.json
```

Output saved to `images/figures/forecast_comparisons/`:
```
20260311_143022_Chronos2_vs_TTM_5pat.png   ← figure
20260311_143022_Chronos2_vs_TTM_5pat.json  ← sidecar (reproducibility metadata)
```

---

## Common recipes

### Single-environment shortcut
When comparing two checkpoints of the same model type (same env), both can run
in one command and the plot is generated automatically:

```bash
source scripts/setup_model_env.sh ttm
python scripts/analysis/compare_forecasts.py \
    --model ttm::TTM-zeroshot \
    --model ttm:trained_models/artifacts/ttm-ft-iob:TTM-IOB
```

### Specific patients
```bash
python scripts/analysis/compare_forecasts.py \
    --model chronos2::Chronos2 \
    --no-plot \
    --patients bro_22 bro_65 bro_75 bro_93 bro_121
```

### Custom percentile columns
```bash
python scripts/analysis/compare_forecasts.py \
    --from-results results/forecast_comparisons/a3f82c1d.json \
                   results/forecast_comparisons/9b14e702.json \
    --percentiles 95 75 50 25 5
```

### Fine-tuned vs zero-shot (cross-env)
```bash
# Stage 1a: zero-shot TTM
source scripts/setup_model_env.sh ttm
python scripts/analysis/compare_forecasts.py --model ttm::TTM-ZS --no-plot

# Stage 1b: fine-tuned TTM checkpoint (same env)
python scripts/analysis/compare_forecasts.py \
    --model ttm:trained_models/artifacts/ttm-ft-run42:TTM-FT \
    --no-plot

# Stage 2
python scripts/analysis/compare_forecasts.py \
    --from-results \
        results/forecast_comparisons/<zs-hash>.json \
        results/forecast_comparisons/<ft-hash>.json
```

### Find cached results
```bash
python -c "
import json, pathlib
for f in sorted(pathlib.Path('results/forecast_comparisons').glob('*.json')):
    m = json.loads(f.read_text())['metadata']
    print(f, m['label'], m['dataset'], 'seed=' + str(m['seed']))
"
```

---

## All options

| Flag | Default | Description |
|------|---------|-------------|
| `--model SPEC` | — | `type:checkpoint:label` (repeatable). Runs inference. |
| `--from-results JSON [...]` | — | Paths to cached JSONs. Skips inference. |
| `--dataset` | `brown_2019` | Dataset name. |
| `--config-dir` | `configs/data/holdout_10pct` | Holdout config directory. |
| `--n-patients` | `5` | Number of patients to randomly sample. Ignored if `--patients` given. |
| `--patients` | all holdout | Explicit patient IDs (overrides `--n-patients`). |
| `--seed` | `42` | Controls patient sampling and reproducibility. |
| `--forecast-length` | `96` | Forecast horizon in steps (96 = 8 h at 5-min). |
| `--context-length` | `512` | Context window in steps (~42.7 h at 5-min). |
| `--covariate-cols` | auto-detected | Extra covariate columns (e.g. `iob`). Fine-tuned checkpoints also auto-detected. |
| `--percentiles` | `90 70 30 10` | RMSE percentile columns in the grid (plot mode). |
| `--no-plot` | false | Cache inference results without plotting. |
| `--results-dir` | `results/forecast_comparisons` | Directory for cached inference JSONs. |
| `--output-dir` | `images/figures/forecast_comparisons` | Directory for figure and sidecar output. |
| `--output-name` | auto (timestamped) | Override output filename. |

### Model spec format
```
type:checkpoint:label
```
- **type** — model type: `chronos2`, `ttm`, `sundial`, `timegrad`, `timesfm`, `tide`
- **checkpoint** — path to fine-tuned checkpoint, or empty for zero-shot (e.g. `ttm::MyLabel`)
- **label** — display name in the plot (defaults to `type` or `type-ft`)

---

## Output files

### Cached inference JSON (`results/forecast_comparisons/<hash>.json`)
```json
{
  "metadata": {
    "model_type": "chronos2",
    "checkpoint": null,
    "label": "Chronos2",
    "dataset": "brown_2019",
    "seed": 42,
    "forecast_length": 96,
    "patients": ["bro_22", "bro_65", "bro_75", "bro_93", "bro_121"],
    ...
  },
  "episodes": [
    {
      "patient_id": "bro_22",
      "anchor": "2020-03-14 00:00:00",
      "context_bg": [6.1, 5.9, ...],
      "target_bg": [5.8, 5.5, ...],
      "forecast": [5.9, 5.6, ...],
      "rmse": 0.312
    },
    ...
  ]
}
```

The filename hash encodes all run parameters — same parameters always map to the
same file. The `metadata` block contains everything needed to reproduce the run.

### Figure sidecar JSON (`images/figures/forecast_comparisons/<timestamp>_*.json`)
Records the exact result files, percentile settings, and model metadata that
produced the figure. Sufficient to fully reproduce or explain any plot.

---

## Notes

- **Percentile semantics:** percentiles are computed per-patient using the
  *first* listed model's RMSE as the ranking signal. P90 = the episode where
  that model struggled most; P10 = where it did best.
- **Episode alignment:** episodes are matched across result files by
  `(patient_id, anchor)` timestamp, not by array index. A failed episode in one
  model's run simply won't appear in the shared set.
- **Iterating on plots:** re-run Stage 2 with different `--percentiles`,
  `--output-name`, or a different subset of `--from-results` without re-running
  inference.
