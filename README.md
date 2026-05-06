# Probabilistic Nocturnal Hypoglycemia Forecasting

Companion code for an anonymous submission on **probabilistic short-horizon CGM forecasting around midnight-anchored nocturnal windows**, evaluated across fourteen forecasting models (classical baselines, statistical methods, deep encoders, and time-series foundation models).

The pipeline:

1. Loads continuous glucose monitor (CGM) data from four publicly available studies (via the [Awesome-CGM](https://github.com/IrinaStatsLab/Awesome-CGM) project).
2. Splits each cohort into a per-patient 90 / 10 train / hold-out partition (configurable).
3. Trains or zero-shot evaluates a model on midnight-anchored episodes (default context 512 steps ≈ 42 h, default forecast 96 steps = 8 h).
4. Computes deterministic (RMSE, MAE, DILATE) and probabilistic (WQL, PIT) forecasting metrics, plus episode-level hypoglycemia classification (AUROC, AUPRC) for the binary task `1{any forecast-window step has actual BG < 3.9 mmol/L}`.
5. Aggregates per-model / per-dataset results into `results/grand_summary/`.

---

## Repository layout

```
configs/                YAML model + data configs (1 dir per model)
src/
  data/                 dataset registry, cache manager, per-cohort cleaners
  models/               14 model wrappers behind a single registry/factory
  evaluation/           midnight-anchored episode builder + metrics
  experiments/          experiment runner used by all CLI entry points
examples/
  end_to_end_workflow.py     reproducible 7-step pipeline
                             (1: generate holdout configs → 2: validate configs → 3: load data
                              → 4: zero-shot eval → 5: fine-tune → 6: reload checkpoint
                              → 7: resume training)
scripts/
  experiments/nocturnal_hypo_eval.py    main per-(model × dataset) evaluation entry point
  analysis/build_grand_summary.py             aggregate all completed runs into one table
  analysis/build_best_episode_classification.py   AUROC / AUPRC table for the best run per (model x dataset x cov bucket)
  analysis/build_pit_*.py                     PIT calibration figures
  data_processing_scripts/generate_holdout_configs.py
tests/                  lightweight tests (data registry, cache, normalization, utils)
results/grand_summary/  pre-computed reference outputs from the paper
```

---

## Installation

The supported models do not share a single set of dependency pins (e.g. `chronos2`, `sundial`, `timesfm`, `ttm`, `moirai`, `toto` each pull in incompatible versions of `torch`, `transformers`, or `gluonts`). To handle this, the repo uses **one main virtual env, one shared AutoGluon env, and one per-model virtual env for each remaining model**, all driven by `pyproject.toml` extras and a `Makefile`.

Requirements: Python 3.11+, `make`, ≥40 GB free disk for all model envs, GPU recommended (most reported runs use a single 96 GB-class card).

```bash
# 1. Main env: data loading, evaluation, analysis only.
make venv-base

# 2. Shared env for AutoGluon-backed models
#    (deepar, patchtst, tft, naive_baseline, statistical, tide):
make venv-autogluon

# 3. Per-model envs (run only the ones you need):
make venv-ttm
make venv-sundial
make venv-chronos2
make venv-timesfm
make venv-moirai
make venv-moment
make venv-timegrad
make venv-toto

# Or build them all at once:
make venv-all-models
```

Each command above creates `.venvs/<extras>/` from `pip install -e ".[<extras>]"`.

---

## Data

All four datasets come from the public [Awesome-CGM](https://github.com/IrinaStatsLab/Awesome-CGM) collection. They must be downloaded **manually** (each requires accepting the source registry's terms) and placed under `cache/data/`:

| Dataset (registry name) | Description                                      | Source wiki                                                                          |
|-------------------------|--------------------------------------------------|--------------------------------------------------------------------------------------|
| `aleppo_2017`           | Aleppo 2017 trial                                | https://github.com/IrinaStatsLab/Awesome-CGM/wiki/Aleppo-(2017)                      |
| `brown_2019`            | Brown 2019 (DCLP3, closed-loop vs sensor-aug.)   | https://github.com/IrinaStatsLab/Awesome-CGM/wiki/Brown-(2019)                       |
| `lynch_2022`            | Lynch 2022 (IOBP2 RCT)                           | https://github.com/IrinaStatsLab/Awesome-CGM/wiki/Lynch-2022                         |
| `tamborlane_2008`       | Tamborlane 2008                                  | https://github.com/IrinaStatsLab/Awesome-CGM/wiki/Tamborlane-(2008)                  |

Expected on-disk layout:

```
cache/data/
  aleppo_2017/raw/<unzipped study files>/
  brown_2019/raw/<unzipped study files>/
  lynch_2022/raw/<unzipped study files>/
  tamborlane_2008/raw/<unzipped study files>/
```

The first time a dataset is requested, `src.data.cache_manager.CacheManager` reads the raw files, runs the per-cohort cleaner in `src/data/diabetes_datasets/<name>/`, and writes a normalized parquet cache under `cache/data/<name>/processed/`.

Holdout splits are deterministic and pre-generated under `configs/data/holdout_10pct/<dataset>.yaml`. To regenerate (or to make a different split):

```bash
./.venv/bin/python scripts/data_processing_scripts/generate_holdout_configs.py \
    --datasets aleppo_2017 brown_2019 lynch_2022 tamborlane_2008 \
    --temporal-pct 0.10 \
    --patient-pct 0.10 \
    --output-dir configs/data/holdout_10pct
```

---

## Running an evaluation

The recommended path for reviewers is **fine-tune with `end_to_end_workflow.py`**, then **evaluate with `nocturnal_hypo_eval.py`**, then aggregate. All three steps are shown below.

### 1. Fine-tune a model

`examples/end_to_end_workflow.py` runs all seven pipeline phases in one shot (1: generate holdout configs → 2: validate configs → 3: load data → 4: zero-shot eval → 5: fine-tune → 6: reload checkpoint → 7: resume training). Useful for the trainable models (`ttm`, `timegrad`, `tide`, `moment`, `chronos2`).

```bash
# Example: fine-tune TTM (CGM-only) on Aleppo 2017
./.venvs/ttm/bin/python examples/end_to_end_workflow.py \
    --model-type ttm \
    --datasets aleppo_2017 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/ttm/00_fine_tune_cgm_only.yaml
```

The trained checkpoint lands under `trained_models/artifacts/<model>/<timestamp>_RID..._holdout_workflow/model.pt` along with config snapshots, training metadata, and per-phase forecast plots.

### 2. Per (model × dataset) nocturnal evaluation

`scripts/experiments/nocturnal_hypo_eval.py` is the **main entry point used to generate the paper's results**. Each invocation evaluates a single model on a single dataset's hold-out patients over midnight-anchored episodes.

```bash
# Example: evaluate the TTM checkpoint trained above
./.venvs/ttm/bin/python scripts/experiments/nocturnal_hypo_eval.py \
    --model ttm \
    --model-config configs/models/ttm/00_fine_tune_cgm_only.yaml \
    --dataset aleppo_2017 \
    --checkpoint trained_models/artifacts/ttm/<timestamp>_RID..._holdout_workflow/model.pt \
    --context-length 512 \
    --forecast-length 96 \
    --probabilistic
```

```bash
# Example: zero-shot Chronos-2 on Brown 2019 (no checkpoint argument needed)
./.venvs/chronos2/bin/python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --model-config configs/models/chronos2/00_bg_only.yaml \
    --dataset brown_2019 \
    --config-dir configs/data/holdout_10pct \
    --context-length 512 \
    --forecast-length 96 \
    --probabilistic
```

Use the matching `.venvs/<model>/bin/python` for each model. AutoGluon-backed models (`deepar`, `patchtst`, `tft`, `naive_baseline`, `statistical`, `tide`) use the shared `.venvs/autogluon/bin/python`. The main `.venv` is for data loading + analysis scripts only.

> **Note on slow probabilistic evals.** `timegrad` (and to a lesser extent `statistical`/AutoARIMA on the largest cohort, Tamborlane 2008) generate sample-path forecasts per-episode rather than in batch. Even at `--num_samples 50`, a single (model × dataset) cell can take 30+ minutes. Allocate at least 4 h of wall time for these (model, dataset) combinations, or skip `--probabilistic` to fall back to faster point-forecast metrics.

Each run writes three tiers of output to `experiments/nocturnal_forecasting/<ctx>ctx_<fh>fh/<model>/<timestamp>_<dataset>_<mode>/`:

- `results_summary.json` — aggregate RMSE / MAE / WQL / coverage (probabilistic runs also include per-episode hypo probabilities used by the AUROC/AUPRC table)
- `episodes.parquet` — per-episode metrics (one row per midnight window)
- `forecasts.npz` — raw point + quantile forecasts for downstream plotting

### 3. Aggregating results

After multiple runs land under `experiments/`, build the master tables:

```bash
# Step 1: walk experiments/nocturnal_forecasting/ and write summary.csv.
# This is required before the grand-summary scripts; they read summary.csv.
./.venv/bin/python scripts/analysis/summarize_experiments.py
# (or equivalently:  make summary)

# Step 2: pivot tables of RMSE / WQL / coverage / calibration plus per-horizon breakdowns.
./.venv/bin/python scripts/analysis/build_grand_summary.py

# Step 3: episode-level hypoglycemia classification table (AUROC / AUPRC for
# the any-step BG<3.9 mmol/L binary task) — the main results table in the paper.
./.venv/bin/python scripts/analysis/build_best_episode_classification.py \
    --context-length 512
```

Outputs go to `results/grand_summary/`. The `results/grand_summary/` directory in this release contains the reference outputs reported in the paper.

---

## Adding a new dataset

1. Implement a cleaner under `src/data/diabetes_datasets/<your_dataset>/<your_dataset>.py` mirroring `lynch_2022/lynch_2022.py`. It must produce a long-format DataFrame with columns `patient_id, time, glucose` (and optional covariates).
2. Add a `DatasetSourceType` member in `src/data/models.py` and a `DatasetConfig` entry in `src/data/dataset_configs.py`.
3. Register the loader branch in `src/data/diabetes_datasets/data_loader.py::get_loader`.
4. Add an entry in `src/data/utils/patient_id.py::DATASET_PREFIXES` for compact patient IDs.
5. Generate a holdout config:
   `./.venv/bin/python scripts/data_processing_scripts/generate_holdout_configs.py --datasets <your_dataset> --output-dir configs/data/holdout_10pct`.

---

## Adding a new model

1. Create `src/models/<your_model>/{model.py, config.py, __init__.py}` following any existing model (e.g. `src/models/ttm/`) — implement `predict()` and (if probabilistic) `predict_quantiles()`.
2. Register the class in `src/models/registry.py`.
3. If it has unique deps, add a `[project.optional-dependencies].<your_model>` entry in `pyproject.toml`, and add the name to `MODEL_VENVS` in the `Makefile`.
4. Add at least one numbered YAML under `configs/models/<your_model>/00_<variant>.yaml` (see existing model directories for naming conventions).
5. Add a per-model branch in `src/training/strategies/` if its training loop differs from the existing ones.

---

## Tests

```bash
make test          # data-registry + cache + utils + normalization checks (main .venv)
```

---

## License

Released under the MIT License — see [LICENSE](LICENSE).
