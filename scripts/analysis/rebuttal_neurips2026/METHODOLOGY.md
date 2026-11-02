# NeurIPS 2026 E&D #3091 — Rebuttal Analyses: Methodology & Verification Guide

This document specifies exactly what every rebuttal analysis (`A1`–`A9`), the
shared data contract (`common.py`), and the demographics loader (`demographics.py`)
compute, so the work can be independently verified. It is written for a reader who
wants to (a) confirm each number cited in the rebuttal traces to code + on-disk
artifacts, and (b) re-run everything.

All analyses are **pure post-hoc processing** of per-episode evaluation artifacts
already written by the training/eval pipeline. **No models are loaded or re-run**
inside `A1`–`A9`; they read `episodes.parquet` + `forecasts.npz` from completed run
directories. The filesystem is the single source of truth (stale tracking CSVs are
never read for selection).

---

## 0. Environment & how to run

```bash
# from repo root
source .noctprob-venv/bin/activate            # pandas/numpy/scipy/sklearn/matplotlib/pyarrow/yaml

# each analysis is a module (note: run as -m so package imports resolve)
python -m scripts.analysis.rebuttal_neurips2026.a1_significance --n-boot 10000
python -m scripts.analysis.rebuttal_neurips2026.a2_alarm --bootstrap
python -m scripts.analysis.rebuttal_neurips2026.a3_anchoring
python -m scripts.analysis.rebuttal_neurips2026.a4_covariate --n-boot 10000
python -m scripts.analysis.rebuttal_neurips2026.a5_subgroup
python -m scripts.analysis.rebuttal_neurips2026.a6_aggregation
python -m scripts.analysis.rebuttal_neurips2026.a7_rank_significance
python -m scripts.analysis.rebuttal_neurips2026.a7_cd_diagram        # run AFTER a7_rank_significance
python -m scripts.analysis.rebuttal_neurips2026.a8_zeroshot_significance --n-boot 10000
python -m scripts.analysis.rebuttal_neurips2026.a9_cross_metric --n-boot 10000 --datasets all
```

- All CSV/figure outputs land in `outputs/` (and `outputs/figures/` for CD plots).
- Every analysis also writes `outputs/run_index.csv` — the exact set of run
  directories the filesystem scan selected, for provenance.
- Bootstrap analyses take `--n-boot` (default 2000; publication numbers use 10000)
  and `--seed` (default 42, so results are deterministic).
- Only ordering constraint: `a7_cd_diagram` reads `a7_friedman.csv`, so run
  `a7_rank_significance` first. `A1`–`A9` are otherwise independent.

---

## 1. Data contract — `common.py`

### 1.1 On-disk inputs
Each completed run lives at:
```
experiments/nocturnal_forecasting/512ctx_96fh/<model>/<run>/
    episodes.parquet          # 1 row per midnight-anchored episode + per-episode metrics
    forecasts.npz             # raw arrays, row-aligned to episodes.parquet
    experiment_config.json    # cli_args: dataset, covariate_cols, ...
    results_summary.json      # overall_rmse (used for best-condition selection)
```
`forecasts.npz` arrays (row `i` ↔ parquet row `i`):

| key | shape | meaning |
|---|---|---|
| `predictions` | (N, H) | point forecast, mmol/L |
| `actuals` | (N, H) | ground truth, mmol/L |
| `episode_ids` | (N,) | `"<patient_id>::ep<NNN>"` |
| `quantile_forecasts` | (N, Q, H) | quantile forecasts (empty (0,0,0) for point-only models) |
| `quantile_levels` | (Q,) | e.g. `[0.1, 0.2, …, 0.9]` |

Horizon `H = 96` (8 h at 5-min cadence, `STEPS_PER_HOUR = 12`). Hypo threshold
`HYPO_MMOL = 3.9`.

Datasets and rebuttal labels:
`aleppo_2017`→Replace-BG, `brown_2019`→DCLP3, `lynch_2022`→IOBP2,
`tamborlane_2008`→Tamborlane.

### 1.2 `load_run(run_path, …, compute_phypo=True) -> Run`
Reads the parquet + npz, asserts `len(parquet) == len(predictions) == len(episode_ids)`,
and (with `verify=True`) re-derives per-episode RMSE from the npz and asserts it
matches the parquet `rmse` column to `atol=1e-6` (the contractual row-alignment
check). It then augments the per-episode frame with:

| column | definition |
|---|---|
| `pid` | `episode_id.split("::")[0]` (patient id) |
| `split` | `patient` if pid ∈ holdout set else `temporal` (from `configs/data/holdout_10pct/<ds>.yaml → patient_config.holdout_patients`) |
| `has_hypo` | `(actuals < 3.9).any(axis=1)` — the episode-level detection **label** |
| `min_actual`, `min_pred` | per-episode minima |
| `score_point` | `-min_t pred_t` — deterministic point-forecast risk score (defined for **all** models) |
| `score_prob` | `max_t P(BG_t < 3.9)` — probabilistic risk score (only when quantiles exist; else `NaN`) |

`P(BG_t < 3.9)` (`_p_hypo_by_step`) is computed per step by interpolating the
threshold on the empirical CDF implied by the quantile forecast: BG quantile
values are sorted ascending (fixing mild quantile crossings) against the ascending
`quantile_levels`, then `np.interp(3.9, sorted_bg, levels)` with clamping to
`[levels[0], levels[-1]]` outside quantile support. Point-only models (TTM, MOMENT)
carry empty quantile arrays → `score_prob = NaN` → they are excluded from
probabilistic-detection axes automatically.

### 1.3 Run discovery & selection (filesystem = source of truth)
- `scan_runs()` walks `EXP_ROOT/*/*/experiment_config.json`, **skips** dirs whose
  name starts with `_` or contains `excluded`, requires both artifacts to exist,
  and records `(model, dataset, mode, condition, overall_rmse, run_path, mtime)`.
  Cached; never reads/writes the stale `summary.csv`/`best_by_model_dataset.csv`.
- `condition_bucket(covariate_cols)` → `bg_only` (None), `iob` (iob/insulin),
  `iob_cob` (cob/carb) — the paper's G / I / IC conditions.
- `_mode_from_name(run_name)` → `zeroshot` / `finetuned` / `unknown` (the
  checkpoint flag is unreliable, so mode is parsed from the dir name).
- `best_run_path(model, dataset, mode=None)` → the **lowest `overall_rmse`** run
  (ties broken by newest `mtime`). This reproduces the paper's best-condition
  table cell. `run_path_for_condition(...)` selects a specific G/I/IC condition.

### 1.4 Metric estimators (operate on a per-episode frame)
- `pooled_rmse(df) = sqrt(mean_i(rmse_i^2))` — pooled over episodes (NOT a mean of
  RMSEs), matching the paper's overall RMSE.
- `mean_metric(df, col)` — `nanmean`; returns `NaN` if the column is absent
  (point-only models lack `wql`, etc.).
- `auroc(df, score_col)` / `auprc(df, score_col)` — sklearn ROC-AUC / average
  precision of `has_hypo` vs the chosen score; returns `NaN` for single-class sets.

### 1.5 Uncertainty — bootstrap methods
| function | resamples | used by |
|---|---|---|
| `bootstrap_ci` | **episodes** (i.i.d. assumption) | A2, A4 marginal CIs |
| `paired_bootstrap_delta` | shared **episodes** (aligned on `episode_id`); two-sided p | A4 covariate delta |
| `cluster_bootstrap_ci` | **patients** (all of a patient's episodes) | A1 (default), A9 marginal CIs |
| `paired_cluster_bootstrap_delta` | shared **patients** (jointly for A and B); two-sided p | A1 (default), A8, A9 dissociation |

**Why patient clustering:** episodes are nested within patients (each patient
contributes many midnight episodes), so episode-level resampling understates
uncertainty (effective N ≈ #patients, not #episodes). Patient-cluster bootstrap
resamples patients with replacement, preserving within-patient correlation → the
honest, wider interval. This is the primary uncertainty method for the rebuttal.
Two-sided p-value = `2·min(P(boot≤0), P(boot≥0))`, capped at 1.

---

## 2. Demographics — `demographics.py`

`load_demographics(dataset)` returns one row per patient:
`pid, ptid, sex (M/F), age, diag_age, duration`.

**Reviewer-flagged correctness point:** `age` is the patient's **actual age at
enrollment/randomization**, NOT `DiagAge` (age at T1D diagnosis). Source columns
per JAEB public release:

| dataset | age (enrollment) | sex | diag_age |
|---|---|---|---|
| aleppo_2017 (Replace-BG) | `HPtRoster.AgeAsOfEnrollDt` | `HScreening.Gender` | `HScreening.DiagAge` |
| brown_2019 (DCLP3, UTF-16) | `DiabScreening_a.AgeAtEnrollment` | `Gender` | `DiagAge` |
| lynch_2022 (IOBP2) | `IOBP2PtRoster.AgeAsofEnrollDt` | `IOBP2DiabScreening.Sex` | `DiagAge` |
| tamborlane_2008 | `tblAPtSummary.AgeAsOfRandDt` | `Gender` | — (none) |

`_read` tries UTF-8/UTF-16/latin-1 (Brown's screening file is UTF-16). Join key:
`pid = <prefix>_<PtID>` with prefixes `ale/bro/lyn/tam`, matching the episode pids.
`duration = age − diag_age`.

---

## 3. Scoring conventions (important for verification)

Detection can be scored two ways; the rebuttal uses each deliberately:

| analysis | detection score | rationale |
|---|---|---|
| **A1, A5, A9** (detection ranking / significance) | `score_prob = max_t P(BG<3.9)` | matches the paper's Table 5a (author-confirmed); point-only models excluded |
| **A2** (alarm operating points) | `score_point = −min_t μ` (default; `--score prob` available) | clean operating characteristic across ALL models; the 9-quantile `score_prob` saturates threshold sweeps |
| **A3** (quiescent rank stability) | `score_point` | testing whether the AUROC **ranking** is invariant to the quiescent split, not absolute detection |
| **A4** (bg-only leaderboard) | `score_point` (AUROC is a secondary column; leaderboard is ranked by RMSE) | |
| **A6** | compares `*_point` vs `*_prob` aggregations head-to-head | that is the analysis |

If you re-derive detection numbers, use `score_prob` to reproduce the headline
detection results (A1/A5/A9).

---

## 4. Per-analysis specifications

Reviewer key: **R1** = EfDr (Accept 5), **R2** = xU1c (Reject 2),
**R3** = RCHh (Borderline 3), **AC** = meta-review.

### A1 — `a1_significance.py`: bootstrap CIs + paired significance
- **Answers:** R3 W2/Q2 ("no significance analysis"); R2 W1 ("not established").
- **Method:** for each (dataset, model) load the best-condition run
  (`compute_phypo=True`) and compute CIs for **RMSE, WQL, AUROC(`score_prob`),
  AUPRC(`score_prob`)**; detection metrics are skipped when `score_prob` is all-NaN
  (point-only models). Then paired tests for the headline head-to-heads:
  `chronos2 vs patchtst` (RMSE) and `moirai/toto/patchtst vs chronos2`
  (AUROC/AUPRC). `--bootstrap-unit patient` (default; clustered) or `episode`.
- **Outputs:** `a1_metric_cis_{patient|episode}.csv`,
  `a1_paired_tests_{patient|episode}.csv`.
- **Headline result:** 17/20 paired comparisons significant under 10k patient-cluster
  bootstrap (the 3 n.s. are single-dataset detection ties on Replace-BG/IOBP2).
- **Verify:** `a1_paired_tests_patient.csv` → `significant` column; a comparison is
  significant iff its 95% CI `[lo,hi]` excludes 0.

### A2 — `a2_alarm.py`: alarm operating points / confusion matrices
- **Answers:** R1 ("what mistakes? false alarms?"); R2 Q3 (point-forecast
  classification); R3 Q5 (do detection gains reduce misses?); AC Q2 (false-alarm rate).
- **Method:** label `has_hypo`; score `score_point` (default) or `score_prob`.
  Three operating points per (dataset, model): `sens90`, `sens80`
  (highest threshold reaching that recall), and `youden` (max sens+spec−1). Reports
  the full confusion matrix (TP/FP/FN/TN) plus sens/spec/PPV/NPV and
  **false-alarms-per-100-nights**. `--bootstrap` adds percentile CIs at the
  sensitivity setpoints, **re-selecting the threshold inside each resample** so the
  CI reflects setpoint uncertainty.
- **Outputs:** `a2_operating_points_point.csv`, `a2_operating_points_prob.csv`.
- **Verify:** e.g. Moirai/DCLP3 `sens90` row → TP 417 / FP 2553 / FN 46 / TN 793,
  PPV 0.14, NPV 0.95, FP/100 ≈ 67 (the R1 confusion-matrix table).

### A3 — `a3_anchoring.py`: midnight-anchoring sensitivity + intervention timing
- **Answers:** R3 W1/Q1 (nocturnal label noise from awake/eating/dosing).
- **Inputs:** processed per-patient series at
  `/data/shared/cache/data/<ds>/processed/<pid>_full.csv` (columns
  `datetime, dose_units/bolus, food_g/cob`).
- **A3b (timing):** hour-of-day (0–23) histogram of bolus units, bolus events, and
  carb events across evaluated patients; fraction of episode-nights with any
  in-window (00:00–08:00) bolus/carb; median first-bolus hour; share of daily
  boluses that fall overnight.
- **A3 (robustness):** an episode is **active** if it has a bolus **or** carb
  **strictly after** the anchor within the 8 h window (a 00:00 bolus is a bedtime
  dose, not overnight wakefulness); **quiescent** otherwise. Recompute per-model
  metrics on full/quiescent/active and report AUROC **rank stability**
  (Kendall τ, full vs quiescent).
- **Coverage caveat (documented in-code):** bolus available for
  Aleppo/DCLP3/IOBP2; carbs only for Aleppo; Tamborlane (CGM-only) has no
  intervention channel → skipped. Closed-loop cohorts deliver micro-boluses
  automatically, so "quiescent" is cleanest where boluses are discrete (DCLP3).
- **Outputs:** `a3_hour_histogram.csv`, `a3_episode_activity.csv`,
  `a3_quiescent_metrics.csv`.
- **Headline result:** DCLP3 46.5% of nights have a post-midnight bolus; quiescent
  ranking identical (τ = +1.0), metrics shift < 0.01 RMSE.

### A4 — `a4_covariate.py`: bg-only controlled leaderboard + covariate contribution
- **Answers:** R3 W3/Q3 and AC Q1/Q3 ("some models got covariates" confound).
- **Method:** (a) **bg-only leaderboard** — every model on identical univariate
  input (fine-tuned bg-only run via `run_path_for_condition(..., 'bg_only')`),
  RMSE with episode-bootstrap CI. (b) **covariate delta** = `RMSE(bg_only) −
  RMSE(best iob/iob_cob)` via paired episode bootstrap; `delta>0` ⇒ covariates
  help. Compares the **architecture spread** (best−worst bg-only RMSE) to the
  **median covariate gain**.
- **Caveat (in-code):** `tft` bg-only per-episode arrays are unavailable on this
  branch for some datasets → its published Table-3 G value is shown **without CI**
  and flagged (`source="published(no CI)"`, from `build_rerun_manifest.PAPER_RMSE`).
- **Outputs:** `a4_bgonly_leaderboard.csv`, `a4_covariate_delta.csv`.
- **Headline result:** architecture spread 0.50–0.64 RMSE ≫ median covariate gain
  +0.07–0.10.

### A5 — `a5_subgroup.py` (+ `demographics.py`): fairness / subgroups
- **Answers:** R3 W5/Q4 (no subgroup analysis).
- **Method:** stratify episode-level performance by **sex** and by **age**
  (median split on the **true enrollment age** of the evaluated patients, so the
  split adapts per cohort — Lynch includes children). Per subgroup report pooled
  RMSE, AUROC(`score_prob`), AUPRC(`score_prob`); report AUROC **rank stability**
  across the two groups (Kendall τ). `--min-group` (default 100) drops tiny cells.
- **Outputs:** `a5_subgroup.csv` (row per dataset×model×axis×group),
  `a5_demographics.csv` (coverage).
- **Headline result:** rankings stable (Kendall τ +0.6 to +1.0); gaps reported
  honestly (e.g. DCLP3 AUROC higher for female ≈0.73 vs male ≈0.68).

### A6 — `a6_aggregation.py`: episode-risk aggregation robustness
- **Answers:** R2 Q2 (does `max` over-react to isolated spikes vs mean/product?).
- **Method:** for each model, aggregate per-step risk to an episode score five
  ways — `max_point`, `mean_point` (from `−μ`), and `max`, `mean`, `noisy_or`
  (product, = `1−∏(1−p_t)`) from `P(hypo)` — and compute AUROC/AUPRC for each.
  `noisy_or` **is** the product aggregation R2 named.
- **Output:** `a6_aggregation.csv` (row per dataset×model×aggregation).
- **Headline result:** max vs mean vs noisy-or differ by mean |Δ|AUROC ≈ 0.0025
  (worst 0.013); rankings preserved (Kendall τ +0.81 to +1.0).

### A7 — `a7_rank_significance.py`: Friedman + Nemenyi + Wilcoxon–Holm
- **Answers:** R3 W2/Q2 (rank-based multi-model significance, Demšar 2006).
- **Method:** build a **patients × models** matrix of per-patient mean RMSE
  (patients present for all models). Per dataset: (1) **Friedman** omnibus;
  (2) average ranks + **Nemenyi critical difference** (CD) for the CD diagram;
  (3) pairwise **Wilcoxon signed-rank** over patients with **Holm** correction and
  a matched-pairs **rank-biserial effect size** (important given large N: a tiny
  p with negligible effect is not meaningful). Unit = **patient** (RMSE is
  per-patient definable; detection AUROC is not, so detection significance comes
  from A1's cluster bootstrap instead).
- **Outputs:** `a7_friedman.csv` (χ², p, N, avg ranks, CD),
  `a7_posthoc.csv` (per pair: Wilcoxon stat, p_holm, rank-biserial, ΔRMSE).

### A7-CD — `a7_cd_diagram.py`: Critical Difference diagrams
- Reads `a7_friedman.csv`; draws one CD diagram per dataset (models on an
  average-rank axis, cliques within the Nemenyi CD joined by bars). **Run A7 first.**
- **Outputs:** `outputs/figures/cd_<dataset>.png`, `outputs/figures/cd_all.png`.
- These are camera-ready figures; per the E&D link policy they may be linked
  anonymously in the rebuttal only in reply to R3's explicit significance request.

### A8 — `a8_zeroshot_significance.py`: zero-shot vs fine-tuned (contamination check)
- **Answers:** AC Q4 / R data-leakage ("did fine-tuning actually apply / is there
  pretraining contamination?").
- **Method:** for each foundation model with both a zero-shot and a fine-tuned
  run, paired **patient-cluster** bootstrap of `delta = RMSE(ZS) − RMSE(FT)` (and
  WQL) on shared midnight episodes; `delta>0` & CI excluding 0 ⇒ fine-tuning
  significantly helps.
- **Output:** `a8_zeroshot_vs_finetuned.csv`.
- **Headline result:** fine-tuning significant on 23/24 model×dataset cells
  (opposite of what memorization would produce); the lone exception is
  toto/Replace-BG.

### A9 — `a9_cross_metric.py`: cross-METRIC dominance (the core claim)
- **Answers:** the paper's central "no single architecture dominates across
  clinically-relevant metrics" claim; R2/R3 significance; AC Q2.
- **Models:** 8 probabilistic only (`chronos2, moirai, toto, timesfm, patchtst,
  tft, tide, deepar`); point-only excluded (detection needs `score_prob`).
- **Metrics (5 axes):** RMSE, WQL, DILATE-shape (`shape_g001`), AUROC(`score_prob`),
  AUPRC(`score_prob`). Detection uses the probabilistic P(hypo) score behind
  Table 5a. `PRIMARY_DS` = Replace-BG/DCLP3/IOBP2 (Tamborlane de-emphasized;
  `--datasets all` to include it).
- **Method:** (1) per-metric leaderboard with **patient-cluster** 95% CIs and the
  per-metric winner; (2) **Spearman** correlation between the RMSE ranking and each
  other metric's ranking (low/negative ⇒ metrics disagree on the best model);
  (3) **dissociation** — paired patient-cluster tests of `chronos2 vs
  {moirai, toto, patchtst}` on every metric, showing chronos2 is significantly
  better on RMSE/WQL yet significantly worse on shape/detection (a statistical
  double dissociation).
- **Outputs:** `a9_metric_leaderboard.csv`, `a9_rank_correlation.csv`,
  `a9_dissociation.csv`.
- **Headline result:** Spearman(RMSE-rank vs shape/detection-rank) = −0.05 to −0.52
  (vs +0.93–0.98 for RMSE↔WQL); chronos2 vs Moirai on DCLP3/IOBP2 — AUROC
  Δ −0.068/−0.055, AUPRC −0.032/−0.037 (all CIs exclude 0), reproduced exactly by
  the A1 10k paired tests.

---

## 5. Reviewer-point → analysis map

| Reviewer point | Analysis | Key output |
|---|---|---|
| R1 confusion matrix / false alarms | A2 | `a2_operating_points_point.csv` |
| R1 HP/compute protocol | (config files, no analysis) | camera-ready appendix |
| R1 tradeoff / which model for which metric | A9 (+A1) | `a9_rank_correlation.csv` |
| R2 Q1 actionable model×metric recs | A9 | `a9_metric_leaderboard.csv`, `a9_rank_correlation.csv` |
| R2 Q2 max vs mean vs product aggregation | A6 | `a6_aggregation.csv` |
| R2 Q3 point-forecast classification | A2 (point) / A6 (`*_point`) | `a2_operating_points_point.csv` |
| R2 W1 significance / "intuitive" | A1, A7, A9 | paired tests + Friedman |
| R3 W1/Q1 midnight anchoring | A3 | `a3_quiescent_metrics.csv` |
| R3 W2/Q2 significance | A1, A7, A9 | all significance CSVs |
| R3 W3/Q3 covariate fairness | A4 | `a4_bgonly_leaderboard.csv` |
| R3 W4 dataset heterogeneity | A7 CD + A9 | `cd_*.png`, `a9_*` |
| R3 W5/Q4 subgroup/fairness | A5 | `a5_subgroup.csv` |
| R3 Q5 alarm utility | A2 | `a2_operating_points_point.csv` |
| AC Q1 univariate/covariate | A4 (+A3) | `a4_*` |
| AC Q2 divergent metrics/alarms | A9 + A2 | `a9_*`, `a2_*` |
| AC Q3 fair adaptation | A8 | `a8_zeroshot_vs_finetuned.csv` |
| AC Q4 contamination | A8 | `a8_zeroshot_vs_finetuned.csv` |

---

## 6. Verification checklist

1. **Run selection is honest:** open any `outputs/run_index.csv` and confirm the
   selected `run_path`s are real dirs with both `episodes.parquet` and
   `forecasts.npz`, and that `best_run_path` picked the min-`overall_rmse` condition.
2. **Row alignment:** run any loader with `verify=True` (or add it) — the assertion
   re-derives RMSE from `forecasts.npz` and matches the parquet `rmse` column.
3. **Determinism:** all bootstraps use `--seed 42`; re-running reproduces the CSVs.
   (Point estimates like the dissociation Δ are bootstrap-independent; only CI
   widths change with `--n-boot`.)
4. **Scoring:** detection headline numbers use `score_prob` (A1/A5/A9); A2 alarm
   table intentionally uses `score_point` (§3).
5. **Significance definition:** a comparison is "significant" iff its 95% CI
   excludes 0 (`lo>0 or hi<0`), consistent across A1/A4/A8/A9.
6. **Cross-check across analyses:** A1's `moirai_vs_chronos2` AUROC deltas should
   equal (sign-flipped) A9's `chronos2_vs_moirai` deltas on the same dataset.

---

## 7. Supporting files (not analyses)

- `build_rerun_manifest.py` — enumerates which (model, dataset, condition, mode)
  runs were needed to reproduce the paper's cells; holds `PAPER_RMSE` (published
  values used by A4 for validation and for the CI-less `tft` bg-only cells),
  `SUPPORTED_MODELS`, `VENV_OVERRIDE` (`tft` runs in the `chronos2` venv),
  and `POINT_ONLY` (`ttm/moment/timegrad` — no `--probabilistic`).
- `outputs/` — all CSVs + `run_index.csv`; `outputs/figures/` — CD diagrams.
- `_table_backups/` — timestamped snapshots of tracking tables (never overwritten).
- `responses/` — the drafted OpenReview responses (public rebuttals + AC comments).
</content>
