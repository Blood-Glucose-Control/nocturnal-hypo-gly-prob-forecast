# Research Synthesis: Improving Nocturnal Blood Glucose Forecasting with Chronos-2

## Executive Summary

Our current validated baseline achieves **2.347 mmol/L RMSE** on the Brown 2019 AID dataset
(16 holdout patients, `configs/data/holdout_10pct/brown_2019.yaml`, 8-hour nocturnal forecast,
72 × 5-min steps). Experiments confirm that Chronos-2 is nearly blind to the IOB covariate
we provide — a failure mode formalised in the literature as *Driver-Blindness*.

The root cause goes deeper than a simple covariate configuration problem. Brown 2019 is a
**closed-loop AID dataset**: the insulin delivery that drives nocturnal BG is itself determined
by a feedback controller responding to CGM in real time. This means future insulin delivery and
future BG are causally entangled, which structurally limits physics-informed approaches on this
dataset. Understanding this constraint correctly narrows the actionable experiment space.

This document summarises what has been tried, what the constraints are per dataset, and what
experiments are actually actionable in our codebase.

---

## 1. Problem Statement

### 1.1 Task Setup

- **Data**: Brown 2019 DCLP3 (Tandem Control-IQ AID system). 168 total patients (125 with
  pump data, 43 CGM-only). 16 held out for evaluation (hybrid temporal + patient split,
  `configs/data/holdout_10pct/brown_2019.yaml`). Nocturnal episodes: context = 512 steps
  (~42.7h ending at midnight), forecast = 72 steps (00:00–06:00).
- **Model**: Chronos-2 fine-tuned via AutoGluon's `TimeSeriesPredictor` (autogluon.timeseries
  == 1.5.0). Stage 1 checkpoint: 15K steps, lr=1e-5, LoRA rank=8.
- **Covariates**: IOB as a past-only covariate, computed by `src/data/physiological/
  insulin_model/insulin_model.py` (Hovorka 3-compartment). AutoGluon auto-classifies columns
  not listed in `known_covariates_names` as past-only; they are NaN-masked in the forecast
  horizon.

### 1.2 Driver-Blindness — Quantified

Three training runs, identical config except for the covariate treatment:

| Configuration                           | RMSE   | Checkpoint        |
|-----------------------------------------|--------|-------------------|
| IOB as `known_covariates_names` (leaky) | ~1.890 | `20260227_032904` |
| IOB as past-only covariate (baseline)   | 2.347  | `20260227_060306` |
| BG-only, no IOB                         | 2.385  | `20260227_230945` |

**ΔRMSE (IOB vs no-IOB) = 0.038 mmol/L (1.6%).** The model is functionally blind to insulin.

This is consistent with Shakeri et al. (2025), who identify three compounding causes:

1. **Architectural bias (C1)**: transformers exploit strong local BG autocorrelation and
   deweight weaker exogenous signals during attention. With a 512-step context window (42h)
   the model has abundant autocorrelation to exploit.

2. **Pharmacokinetic misrepresentation (C2)**: a bug in our pipeline truncates the IOB curve
   at 3h. Rapid-acting insulin analogues (lispro/aspart/glulisine) have a 5–6h duration of
   action (DIA). The 3–6h tail — the portion most relevant to nocturnal hypoglycaemia — is
   zeroed out.

   **Root cause**: `src/data/physiological/carb_model/constants.py` sets
   `T_ACTION_MAX_MIN = 180`. This constant is imported by both `carb_model.py` and
   `insulin_model/insulin_model.py` (line 15). 180 min is physiologically correct for carbs;
   it is wrong for insulin.

3. **Forecast-horizon blindness (C3)**: because IOB is past-only, AutoGluon NaN-masks it for
   the 72-step forecast window. The model gets zero insulin information about the period it is
   actually predicting.

### 1.3 Why C3 is Hard to Fix on AID Data — The Treatment-Confounder Feedback Problem

The large improvement from the leaky run (~1.890) is misleading about what physics features
can achieve. The "future IOB" values it used weren't just a deterministic projection of
pre-midnight on-board insulin — they included insulin the AID *delivered overnight in response
to the patient's actual BG trajectory*. Control-IQ runs a feedback loop every 5 minutes:

```
BG(t) → Control-IQ prediction algorithm → basal_delivery(t) → BG(t+1)
```

Future insulin delivery is a function of future BG, which is what we are trying to predict.
The circular dependency means genuine future IOB is fundamentally unknowable at midnight — it
encodes what the controller "saw" happening. This is why that run gets ~1.890: the covariate
is almost a proxy for the BG label.

**What we can deterministically compute at midnight:** the decay of insulin already on-board
from pre-midnight doses only. This is non-leaky. But it is **incomplete**: the AID will
continue delivering basal insulin all night (~0.5–2 U/h, so 3–12 units over 6h depending on
the patient's basal rate and AID interventions). That overnight basal component is not
capturable without knowing future BG. The gap between the computable (pre-midnight decay) and
the actual (pre-midnight + overnight AID delivery) grows monotonically over the 6h horizon.

There is also a secondary inaccuracy: our pipeline uses the Hovorka 3-compartment model, but
the data was generated by a system running Control-IQ's internal 5-hour exponential decay
model. The PK curve shapes are different, so even the pre-midnight portion has a systematic
shape mismatch vs what actually drove the AID's decisions during training.

### 1.4 Dataset Taxonomy — Physics Approach Viability Per Dataset

This constraint is **specific to closed-loop AID datasets**. Our four datasets differ
fundamentally:

| Dataset         | Delivery mode                             | Future basal predictable?              | Physics feature viable?           |
|-----------------|-------------------------------------------|----------------------------------------|-----------------------------------|
| Brown 2019      | Control-IQ AID (closed-loop)              | No — feedback-controlled every 5 min   | Limited — pre-midnight decay only |
| Lynch 2022      | iLet Bionic Pancreas (fully autonomous)   | No — more aggressive than Control-IQ   | Limited — same issue              |
| Aleppo 2017     | Open-loop pump / MDI (no AID, ~2015 data) | Yes — programmed basal rate is fixed   | Works cleanly                     |
| Tamborlane 2008 | MDI (2008 — no AID existed)               | N/A — CGM-only, no dose data available | Not applicable                    |

For Brown 2019 and Lynch 2022, any physics-informed known future covariate will be an
incomplete approximation. For Aleppo 2017 (open-loop), the programmed basal rate is a patient
setting that doesn't change in real time — future basal delivery is deterministic and the
physics approach from the literature applies directly.

This means experiments on physics-informed features should ideally be **validated on Aleppo**
first, where the approach is clean, and then assessed on Brown/Lynch where the limitation is
understood and can be discussed as a structural constraint in the paper.

---

## 2. Literature Evidence

### 2.1 Physics-Informed / Hybrid PK-ML Models

| Paper                 | Year | Method                                        | Improvement                           |
|-----------------------|------|-----------------------------------------------|---------------------------------------|
| Georga et al.         | 2015 | PK features + SVR                             | 10–15% RMSE at 120 min                |
| De Bois et al.        | 2021 | Residual learning on Hovorka                  | 15–20% RMSE; robust to ±30% ISF error |
| Daniels et al.        | 2022 | LSTM correcting physiological model           | ~12% RMSE                             |
| Mosquera-Lopez et al. | 2022 | IOB-at-bedtime for nocturnal hypo             | AUC ~0.87                             |
| Oviedo et al.         | 2017 | Review: data-driven models lose skill >90 min | Physics essential for >2h horizons    |

**Important caveat**: all of the above studies used open-loop or MDI cohorts, where future
insulin delivery is either programmed (pump basal) or patient-announced (bolus). None were
conducted on closed-loop AID data. The 10–20% improvement figures are therefore directly
applicable to Aleppo 2017 in our repo, and are optimistic upper bounds for Brown 2019 / Lynch
2022.

De Bois et al. (2021): ±30% ISF error → only ~5% RMSE degradation in the hybrid model.
Parametric accuracy is secondary to structural correctness — the ML component corrects
systematic bias. This argument holds for open-loop settings; on AID data the dominant
inaccuracy is the missing overnight delivery term, not ISF estimation.

### 2.2 Multi-Task / Joint Forecasting

Training on multiple co-evolving targets forces the encoder to retain information about each
series in its latent representation. If the model is penalised for poor IOB predictions during
training, it cannot learn to ignore insulin — the gradient path is preserved.

| Paper          | Year | Result                                                     |
|----------------|------|------------------------------------------------------------|
| De Bois et al. | 2022 | Multi-task BG+IOB on OhioT1DM: 1–3% RMSE improvement       |
| Sun et al.     | 2023 | Joint glucose + insulin: ~5% improvement at 60 min horizon |

In our AutoGluon setup this maps to defining BG and IOB as co-targets with the same `item_id`
(Group ID). Chronos-2's alternating Time Attention / Group Attention architecture handles
cross-series information flow natively.

**Key advantage over Issue 3 on AID data**: joint forecasting doesn't require an explicit
model of future insulin delivery. The model must learn to predict future IOB from the context
window — and in doing so, it implicitly learns the AID's overnight delivery patterns from
training data. The treatment-confounder feedback loop is in the data; the model can learn it.
This makes joint forecasting more dataset-agnostic than physics-informed features.

### 2.3 Evaluation Metrics

**RMSE is misaligned with both our training objective and our clinical goal.**

- Chronos-2 training: hardcoded 21-level pinball loss in `_compute_loss()` (`.venvs/chronos2/
  .../chronos/chronos2/model.py`, lines 499–548), which approximates CRPS. Setting
  `eval_metric="RMSE"` in `TimeSeriesPredictor` affects model selection and early stopping
  only — it does not change training gradients.
- The `"mean"` prediction column from AutoGluon is the 0.5 quantile (median), not the
  mathematical mean. Blood glucose is right-skewed (log-normal), so mean > median. RMSE
  minimisation would pull predictions upward toward the mean — away from the hypo tail.
- RMSE is symmetric: a 2 mmol/L error at BG = 5.5 is penalised identically to one at BG = 3.0
  (life-threatening).

**Correct metrics:**
- **WQL** (Weighted Quantile Loss): normalised discrete approximation of CRPS, what AutoGluon
  natively computes. Directly aligned with the 21-quantile training loss.
- **Brier Score at 3.9 mmol/L**: `BS = E[(P(BG < 3.9) - 1{BG < 3.9})²]`. Directly measures
  calibration of hypoglycaemia probability estimates — the stated project goal.
- **RMSE**: keep as secondary for literature comparability.

### 2.4 Circadian Features

The dawn phenomenon (cortisol + growth hormone surge 3–5 am) causes progressive insulin
resistance through the forecast window. This is a deterministic, zero-leakage signal — wall
clock time is always known — that cannot be inferred from BG autocorrelation alone because
the pattern is circadian, not episode-specific. Applies to all four datasets equally.

---

## 3. Proposed Experiments

### Issue 1 — Bug Fix: IOB tail truncation (< 1 hour, no GPU)

**File:** `src/data/physiological/carb_model/constants.py`

`T_ACTION_MAX_MIN = 180` is imported by `carb_model.py` and `insulin_model/insulin_model.py`
(line 15). Fix: introduce a separate `INSULIN_ACTION_MAX_MIN = 360` in
`src/data/physiological/insulin_model/constants.py` and update the insulin model import.
Every prior experiment used a truncated IOB curve; this is a prerequisite for any insulin
feature work.

**Risk:** None — single constant, no algorithmic change.

---

### Issue 2 — Eval fix: WQL primary + Brier@3.9 (< 2 hours, no GPU)

In `src/evaluation/nocturnal.py`, `evaluate_nocturnal_forecasting()` currently computes RMSE
as the top-level scalar. Changes:

1. Compute WQL from the per-episode quantile forecasts already stored in `per_episode` results.
2. Add `brier_score_39()` helper: threshold quantile CDF at 3.9 mmol/L using linear
   interpolation between the 21 available quantiles, compute Brier Score.
3. Return both in the summary dict; update `scripts/experiments/nocturnal_hypo_eval.py` to
   log them.

No retraining needed. Gives correct signal for all downstream experiments.

---

### Issue 3 — Physics-informed known future covariate (2–3 days, GPU)

**What this is and what it isn't:**

At midnight, we can compute a partial forecast of insulin's BG effect using only pre-midnight
doses. `create_iob_and_ins_availability_cols()` in `insulin_model.py` already propagates each
dose's PK curve forward — so computing a "midnight-only" version means re-running that
simulation using the dose history truncated at midnight. Multiplying by an ISF estimate gives
`expected_insulin_bg_effect[t]`, which is added as `known_covariates_names` in AutoGluon.

**Why this is incomplete on Brown 2019:** the AID will deliver additional basal insulin all
night. That component is not capturable without knowing future BG (see §1.3). The pre-midnight
decay will be accurate for hours 1–2, then increasingly underestimate actual IOB as overnight
basal accumulates. There is also a PK model mismatch: our Hovorka curve vs Control-IQ's
internal 5h exponential.

**Where this works cleanly: Aleppo 2017.** Open-loop patients have a programmed overnight
basal rate (a patient-configured setting that doesn't change in real time). That rate is
available in the pump data. The full forward simulation — pre-midnight on-board decay +
projected programmed basal — is deterministic and accurate. We should validate this experiment
on Aleppo first, then apply to Brown/Lynch with the caveat in place.

**Implementation steps:**
1. After fixing Issue 1, add `compute_expected_bg_effect(dose_history_up_to_t, isf)` to
   `src/data/physiological/insulin_model/` using Euler integration at 5-min resolution,
   outputting a 72-step vector.
2. For Aleppo: include projected programmed basal in the dose history.
3. For Brown/Lynch: use pre-midnight doses only (acknowledge incompleteness).
4. ISF: `1700 / TDD` on past 7 days; fall back to population mean.
5. Add column to nocturnal episode DataFrames; add `known_covariates_names` to
   `configs/models/chronos2/`.

**Expected outcome:**
- Aleppo 2017: 10–15% RMSE reduction (literature-consistent)
- Brown 2019: 3–8% RMSE reduction (pre-midnight insulin effect is still real signal, but
  limited by incomplete overnight basal and PK mismatch)

**Risk:** Low engineering risk. The expected gain on Brown 2019 is uncertain and should not
be reported as 10–20% — that figure comes from open-loop literature.

---

### Issue 4 — Joint BG+IOB panel forecasting (1–2 days, GPU)

**Approach:**

Reformat training data so BG and IOB share the same `item_id` (Group ID), making them
co-targets in AutoGluon's panel forecasting mode. Chronos-2's Group Attention layers
cross-pollinate the two series during training. The model is penalised for poor IOB
predictions → gradient signal from insulin dynamics is preserved → Driver-Blindness is
mitigated.

**Why this is the better bet on AID data:** the model learns to predict future IOB from
the context window — which implicitly captures the AID's overnight delivery patterns from
training data. No explicit physiological simulation of overnight basal is needed. The
treatment-confounder feedback loop is present in the training distribution; the model can
learn it from data in a way that a hand-engineered physics feature cannot.

This is architecturally different from providing IOB as a covariate:
- **Covariate (current)**: IOB shapes the hidden state but has no dedicated loss term; model
  can and does learn to ignore it.
- **Co-target (proposed)**: IOB has its own pinball loss term; the model cannot ignore insulin
  dynamics without paying a direct training cost.

**Leakage check:** predicting future IOB from past delivery patterns is a valid forecasting
task. The model learns to extrapolate the overnight IOB trajectory from prior context — this
is data it has at inference time.

**Implementation steps:**
1. Extend the nocturnal episode builder to emit parallel `item_id` rows for BG and IOB
   sharing the same Group ID.
2. At inference, select only the BG output head for evaluation.
3. Retrain Stage 1 with panel format.

**Expected outcome:** 2–6% RMSE reduction on Brown 2019 (more confident than Issue 3 for
this dataset); potentially larger gains if the model successfully internalises AID dynamics.
**Risk:** Moderate — novel application of Chronos-2 Group IDs in our setup; outcome uncertain.

---

### Issue 5 — Lower-tail quantile weighting (1 hour + training run, combinable)

Chronos-2's `_compute_loss()` (`.venvs/chronos2/.../chronos/chronos2/model.py`, lines 499–548)
applies uniform weights across its 21 quantile levels `[0.01, 0.05, 0.1, ..., 0.9, 0.95, 0.99]`.
A 5-line edit doubles the weight on quantiles ≤ 0.2:

```python
quantile_weights = torch.ones_like(self.quantiles)
quantile_weights[self.quantiles <= 0.2] = 2.0
quantile_weights = rearrange(quantile_weights, "num_quantiles -> 1 num_quantiles 1")
quantile_loss = quantile_loss * quantile_weights
```

Combine with whichever training run comes first (Issue 3 or 4). Applies to all datasets
equally — no dependency on delivery mode.

**Expected outcome:** Improved Brier@3.9; minimal RMSE change.
**Risk:** Low. 2× is conservative.

---

### Issue 6 — sin/cos time-of-day known future covariate (0.5 days + training run, combinable)

`sin(2π·t/288)` and `cos(2π·t/288)` (288 = steps/day at 5-min resolution). Zero-leakage,
zero-cost known future covariates encoding circadian phase. Relevant to all four datasets.
Combine with Issue 3 training run.

---

## 4. Negative Result: Per-Patient LoRA Fine-Tuning

Two-stage protocol: Stage 1 = population checkpoint (`20260227_060306`); Stage 2 = per-patient
LoRA adaptation on all data before the test window.

| Steps | Improved | Regressed | Median ΔRMSE |
|-------|----------|-----------|--------------|
| 5000  | 2/16     | 14/16     | −0.35        |
| 1000  | 2/16     | 14/16     | −0.28        |
| 500   | 2/16     | 14/16     | −0.22        |
| 200   | 2/16     | 14/16     | −0.20        |

Worst case: `bro_91` regressed 1.057 RMSE at 200 steps.

**Root cause:** intra-patient temporal distribution shift. The fine-tune window covers months
1–5 of a patient's history; the test window is the final 14 days. Patient behaviour, pump
settings, and physiology drift over months — the population prior generalises better than
stale patient-specific adaptation.

Fewer steps reduce regression magnitude but never cross to net improvement, ruling out
hyperparameter tuning as a fix. The result is **architecturally clean**: static retrospective
LoRA cannot track non-stationary glucose dynamics. Future directions: online/continual
learning, or meta-learning (MAML) to learn a fast-adaptation prior.

Full per-patient RMSE table: `per-patient-finetune-results.md`.
Artifact: `trained_models/artifacts/chronos2/2026-03-01_00:28_RID20260301_002839_1894918_per_patient_finetune/`

---

## 5. Scheduling

```
Issue 1 (bug fix, 30 min — prerequisite for all insulin experiments)
│
├── Issue 2 (eval fix, 2 hrs, no GPU — do immediately)
│
├── Issue 3 (Experiment A — validate on Aleppo first, then Brown)
│     2–3 days eng + ~6h GPU per dataset
│
└── Issue 4 (Experiment B — primary experiment for Brown 2019)
      1–2 days eng + ~6h GPU

Issues 3 and 4 are independent and can run in parallel.
Issues 5 + 6 fold into whichever training run confirms positive signal (marginal eng overhead).
```

---

## 6. Summary

| # | Issue                                   | Dataset scope                           | GPU            | Eng effort | Expected Δ RMSE               |
|---|-----------------------------------------|-----------------------------------------|----------------|------------|-------------------------------|
| 1 | Fix `T_ACTION_MAX_MIN` (180→360)        | All                                     | No             | 30 min     | Prerequisite                  |
| 2 | Switch to WQL + Brier@3.9               | All                                     | No             | 2 hrs      | Better signal                 |
| 3 | Physics-informed known future covariate | **Aleppo: clean. Brown/Lynch: limited** | Yes            | 2–3 days   | Aleppo: −10–15%. Brown: −3–8% |
| 4 | Joint BG+IOB panel forecasting          | All (best on AID datasets)              | Yes            | 1–2 days   | −2–6%                         |
| 5 | Lower-tail quantile weighting           | All                                     | Yes (with 3/4) | 1 hr       | Brier calibration             |
| 6 | sin/cos time-of-day covariate           | All                                     | Yes (with 3)   | 0.5 days   | Moderate                      |

**Total engineering (excl. GPU wall-clock):** ~5–7 days
**Target on Brown 2019:** RMSE 2.347 → ~2.1–2.2 mmol/L (conservative given AID constraints)
**Target on Aleppo 2017:** 10–15% improvement if physics feature works as literature predicts

---

## 8. Evaluation Methodology Decisions

### 8.1 Probabilistic Interval Levels: 50%, 90%, 95%

Coverage and sharpness are computed at three nominal levels:

| Level | Interpretation | Rationale |
|-------|---------------|-----------|
| 50%   | Every other night the actual BG falls inside the interval | Standard narrow interval; universally reported |
| 90%   | 9 out of 10 nights inside the interval | Clinical resonance ("9 in 10 nights safe"); widely used in ML forecasting literature |
| 95%   | 19 out of 20 nights inside the interval | Aligns with M4/M5 competition standard; comparable to published baselines |

80% was considered but rejected: non-standard (no literature comparison point), and at this
clinical safety application the 80%/90%/95% gap is mostly a tail-calibration diagnostic that
90% already captures.

### 8.2 DILATE Metrics: Per-Episode Scalar, Not Per-Time-Step

DILATE (Soft-DTW shape + TDI temporal) is computed as a scalar per episode because it is a
holistic trajectory measure — it does not decompose meaningfully by time step. We compute it
at three gamma values (0.001, 0.01, 0.1) to capture sensitivity across alignment softness.

A `--no-dilate` flag is available in `nocturnal_hypo_eval.py` to skip DILATE for large
hyperparameter sweeps where the O(n_episodes × forecast_length²) cost is prohibitive.

### 8.3 Per-Time-Step Coverage and Sharpness (Future Work)

Coverage and sharpness are also meaningful at each forecast horizon step (e.g., step 5 = 25 min,
step 72 = 6 hrs). This would reveal *temporal miscalibration* — whether intervals are
well-calibrated early but collapse or blow up later overnight.

**Decision:** Implement as a post-loop summary over stacked arrays, not per-episode. After the
episode loop, stack `_q_forecasts` (n_eps, n_q, fh) and `_actuals_array` (n_eps, fh), then for
each time step t compute the mean coverage/sharpness across all episodes simultaneously.
Result: 1-D arrays of length `forecast_length` stored in Tier 3 `forecasts.npz` under keys
`coverage_by_step_{50,90,95}` and `sharpness_by_step_{50,90,95}` (not in `summary.csv` —
wrong shape for a flat table). Primary use: calibration plots (x = forecast horizon in minutes,
y = coverage) for the paper's supplementary material.

**Status:** Implemented. See `src/evaluation/metrics/probabilistic.py`
(`compute_coverage_by_step`, `compute_sharpness_by_step`) and `src/evaluation/nocturnal.py`
(post-loop probabilistic block). Arrays are saved via `src/evaluation/storage.py`
`_write_tier3` into `forecasts.npz`.

---

## 7. References

- Georga et al. (2015). *Multivariate Prediction of Subcutaneous Glucose Concentration in T1DM
  Patients Based on Support Vector Regression.* IEEE JBHI.
- De Bois et al. (2021). *Residual learning for blood glucose forecasting.* EMBC.
- De Bois et al. (2022). *Multi-task learning for blood glucose forecasting.* EMBC.
- Mosquera-Lopez et al. (2022). *Nocturnal hypoglycaemia prediction.* Nature Digital Medicine.
- Daniels et al. (2022). *Deep hybrid physiological–data-driven glucose prediction.* Frontiers
  in Physiology.
- Oviedo et al. (2017). *A review of personalised blood glucose prediction.* Biomedical
  Engineering Online.
- Shakeri et al. (2025). *Driver-Blindness in time-series foundation models.* (preprint)
- Gneiting & Raftery (2007). *Strictly proper scoring rules, prediction, and estimation.*
  Journal of the American Statistical Association.
- Sun et al. (2023). *Joint glucose and insulin prediction.* (conference)
- Rubin-Falcone et al. (2020). *Deep residual time-series for BG prediction.* IEEE JBHI.
