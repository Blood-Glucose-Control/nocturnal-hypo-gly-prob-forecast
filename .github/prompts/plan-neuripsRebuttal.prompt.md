# Plan: NeurIPS 2026 E&D Rebuttal — Nocturnal Hypo Benchmark (#3091)

## ✅ STATUS (2026-07-25) — ANALYSES DONE; DRAFTING REMAINS
**All analysis code complete + validated** under `scripts/analysis/rebuttal_neurips2026/` (outputs in `outputs/`):
- A1 significance (bootstrap CIs + paired tests) — now **patient-cluster** primary (`--bootstrap-unit patient`)
- A2 alarm operating points · A3/A3b anchoring + intervention timing · A4 bg-only leaderboard + covariate deltas
- A5 subgroup (true enrollment age + sex) · A6 aggregation ablation
- **A7 Friedman + Wilcoxon-Holm + rank-biserial effect sizes + Nemenyi CD** · **A7-CD diagrams** (`outputs/figures/`)
- **A8 zero-shot vs fine-tuned significance**
- **Reruns:** 32 foundation/ttm/toto + 3 tft bg-only, all validated to the paper's cells (±0.016 worst)
- **tft ported** from feat/autogluon-baselines (runs in `.venvs/chronos2`); factory + CLI wired
- **Clustering quantified:** ICC 0.11–0.21, design effect 2.5–6× → patient-cluster is the honest CI (12/13 headline paired results still significant)

**IN PROGRESS:** publication-grade n_boot=10000 runs in tmux `pubstats` (A1 patient/episode, A4, A8, A2).

**REMAINING WORK:**
1. **Non-coding content to write** (N1–N4 below): data-leakage argument, HP/compute protocol, contemporaneous cites, novelty reframe.
2. **DRAFT the rebuttal** (Draft phases 2–4 below): AC 3-category framing → R1/R2/R3 10k replies → AC confidential 5k + meta-review confidential 5k.
3. **CD-diagram hosting decision** (anonymized link, only in reply to R3 who requested significance analysis).
4. **Optional:** commit the tft restoration (`src/models/tft/*`, `autogluon_base.py`, factory + eval edits).
5. **Possible thin spot — R3 W4 dataset heterogeneity** ("why models succeed/fail across cohorts/devices"): address in text via dataset/device characteristics + A5; no new analysis needed.

---

Your strongest move is to **lead with statistical significance + alarm-utility numbers** (satisfies the shared concern across R2/R3/AC in one stroke), then reframe the AC's four questions around evidence you already have (covariate ablations, point-forecast detection, context ablation) plus a handful of new inline analyses. No PDF edits are allowed, so every new result must be quotable as text inside the 10k-char rebuttals — which is fine, because the highest-leverage asks (CIs, false-alarm rates) are compact numbers.

**Constraints locked in:** 10k chars/review (markdown, no links, no uploads), 5k AC-confidential comments, no artifact revisions. Scores R1=5, R2=2 (shallow), R3=3 (deepest), AC meta-review is LLM-written with inaccuracies and is the key reader.

## AC's 3 categories (rebuttal spine)
- **A — Univariate vs multimodal (AC Q1):** rebut — covariates *are* included as controlled I/IC ablations. → A3, A4
- **B — Classification/alarm utility (AC Q2):** the one concern shared by *all* reviewers. → A2, A6
- **C — Fair evaluation: context/HP/preprocessing (AC Q3) + data leakage (AC Q4).** → A4, N1, N2
- **Cross-cutting #1 priority:** statistical significance (A1) simultaneously answers R3's W2/Q2 *and* R2's "differences are intuitive/not established."

## Comment → category → analysis map
- **R1:** HP/FLOPs docs (C/N2); confusion-matrix/false-alarm question (B/A2); contemporaneous cites (N3). R1 already accepts.
- **R2:** novelty/dataset-contribution (N4 rebut w/ E&D guidelines); Q2 max-aggregation (A6); Q3 deterministic point-forecast classification — **already done in Table 5a** (rebut, point to it).
- **R3:** W1/Q1 midnight anchoring (A3); W2/Q2 significance (A1); W3/Q3 covariate fairness (A4); W5/Q4 subgroup (A5); dataset-contribution (N4); clinical claims temper (concede + B/A2).
- **AC-only:** Q4 data leakage (N1).

## Analyses to run (inline-reportable numbers)
1. **A1 — Bootstrap CIs + paired significance tests** *(HIGH)*. New `scripts/analysis/` script reusing `src/experiments/nocturnal/holdout_split_analysis.py` (`load_run_episode_classification` joins episodes.parquet + forecasts.npz) and `src/evaluation/metrics`. Paired bootstrap over episodes for RMSE/WQL/AUROC/AUPRC deltas on headline claims (attention best on RMSE/WQL; Moirai/Toto best on shape/detection). → R3 W2/Q2, R2.
2. **A2 — Alarm operating-point table** *(HIGH)*: sens/spec/PPV/NPV + false-alarms-per-night, reusing the `analyze_hypo_detection` pattern from `docs-internal/notebooks/4.14-ss-moirai-forecasting.ipynb`, at fixed operating points for top models across datasets. → R1, R2Q3, R3Q5, AC Q2.
3. **A3 — Midnight-anchoring sensitivity** *(HIGH — R3's #1 concern)*: define "quiescent nights" (no bolus/carb in the 8h window) as a sleep proxy; recompute metrics + Kendall-τ rank stability vs. full set. → R3 W1/Q1, AC Q1.
   - **A3b — Hour-of-day insulin/carb activity histograms** (per dataset): bin bolus events and carb intake by clock hour (0–23) across all patients to characterize when overnight interventions actually occur relative to the midnight anchor. Directly motivates the "quiescent night" cutoff and quantifies label noise. Since OpenReview allows no image uploads, report as inline summary stats (e.g., "% of boluses/carbs after 01:00, after 02:00; median time of last overnight bolus") and keep the figures for the camera-ready appendix. Data: bolus/`dose_units` + `food_g`/COB columns from processed patient frames (same loaders as A4). → R3 W1/Q1, AC Q1.
4. **A4 — bg-only controlled leaderboard + covariate delta** *(MED)*: extract bg_only condition for ALL models from `results/grand_summary/` via `src/experiments/nocturnal/grand_summary.py`; compute G→I/IC deltas to quantify architecture vs. covariate contribution. → R3 W3/Q3, AC Q1/Q3.
5. **A5 — Subgroup (sex primary, age where available)** *(MED, partial)*: feasible via Lynch `Sex`/`DiagAge` and Aleppo raw `HPtRoster`/`HScreening`/`HLocalHbA1c`. Sex-split metrics for top models; median-age split for Lynch+Aleppo. → R3 W5/Q4.
6. **A6 — max vs mean/product aggregation ablation** *(LOW)*: recompute episode risk under mean/product; compare AUROC/AUPRC. → R2 Q2.

## Non-coding rebuttal content
- **N1 — Data leakage (AC Q4):** JAEB datasets are credentialed/gated, absent from documented TSFM pretraining corpora (Chronos/Moirai/TimesFM); ZS < FT performance argues against memorization.
- **N2 — HP/compute protocol (R1, AC Q3):** document search budget/context windows/patch sizes from `configs/`; cite existing context ablation `scripts/experiments/nocturnal_hypo_eval_ctx_ablation.py` (same anchors across ctx).
- **N3 — Contemporaneous cites (R1):** commit to citing "From Prediction to Practice" + "DiaData" in camera-ready (cannot edit PDF now).
- **N4 — Novelty/contribution reframe (R2, R3):** E&D guidelines — Benchmark Design & Evaluation Methodology originality does NOT require beating a baseline; "new framing is sufficient." Not a dataset-contribution paper.

## Reviewer-vs-reviewer plays
- R2 "not novel" ↔ R1 "hope this becomes the standard" + R3 "valuable to both communities".
- AC/R3 covariate-fairness ↔ paper reports bg_only for all models + controlled ablations; R1 found the comparison reproducible.
- R2 "differences intuitive" ↔ A1 significance shows they're real; R1/R3 call the divergence "important/valuable".
- R2 Q3 deterministic classification ↔ Table 5a already does exactly this (s_i = −min_t μ̂_t for all models).

## Draft phases
0. Consolidate comments into the 3 categories (mapping table above).
1. Run A1–A6 (parallelize A1/A2/A4/A6 first; A3/A5 need extra data extraction).
2. Draft AC-facing framing (3 categories) → reused across rebuttals + meta-review confidential.
3. Draft per-reviewer 10k rebuttals reusing numbers.
4. AC confidential (5k): flag meta-review inaccuracies (covariate framing, context alignment already done).

## Verification
- Bootstrap point estimates recombine to published table values (RMSE/WQL within rounding); CIs contain point estimates.
- AUROC CIs contain Table 5a values; paired-test p-values sign-consistent with table ordering.
- Quiescent-subset episode counts + base rates sane; rank-τ reported with n.
- Char-count each rebuttal ≤10k; AC comments ≤5k; no links present.

## Open decisions
1. **AUROC significance:** DeLong (fast, parametric) vs paired bootstrap (consistent w/ RMSE/WQL). Rec: paired bootstrap for consistency.
2. **Subgroup depth:** sex-across-all + age for Lynch/Aleppo; defer device/HbA1c as future work.
3. **A6 priority if time-constrained:** drop first (lowest leverage — R2).
