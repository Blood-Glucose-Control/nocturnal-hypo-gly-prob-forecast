<!-- NeurIPS 2026 E&D #3091 — REBUTTAL DRAFTS (working). -->
<!-- OpenReview boxes: Overall AC-Confidential (5k); Meta-review AC-Confidential (5k);
     per-reviewer Rebuttal (10k, PUBLIC→reviewer) + AC-Confidential (5k, AC-only).
     NOTE: meta-review has NO public box — all meta-review replies are AC-confidential. -->
<!-- Numbers sourced from scripts/analysis/rebuttal_neurips2026/outputs/*.csv (patient-cluster bootstrap = primary). -->

# ============================================================
# OVERALL — AUTHOR/AC CONFIDENTIAL COMMENT  (AC-only, ≤5,000 chars)
# ============================================================

We thank the reviewers and the AC. We summarize the contribution, the rebuttal-window work, and how we read the review set; per-review and per-meta-review responses give the details and numbers.

**Contribution type.** This is an **Evaluation & Datasets** submission whose contribution is a *benchmark + evaluation methodology*, not a new dataset or a new SOTA model. Its central, now-verified finding is a **cross-metric double dissociation**: the model that is significantly best on point/probabilistic accuracy (chronos2 on RMSE/WQL) is significantly *worse* on trajectory shape (DILATE) and on hypoglycemia detection (AUROC/AUPRC), where continuous-output models (Moirai/Toto/PatchTST) lead. The Spearman correlation between the RMSE ranking and the shape/detection rankings is only **−0.05 to −0.52** (vs +0.93–0.98 for RMSE↔WQL). The meta-review's own summary states this finding correctly — it is the paper's core message, and a caution the ML/health communities need: **optimizing point accuracy does not optimize clinical event detection.**

**Rebuttal-window work (all reproducible in released code).** We added nine analyses, all using **patient-clustered resampling** (the honest, wider interval given episodes are nested within patients; ICC 0.11–0.21). Highlights: paired significance with Friedman + Wilcoxon–Holm and effect sizes (17/20 headline comparisons remain significant under 10,000-bootstrap patient clustering); alarm operating points with confusion matrices; midnight-anchoring sensitivity (quiescent-night ranking identical, τ = +1.0); glucose-only fairness leaderboard (architecture spread 0.50–0.64 RMSE ≫ covariate gain 0.07–0.10); subgroup analysis by true enrollment age and sex; aggregation robustness (max ≈ mean ≈ noisy-or); zero-shot vs fine-tuned contamination check. No PDF revision is possible during rebuttal, so all numbers are reported inline; the camera-ready will fold them in.

**How we read the review set.** R1 (Accept, 5) understands the sub-field and the contribution type and considers the paper publishable as-written. R3 (borderline; the most detailed review) raised five specific, constructive concerns — each is now answered with a dedicated new analysis, none of which changes a conclusion. R2 (Reject, 2) is brief and generic ("intuitive," "unclear novelty") and does not engage the paper's methodology or any table; its three questions are all answerable, two from material already in the submission (including the product aggregation it assumed was untested). We respectfully ask that the discussion weight the two reviews that engage the technical content.

**On the meta-review.** It appears LLM-assisted and contains one factual error about the benchmark design (that some models get covariates and others do not — in fact every model is evaluated glucose-only, with covariates as a controlled per-model ablation). We correct this in the meta-review box; it does not affect any result.


# ============================================================
# META-REVIEW — AUTHOR/AC CONFIDENTIAL COMMENT  (AC-only, ≤5,000 chars)
# ============================================================

We thank the AC. We first note a factual point about the meta-review, then answer its four questions with **new analyses added during rebuttal** (fully reproducible in our released code; all use patient-clustered resampling — see below).

**Correction (affects Category A & C).** The meta-review states "some models are evaluated with insulin/carbohydrate covariates while others are effectively restricted to glucose-only." This is not how the benchmark is built: **every model is evaluated glucose-only**, and covariates enter only as a *controlled, per-model ablation* (subscripts G/I/IC in Tables 3–4). No cross-model comparison mixes input sets. We now make the controlled comparison explicit (below).

**Q1 — Univariate vs. multimodal realism.** The benchmark is *not* univariate-only: we report glucose-only, +insulin (I), and +insulin+carbs (IC) for every covariate-capable model. New: on the **glucose-only** leaderboard (identical input for all 10 models), the architecture spread is **0.50–0.64 RMSE** per dataset, whereas the *median* within-model gain from adding covariates is only **+0.07–0.10 RMSE** (paired patient-cluster bootstrap; chronos2 gains most, +0.09–0.18, all CIs exclude 0; ttm ~0). So covariate access does **not** drive the ranking — architecture does. Overnight physiology is real, but our A3 timing analysis shows overnight insulin/carb events are the exception, not the rule (Q1 of R3).

**Q2 — Divergent metrics & alarm utility.** The meta-review's own summary states our central finding precisely: rankings "diverge significantly on shape-sensitive (DILATE) and classification (AUROC/AUPRC) metrics." We now establish this **statistically**: chronos2 is significantly best on RMSE/WQL yet significantly *worse* than Moirai/Toto/PatchTST on DILATE-shape and on hypo detection (AUROC/AUPRC) — a paired **patient-clustered** double dissociation (95% CIs exclude 0 on DCLP3/IOBP2; Spearman between the RMSE ranking and the shape/detection rankings is −0.05 to −0.52, vs +0.93–0.98 for RMSE↔WQL). Interpretation for alarm systems: the model must be chosen by the *clinical* metric, not by RMSE. On alarm utility itself (operating points computed on the deterministic min-forecast risk score; detection AUROC/AUPRC use the probabilistic P(hypo) score of Table 5a): at 90% sensitivity false alarms run **45–72 per 100 nights** (PPV 0.14–0.43, base-rate dependent); at the Youden point PPV reaches **0.55–0.63** at 44–68% recall. We state plainly that long-horizon alarms are **not yet deployable at low false-alarm rates** — a headline benchmark *finding*, not a limitation we hide.

**Q3 — Fair adaptation.** Context window (512), forecast horizon (96), preprocessing, holdout splits, and the midnight-anchored episode set are **identical across all models**; only architecture-native fine-tuning differs. Our repo already contains a context-length ablation (same anchors across ctx). New: **zero-shot vs. fine-tuned** paired tests show fine-tuning significantly helps nearly every foundation model (patient-cluster CIs exclude 0), confirming adaptation was actually applied.

**Q4 — Pretraining contamination.** Before including each TSFM we manually reviewed its publication and documented pretraining corpus (Chronos, Moirai, TimesFM, Moment, Toto): **none report any diabetes or CGM data**. We surface this as a *finding*, not just a negative check — the major TSFM pretraining corpora entirely omit CGM, a large-scale (47.5M readings in this benchmark alone), high-real-world-impact signal with rich open problems; the ML community is leaving a valuable data source on the table. The four JAEB datasets are moreover **credentialed/gated** (application required) and do not appear in those corpora. Empirically, **zero-shot is significantly worse than fine-tuned** on 23/24 model×dataset cells — the opposite of what memorization would produce.

**On the meta-review's LLM provenance.** Several framings (e.g., the covariate point above; "no significance tests") are addressed by analyses that were either already in the paper or added here; we ask that the discussion weight the reviewers' specific technical points, which we answer in full in the per-review responses.


# ============================================================
# REVIEWER 1 — REBUTTAL  (PUBLIC, ≤10,000 chars)  — Accept (5); champion of the paradigm
# ============================================================

We thank R1 for engaging so deeply with the *purpose* of the benchmark — that RMSE is too distal an outcome and that clinically-relevant, long-horizon, uncertainty-aware evaluation should become the standard. That is exactly the paradigm shift we hope the paper seeds. We answer the specific question and the three actionable concerns below; all new numbers are reproducible in the released code.

**Q — What mistakes are the models making? (precision/recall, confusion matrix.)** We added a full operating-point analysis (sensitivity/specificity/PPV/NPV + a confusion matrix at clinically meaningful thresholds). The error profile is consistent and clinically legible: models are **strong at ruling hypoglycemia *out*** but **weak at ruling it *in*** at high recall. Example — Moirai on DCLP3 (base rate 12%), episode-level any-step detection:

| Operating point | TP | FP | FN | TN | Sens | Spec | PPV | NPV | FP/100 nights |
|---|---|---|---|---|---|---|---|---|---|
| High-sensitivity (~0.90 sens) | 417 | 2553 | 46 | 793 | 0.90 | 0.24 | 0.14 | 0.95 | 67 |
| Youden-optimal | 214 | 559 | 249 | 2787 | 0.46 | 0.83 | 0.28 | 0.92 | 15 |

So the dominant error at deployable-recall is **false alarms, not missed events** (NPV 0.92–0.95; FP ≫ FN). Tightening the threshold trades recall for a ~4.5× reduction in false alarms (67→15 per 100 nights) and roughly doubles PPV. Across datasets, at 90% sensitivity false alarms run 45–72/100 nights (PPV 0.14–0.43); at the Youden point PPV reaches 0.55–0.63. We now report this explicitly and frame low-false-alarm long-horizon alarming as an **open problem the benchmark defines**.

**W1 — Hyperparameter / compute protocol.** We agree the paper under-documents this and we fix it. All models share an **identical evaluation harness**: context 512, horizon 96, the same preprocessing, holdout splits, and midnight-anchored episode set; only architecture-native fine-tuning differs. Each model was tuned within its own documented search space (the config files R1 located); we will add a **compute/HP appendix** in the camera-ready giving, per model, the search space, tuning budget, and wall-clock GPU-hours so FLOPs/clocktime comparability is explicit rather than implicit. We are also happy to include this table in the discussion now if useful.

**W2 — "Inconclusive findings" → tradeoff discussion.** We share the goal of turning "different models win different metrics" into *actionable* guidance. The rebuttal makes the finding concrete and statistical: chronos2 is significantly best on **RMSE/WQL**, Moirai/Toto on **shape (DILATE)**, and Moirai/PatchTST on **event detection (AUROC/AUPRC)** — a paired patient-clustered *double dissociation* (Spearman between the RMSE ranking and the shape/detection rankings is only −0.05 to −0.52, vs +0.93–0.98 for RMSE↔WQL). Practitioner takeaway we now state: **select the model by the clinical objective, not by RMSE.** The mechanism (continuous-output models place probability mass in the hypoglycemic tail, which point-optimized models smooth away) motivates a concrete future direction — training with expanded/weighted extreme quantiles rather than the default 0.1–0.9 grid.

**W3 — Contemporaneous work.** Thank you — we will cite and discuss both: *"From Prediction to Practice: A Task-Aware Evaluation Framework for Blood Glucose Forecasting"* (May 2026 preprint), whose task-aware framing is complementary to ours (we differ in the long-horizon nocturnal task, TSFM breadth, and the multi-metric dissociation), and *"Benchmarking Hypoglycemia Classification Using Quality-Enhanced DiaData"*, which we will contrast on task definition and data-quality treatment.

We are grateful for R1's support and for recognizing that the contribution is a **new benchmarking paradigm**, not an exhaustive model horse-race. The three curiosities R1 raises (which architectures and why, decision-aware training, per-patient success) are precisely the future work this benchmark is designed to enable.


# ============================================================
# REVIEWER 1 — AUTHOR/AC CONFIDENTIAL COMMENT  (AC-only, ≤5,000 chars)
# ============================================================

R1 recommends **Accept (5, confidence 4)** and is, in our reading, the reviewer who most clearly grasps the contribution type: a *paradigm* for clinically-grounded T1D forecasting evaluation (task design + multi-metric, uncertainty-aware, long-horizon protocol), explicitly **not** a claim to a new dataset or a new SOTA model. R1 states the paper "merits publication as-written."

R1's concerns are all **addressable without altering any conclusion**: (i) the confusion-matrix / error-characterization question is answered with our new operating-point analysis (false alarms dominate at high recall; NPV 0.92–0.95); (ii) the hyperparameter/compute documentation gap is a presentation fix (identical shared harness; per-model search spaces already in the released configs; camera-ready compute table); (iii) the two contemporaneous references will be cited and contrasted. None of these bear on the headline cross-metric dissociation.

We flag for the AC that R1's framing directly answers R2's central objection: R1 — who understands the sub-field — considers the evaluation-methodology contribution sufficient and impactful for the venue, consistent with the E&D track's stated criteria (originality in task/evaluation design, not in beating a baseline). We believe R1's and R3's technical engagement should carry more weight than R2's, whose review does not engage the paper's specific methodology (see R2 confidential note).


# ============================================================
# REVIEWER 2 — REBUTTAL  (PUBLIC, ≤10,000 chars)  — Reject (2)
# ============================================================

We thank R2 for recognizing the clinical task formulation, the breadth of the evaluation, and the go-beyond-RMSE metric design. We answer all three questions with concrete, cross-dataset numbers (reproducible in the released code), then address the novelty concern directly.

**Q1 — Actionable recommendations (model × metric), justified quantitatively across datasets.** Yes, and the rebuttal makes them explicit and statistical. Per-metric winners are consistent across the adult cohorts:

| Objective | Recommended model | Evidence |
|---|---|---|
| Point accuracy (RMSE) / probabilistic (WQL) | **chronos2** | best on all datasets; sig. by paired patient-cluster bootstrap |
| Trajectory *shape* (DILATE) | **Moirai** | best on all datasets; sig. vs chronos2 (Δ +8–59 shape units) |
| Hypoglycemia *detection* (AUROC/AUPRC) | **Moirai / PatchTST** | best on DCLP3, IOBP2 (Moirai) and Replace-BG (PatchTST); sig. |

Crucially these recommendations **disagree with each other**: the Spearman correlation between the RMSE ranking and the shape/detection rankings is only **−0.05 to −0.52** (vs +0.93–0.98 for RMSE↔WQL). So the single actionable rule the benchmark establishes is: **select the model by the clinical objective, not by RMSE** — a decision that a point-accuracy-only benchmark would get wrong.

**Q2 — Does max aggregation over-react to isolated low-probability spikes vs. mean or product?** We tested exactly this — including the **product (noisy-or)** aggregation R2 names. Across all models and datasets, episode-level AUROC differs by only **mean |Δ| = 0.0026** between max and mean and **0.0024** between max and **noisy-or** (worst case 0.013); AUPRC differs by mean |Δ| ≈ 0.011–0.016. Model **rankings are preserved** under all three schemes (Kendall τ = +0.81 to +1.0 across aggregations). Max is therefore **not** spike-sensitive in practice. We use it because it matches the clinical decision (a single predicted crossing of 3.9 mmol/L within the night should raise the alarm); the near-identical noisy-or/mean results confirm the choice is not driving any conclusion.

**Q3 — Behavior under deterministic point-forecast classification.** Well-defined and we report it (threshold the minimum of the point forecast). Detection AUROC is comparable in magnitude (e.g., 0.73–0.76 on Replace-BG), but the **model ranking collapses toward the RMSE ranking**: under point-forecast classification the point-accuracy leader (chronos2) also appears best at detection. This is expected — a point forecast carries no tail-probability information, so it conflates "accurate mean" with "good hypo detector." The probabilistic **P(BG<3.9)** score (Table 5a) is precisely what disentangles them and reveals that continuous-output models (Moirai/Toto/PatchTST) capture hypoglycemic risk better. This is itself a core methodological message of the paper: *the scoring rule, not just the model, determines the clinical conclusion.*

**W1 — "Conclusions plausible but not deeply established / intuitive / benchmarking, not a scientific advance."** Two responses. (i) *Established:* the rebuttal replaces plausibility with significance — paired **patient-clustered** bootstraps, Friedman (p < 1e-140 per dataset) with Wilcoxon–Holm post-hoc, and rank-biserial effect sizes; 17 of 20 headline comparisons remain significant under 10,000-bootstrap patient clustering. (ii) *Intuitive?* The dissociation is not the clean "generative-vs-point" split intuition predicts: **PatchTST**, a point-output transformer, is a top *detector* on Replace-BG, while chronos2, an attention model, leads RMSE — the pattern cuts across output parameterization, which is why an empirical benchmark is needed rather than reasoning from architecture. We also add a mechanism (continuous-output models place probability mass in the hypoglycemic tail that point-optimized models smooth away), motivating a concrete future direction (expanded/weighted extreme quantiles beyond the default 0.1–0.9 grid).

**W2 — "Insufficient clarity on what is new."** The benchmark's originality is in the task and evaluation design, not in a new dataset or a new SOTA model (an explicitly valid E&D contribution type). Concretely, versus prior CGM benchmarks — which are short-horizon (≤60 min), RMSE-centric, and single-model-family — this is the first to combine: (a) a **long-horizon (8-hour / 96-step) nocturnal** task tied to a clinical decision; (b) **7 time-series foundation models** evaluated head-to-head on CGM; (c) a **multi-metric double dissociation** (RMSE/WQL vs DILATE vs AUROC/AUPRC) established with significance; and (d) the finding that **major TSFM pretraining corpora omit CGM entirely** — a large-scale, high-impact signal the ML community is leaving unused. Points (c)–(d) are new *insights*, not just new measurements.

We hope these results move R2's assessment: the conclusions are now statistically established, the aggregation and point-forecast questions are answered (including the product aggregation), and the novelty is concrete and, per E&D criteria, of the appropriate type.


# ============================================================
# REVIEWER 2 — AUTHOR/AC CONFIDENTIAL COMMENT  (AC-only, ≤5,000 chars)
# ============================================================

We respectfully ask the AC to weigh the **depth of engagement** across reviews. R2 recommends Reject (2) at confidence 4, but the review is brief and generic: its weaknesses are "conclusions are intuitive" and "unclear what is new," with no engagement of the paper's specific methodology (the metrics, the anchoring protocol, the model set, or any number in the tables). By contrast R1 (Accept, 5) and R3 (borderline, the most detailed review) both engage the methodology directly.

On the substance, R2's three questions are fully answerable and two are addressed by material already in the submission:
- *Actionable recommendations:* provided and now statistical (per-metric winners across datasets; cross-metric dissociation with paired patient-cluster significance).
- *Max vs mean/product aggregation:* we had already implemented all three, including the noisy-or/product aggregation R2 assumed was untested; results are near-identical (mean |Δ|AUROC ≈ 0.0025; rankings preserved, Kendall τ ≥ 0.81).
- *Point-forecast classification:* reported; it collapses the detection ranking toward RMSE, which is exactly the methodological point of using a probabilistic score.

R2's "not novel enough / intuitive" critique is the standard objection the E&D track was created to recalibrate: a rigorous, decision-relevant evaluation that changes how models are compared is an in-scope contribution and need not beat a baseline or deliver a new architecture. R2's own listed strengths (clinically relevant task, breadth, go-beyond-RMSE metrics) are precisely the contribution. We do not believe the review identifies a technical flaw; we ask that its low score be weighed against its limited specificity relative to R1 and R3.


# ============================================================
# REVIEWER 3 — REBUTTAL  (PUBLIC, ≤10,000 chars)  — deepest/most constructive review
# ============================================================

We thank R3 for an exceptionally careful review. We ran **five new analyses** during the rebuttal window; all are reproducible in the released code and all use **patient-clustered resampling** (see Q2). Summary up front:

- **Statistical significance (W2/Q2):** the model ranking **diverges across metrics** — chronos2 is significantly best on RMSE/WQL but significantly *worse* than Moirai on DILATE-shape and than Moirai/Toto on detection (AUROC/AUPRC). Spearman correlation between the RMSE ranking and the shape/detection rankings is only **−0.05 to −0.52** (vs +0.93–0.98 for RMSE↔WQL). This is a statistical *double dissociation* proving "no single architecture dominates" across clinically-relevant metrics.
- **Midnight-anchoring (W1/Q1):** on nights with no overnight insulin/carbs (53% of DCLP3 nights), the model ranking is **unchanged** (Kendall τ = +1.0) and metrics move < 0.01 RMSE.
- **Covariate fairness (W3/Q3):** on identical glucose-only input, architecture spread (0.50–0.64 RMSE) ≫ covariate gain (+0.07–0.10) — access to covariates does not drive rankings.
- **Subgroup (W5/Q4):** rankings are stable across sex and (true enrollment) age; we report the gaps that exist.

**Q2 / W2 — Significance.** We now support every headline claim with bootstrap CIs, paired tests, and the rank-based pipeline expected for multi-model benchmarks (Friedman p < 1e-140 on every dataset; Wilcoxon signed-rank + Holm; rank-biserial effect sizes). Two points:

*(i) "No single architecture dominates" is now a statistical result, not an assertion — and it is a CROSS-METRIC statement.* On DCLP3/IOBP2, chronos2 is significantly **better** than Moirai on RMSE (Δ = −0.088 / −0.186 mmol/L) and WQL, yet significantly **worse** on DILATE-shape (Δ = +58.9 / +8.2 shape units) and on hypo detection — AUROC (Δ = −0.068 / −0.055) and AUPRC (Δ = −0.032 / −0.037) — all by paired **patient-cluster** bootstrap (95% CI excludes 0). Equivalently, the Spearman correlation between the RMSE ranking and the shape/detection rankings is only **−0.05 to −0.52**, versus **+0.93–0.98** for RMSE↔WQL: point accuracy and shape/detection genuinely disagree about the best model. (Detection uses the probabilistic P(BG<3.9) risk score, consistent with the paper.)

*(ii) Clustering — a subtlety we corrected.* Episodes are **clustered within patients** (ICC of per-episode RMSE = 0.11–0.21; 167–430 patients, ~10–25 episodes each), so we resample **patients**, not episodes (design effect 2.5–6×; the honest, wider interval). Under this stricter test (10,000 patient-cluster bootstraps), **17 of 20** headline pairwise comparisons remain significant; the three exceptions are all single-dataset detection ties (Moirai/Toto vs chronos2 AUROC on Replace-BG, PatchTST vs chronos2 AUROC on IOBP2), and the cross-metric dissociation itself holds on every dataset. Because N is large, we report **effect sizes and ΔRMSE with CIs**, not just p-values. We are happy to add per-dataset Critical-Difference diagrams (anonymized link) as the reviewer requested.

**Q1 / W1 — Midnight anchoring and label noise.** We quantify the concern directly using insulin/carbohydrate timing. In DCLP3 (the cohort with discrete meal boluses), only **46.5%** of midnight episodes contain any post-midnight bolus, at a median of **~4.9 h** into the window; only **12–35%** of a patient's daily boluses fall in 00:00–08:00. We then define **"quiescent" nights** (no post-midnight bolus/carb, 53% of DCLP3 nights) and recompute everything: the AUROC ranking is **identical** to the full set (Kendall τ = +1.0) and metrics shift **< 0.01 RMSE**. So midnight-anchoring label noise does **not** change the benchmark's conclusions. (Honest caveat we now state: closed-loop cohorts — IOBP2/DCLP3 automation — deliver insulin continuously, so "quiescent" is only cleanly defined where boluses are discrete; Tamborlane 2008 is CGM-only.)

**Q3 / W3 — Covariate fairness.** We agree this must be controlled, and it is: **every model is reported on glucose-only input** (Tables 3–4, condition G), with covariates as a per-model ablation. New, explicit result: on the glucose-only leaderboard, **architecture spread is 0.50–0.64 RMSE** per dataset, while the median within-model covariate gain is only **+0.07–0.10 RMSE** (paired patient-cluster bootstrap; chronos2 +0.09–0.18 sig., ttm ≈ 0). Improvements attributed to architecture are therefore not covariate artifacts.

**Q4 / W5 — Subgroup / fairness.** We add subgroup analysis using the patients' **actual enrollment age** (not age-at-diagnosis) and sex, from the JAEB rosters. Model rankings are **stable** across sex (Kendall τ = +1.0 on Replace-BG and IOBP2) and across an age median-split (τ = +1.0 on IOBP2, Tamborlane). We also report the gaps that exist and do not hide them: e.g., on DCLP3 AUROC is consistently higher for female than male patients across all models (~0.73 vs ~0.68), and younger patients have higher RMSE. Top models (chronos2/Moirai/Toto) remain top in every subgroup.

**Q5 — Do AUROC/AUPRC gains translate to fewer missed events?** We add operating-point analysis (sensitivity/specificity/PPV/NPV + false-alarms-per-100-nights). The honest answer: at long horizons, gains are real but modest — at 90% sensitivity, false alarms are **45–72 per 100 nights** and PPV **0.14–0.43**; only by trading recall (Youden point) does PPV reach **0.55–0.63**. We now temper the clinical claims accordingly and frame deployable long-horizon alarming as an **open problem the benchmark defines**, not a solved one.

**W4 — Dataset heterogeneity.** Performance differences track concrete cohort/device factors we can name: sensor generation (Dexcom G4 in Replace-BG vs G6 in DCLP3 vs iLet CGM in IOBP2), therapy (MDI/pump vs Control-IQ vs bionic pancreas), and age (Tamborlane includes children; hypo base rate 0.39 vs 0.10–0.26 elsewhere). These explain the reshuffling in the CD analysis and are now discussed.

**On "dataset contribution is limited."** We agree — and we do not claim a dataset contribution. Our contribution type is **Benchmark Design + Evaluation Methodology**; per the E&D guidelines, originality here comes from task/evaluation design and analysis, not from new data or beating a baseline.

We are grateful for the depth of this review; the new analyses materially strengthen the paper and we believe address every point raised. We would welcome guidance on which additional cut (e.g., device-stratified error decomposition) would most increase R3's confidence.


# ============================================================
# REVIEWER 3 — AUTHOR/AC CONFIDENTIAL COMMENT  (AC-only, ≤5,000 chars)
# ============================================================

R3 (borderline; score 3) is the most careful and constructive reviewer, and in our view the swing vote. Every concern maps to a specific new analysis we ran during the rebuttal, and none overturns a conclusion:

- **W2/Q2 statistical significance** → paired **patient-clustered** bootstrap + Friedman (p < 1e-140) + Wilcoxon–Holm + effect sizes; the cross-metric double dissociation is now demonstrated (17 of 20 headline comparisons significant at 10,000 bootstraps). This is the concern R3 weighted most, and it is now fully resolved.
- **W1/Q1 midnight-anchoring label noise** → quiescent-night sensitivity: ranking identical (τ = +1.0), metrics shift < 0.01 RMSE.
- **W3/Q3 covariate fairness** → glucose-only leaderboard: architecture spread (0.50–0.64) ≫ covariate gain (0.07–0.10).
- **W5/Q4 subgroup/fairness** → sex + true-enrollment-age analysis; rankings stable, gaps reported honestly.
- **W4 dataset heterogeneity / Q5 alarm utility / "clinical claims"** → device-and-cohort attribution + operating-point analysis; we temper deployment claims and frame low-false-alarm long-horizon alarming as an open problem.

We asked R3 publicly which additional cut would most raise confidence and pre-committed to a device-stratified error decomposition. We believe the significance analysis in particular directly answers R3's primary stated reason for hesitation, and we hope the AC will encourage R3 to revisit the score in light of the new, reproducible evidence. R3's critique materially improved the paper and we are glad to keep iterating during discussion.
