We thank Reviewer xU1c for recognizing the clinical task, the breadth of evaluation, and that we go beyond RMSE metrics. We answer all three questions with cross-dataset numbers, then the novelty concern.

**Q1: Actionable recommendations (model + metric), across datasets.** Yes, and we added statistical support based on Rev. RCHh's concerns. Per-metric winners are consistent across cohorts:

| Objective | Recommended model | Evidence |
|---|---|---|
| Accuracy (RMSE) / probabilistic (WQL) | **chronos2** | best on 3/4 datasets (PatchTST wins Tam.); sig. (paired patient-cluster bootstrap) |
| Trajectory *shape* (DILATE) | **Moirai** | best on all datasets; sig. vs chronos2 ranges (+8–59) |
| Hypo *detection* (ROC/PRC) | **Moirai / PatchTST** | Moirai on DCLP3/IOBP2, PatchTST on Replace-BG; sig. |

These recommendations **disagree**: Spearman between the RMSE ranking and the shape/detection rankings is only **−0.05 to −0.52** (vs +0.93–0.98 for RMSE to WQL). The actionable recommendation: **choose the model by the clinical objective, not by RMSE**.

**Q2: Max vs mean vs product aggregation.** We tested all three. Across all 7 prob. models + 4 datasets, swapping max for either alternative shifts episode AUROC by **mean |diff| ≈ 0.0025** (worst 0.013) and AUPRC by **mean |diff| ≈ 0.011–0.016** (worst 0.06). **AUROC model rankings are preserved on every dataset** (Kendall τ = +0.81 to +1.0). AUPRC rankings also agree closely on 3 of 4 datasets (τ = +0.81 to +1.0); the exception is DCLP3, where several models trade adjacent positions within a narrow AUPRC band (τ ≈ +0.52 for max vs mean). Max does **not** appear spike-sensitive. We also used max because it more closely resembles the clinical decision (a single predicted crossing of 3.9 mmol/L should trigger the alarm), and the near-identical mean results support that it drives no conclusion. We also felt product aggregation might not be appropriate here because it would assume step-wise independence.

**Q3: Deterministic point-forecast classification.** We assessed this for you by ranking under $score = -\min_t \mu_t$. As expected, it shifts winners back to strong point forecasters, but its threshold (e.g. "alert if min predicted BG < 4.5 mmol/L") is less interpretable than a distribution and gives weaker decision support around correction dose (5g vs 15g) or absorption speed (dextrose vs fruit). Empirically (AUROC), a head-to-head between our RMSE winner (chronos2) and our probabilistic-detection winner (Moirai), each with its best run per dataset:

| Dataset | chronos2 point | chronos2 prob | Moirai point | Moirai prob |
|---|---|---|---|---|
| Replace-BG | **0.756** | 0.712 | 0.752 | 0.727 |
| DCLP3 | 0.703 | 0.652 | 0.698 | **0.720** |
| IOBP2 | 0.670 | 0.650 | 0.656 | **0.705** |
| Tamborlane | 0.680 | 0.651 | 0.675 | **0.685** |

Under point classification chronos2 appears to be an improvement on Replace-BG dataset, but does not appear to be as robust across datasets. These results may not hold if greater emphasis was placed on modeling the tails of our distributions during fine-tuning. Further investigation is warranted.

**W1: Not deeply established."**
(i) *Established:* we added paired significance testing with patient-cluster bootstraps (10,000 resamples), Friedman + Wilcoxon–Holm + rank-biserial effect sizes. chronos2 significantly beats PatchTST on RMSE on 3/4 datasets (PatchTST wins Tamborlane by 0.029 mmol/L, sig.); Moirai significantly beats chronos2 on AUPRC on 4/4 and on AUROC on 3/4; PatchTST beats chronos2 on AUROC on 3/4. We can further expand on this analysis for camera ready.
(ii) *Causal Analysis* The mechanism (continuous-output models place mass in the hypo tail that point-optimized models smooth away) could motivate a concrete direction: expanded/weighted extreme quantiles beyond the default 0.1–0.9 grid in future released TSFMs.

**W2: "What is new."** We agree the framing can be clearer, and thank you for identifying this issue. Versus prior CGM benchmarks **(short-horizon 15-120 min, RMSE-centric)**, this is the first to combine: (a) a **long-horizon 8-h/96-step nocturnal** task tied to a clinical decision; (b) **7 TSFMs** directly compared on CGM data; (c) a **multi-metric double dissociation** (now with significance); (d) the finding that **major TSFM pretraining corpora omit CGM entirely** and (e) **long context windows** our ablation in the appendix demonstrates the value of having longer context windows, largely ignored in other CGM studies. We also believe our formulation is likely to lead to more interpretable and trustworthy tools for patients, but would need to be evaluated in clinical settings after our work inspires new architectures that can win across metrics. We will state these explicitly in camera-ready.

We hope this moves your assessment: the conclusions will now be statistically established, the aggregation and point-forecast questions answered, and the novelty more concrete and of the appropriate E&D type.
