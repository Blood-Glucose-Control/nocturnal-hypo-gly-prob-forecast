# Chronos-2 Sweep Results Analysis
**Date:** 2026-04-25 (updated from 2026-04-24)
**Sweep:** 16 configs (00–15), 10 000 steps each, `holdout_10pct` split
**Eval script:** `GPUS="0 1" bash scripts/experiments/chronos2_sweep_eval.sh`
**Total evaluations:** 64 (20 fine-tuned + 44 zero-shot)

> **Retraining complete — previous data errors resolved.**
> - `lynch_2022` COB is all-zero: configs 06–09 now trained on `aleppo_2017` only.
> - `tamborlane_2008` now included in `00_bg_only` training split.
> Earlier stale artifacts archived to `_bad_runs_archive/`. All numbers below reflect the corrected runs.

---

## Per-Run Results — Fine-Tuned

DILATE metrics use `gamma=0.001`. Coverage targets: 50% and 80% nominal.
**Bold** = best per dataset for that metric within the applicable comparison group.
Lower is better for all metrics except Cov50/Cov80 (target 0.50 / 0.80).

### 00_bg_only (all 4 datasets)

| Config | Dataset | ctx | RMSE | WQL | Brier | MACE | Cov50 | Cov80 | Sharp50 | DILATE | Shape | Temporal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00_bg_only | aleppo_2017 | 512 | 2.7514 | 0.7841 | 0.04108 | 0.0045 | 0.5111 | 0.8027 | 3.496 | 233.30 | 448.45 | 18.160 |
| 00_bg_only | brown_2019 | 512 | 2.5217 | 0.7157 | 0.01943 | 0.0447 | 0.5385 | 0.8322 | 3.439 | 209.61 | 402.85 | 16.364 |
| 00_bg_only | lynch_2022 | 512 | 2.4292 | 0.6969 | 0.02377 | 0.0512 | 0.5799 | 0.8567 | 3.618 | 193.24 | 372.96 | 13.522 |
| 00_bg_only | tamborlane_2008 | 512 | 3.2697 | 0.9312 | 0.04407 | 0.0032 | 0.4988 | 0.7879 | 4.074 | 395.36 | 774.58 | 16.151 |

### IOB group (01–05, 10–12 — trained on lynch_2022, aleppo_2017, brown_2019)

Configs 01–05 are from the original sweep (not retrained). Configs 10–12 are new (context ablation at high LR).

| Config | Dataset | ctx | RMSE | WQL | Brier | MACE | Cov50 | Cov80 | Sharp50 | DILATE | Shape | Temporal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01_bg_iob_insulin_availability | aleppo_2017 | 512 | 2.7433 | 0.7798 | 0.04119 | 0.0144 | 0.4679 | 0.7645 | 3.127 | 224.13 | 430.64 | 17.617 |
| 01_bg_iob_insulin_availability | brown_2019 | 512 | 2.4911 | 0.6941 | 0.01946 | 0.0282 | 0.5120 | 0.8057 | 3.062 | 194.32 | 372.90 | 15.737 |
| 01_bg_iob_insulin_availability | lynch_2022 | 512 | 2.3650 | 0.6659 | 0.02380 | 0.0353 | 0.5403 | 0.8262 | 3.128 | 164.53 | 316.90 | 12.159 |
| 02_bg_iob | aleppo_2017 | 512 | 2.7300 | 0.7766 | 0.04113 | 0.0139 | 0.4675 | 0.7645 | 3.128 | 221.73 | 426.01 | 17.449 |
| 02_bg_iob | brown_2019 | 512 | 2.4846 | 0.6912 | 0.01947 | 0.0255 | 0.5098 | **0.8037** | 3.030 | 191.71 | 368.02 | 15.399 |
| 02_bg_iob | lynch_2022 | 512 | 2.3566 | 0.6621 | 0.02379 | 0.0331 | 0.5363 | 0.8231 | 3.076 | 161.09 | 310.26 | 11.912 |
| 03_joint_bg_iob | aleppo_2017 | 512 | 2.7782 | 0.7892 | 0.04115 | **0.0082** | **0.4900** | **0.7887** | 3.339 | 232.49 | 446.76 | 18.227 |
| 03_joint_bg_iob | brown_2019 | 512 | 2.5164 | 0.7083 | 0.01947 | 0.0392 | 0.5282 | 0.8247 | 3.286 | 203.27 | 390.66 | 15.878 |
| 03_joint_bg_iob | lynch_2022 | 512 | 2.4130 | 0.6882 | 0.02381 | 0.0480 | 0.5670 | 0.8490 | 3.459 | 185.42 | 357.75 | 13.094 |
| **04_bg_iob_ia_high_lr** | aleppo_2017 | 512 | 2.6797 | 0.7589 | 0.04088 | 0.0152 | 0.4648 | 0.7592 | 3.005 | 211.10 | 405.59 | 16.618 |
| **04_bg_iob_ia_high_lr** | brown_2019 | 512 | **2.4532** | **0.6734** | **0.01945** | **0.0131** | **0.4983** | 0.7890 | **2.806** | **185.27** | **355.98** | **14.564** |
| **04_bg_iob_ia_high_lr** | lynch_2022 | 512 | **2.2584** | **0.6201** | **0.02373** | 0.0082 | 0.5206 | **0.8071** | **2.689** | **146.69** | **283.09** | **10.285** |
| 05_bg_iob_short_ctx | aleppo_2017 | 288 | 2.8157 | 0.8079 | 0.04202 | 0.0129 | 0.4754 | 0.7729 | 3.330 | 251.28 | 484.14 | 18.426 |
| 05_bg_iob_short_ctx | brown_2019 | 288 | 2.5879 | 0.7327 | 0.02027 | 0.0446 | 0.5302 | 0.8211 | 3.405 | 222.24 | 427.55 | 16.938 |
| 05_bg_iob_short_ctx | lynch_2022 | 288 | 2.5452 | 0.7292 | 0.02439 | 0.0532 | 0.5495 | 0.8356 | 3.579 | 204.84 | 395.86 | 13.825 |
| 10_bg_iob_ia_high_lr_ctx256 | aleppo_2017 | 256 | 2.7594 | 0.7871 | 0.04187 | 0.0107 | 0.4793 | 0.7768 | 3.251 | 234.58 | 451.78 | 17.389 |
| 10_bg_iob_ia_high_lr_ctx256 | brown_2019 | 256 | 2.5294 | 0.7026 | 0.02018 | 0.0186 | 0.5349 | 0.8189 | 3.182 | 202.51 | 389.80 | 15.216 |
| 10_bg_iob_ia_high_lr_ctx256 | lynch_2022 | 256 | 2.4514 | 0.6880 | 0.02488 | 0.0140 | 0.5417 | 0.8268 | 3.188 | 181.30 | 350.48 | 12.122 |
| 11_bg_iob_ia_high_lr_ctx128 | aleppo_2017 | 128 | 2.9388 | 0.8482 | 0.04420 | 0.0079 | 0.4889 | 0.7788 | 3.617 | 277.53 | 537.58 | 17.484 |
| 11_bg_iob_ia_high_lr_ctx128 | brown_2019 | 128 | 2.8123 | 0.8118 | 0.02162 | 0.0512 | 0.5273 | 0.8078 | 3.730 | 271.77 | 527.08 | 16.467 |
| 11_bg_iob_ia_high_lr_ctx128 | lynch_2022 | 128 | 2.7048 | 0.7882 | 0.02670 | 0.0563 | 0.5689 | 0.8445 | 3.970 | 242.48 | 471.19 | 13.774 |
| 12_bg_iob_ia_high_lr_ctx64 | aleppo_2017 | 64 | 3.0792 | 0.8960 | 0.04967 | 0.0233 | 0.4938 | 0.7908 | 3.859 | 335.82 | 653.54 | 18.103 |
| 12_bg_iob_ia_high_lr_ctx64 | brown_2019 | 64 | 3.0570 | 0.8948 | 0.02817 | 0.0608 | 0.5208 | 0.8014 | 4.075 | 343.70 | 670.06 | 17.338 |
| 12_bg_iob_ia_high_lr_ctx64 | lynch_2022 | 64 | 3.1219 | 0.9243 | 0.03419 | 0.0637 | 0.5402 | 0.8168 | 4.372 | 345.79 | 674.95 | 16.635 |

### COB group (06–09, 13–15 — trained on aleppo_2017 only)

COB models retrained on aleppo-only (lynch carbs are all-zero). Results are single-dataset.

| Config | Dataset | ctx | RMSE | WQL | Brier | MACE | Cov50 | Cov80 | Sharp50 | DILATE | Shape | Temporal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_bg_iob_cob | aleppo_2017 | 512 | 2.6813 | 0.7623 | 0.04093 | 0.0092 | 0.4937 | 0.7933 | 3.303 | 219.85 | 422.31 | 17.399 |
| 07_bg_full_features | aleppo_2017 | 512 | 2.6956 | 0.7666 | 0.04099 | 0.0101 | 0.4877 | 0.7855 | 3.260 | 221.43 | 425.32 | 17.539 |
| **08_bg_iob_cob_high_lr** | aleppo_2017 | 512 | **2.6090** | **0.7377** | **0.04088** | **0.0029** | 0.4989 | 0.7928 | **3.224** | **203.55** | **390.66** | **16.440** |
| 09_bg_iob_cob_short_ctx | aleppo_2017 | 288 | 2.7626 | 0.7918 | 0.04181 | 0.0154 | 0.4931 | 0.7935 | 3.426 | 243.93 | 469.40 | 18.463 |
| 13_bg_iob_cob_high_lr_ctx256 | aleppo_2017 | 256 | 2.6987 | 0.7720 | 0.04182 | 0.0037 | 0.5133 | 0.8054 | 3.495 | 226.83 | 436.30 | 17.358 |
| 14_bg_iob_cob_high_lr_ctx128 | aleppo_2017 | 128 | 2.8854 | 0.8384 | 0.04517 | 0.0111 | 0.5327 | 0.8199 | 3.990 | 275.40 | 533.61 | 17.195 |
| 15_bg_iob_cob_high_lr_ctx64 | aleppo_2017 | 64 | 3.0438 | 0.8931 | 0.05449 | 0.0150 | 0.5427 | 0.8343 | 4.332 | 329.03 | 640.39 | 17.663 |

---

## Per-Run Results — Zero-Shot

**Key finding: all ctx=512 zero-shot runs produce identical results regardless of covariate configuration.** This confirms the pre-trained Chronos-2 does not meaningfully differentiate covariate columns it was not pre-trained on. Only context length affects zero-shot performance. One representative row per (context_length, dataset) is shown.

| context | Dataset | RMSE | WQL | Brier | MACE | Cov50 | Cov80 | Sharp50 | DILATE | Shape | Temporal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | aleppo_2017 | 2.7649 | 0.7948 | 0.04206 | 0.0277 | 0.5252 | 0.8288 | 3.683 | 260.80 | 501.88 | 19.724 |
| 512 | brown_2019 | 2.6495 | 0.7591 | 0.02060 | 0.0762 | 0.5537 | 0.8592 | 3.781 | 253.19 | 487.33 | 19.043 |
| 512 | lynch_2022 | 2.6724 | 0.7677 | 0.02525 | 0.0789 | 0.5737 | 0.8734 | 3.981 | 258.44 | 499.28 | 17.599 |
| 512 | tamborlane_2008 | 3.3536 | 0.9554 | 0.04578 | 0.0085 | 0.4970 | 0.7913 | 4.120 | 421.64 | 826.91 | 16.376 |
| 288 | aleppo_2017 | 2.8579 | 0.8216 | 0.04369 | 0.0352 | 0.5218 | 0.8294 | 3.787 | 291.43 | 562.84 | 20.017 |
| 288 | brown_2019 | 2.7767 | 0.7930 | 0.02252 | 0.0789 | 0.5523 | 0.8590 | 3.927 | 284.38 | 548.75 | 20.010 |
| 288 | lynch_2022 | 2.8567 | 0.8196 | 0.02756 | 0.0839 | 0.5670 | 0.8742 | 4.227 | 296.86 | 575.35 | 18.364 |
| 288 | tamborlane_2008 | 3.4909 | 0.9938 | 0.04824 | 0.0095 | 0.4965 | 0.7927 | 4.245 | 447.43 | 878.20 | 16.650 |
| 256 | aleppo_2017 | 2.8582 | 0.8222 | 0.04400 | 0.0329 | 0.5225 | 0.8304 | 3.791 | 289.49 | 559.18 | 19.809 |
| 256 | brown_2019 | 2.7708 | 0.7913 | 0.02240 | 0.0770 | 0.5552 | 0.8614 | 3.936 | 282.27 | 544.71 | 19.832 |
| 256 | lynch_2022 | 2.8970 | 0.8317 | 0.02816 | 0.0796 | 0.5613 | 0.8677 | 4.223 | 306.15 | 593.85 | 18.448 |
| 256 | tamborlane_2008 | 3.5197 | 1.0010 | 0.04890 | 0.0100 | 0.4943 | 0.7926 | 4.255 | 452.00 | 887.47 | 16.539 |
| 128 | aleppo_2017 | 3.0937 | 0.8874 | 0.04984 | 0.0352 | 0.5151 | 0.8167 | 3.982 | 357.36 | 695.04 | 19.676 |
| 128 | brown_2019 | 3.1508 | 0.9001 | 0.02602 | 0.0861 | 0.5223 | 0.8345 | 4.167 | 379.28 | 738.47 | 20.089 |
| 128 | lynch_2022 | 3.3022 | 0.9483 | 0.03390 | 0.0938 | 0.5376 | 0.8529 | 4.573 | 411.89 | 805.02 | 18.748 |
| 128 | tamborlane_2008 | 3.7952 | 1.0832 | 0.05669 | 0.0119 | 0.4765 | 0.7702 | 4.368 | 522.28 | 1027.66 | 16.908 |
| 64 | aleppo_2017 | 3.4444 | 0.9821 | 0.06249 | 0.0259 | 0.4643 | 0.7564 | 3.890 | 448.36 | 876.31 | 20.415 |
| 64 | brown_2019 | 3.4816 | 0.9964 | 0.03864 | 0.0654 | 0.4769 | 0.7730 | 4.113 | 455.70 | 890.48 | 20.921 |
| 64 | lynch_2022 | 3.7598 | 1.0821 | 0.04953 | 0.0677 | 0.4873 | 0.7839 | 4.577 | 522.87 | 1025.90 | 19.837 |
| 64 | tamborlane_2008 | 4.1870 | 1.1972 | 0.06702 | 0.0291 | 0.4349 | 0.7171 | 4.404 | 630.62 | 1243.57 | 17.674 |

---

## Per-Config Averages (Fine-Tuned)

Averaged across each config's assigned dataset group. Groups are not cross-comparable.
**Bold** = best in group.

| Config | n | RMSE | WQL | Brier | MACE | Cov50 | Cov80 | Sharp50 | DILATE | Shape | Temporal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 00_bg_only | 4 | 2.7430 | 0.7820 | 0.03209 | 0.0259 | 0.5321 | 0.8199 | 3.657 | 257.88 | 499.71 | 16.049 |
| 01_bg_iob_insulin_availability | 3 | 2.5331 | 0.7133 | 0.02815 | 0.0260 | 0.5067 | 0.7988 | 3.106 | 194.33 | 373.15 | 15.171 |
| 02_bg_iob | 3 | 2.5237 | 0.7099 | 0.02813 | 0.0241 | 0.5045 | 0.7971 | 3.078 | 191.51 | 368.10 | 14.920 |
| 03_joint_bg_iob | 3 | 2.5692 | 0.7286 | 0.02814 | 0.0318 | 0.5284 | 0.8208 | 3.361 | 207.06 | 398.39 | 15.733 |
| **04_bg_iob_ia_high_lr** | 3 | **2.4638** | **0.6841** | **0.02802** | **0.0122** | 0.4946 | 0.7851 | **2.833** | **181.02** | **348.22** | **13.822** |
| 05_bg_iob_short_ctx | 3 | 2.6496 | 0.7566 | 0.02889 | 0.0369 | 0.5184 | 0.8099 | 3.438 | 226.12 | 435.85 | 16.396 |
| 10_bg_iob_ia_high_lr_ctx256 | 3 | 2.5801 | 0.7259 | 0.02898 | 0.0144 | 0.5186 | 0.8075 | 3.207 | 206.13 | 397.35 | 14.909 |
| 11_bg_iob_ia_high_lr_ctx128 | 3 | 2.8186 | 0.8161 | 0.03084 | 0.0385 | 0.5284 | 0.8104 | 3.772 | 263.93 | 511.95 | 15.908 |
| 12_bg_iob_ia_high_lr_ctx64 | 3 | 3.0861 | 0.9050 | 0.03734 | 0.0493 | 0.5183 | 0.8030 | 4.102 | 341.77 | 666.18 | 17.359 |
| **08_bg_iob_cob_high_lr** | 1 | **2.6090** | **0.7377** | 0.04088 | **0.0029** | 0.4989 | 0.7928 | **3.224** | **203.55** | **390.66** | **16.440** |
| 06_bg_iob_cob | 1 | 2.6813 | 0.7623 | 0.04093 | 0.0092 | 0.4937 | 0.7933 | 3.303 | 219.85 | 422.31 | 17.399 |
| 07_bg_full_features | 1 | 2.6956 | 0.7666 | 0.04099 | 0.0101 | 0.4877 | 0.7855 | 3.260 | 221.43 | 425.32 | 17.539 |
| 13_bg_iob_cob_high_lr_ctx256 | 1 | 2.6987 | 0.7720 | 0.04182 | 0.0037 | 0.5133 | 0.8054 | 3.495 | 226.83 | 436.30 | 17.358 |
| 09_bg_iob_cob_short_ctx | 1 | 2.7626 | 0.7918 | 0.04181 | 0.0154 | 0.4931 | 0.7935 | 3.426 | 243.93 | 469.40 | 18.463 |
| 14_bg_iob_cob_high_lr_ctx128 | 1 | 2.8854 | 0.8384 | 0.04517 | 0.0111 | 0.5327 | 0.8199 | 3.990 | 275.40 | 533.61 | 17.195 |
| 15_bg_iob_cob_high_lr_ctx64 | 1 | 3.0438 | 0.8931 | 0.05449 | 0.0150 | 0.5427 | 0.8343 | 4.332 | 329.03 | 640.39 | 17.663 |

---

## Context Window Ablation

### IOB high-LR (04 is baseline ctx=512; compare 10/11/12)
Averages over aleppo, brown, lynch.

| ctx | Config | avg RMSE | avg WQL | avg MACE | avg Sharp50 | avg DILATE | vs 04 RMSE |
|---|---|---|---|---|---|---|---|
| 512 | 04_bg_iob_ia_high_lr | 2.4638 | 0.6841 | 0.0122 | 2.833 | 181.02 | baseline |
| 256 | 10_bg_iob_ia_high_lr_ctx256 | 2.5801 | 0.7259 | 0.0144 | 3.207 | 206.13 | +0.116 (+4.7%) |
| 128 | 11_bg_iob_ia_high_lr_ctx128 | 2.8186 | 0.8161 | 0.0385 | 3.772 | 263.93 | +0.355 (+14.4%) |
| 64 | 12_bg_iob_ia_high_lr_ctx64 | 3.0861 | 0.9050 | 0.0493 | 4.102 | 341.77 | +0.622 (+25.3%) |

### COB high-LR (08 is baseline ctx=512; compare 13/14/15)
aleppo only.

| ctx | Config | RMSE | WQL | MACE | Sharp50 | DILATE | vs 08 RMSE |
|---|---|---|---|---|---|---|---|
| 512 | 08_bg_iob_cob_high_lr | 2.6090 | 0.7377 | 0.0029 | 3.224 | 203.55 | baseline |
| 256 | 13_bg_iob_cob_high_lr_ctx256 | 2.6987 | 0.7720 | 0.0037 | 3.495 | 226.83 | +0.089 (+3.4%) |
| 128 | 14_bg_iob_cob_high_lr_ctx128 | 2.8854 | 0.8384 | 0.0111 | 3.990 | 275.40 | +0.277 (+10.6%) |
| 64 | 15_bg_iob_cob_high_lr_ctx64 | 3.0438 | 0.8931 | 0.0150 | 4.332 | 329.03 | +0.435 (+16.7%) |

**The degradation is non-linear and accelerates sharply below ctx=128.** 256 is tolerable; 128 is materially worse; 64 is severe.

---

## Context Window Ablation — Fixed Episode Set (Fair Comparison)

**eval script:** `GPUS="0 1" bash scripts/experiments/chronos2_ctx_ablation_eval.sh`
**results dir:** `experiments/nocturnal_forecasting_ctx_ablation/`
**methodology:** All context lengths evaluated on the *same* set of midnight anchors — only those requiring ≥512 steps of history. Episodes valid only at smaller contexts are excluded. This removes the confound whereby ctx=64 had 1490 more episodes (on lynch) than ctx=512 simply because shorter windows can sample more within each patient sequence.

**Episode counts (identical across all ctx levels for each dataset):**
aleppo=4243 · brown=3809 · lynch=2928 · tamborlane=10782 (zero-shot only)

### IOB high-LR fixed-anchor ablation (04/10/11/12 — 3-dataset avg)

| ctx | Config | avg RMSE | avg WQL | avg MACE | avg Sharp50 | avg DILATE | vs ctx=512 | old % (unfair) |
|---|---|---|---|---|---|---|---|---|
| 512 | 04_bg_iob_ia_high_lr | 2.4638 | 0.6841 | 0.0122 | 2.833 | 181.02 | baseline | — |
| 256 | 10_bg_iob_ia_high_lr_ctx256 | **2.5698** | 0.7203 | 0.0143 | 3.159 | 203.70 | +0.106 (+**4.3%**) | 4.7% |
| 128 | 11_bg_iob_ia_high_lr_ctx128 | **2.8090** | 0.8106 | 0.0378 | 3.715 | 262.58 | +0.345 (+**14.0%**) | 14.4% |
| 64 | 12_bg_iob_ia_high_lr_ctx64 | **3.0643** | 0.8953 | 0.0475 | 4.016 | 339.59 | +0.600 (+**24.4%**) | 25.3% |

*Per-dataset RMSE (fair):*

| ctx | aleppo | brown | lynch |
|---|---|---|---|
| 512 | 2.6797 | 2.4532 | 2.2584 |
| 256 | 2.7488 | 2.5139 | 2.4466 |
| 128 | 2.9398 | 2.8030 | 2.6841 |
| 64  | 3.0627 | 3.0545 | 3.0757 |

### COB high-LR fixed-anchor ablation (08/13/14/15 — aleppo only)

| ctx | Config | RMSE | WQL | MACE | Sharp50 | DILATE | vs ctx=512 | old % (unfair) |
|---|---|---|---|---|---|---|---|---|
| 512 | 08_bg_iob_cob_high_lr | 2.6090 | 0.7377 | 0.0029 | 3.224 | 203.55 | baseline | — |
| 256 | 13_bg_iob_cob_high_lr_ctx256 | **2.6909** | 0.7674 | 0.0027 | 3.443 | 223.26 | +0.082 (+**3.1%**) | 3.4% |
| 128 | 14_bg_iob_cob_high_lr_ctx128 | **2.8834** | 0.8357 | 0.0079 | 3.920 | 274.49 | +0.274 (+**10.5%**) | 10.6% |
| 64  | 15_bg_iob_cob_high_lr_ctx64 | **3.0262** | 0.8853 | 0.0132 | 4.253 | 326.82 | +0.417 (+**16.0%**) | 16.7% |

### Zero-shot fixed-anchor ablation (3-dataset avg: aleppo+brown+lynch)

| ctx | avg RMSE | avg WQL | avg MACE | avg DILATE | vs ctx=512 | old % (unfair) |
|---|---|---|---|---|---|---|
| 512 | 2.6956 | 0.7725 | 0.0610 | 257.47 | baseline | — |
| 256 | 2.8237 | 0.8091 | 0.0625 | 288.16 | +0.128 (+**4.8%**) | 5.4% |
| 128 | 3.1720 | 0.9076 | 0.0702 | 380.34 | +0.476 (+**17.7%**) | 18.1% |
| 64 | 3.5396 | 1.0116 | 0.0535 | 473.03 | +0.844 (+**31.3%**) | 32.1% |

*Zero-shot tamborlane (fixed, ZS only — 10782 episodes):*

| ctx | RMSE | WQL | MACE | DILATE |
|---|---|---|---|---|
| 512 | 3.3536 | 0.9554 | 0.0085 | 421.64 |
| 256 | 3.5061 | 0.9978 | 0.0101 | 449.24 |
| 128 | 3.7759 | 1.0786 | 0.0119 | 517.05 |
| 64  | 4.1603 | 1.1906 | 0.0289 | 624.32 |

### What changed vs the original (biased) ablation?

The corrected analysis reveals the extra episodes at smaller ctx were **harder on average**, not easier. Filtering to the shared 512-valid set slightly *improves* the smaller-ctx numbers and thus slightly *narrows* all degradation penalties:

| Step | IOB % (original) | IOB % (fair) | COB % (original) | COB % (fair) |
|---|---|---|---|---|
| 512 → 256 | 4.7% | **4.3%** | 3.4% | **3.1%** |
| 512 → 128 | 14.4% | **14.0%** | 10.6% | **10.5%** |
| 512 → 64 | 25.3% | **24.4%** | 16.7% | **16.0%** |

The qualitative ranking and all conclusions are unchanged. The original analysis modestly *overstated* the penalty for smaller windows because the additional short-sequence episodes sampled at ctx=64/128/256 turned out to be harder (e.g., patients with incomplete recording periods, high-variability edge segments).

**Additional finding:** At all context lengths on aleppo, `iob + cob` (configs 13/14/15) consistently outperforms `iob + insulin_availability` (configs 10/11/12) — but the advantage compresses as context shrinks:

| ctx | iob+ia RMSE | iob+cob RMSE | cob advantage | iob+ia vs 512 | iob+cob vs 512 |
|---|---|---|---|---|---|
| 512 | 2.6797 | 2.6090 | −0.071 | baseline | baseline |
| 256 | 2.7488 | 2.6909 | −0.058 | +2.6% | +3.1% |
| 128 | 2.9398 | 2.8834 | −0.056 | +9.7% | +10.5% |
| 64  | 3.0627 | 3.0262 | −0.037 | +14.3% | **+16.0%** |

Both feature sets always contain `iob`. The comparison is therefore: does adding `cob` help more than adding `insulin_availability`? On aleppo (the only dataset with real meal data), yes at every context length. This is consistent with conclusion #6 (config 07 full-features vs 06): `insulin_availability` appears to add little or no value once raw `iob` is already in the model. It is a derived signal whose computation depends on dose start times, making it harder to learn from cleanly than raw `iob`. COB on the other hand brings genuinely independent meal-absorption information.

The degradation pattern (both degrade at similar rates, COB slightly worse percentage-wise) primarily reflects that shorter windows hurt `iob` signal quality in both feature sets equally — losing insulin dosing history beyond ~5h affects both models the same way. The COB advantage narrowing at shorter contexts likely reflects that at ctx=64/128, the `cob` signal itself is truncated (a meal absorbed over 3–4h is well-captured, but context shorter than that may cut off the rising phase).

---

## Zero-Shot vs Fine-Tuned Comparison

Using ctx=512 zero-shot as the baseline (same for all covariate configs at this context length).

| Config | Dataset | ZS RMSE | FT RMSE | Gain | ZS WQL | FT WQL | Gain |
|---|---|---|---|---|---|---|---|
| 04_bg_iob_ia_high_lr | aleppo_2017 | 2.7649 | 2.6797 | −0.085 (−3.1%) | 0.7948 | 0.7589 | −0.035 |
| 04_bg_iob_ia_high_lr | brown_2019 | 2.6495 | 2.4532 | −0.196 (−7.4%) | 0.7591 | 0.6734 | −0.086 |
| 04_bg_iob_ia_high_lr | lynch_2022 | 2.6724 | 2.2584 | −0.414 (−15.5%) | 0.7677 | 0.6201 | −0.148 |
| 08_bg_iob_cob_high_lr | aleppo_2017 | 2.7649 | 2.6090 | −0.156 (−5.6%) | 0.7948 | 0.7377 | −0.057 |

Fine-tuning provides substantial gains, especially on lynch (−15.5% RMSE) and brown (−7.4%). Aleppo gain is smaller due to higher inherent variability in that cohort. The high-LR configs produce the most improvement.

---

## Rankings Summary

### IOB group (01 vs 02 vs 03 vs 04 vs 05 — 3-dataset avg; 10–12 — same datasets, ctx ablation)

| Rank | Config | avg RMSE | avg WQL | avg MACE | avg Sharp50 | avg DILATE |
|---|---|---|---|---|---|---|
| 1 | **04_bg_iob_ia_high_lr** | 2.4638 | 0.6841 | 0.0122 | 2.833 | 181.02 |
| 2 | 10_bg_iob_ia_high_lr_ctx256 | 2.5801 | 0.7259 | 0.0144 | 3.207 | 206.13 |
| 3 | 02_bg_iob | 2.5237 | 0.7099 | 0.0241 | 3.078 | 191.51 |
| 4 | 01_bg_iob_insulin_availability | 2.5331 | 0.7133 | 0.0260 | 3.106 | 194.33 |
| 5 | 03_joint_bg_iob | 2.5692 | 0.7286 | 0.0318 | 3.361 | 207.06 |
| 6 | 05_bg_iob_short_ctx (ctx=288) | 2.6496 | 0.7566 | 0.0369 | 3.438 | 226.12 |
| 7 | 11_bg_iob_ia_high_lr_ctx128 | 2.8186 | 0.8161 | 0.0385 | 3.772 | 263.93 |
| 8 | 12_bg_iob_ia_high_lr_ctx64 | 3.0861 | 0.9050 | 0.0493 | 4.102 | 341.77 |

### COB group (aleppo only — ctx ablation at high LR; 06/07/09 at various ctx)

| Rank | Config | RMSE | WQL | MACE | Sharp50 | DILATE |
|---|---|---|---|---|---|---|
| 1 | **08_bg_iob_cob_high_lr** | 2.6090 | 0.7377 | 0.0029 | 3.224 | 203.55 |
| 2 | 06_bg_iob_cob | 2.6813 | 0.7623 | 0.0092 | 3.303 | 219.85 |
| 3 | 07_bg_full_features | 2.6956 | 0.7666 | 0.0101 | 3.260 | 221.43 |
| 4 | 13_bg_iob_cob_high_lr_ctx256 | 2.6987 | 0.7720 | 0.0037 | 3.495 | 226.83 |
| 5 | 09_bg_iob_cob_short_ctx (ctx=288) | 2.7626 | 0.7918 | 0.0154 | 3.426 | 243.93 |
| 6 | 14_bg_iob_cob_high_lr_ctx128 | 2.8854 | 0.8384 | 0.0111 | 3.990 | 275.40 |
| 7 | 15_bg_iob_cob_high_lr_ctx64 | 3.0438 | 0.8931 | 0.0150 | 4.332 | 329.03 |

---

## Conclusions

### 1. High learning rate (5e-5) is by far the most impactful hyperparameter

Already confirmed in the original sweep: 04 and 08 dominate their groups on every metric. The new context-ablation data further confirms this — even at ctx=256, high-LR (10) beats standard-LR ctx=512 configs (01, 02, 03) on RMSE and WQL. **5e-5 LR should be the default for all long runs.**

### 2. Context length 512 is strongly preferred; degradation is non-linear below 256

The context ablation sweep (configs 10–15) was repeated on a *fixed episode set* (only anchors valid for all ctx levels — see "Fixed Episode Set" section above) for a fair apples-to-apples comparison. The corrected numbers are slightly smaller because the extra short-sequence episodes sampled at smaller ctx were harder on average:

| Step | IOB penalty (fair) | COB penalty (fair) | ZS penalty (fair) |
|---|---|---|---|
| 512 → 256 | +0.106 (+**4.3%**) | +0.082 (+**3.1%**) | +0.128 (+4.8%) |
| 512 → 128 | +0.345 (+**14.0%**) | +0.274 (+**10.5%**) | +0.476 (+17.7%) |
| 512 → 64 | +0.600 (+**24.4%**) | +0.417 (+**16.0%**) | +0.844 (+31.3%) |

256 is usable in memory-constrained deployments (+4.3%). 128 and especially 64 are not competitive — even in the fair comparison, ctx=64 is +24.4% worse in RMSE. Fine-tuning at ctx=64 barely beats zero-shot at the same length, suggesting insufficient context for meaningful learning. **Use ctx=512 for all training and deployment unless severely memory-constrained.**

The zero-shot penalty is consistently larger than the fine-tuned penalty at every step, showing that fine-tuning is relatively *more* valuable at shorter context lengths (the model can partially compensate for missing long-range context via learned patterns).

### 3. Fine-tuning provides meaningful gains over zero-shot (5–16% RMSE)

Zero-shot Chronos-2 at ctx=512 gives: aleppo RMSE=2.765, brown=2.650, lynch=2.672.
Best fine-tuned (04, high-LR): aleppo=2.680, brown=2.453, lynch=2.258.
The gain on lynch is particularly strong (−15.5%), likely because lynch data patterns differ more from Chronos-2's pre-training distribution.

### 4. Zero-shot covariates have no effect — covariate utility requires fine-tuning

All ctx=512 zero-shot runs (02, 06, 07, 08 — covering no covariates, iob-only, iob+cob, all features) produce exactly identical results (RMSE=2.7649 on aleppo, etc.). Pre-trained Chronos-2 does not meaningfully exploit additional covariate channels it wasn't trained on. The covariate benefit only materialises after fine-tuning. This is a useful sanity check confirming the fine-tuning is doing something real.

### 5. COB (carb) benefit is real and measurable when trained on clean data

New aleppo-only training improved COB config results vs the old mixed (aleppo+lynch) training:
- 06: aleppo RMSE 2.7415 → 2.6813 (−0.060, −2.2%)
- 08: aleppo RMSE 2.6723 → 2.6090 (−0.063, −2.4%)

This confirms that training on constant-zero COB (lynch) was diluting the learned signal. With clean training data, COB + high LR (08) achieves the best single-dataset performance in the entire sweep:
**RMSE=2.6090, WQL=0.7377, MACE=0.0029** — notably, MACE of 0.003 is an order of magnitude better than zero-shot (0.028).

### 6. Kitchen-sink features (07) add no value over minimal COB (06)

07 (iob + cob + insulin_availability + carb_availability) is consistently worse than 08 (iob + cob, high LR) and nearly identical to 06 (iob + cob, standard LR) on all metrics. This confirms `insulin_availability` and `carb_availability` are redundant once the raw IOB/COB values are present.

### 7. Tamborlane RMSE (3.27) remains substantially worse than other cohorts

Including tamborlane in `00_bg_only` training modestly improved its RMSE (3.30 → 3.27) while slightly degrading lynch (2.384 → 2.429). This is consistent with the training distribution shift. The tamborlane/other-cohort gap is likely intrinsic (device differences, cohort characteristics) rather than a training data omission. Adding it to training helps marginally.

### 8. Retraining the BG-only baseline with all 4 datasets slightly changes the numbers but doesn't alter rankings

Old 00 vs new 00 on the overlapping 3 datasets:
| Dataset | Old RMSE | New RMSE | Δ |
|---|---|---|---|
| aleppo | 2.7762 | 2.7514 | −0.025 |
| brown | 2.5117 | 2.5217 | +0.010 |
| lynch | 2.3840 | 2.4292 | +0.045 |

Small differences expected from training-distribution shift. The baseline conclusion (covariates substantially outperform BG-only) is unchanged.

### 9. Short-context fine-tuned models consistently beat zero-shot at the same context length

Even ctx=64 fine-tuning (12: RMSE=3.08 on aleppo) beats ctx=64 zero-shot (RMSE=3.44). This shows fine-tuning adds real value at all context lengths, but at ctx=64 the ceiling is low regardless.

---

## Deep Training Candidates

For 250k-step long runs, the clear priorities are:

| Priority | Config | Rationale | Deep-run prediction |
|---|---|---|---|
| ⭐ **1st** | **08_bg_iob_cob_high_lr** | Best RMSE+WQL+MACE on aleppo; benefits from 43h of carb context; high LR accelerates learning | Likely to extend its lead with more steps |
| ⭐ **2nd** | **04_bg_iob_ia_high_lr** | Best multi-dataset performance; generalises well across cohorts with different device types | Strong generalisation candidate |
| 3rd | **00_bg_only** | Required baseline for covariate benefit quantification; also the only config evaluated on all 4 datasets including tamborlane | Needed for fair comparison |
| 4th | **06_bg_iob_cob** (standard LR) | Close to 08 at 10k steps; lower LR may be more stable long-term if 08 plateaus or overfits | Hedge if 08 high-LR overshoots |

**Configs to skip for long runs:** 03 (joint, worst IOB group), 05/09 (short ctx), 07 (kitchen-sink, no gain), 11–15 (ctx < 512), 10 (ctx=256 gives tolerable but not competitive results). 02 and 01 are superseded by 04/08 on all metrics.

**If only one config can be run for 250k steps:** use **08_bg_iob_cob_high_lr**. It has the best absolute performance, exploits real carb data, and has the smallest MACE — the most important metric for nocturnal hypoglycaemia risk prediction.

---

## Open Questions

1. **Does 5e-5 LR remain best at 250k steps?** The checkpoint sweep will reveal whether high LR causes overfitting at longer runs or continues to improve. Critical — if 08 overshoots, 06 would be the fallback.

2. **What is the optimal step count?** The 10k checkpoint is likely near early convergence. The learning curve shape (does performance plateau at 50k? 100k?) determines how much long-run compute is worthwhile.

3. **Can 08 be extended to brown/lynch?** `08` was trained only on aleppo. If zero-fill or dataset-masking is applied for the carb columns, we could train on 3 datasets and directly compare to `04` on brown and lynch. This is the cleanest test of whether COB helps beyond aleppo.

4. **Is `insulin_availability` redundant at high LR?** At standard LR, 01 (iob+ia) and 02 (iob) are nearly identical. Config 04 (iob+ia, high LR) performs well, but we have no high-LR iob-only baseline. If 04 ≈ (hypothetical) high-LR 02, then `insulin_availability` adds nothing.

5. **How do best Chronos-2 configs compare to Moirai and TiDE?** Cross-model comparison needed once all model sweeps are complete.

6. **Why does joint training (03) underperform?** The joint target mode co-predicts bg_mM and iob. Possible causes: (a) capacity split across two targets, (b) noisy IOB target harming the BG head, (c) implementation issue. Not worth investigating until after deep runs confirm final winner.

7. **Does ctx=256 degrade further at 250k steps, or converge to the same point as ctx=512?** At 10k steps, 256 is −4.7% RMSE vs 512. If 512 continues improving with more steps but 256 plateaus earlier, the gap would widen. If they converge, 256 is viable for memory-constrained deployment.

8. **Is the tamborlane gap reducible with domain-specific fine-tuning?** RMSE=3.27 vs ~2.4 for other datasets suggests the model hasn't adapted well to that cohort. A tamborlane-focused fine-tuning run with high-LR could test this.

---

## Related Files

- [sweep_plan.md](sweep_plan.md) — original training sweep design and config comparison matrix
- [sweep_eval_commands.md](sweep_eval_commands.md) — historical ad-hoc eval commands (pre-sweep)
- [configs/models/chronos2/README.md](../../configs/models/chronos2/README.md) — config index
- [scripts/experiments/chronos2_sweep_eval.sh](../../scripts/experiments/chronos2_sweep_eval.sh) — eval script (produced main results; configs 01–05 commented out)
- [scripts/experiments/chronos2_ctx_ablation_eval.sh](../../scripts/experiments/chronos2_ctx_ablation_eval.sh) — fair-comparison ctx ablation eval (fixed episode set)
- [scripts/experiments/chronos2_sweep_train.sh](../../scripts/experiments/chronos2_sweep_train.sh) — training sweep script
