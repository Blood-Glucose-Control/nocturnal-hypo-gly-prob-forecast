# Chronos-2 Sweep Configs

## Covariate Columns

| Column | Meaning | Datasets with this feature |
|---|---|---|
| `iob` | Insulin-on-board (units) | aleppo_2017, brown_2019, lynch_2022 |
| `insulin_availability` | Derived insulin curve (0–1) | aleppo_2017, brown_2019, lynch_2022 |
| `cob` | Carbs-on-board (g) | aleppo_2017 only (lynch_2022 carbs are all-zero) |
| `carb_availability` | Derived carb curve (0–1) | aleppo_2017 only (lynch_2022 carbs are all-zero) |

---

## Sweep Config Index

All sweep configs share: `model_path: autogluon/chronos-2`, `fine_tune_steps: 10 000`,
`batch_size: 256`, `forecast_length: 96`, `freeze_backbone: false`, checkpoints every 2 000 steps.

| File | ctx | LR | Covariates | Eval datasets | Purpose |
|---|---|---|---|---|---|
| `00_bg_only.yaml` | 512 | 1e-5 | — | ALL (4) | BG-only baseline |
| `01_bg_iob_insulin_availability.yaml` | 512 | 1e-5 | `iob`, `insulin_availability` | IOB (3) | Covariate baseline |
| `02_bg_iob.yaml` | 512 | 1e-5 | `iob` | IOB (3) | Isolates raw IOB vs `insulin_availability` |
| `03_joint_bg_iob.yaml` | 512 | 1e-5 | *(joint: bg_mM + iob)* | IOB (3) | Multi-target panel mode |
| `04_bg_iob_ia_high_lr.yaml` | 512 | 5e-5 | `iob`, `insulin_availability` | IOB (3) | LR ablation of 01 |
| `05_bg_iob_short_ctx.yaml` | 288 | 1e-5 | `iob`, `insulin_availability` | IOB (3) | Context ablation of 01 (~24 h, calendar-aligned) |
| `06_bg_iob_cob.yaml` | 512 | 1e-5 | `iob`, `cob` | COB (1) | Carb baseline |
| `07_bg_full_features.yaml` | 512 | 1e-5 | `iob`, `cob`, `insulin_availability`, `carb_availability` | COB (1) | Kitchen-sink |
| `08_bg_iob_cob_high_lr.yaml` | 512 | 5e-5 | `iob`, `cob` | COB (1) | LR ablation of 06 |
| `09_bg_iob_cob_short_ctx.yaml` | 288 | 1e-5 | `iob`, `cob` | COB (1) | Context ablation of 06 (~24 h, calendar-aligned) |
| `10_bg_iob_ia_high_lr_ctx256.yaml` | 256 | 5e-5 | `iob`, `insulin_availability` | IOB (3) | Context ablation ladder: 04 at 2⁸ (~21.3 h) |
| `11_bg_iob_ia_high_lr_ctx128.yaml` | 128 | 5e-5 | `iob`, `insulin_availability` | IOB (3) | Context ablation ladder: 04 at 2⁷ (~10.7 h) |
| `12_bg_iob_ia_high_lr_ctx64.yaml` | 64 | 5e-5 | `iob`, `insulin_availability` | IOB (3) | Context ablation ladder: 04 at 2⁶ (~5.3 h) |
| `13_bg_iob_cob_high_lr_ctx256.yaml` | 256 | 5e-5 | `iob`, `cob` | COB (1) | Context ablation ladder: 08 at 2⁸ (~21.3 h) |
| `14_bg_iob_cob_high_lr_ctx128.yaml` | 128 | 5e-5 | `iob`, `cob` | COB (1) | Context ablation ladder: 08 at 2⁷ (~10.7 h) |
| `15_bg_iob_cob_high_lr_ctx64.yaml` | 64 | 5e-5 | `iob`, `cob` | COB (1) | Context ablation ladder: 08 at 2⁶ (~5.3 h) |

**Dataset groups:**
- **ALL** (4): `lynch_2022`, `aleppo_2017`, `brown_2019`, `tamborlane_2008`
- **IOB** (3): `lynch_2022`, `aleppo_2017`, `brown_2019` *(tamborlane has no insulin data)*
- **COB** (1): `aleppo_2017` only *(lynch carbs are all-zero; brown has no meal data; tamborlane excluded)*

---

## Expected Evaluation Results

After running `chronos2_sweep_eval.sh`, you should see **35 fine-tuned + 60 zero-shot = 95 total result directories**:

### Fine-tuned evaluations

| Dataset group | Configs | Datasets | Count |
|---|---|---|---|
| ALL | `00` | 4 | 4 |
| IOB | `01`, `02`, `03`, `04`, `05` | 3 | 15 |
| COB | `06`, `07`, `08`, `09` | 1 | 4 |
| IOB ctx ablation | `10`, `11`, `12` | 3 | 9 |
| COB ctx ablation | `13`, `14`, `15` | 1 | 3 |
| **Total fine-tuned** | | | **35** |

### Zero-shot evaluations

All configs except `03_joint_bg_iob` (joint target not applicable zero-shot) × all 4 datasets:

| Configs | Datasets | Count |
|---|---|---|
| `00`–`02`, `04`–`15` (15 configs) | 4 | **60** |

---

## Utility Configs (not part of sweep)

| File | Steps | Purpose |
|---|---|---|
| `98_checkpoint_smoketest.yaml` | 200 | Verifies checkpoint materialisation before launching the full sweep |
| `99_250k_checkpoints.yaml` | 250 000 | Long run on the winning config for checkpoint-step analysis |
| `bg_only_test.yaml` | 100 | End-to-end CI smoketest |
