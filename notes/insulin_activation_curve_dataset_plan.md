# Insulin Activation Curve, IOB, and COB — Per-Dataset Plan

## Context

The Hovorka 3-compartment model in `src/data/physiological/insulin_model/insulin_model.py`
uses fixed population-average constants (`TMAX=55`, `KE=0.138`, `T_ACTION_MAX_MIN=180`).
The 180-minute action window is too short for rapid-acting analogs (should be 360 min).
Insulin type and patient BMI data exist in the raw files for 3 of 5 datasets but are
currently unused. The carb model has additional structural issues documented below.

---

## Model Selection: Hovorka vs. Dalla Man 2007

### Insulin model: keep Hovorka 2004

The 3-compartment Hovorka model is well-validated for subcutaneous insulin delivery
and the main deficiencies are parametric, not structural. Switching to the Dalla Man
2007 insulin model would add significant parameter complexity for marginal accuracy gain.
The correct fix is per-insulin-type constants and BMI-based scaling, not a model swap.

### Carb model: upgrade to Dalla Man 2007

The current 2-compartment carb model uses a hardcoded `tmax=40` derived from a liquid
glucose challenge (Hovorka 2004). For solid mixed meals — which is what all five datasets
contain — Dalla Man 2007 is meaningfully more accurate. The key improvement is replacing
the fixed time constant with a nonlinear (saturable) gastric emptying rate that scales
with meal size:

$$k_{empt}(Q_{sto}) = k_{min} + \frac{k_{max} - k_{min}}{2} \left[ \tanh\left(\alpha (Q_{sto} - b \cdot D)\right) - \tanh\left(\beta (Q_{sto} - c \cdot D)\right) + 2 \right]$$

Where $Q_{sto}$ is total stomach content (g), $D$ is meal size (g), and $\alpha, \beta, b, c$
are published population parameters (Dalla Man et al. 2007, IEEE TBME).

This naturally handles the fact that larger meals empty more slowly — no separate
parameterization needed per meal size.

**Source:** Dalla Man C, Rizza RA, Cobelli C. "Meal simulation model of the
glucose-insulin system." IEEE Trans Biomed Eng. 2007;54(10):1740-9.

---

## IOB Revisions

The Hovorka IOB model structure is correct (`iob = insulin - eliminated`, tracking
the q4 elimination compartment). All required changes are parametric:

1. **Extend `T_ACTION_MAX_MIN` from 180 → 360 min** for rapid-acting analogs (minimum
   fix, no other changes needed).
2. **Per-insulin-type parameters** — use the lookup table below once insulin brand
   is resolved from raw files per dataset.
3. **BMI-based TMAX scaling** where weight/height is available (Aleppo, Brown, Lynch):
   `TMAX_patient = TMAX_population × (1 + 0.006 × (BMI - 25))`
4. **Decouple `T_ACTION_MAX_MIN`** — currently shared between carb and insulin models
   via `carb_model/constants.py`. Must be split into separate constants before
   changing either independently.

### Insulin type parameter lookup table

| Insulin Type        | Examples                  | `TMAX` (min) | `KE` (min⁻¹) | `T_ACTION_MAX_MIN` (min) |
| ------------------- | ------------------------- | ------------ | ------------ | ------------------------ |
| Ultra-rapid analog  | Fiasp, Lyumjev            | 45           | 0.155        | 300                      |
| Rapid-acting analog | Aspart, Lispro, Glulisine | 55           | 0.138        | 360                      |
| Regular human (R)   | Humulin R, Novolin R      | 90           | 0.100        | 480                      |
| NPH basal           | Humulin N, Novolin N      | 180          | 0.050        | 900                      |
| Long-acting analog  | Glargine, Detemir         | 240          | 0.035        | 1440                     |
| Ultra-long-acting   | Degludec (Tresiba)        | 300          | 0.025        | 2160                     |
| Inhaled             | Afrezza                   | 20           | 0.200        | 150                      |

Sources: Wilinska et al. 2010 (IEEE TBME), Schiavon et al. 2014 (IEEE TBME),
Dalla Man et al. 2007 (IEEE TBME).

---

## COB Revisions

The carb model in `src/data/physiological/carb_model/carb_model.py` has four issues
to fix, independent of the Dalla Man upgrade:

1. **`tmax=40` is hardcoded inside the function body** — not in `constants.py`.
   Must be extracted to a named constant before any other changes.

2. **`T_ACTION_MAX_MIN=180` is too short for carbs.** A medium mixed meal has
   meaningful glucose impact for 3–5 hours; high-fat meals up to 6–8 hours.
   Target: `T_ACTION_MAX_MIN_CARB = 360` as minimum. Decouple from insulin constant first.

3. **`CARB_ABSORPTION=0.8` is a fixed population mean.** Gluroo provides
   `food_glycemic_index` per meal event in the messages table — currently unused.
   This should modulate the absorption fraction where available. High-GI meals
   absorb faster (higher effective fraction and lower tmax); low-GI meals slower.

4. **Upgrade to Dalla Man 2007 gastric emptying model** (see above) to replace the
   fixed `tmax` with a meal-size-dependent nonlinear rate. Use published population
   parameters — no refitting required.

---

## Raw Data Analysis Per Dataset

### Gluroo

**Insulin type:** `rapid_insulin`, `long_insulin`, `regular_insulin` — free-text brand
name strings in the `groups` DB table per patient (e.g., "Novolog", "Humalog"). Set at
onboarding, not per-dose event. The `messages` table has `dose_type` to distinguish
bolus vs. correction vs. basal.

**Demographics:** `age_at_onboarding`, `gender`, `time_since_diagnosis_lower/upper_days`
in `groups`. No weight/height/BMI available.

**Carbs:** Yes — `food_g` and `food_glycemic_index` in messages.

**Plan:** Map patient-level `rapid_insulin`/`long_insulin` strings to the parameter
lookup table. No BMI scaling possible without weight data. Use `food_glycemic_index`
to modulate `CARB_ABSORPTION` in COB calculation.

---

### Aleppo 2017

**Insulin type:** `HInsulin.txt` has `InsName` (e.g., "Novolog (Aspart)", "Humalog
(Lispro)"), `InsRoute` (Pump/Injection), and start/stop dates — so the active insulin
during any time window per patient is determinable. **Currently entirely ignored by the
cleaner.**

**Demographics:** `HScreening.txt` has `Weight` (kg), `Height` (cm), `Gender`, `DiagAge`,
`CGMUseDevice`. BMI is fully calculable. **Currently ignored.**

**Carbs:** Yes — `foodG` in the bolus/wizard device data. No glycemic index available.

**Plan:** Join `HInsulin.txt` by `PtID` + date range for time-varying per-patient
insulin type. Join `HScreening.txt` for BMI-based TMAX scaling. Richest dataset for
individualized curves among the five.

---

### Brown 2019 (DCLP3)

**Insulin type:** `Insulin_a.txt` has insulin name (e.g., "Novolog (Aspart)", "Lantus
(Glargine)"), `InsRoute`, and start/stop dates — same structure as Aleppo.
**Currently ignored.**

**Demographics:** `DiabPhysExam_a.txt` has `Weight` (lbs), `Height` (cm) per visit —
BMI calculable. **Currently ignored.**

**Carbs:** No — DCLP3 never collected meal/carb data in its released files.

**Plan:** Join `Insulin_a.txt` for per-patient time-varying insulin type. Join
`DiabPhysExam_a.txt` for BMI. No COB calculation possible — exclude carb features
for this dataset.

---

### Lynch 2022 (IOBP2)

**Insulin type:** `IOBP2Insulin.txt` has `InsulinName` (e.g., "Humalog (Lispro)",
"Degludec (Tresiba)", "Novolog (Aspart)"), `InsRoute`, and start/stop dates. Patients
may be on both a rapid-acting pump insulin and a long-acting basal injection concurrently.
**Currently ignored.**

**Demographics:** `IOBP2HeightWeight.txt` has `Weight` (lbs), `Height` (cm/in) per visit
— BMI fully calculable. **Currently ignored.** (The cleaner only pulls `PtWeight` from
the iLet device table — prefer the clinical measurement file instead.)

**Carbs:** Yes — `MealSize` (iLet carb entries) already pulled into the cleaner.
No glycemic index available.

**Plan:** Same join pattern as Brown/Aleppo for `IOBP2Insulin.txt`. Use
`IOBP2HeightWeight.txt` for BMI. iLet doses are near-continuous micro-deliveries —
the existing superposition approach in `create_iob_and_ins_availability_cols` is correct.

---

### Tamborlane 2008

**Insulin type:** No insulin delivery file exists in the raw data. This is the JDRF
CGM RCT — only CGM readings were recorded and released.

**Demographics:** `tblALabHbA1c.csv` has longitudinal HbA1c. `tblAAddlCEData.csv` has
blood pressure but no weight/height. No BMI data present.

**Carbs:** No.

**Plan:** Nothing is possible for insulin or carb features. Use for CGM-trajectory
features only, with insulin and carb feature columns excluded or zeroed.

---

## Summary Table

| Dataset         | Insulin name in raw?           | BMI calculable? | Carbs? | GI available? | Files to join                               |
| --------------- | ------------------------------ | --------------- | ------ | ------------- | ------------------------------------------- |
| Gluroo          | Yes (per patient, onboarding)  | No              | Yes    | Yes           | `groups` DB table                           |
| Aleppo 2017     | Yes (per patient, time-ranged) | Yes             | Yes    | No            | `HInsulin.txt`, `HScreening.txt`            |
| Brown 2019      | Yes (per patient, time-ranged) | Yes             | No     | No            | `Insulin_a.txt`, `DiabPhysExam_a.txt`       |
| Lynch 2022      | Yes (per patient, time-ranged) | Yes             | Yes    | No            | `IOBP2Insulin.txt`, `IOBP2HeightWeight.txt` |
| Tamborlane 2008 | No                             | No              | No     | No            | —                                           |

For Aleppo, Brown, and Lynch the insulin type and BMI raw files share identical
structure — implement once as a reusable join utility and apply across all three
dataset cleaners.
