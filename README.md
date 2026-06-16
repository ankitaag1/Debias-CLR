# Debias-CLR — Contrastive Debiasing for Fair Healthcare Predictions
Code for the analysis done in the published paper 'Debias-CLR: A Contrastive Learning Based Debiasing Method for Algorithmic Fairness in Healthcare Applications' https://ieeexplore.ieee.org/abstract/document/10825827

A runnable implementation of **Debias-CLR**, an in-processing debiasing method that uses
contrastive learning to reduce demographic bias in clinical prediction models, applied to a
heart-failure EMR cohort and evaluated on **length-of-stay** prediction.

The notebook reproduces the paper's pipeline end to end: Clinical BERT text embeddings + an
LSTM-autoencoder vital block → mutual-information feature selection → SMOTE balancing →
counterfactual positive generation → NT-Xent + LARS contrastive training → SC-WEAT bias
measurement → optional cutout regularisation (**Debias-CLR-R**), with two independent debias models
(one for **gender**, one for **ethnicity**).

---

## Files you provide (required at runtime)

The pipeline needs **three input files**, located automatically by `_find()` (working directory,
then the upload dir). 

| File | Config var | Description |
|---|---|---|
| EHR export | `DATA_PATH` (`clinical_notes_dataset.csv`) | One row per procedure record, in the schema below. |
| CCSR mapping | `CCSR_PATH` (`code_to_CCSR.csv`) | AHRQ CCSR file mapping decimal-free ICD-10-CM `Code` → `Category` name (one code may map to several). |
| Vitals export | `VITALS_PATH` (`vitals.csv`) | Long-format physiological vitals, one row per reading, in the schema below. |

**AHRQ CCSR mapping:** obtain the *"CCSR for ICD-10-CM Diagnoses"* tool from
<https://www.hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp> and prepare a two-column CSV with
headers `Code` (decimal-free ICD-10-CM code) and `Category` (CCSR category name).

### EHR export format

CSV or Excel, one row per procedure record (several rows roll up to one `encounter_id`):

| Column | Meaning |
|---|---|
| `record_id` | Patient identifier. |
| `redcap_repeat_instrument` | REDCap repeating instrument name (e.g. `procedures`). |
| `redcap_repeat_instance` | Per-patient running counter of procedure rows (not used for grouping). |
| `encounter_id` | Hospitalization key — several procedure rows roll up to one `encounter_id`. |
| `Admission Date` | Admission date (any parseable date format). |
| `Discharge Date` | Discharge date. |
| `primary diagnosis` | ICD-10-CM code(s); may contain multiple codes separated by `;`. |
| `secondary diagnosis` | ICD-10-CM code(s), `;`-separated. |
| `procedure report` | Free-text report; the `IMPRESSION:`/`CONCLUSION:` section (else `FINDINGS:`) is extracted. |
| `gender` | Sensitive attribute (e.g. `Male`/`Female`); read directly when `DEMOGRAPHICS = "file"`. |
| `ethnicity` | Sensitive attribute (e.g. `Hispanic`/`Non-Hispanic`); imbalanced classes are SMOTE-balanced. |
| `diag_topic` | Diagnostic phenotype-topic **name** (one of 12), mapped to a code 1–12 in code. |
| `proc_topic` | Procedure phenotype-topic **name** (one of 12), mapped to a code 1–12 in code. |

### Vitals export format

Long format, one row per reading:

| Column | Meaning |
|---|---|
| `record_id` | Patient identifier. |
| `redcap_repeat_instrument`, `redcap_repeat_instance` | REDCap repeat metadata. |
| `encounter_id` | Hospitalization key (matches `encounter_id` in the EHR export). |
| `VITAL_NAME` | One of `Pulse Rate`, `BP Systolic`, `BP Diastolic`, `Respiratory Rate`, `Temp (DegC)`, `SPO2`. |
| `VITAL_DATE` | Timestamp of the reading. |
| `VITAL_VALUE`, `VITAL_UNITS` | Reading value and units. |

---

### Requirements

```
numpy
pandas
scikit-learn
imbalanced-learn
torch
transformers          # Clinical BERT (requires internet/model access)
jupyter
nbformat
```

> **Clinical BERT is mandatory.** Text is embedded only with `medicalai/ClinicalBERT`.

---

## Configuration

Set in the first cell:

| Switch | Default | Meaning |
|---|---|---|
| `CLINICAL_BERT_MODEL` | `"medicalai/ClinicalBERT"` | HF model used for diagnostic & procedure text |
| `HAS_VITALS` | `True` | include the 600-d LSTM-autoencoder vital block |
| `DEMOGRAPHICS` | `"file"` | read `gender`/`ethnicity` from dataset columns (assumed present) |
| `TEXT_EMB_DIM` | `768` | Clinical BERT embedding size per text field |
| `N_PHEN` | `12` | phenotype topics per target |
| `EPOCHS` | `60` | contrastive-training epochs (raise toward 100 for paper-level results) |
| `SEED` | `42` | global reproducibility seed |

---

## Pipeline overview

**Section 2 — Data preparation**
- **2.1 CCSR mapping.** Combine primary + secondary ICD-10 codes per encounter, map to CCSR
  categories via `codes_to_text`, drop *Heart failure*. Diagnoses are constant within an encounter;
  all of an encounter's procedure reports have their Impression/Conclusion (else Findings) extracted
  and **concatenated** into one text. Encounters missing diagnoses or a procedure report are dropped.
- **2.2 Length of stay.** Days from admission→discharge, binned into 5 ordinal classes
  (0–1, 2–7, 8–14, 15–21, >21).
- **2.3 Text embeddings.** Clinical BERT (768-d) for diagnostic text and for the concatenated
  procedure text.
- **2.4 Phenotype topics.** 12 diagnostic + 12 procedure topic **names** are read from the dataset
  columns and mapped to codes **1–12**; per-encounter one-hot membership is the SC-WEAT target.
- **2.4b Vitals.** Six vitals (pulse, systolic/diastolic BP, respiratory rate, temp °C, SpO₂) from
  `vitals.csv`; out-of-range values dropped, each vital ordered by time and resampled, then an
  LSTM autoencoder encodes each to 100-d (**6 × 100 = 600-d**).
- **2.5 Demographics.** `gender` (F→0/M→1) and `ethnicity` (Hispanic→0/Non-Hispanic→1) read from the
  dataset.
- **2.6 Feature matrix.** `X = [diag 768 | proc 768 | vitals 600] = 2,136-d` per encounter.
- **2.7 Dataset after embeddings.** A table pairing the metadata (record id, encounter, LOS, gender,
  ethnicity, topics) with all embedding columns, also written to `dataset_after_embeddings.csv`.

**Sections 3–7 — Debias-CLR**
- Mutual-information selection of the top-50% sensitive features (Table I).
- **SMOTE** balancing for the imbalanced ethnicity attribute.
- Counterfactual positive generation (Algorithm 1).
- Contrastive training with **NT-Xent loss + LARS** optimiser; residual encoder preserves
  task-relevant signal.
- **SC-WEAT** effect size (Algorithm 2) — ~0 means fair.
- **Cutout** regularisation → **Debias-CLR-R**.
- Two independent models are trained: one debiasing **gender**, one debiasing **ethnicity**.

**Sections 8–9 — Evaluation**
- **Table I** — sensitive-attribute prediction accuracy (5 classifiers).
- **Table II** — SC-WEAT vs. training-data fraction.
- **Tables III/IV** — length-of-stay prediction (Accuracy, MCC, Cohen's Kappa) on raw vs.
  debiased embeddings, for gender and ethnicity.

---
