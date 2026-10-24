# Diabetes Risk Screening

LightGBM screening aid trained on the CDC BRFSS diabetes survey (UCI id 891,
`diabetes_data.csv`). Takes either direct survey-style features or raw lab OCR
JSON and returns a calibrated risk score with a screening recommendation.

> **Screening aid only — not a diagnosis.**

## Layout

```
production/                       # pipeline (import as `from production.…`)
  common.py                       # config, splits, calibration + missingness helpers
  train_final.py                  # train + calibrate + package model → artifacts/
  predict.py                      # predict_risk() inference API (NaN-tolerant)
  lab_adapter.py                  # OCR JSON → CDC features (universal lab-name inference)
  pipeline.py                     # end-to-end: report/JSON → risk → advice
  advisor.py                      # AI advice (Claude) with rule-based fallback (en/ar)
  integrations/                   # bridge between lab_ocr and the model bundle
  lab_ocr/                        # FastAPI OCR + review UI + model-spec predictor
    models/diabetes_spec.json     # points the OCR predictor at our bundle
    tests/                        # OCR suite (pytest)
  tests/                          # advisor + integration tests
experiments/                      # benchmarks & analysis (documentation, not deployed)
  main.py                         # LogReg vs RandomForest vs GBDT comparison
  TreeBoosts.py                   # LightGBM vs CatBoost comparison
  analysis.py                     # threshold strategy + SHAP importance
  calibration.py                  # reliability curves + calibrator selection
  Eda.py                          # exploratory analysis + feature ideas
artifacts/                        # diabetes_risk_model.joblib, metrics.json
lab_ocr_sample.json               # OCR contract example 
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# OCR also needs the Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki
# AI advice needs a key:  cp production/.env.example production/.env  (never commit .env)
```

## Usage

```bash
# (re)train — LightGBM on full + lab-masked copies, calibrate, package
python -m production.train_final

# end-to-end from OCR JSON or a PDF/image report (add --advice for AI advice)
python -m production.pipeline --json lab_ocr_sample.json --advice --lang ar
python -m production.pipeline --file report.pdf

# OCR web app (upload → review → predict), with optional AI advice
LAB_OCR_MODEL_SPEC=production/lab_ocr/models/diabetes_spec.json python -m production.lab_ocr serve

# inference demos
python -m production.predict
python -m experiments.Eda
python -m experiments.calibration

# tests (OCR mocked, no internet)
python -m pytest production/tests production/lab_ocr/tests -q
```

```python
from production.predict import predict_risk
from production.pipeline import predict_from_lab_json, predict_from_report

# direct 21 CDC features (see predict.py docstring for coding)
predict_risk({"HighBP": 1, "HighChol": 1, ...})

# raw lab OCR JSON, any lab naming/units — see lab_ocr_sample.json
predict_from_lab_json([
    {"test": "Blood Pressure", "value": "138/86", "unit": "mmHg"},
    {"test": "Total Cholesterol", "value": 212, "unit": "mg/dL"},
], advice=True)
```

## Modules

- **OCR (`production.lab_ocr`)** — FastAPI backend + static review UI + CLI. Extracts values
  from PDFs/images (selectable-text PDFs skip OCR), maps test names, requires the user to
  confirm values before predicting, and never converts units silently.
- **Adapter (`production.lab_adapter`)** — universal lab-name inference (aliases + fuzzy match),
  normalization to the CDC feature set with AHA cutoffs, and clinical ADA evaluation of glycemic
  signals (HbA1c / glucose) with diagnostic safety alerts.
- **Bridge (`production.integrations`)** — lets the OCR predictor call the calibrated,
  NaN-tolerant production bundle instead of a plain estimator, incorporating glycemic safety overrides.
- **Advisor (`production.advisor`)** — AI advice (Anthropic Claude, Google Gemini, or OpenAI)
  in English/Arabic, with a rule-based fallback so the app never breaks. It contextualizes risk scores,
  laboratory findings (`lab_signals`), and clinical alerts while preserving privacy.

## Model notes

- Missing-robust: survey-only fields may be `NaN`; LightGBM handles them natively
  (trained with lab-missingness augmentation). Labs-only stress ROC-AUC 0.79 vs
  0.83 full input — see `artifacts/metrics.json`.
- Binning follows AHA cutoffs (sys ≥ 130 or dia ≥ 80 → `HighBP=1`;
  total chol ≥ 200 mg/dL → `HighChol=1`).
- Glucose/HbA1c are extracted to `lab_signals`. Because the CDC-21 survey model lacks a glucose slot,
  an automated clinical rule layer evaluates glycemic signals against ADA diagnostic thresholds
  (HbA1c ≥ 6.5% or glucose ≥ 126 mg/dL elevates risk to High with immediate clinical testing recommendation).
- Future diseases: add alias groups in `lab_adapter.py` + new model heads;
  `experiments/` stays as the audit trail.

