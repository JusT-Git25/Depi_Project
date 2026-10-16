# Diabetes Risk Screening

LightGBM screening aid trained on the CDC BRFSS diabetes survey (UCI id 891,
`diabetes_data.csv`). Takes either direct survey-style features or raw lab OCR
JSON and returns a calibrated risk score with a screening recommendation.

> **Screening aid only — not a diagnosis.**

## Layout

```
production/            # pipeline (import as `from production.…`)
  common.py            # config, splits, calibration + missingness helpers
  train_final.py       # train + calibrate + package model → artifacts/
  predict.py           # predict_risk() inference API (NaN-tolerant)
  lab_adapter.py       # OCR JSON → CDC features → predict_from_labs()
experiments/           # benchmarks & analysis (documentation, not deployed)
  main.py              # LogReg vs RandomForest vs GBDT comparison
  TreeBoosts.py        # LightGBM vs CatBoost comparison
  analysis.py          # threshold strategy + SHAP importance
  calibration.py       # reliability curves + calibrator selection
  Eda.py               # exploratory analysis + feature ideas
artifacts/             # diabetes_risk_model.joblib, metrics.json
lab_ocr_sample.json    # OCR contract example (give this to the OCR team)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# (re)train — LightGBM on full + lab-masked copies, calibrate, package
python -m production.train_final

# inference demos
python -m production.predict
python -m experiments.Eda
python -m experiments.calibration
```

```python
from production.predict import predict_risk
from production.lab_adapter import predict_from_labs

# direct 21 CDC features (see predict.py docstring for coding)
predict_risk({"HighBP": 1, "HighChol": 1, ...})

# raw lab OCR JSON, any lab naming/units — see lab_ocr_sample.json
predict_from_labs([
    {"test": "Blood Pressure", "value": "138/86", "unit": "mmHg"},
    {"test": "Total Cholesterol", "value": 212, "unit": "mg/dL"},
])
```

## Model notes

- Missing-robust: survey-only fields may be `NaN`; LightGBM handles them natively
  (trained with lab-missingness augmentation). Labs-only stress ROC-AUC 0.79 vs
  0.83 full input — see `artifacts/metrics.json`.
- Binning follows AHA cutoffs (sys ≥ 130 or dia ≥ 80 → `HighBP=1`;
  total chol ≥ 200 mg/dL → `HighChol=1`).
- Glucose/HbA1c are extracted to `lab_signals` but are **not** model inputs (the
  CDC-21 feature set has no glucose slot) — needs paired lab+outcome data.
- Future diseases: add alias groups in `lab_adapter.py` + new model heads;
  `experiments/` stays as the audit trail.
