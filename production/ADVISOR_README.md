# AI Advice module (Asser's task)

`advisor.py` takes the output of `predict_risk()` and asks an AI model to write clear health advice
(in English or Arabic). If the AI is unavailable it returns rule-based advice instead, so the app never breaks.

## Setup
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and put your key in it (**never commit `.env`**; it is in `.gitignore`).
   Or set it in the shell: `export ANTHROPIC_API_KEY=...`
3. Make sure the model exists: `python train_final.py`

## Use
```python
from advisor import predict_with_advice, generate_advice

result = predict_with_advice(patient, language="ar")   # "en" or "ar"
print(result["risk_percent"], result["risk_level"])
print(result["advice"]["summary"])

# If you already have a prediction (e.g. from the OCR -> JSON -> normalization flow):
advice = generate_advice(prediction, patient, language="en")
```
Quick demo: `python advisor.py --lang ar`   |   Tests (no key needed): `python -m pytest tests -q`

## What `advice` contains
| field | meaning |
|---|---|
| `source` | `"ai"` or `"fallback"` (and `fallback_reason` says why) |
| `summary`, `key_factors`, `lifestyle_tips`, `next_steps`, `when_to_seek_care` | the advice text |
| `clinical_recommendation`, `disclaimer` | always written by our code, never by the AI |

## What is sent to the AI
Risk %, risk level, whether it was flagged, the top 3 risk factors and a short profile (age band, sex, BMI,
blood pressure, cholesterol, smoking, activity, diet, general health...). No names, IDs, income, education or
raw lab files. See `build_payload()`.

## Integration points
- **Justin (normalization/JSON):** the output must be a dict with the 21 columns `predict_risk()` expects.
- **Ahmed (OCR):** nothing to change; his JSON goes through Justin's normalizer, then `predict_with_advice`.
- **Menna & Nour (disease 2):** `generate_advice(prediction, patient)` only needs `risk_percent`, `risk_level`,
  `flagged_for_screening`, `top_risk_factors` and `recommendation`. For a new disease add its labels to
  `FEATURE_LABELS` and tips to `_TIPS`, and change the wording in `SYSTEM_PROMPT` (it says "diabetes").
- **Another AI provider:** only `_call_llm()` needs to change.
