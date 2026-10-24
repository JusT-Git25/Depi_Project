# Lab Report Analyzer (OCR + risk prediction)

The user uploads a lab report (PDF or image). The app reads the values with OCR, lets the user check and fix them, then runs your trained model on the confirmed values and shows the prediction. No Streamlit: it is a FastAPI backend with a small static web page.

## Flow

1. Upload a PDF or image. Extension, size and PDF page count are validated.
2. PDFs with selectable text skip OCR. Scanned pages and images go through Tesseract (local, no cloud).
3. Rows are parsed deterministically (name, value, unit, reference range, flag). Test names are mapped to standard names.
4. The user reviews and edits the table, then ticks "I checked these values".
5. `POST /api/predict` maps the confirmed values to your model's features and runs the model.

## Safety rules built into the prediction step

- Values the user did not confirm are not used.
- A value whose unit is missing or different from `expected_units` is excluded. Units are never converted.
- Values like `<0.5` (not exact) and non-numeric values like `Positive` are excluded.
- If any required feature is missing and has no `fill_values` default, **no prediction is made** (`status: "insufficient_data"`) and the missing features are listed.
- The model file is only loaded, never retrained or changed. It is loaded with joblib (pickle), so only load model files you made or fully trust.
- The response always carries a "not a diagnosis" disclaimer. A model marked `"demo": true` shows a red banner in the UI.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[ocr,api,predict,test]"
```

Install Tesseract separately (UB Mannheim installer) and check `tesseract --version`. If it is not on PATH:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Default OCR language is `eng`. Set `LAB_OCR_LANGUAGE` for others (for example `ara+eng`, if installed).

## Plug in your model

1. Copy `models/model_spec.example.json` and edit it to match your real model:
   - `model_path`: your `.joblib` file (relative to the spec file).
   - `feature_order`: the exact column names and order the model was trained with.
   - `feature_mapping`: standard test name from OCR -> model feature name.
   - `expected_units`: the unit each feature was trained in.
   - `fill_values` (optional): default for a feature that may be missing. Leave empty to require it.
   - `labels`: class value (as text) -> name to show.
2. The model should accept a pandas DataFrame with those columns (a scikit-learn Pipeline works well) and ideally expose `predict_proba`.
3. Add any new test names or OCR misspellings to `lab_ocr/mappings/test_names.json` (keys are case-insensitive).

To try the whole flow first, create a **demo** model (synthetic data, no medical meaning):

```powershell
python scripts/make_demo_model.py
```

## Run

```powershell
python -m lab_ocr serve --model-spec models/demo_spec.json
```

Open http://127.0.0.1:8000. You can also set `LAB_OCR_MODEL_SPEC` instead of the flag. Without a model, the page still extracts values and exports JSON/CSV.

Command line, no browser:

```powershell
python -m lab_ocr analyze report.pdf --model-spec models/demo_spec.json
```

It prints the extracted values and asks you to confirm them before predicting (`--yes` skips the question).

## API

| Route | Purpose |
|---|---|
| `GET /api/health` | Model status and known test names |
| `POST /api/extract` | Multipart `file` -> extracted results (JSON) |
| `POST /api/predict` | `{results, reviewed_all}` -> prediction. `reviewed_all: true` means the user confirmed every value |
| `POST /api/export/{json,csv}` | Download the (edited) results |

Tests: `pytest` (OCR is mocked, no internet needed).

## Limits

- OCR can misread values (for example `HbA1c` read as `HbAlc`). That is why the review step exists.
- Extraction handles common table and `Name: value unit` layouts, not every lab format.
- Reference ranges and flags are copied from the report. The software does not decide what is normal.
- Nothing is saved: reports are processed in memory and the page keeps data only until it is closed or discarded.
- The server has no login. Before real patient data, put it behind your app's authentication, HTTPS, and your retention and audit rules. Never commit real reports.
- The prediction is only as good as your model and its training data. Validate it clinically before real use.
