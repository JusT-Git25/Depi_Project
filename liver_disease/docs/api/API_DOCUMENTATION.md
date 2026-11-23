# API Documentation — Liver Disease Prediction REST API

- **Framework**: FastAPI (`v1.0.0`)
- **Interactive Swagger UI**: `GET /docs`
- **ReDoc**: `GET /redoc`
- **OpenAPI JSON Schema**: `GET /openapi.json`
- **Authentication Header**: `X-API-Key: <LIVER_API_KEY>`

> **Academic Research Disclaimer**: Predictions returned by this API are statistical outputs of an academic ML prototype trained on UCI ILPD and are **not** clinically validated diagnostic results.

---

## Authentication

All endpoints except `/health`, `/docs`, `/redoc`, and `/openapi.json` require the `X-API-Key` HTTP header. The server validates the header against the `LIVER_API_KEY` environment variable using constant-time comparison (`hmac.compare_digest`).

- Missing or invalid `X-API-Key` → **HTTP 401 Unauthorized**
- Unconfigured `LIVER_API_KEY` on server → **HTTP 503 Service Unavailable** (fails closed)

---

## Endpoints

### 1. `GET /health`

Health and model readiness probe (unauthenticated for load balancers / Docker `HEALTHCHECK`).

**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "model_loaded": true,
  "model_version": "1.0.0",
  "model_name": "SoftVoting_ET_RF_LR"
}
```

---

### 2. `GET /model-info` *(Auth Required)*

Returns model version, architecture, decision threshold, risk tiers, required features with medical units and validation bounds, global SHAP importance, dataset provenance, and evaluation metrics.

**Headers**: `X-API-Key: <YOUR_API_KEY>`

**Response (`200 OK`)** *(abbreviated)*:
```json
{
  "model_version": "1.0.0",
  "model_name": "SoftVoting_ET_RF_LR",
  "model_type": "VotingClassifier",
  "calibration_method": "none",
  "decision_threshold": 0.57,
  "default_threshold": 0.5,
  "risk_category_bands": {
    "Low": "< 0.35",
    "Moderate": "0.35 – < 0.65",
    "High": ">= 0.65"
  },
  "target_encoding": {
    "1": "Liver Disease",
    "2": "No Liver Disease"
  },
  "explainability_supported": true,
  "explainer_type": "KernelExplainer",
  "required_features": [
    {
      "api_name": "age",
      "raw_csv_name": "Age",
      "display_name": "Age",
      "dtype": "int",
      "unit": "years",
      "min_value": 1.0,
      "max_value": 120.0,
      "normal_reference_range": "1–120 years"
    }
  ]
}
```

---

### 3. `POST /predict` *(Auth Required)*

Predict liver disease class, model probability score, and qualitative risk category for one patient.

**Headers**:
- `Content-Type: application/json`
- `X-API-Key: <YOUR_API_KEY>`

**Request Body** (all 10 fields strictly required; extra fields rejected with HTTP 422):
```json
{
  "age": 62,
  "gender": "Male",
  "total_bilirubin": 10.9,
  "direct_bilirubin": 5.5,
  "alkaline_phosphotase": 699.0,
  "alamine_aminotransferase": 64.0,
  "aspartate_aminotransferase": 100.0,
  "total_proteins": 7.5,
  "albumin": 3.2,
  "albumin_and_globulin_ratio": 0.74
}
```

**Response (`200 OK`)**:
```json
{
  "prediction_id": "7f3c8e58-1b2a-4c6e-8d19-2a9c5b4f0e11",
  "prediction": 1,
  "prediction_label": "Liver Disease",
  "probability": 0.9823,
  "risk_category": "High",
  "decision_threshold": 0.57,
  "model_version": "1.0.0",
  "disclaimer": "ACADEMIC RESEARCH PROTOTYPE DISCLAIMER: ..."
}
```

**Validation Errors (`422 Unprocessable Content`)**:
- Missing any required feature
- Extra/unexpected JSON field
- Invalid `gender` (must be `"Male"` or `"Female"`)
- Out-of-range numeric measurement (e.g., negative bilirubin or `age > 120`)
- Non-numeric, boolean, `NaN`, or `Infinity` values

---

### 4. `POST /explain` *(Auth Required)*

Returns the same prediction output as `/predict` plus individual **SHAP** local feature attributions ranked by absolute impact.

**Headers**:
- `Content-Type: application/json`
- `X-API-Key: <YOUR_API_KEY>`

**Request Body**: Identical to `POST /predict`.

**Response (`200 OK`)** *(abbreviated)*:
```json
{
  "prediction_id": "2c91a4d1-90a2-4c77-a1b8-6e8b9213c8a4",
  "prediction": 1,
  "prediction_label": "Liver Disease",
  "probability": 0.9823,
  "risk_category": "High",
  "decision_threshold": 0.57,
  "model_version": "1.0.0",
  "disclaimer": "ACADEMIC RESEARCH PROTOTYPE DISCLAIMER: ...",
  "explanation": {
    "explainer_type": "KernelExplainer",
    "base_value": 0.70614,
    "top_contributing_features": [
      {
        "feature": "DB",
        "raw_value": 5.5,
        "transformed_value": 1.4481,
        "shap_value": 0.0752,
        "abs_impact": 0.0752,
        "direction": "increases_risk"
      }
    ],
    "all_feature_contributions": [],
    "limitations_note": "Feature attributions (SHAP values) quantify each feature's statistical contribution to this model's output relative to the ILPD training baseline. Correlated biomarkers (e.g., Total vs Direct Bilirubin, AST vs ALT) share attribution weight, and feature importance is NOT evidence of biological causation or clinical diagnosis.",
    "disclaimer": "ACADEMIC RESEARCH PROTOTYPE DISCLAIMER: ..."
  }
}
```

---

### 5. `POST /predict/batch` *(Auth Required)*

Accepts a `.csv` file upload via `multipart/form-data`, validates schema and every row strictly, returns predictions for all valid rows, and explicitly reports any invalid rows (never silently accepting malformed medical records).

**Headers**:
- `X-API-Key: <YOUR_API_KEY>`

**Accepted CSV Header Schemas** (exact match required; extra or missing columns return HTTP 422):
1. Canonical API columns: `age,gender,total_bilirubin,direct_bilirubin,alkaline_phosphotase,alamine_aminotransferase,aspartate_aminotransferase,total_proteins,albumin,albumin_and_globulin_ratio`
2. Raw ILPD columns: `Age,Gender,TB,DB,Alkphos,Sgpt,Sgot,TP,ALB,A/G Ratio`

**Response (`200 OK`)**:
```json
{
  "batch_id": "b62f12d4-3c98-4027-9d4e-01d23b6f4f9c",
  "model_version": "1.0.0",
  "total_rows": 3,
  "valid_count": 2,
  "invalid_count": 1,
  "predictions": [
    {
      "row_index": 0,
      "prediction_id": "a1b2c3d4-...",
      "prediction": 1,
      "prediction_label": "Liver Disease",
      "probability": 0.9823,
      "risk_category": "High",
      "decision_threshold": 0.57,
      "model_version": "1.0.0"
    }
  ],
  "invalid_records": [
    {
      "row_index": 2,
      "errors": ["total_bilirubin: Input should be greater than or equal to 0.05"]
    }
  ],
  "disclaimer": "ACADEMIC RESEARCH PROTOTYPE DISCLAIMER: ..."
}
```

**Error Responses**:
- `400 Bad Request`: Non-`.csv` file, empty file, invalid UTF-8, or malformed CSV syntax.
- `401 Unauthorized`: Missing or invalid `X-API-Key`.
- `413 Payload Too Large`: Row count exceeds `MAX_BATCH_ROWS` (default `5000`).
- `422 Unprocessable Content`: Missing/unexpected columns, or 100% of rows failed medical input validation.
