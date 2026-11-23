# Backend Integration Guide — Liver Disease Prediction API

This guide is written specifically for the **external Backend Engineering team** integrating the Liver Disease Prediction ML service into a web application, mobile backend, or hospital-research portal without touching any internal ML training code.

---

## 1. Architecture Separation

```text
External Backend / Client App
        │
        │  HTTPS + Header: X-API-Key: <SECRET>
        ▼
FastAPI ML Microservice (port 8000)
  ├── GET  /health           (No auth — container/load-balancer readiness probe)
  ├── GET  /model-info       (Auth required — schema, units, metrics, version)
  ├── POST /predict          (Auth required — single patient prediction)
  ├── POST /predict/batch    (Auth required — CSV upload batch predictions)
  ├── POST /explain          (Auth required — single prediction + SHAP attributions)
  └── GET  /docs             (Interactive Swagger UI)
```

- The ML service is **stateless** and **independently deployable** (`Dockerfile`).
- It loads the frozen, versioned inference artifact (`liver_disease_pipeline.joblib`) once on startup and **never retrains at runtime**.
- Every prediction response returns a unique `prediction_id` (UUID v4) and `model_version` for audit logging in your primary database.

---

## 2. Authentication

All protected endpoints require the `X-API-Key` HTTP header:

```http
X-API-Key: YOUR_SECRET_API_KEY
```

### Security Rules for Backend Teams
1. Store the API key in your backend's environment variables or secret manager — **never** expose it in browser/mobile client-side JavaScript.
2. Always proxy requests from your frontend through your Backend server to this ML API over HTTPS.
3. Missing or invalid credentials return **HTTP 401 Unauthorized**.
4. If the ML server itself is missing `LIVER_API_KEY`, it fails closed with **HTTP 503 Service Unavailable**.

---

## 3. Required Input Schema & Medical Units

All 10 features are **strictly required** for JSON requests. Unexpected extra fields are rejected with **HTTP 422**.

| Field | Type | Unit | Allowed Range | Normal Reference |
| :--- | :--- | :--- | :--- | :--- |
| `age` | `int` | years | 1 – 120 | 1–120 years |
| `gender` | `string` | categorical | `"Male"` or `"Female"` | — |
| `total_bilirubin` | `float` | mg/dL | 0.05 – 100.0 | 0.1–1.2 mg/dL |
| `direct_bilirubin` | `float` | mg/dL | 0.0 – 50.0 | 0.0–0.3 mg/dL |
| `alkaline_phosphotase` | `float` | IU/L | 10.0 – 5000.0 | 44–147 IU/L |
| `alamine_aminotransferase` | `float` | IU/L | 1.0 – 5000.0 | 7–56 IU/L |
| `aspartate_aminotransferase` | `float` | IU/L | 1.0 – 10000.0 | 10–40 IU/L |
| `total_proteins` | `float` | g/dL | 1.0 – 20.0 | 6.0–8.3 g/dL |
| `albumin` | `float` | g/dL | 0.5 – 12.0 | 3.5–5.0 g/dL |
| `albumin_and_globulin_ratio` | `float` | dimensionless ratio | 0.1 – 10.0 | 1.0–2.5 |

### Output Semantics
- `prediction`: `1` (`"Liver Disease"`) or `2` (`"No Liver Disease"`), matching the original ILPD target convention.
- `probability`: Model risk score $P(\text{Liver Disease}) \in [0, 1]$ (not a clinically calibrated diagnostic probability).
- `risk_category`:
  - `"Low"`: `probability < 0.35`
  - `"Moderate"`: `0.35 <= probability < 0.65`
  - `"High"`: `probability >= 0.65`
- `decision_threshold`: The validation-selected binary classification cutoff used to assign `prediction` (`1` vs `2`).

---

## 4. Production Python `requests` Integration Client

Drop-in client wrapper your Backend team can use immediately:

```python
import os
from pathlib import Path
from typing import Any, Dict

import requests


class LiverDiseaseAPIClient:
    """Reusable HTTP client for the Liver Disease Prediction ML microservice."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})

    def health(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

    def model_info(self) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/model-info", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def predict(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.post(
            f"{self.base_url}/predict",
            json=patient,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def explain(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.post(
            f"{self.base_url}/explain",
            json=patient,
            timeout=max(self.timeout, 25.0),
        )
        resp.raise_for_status()
        return resp.json()

    def predict_batch_csv(self, csv_path: Path) -> Dict[str, Any]:
        with csv_path.open("rb") as fh:
            resp = self.session.post(
                f"{self.base_url}/predict/batch",
                files={"file": (csv_path.name, fh, "text/csv")},
                timeout=60.0,
            )
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    client = LiverDiseaseAPIClient(
        base_url=os.getenv("LIVER_API_URL", "http://localhost:8000"),
        api_key=os.environ["LIVER_API_KEY"],
    )

    print("Health:", client.health())

    sample_patient = {
        "age": 62,
        "gender": "Male",
        "total_bilirubin": 10.9,
        "direct_bilirubin": 5.5,
        "alkaline_phosphotase": 699.0,
        "alamine_aminotransferase": 64.0,
        "aspartate_aminotransferase": 100.0,
        "total_proteins": 7.5,
        "albumin": 3.2,
        "albumin_and_globulin_ratio": 0.74,
    }

    prediction = client.predict(sample_patient)
    print("Prediction:", prediction)

    explanation = client.explain(sample_patient)
    print("Top SHAP drivers:", explanation["explanation"]["top_contributing_features"][:3])
```

---

## 5. `curl` & Postman Verification Examples

### Health Check
```bash
curl -s http://localhost:8000/health
```

### Single Prediction
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "age": 62,
    "gender": "Male",
    "total_bilirubin": 10.9,
    "direct_bilirubin": 5.5,
    "alkaline_phosphotase": 699,
    "alamine_aminotransferase": 64,
    "aspartate_aminotransferase": 100,
    "total_proteins": 7.5,
    "albumin": 3.2,
    "albumin_and_globulin_ratio": 0.74
  }'
```

### Batch CSV Upload
```bash
curl -s -X POST http://localhost:8000/predict/batch \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@patients.csv;type=text/csv"
```

### Postman Setup
1. Create a Postman Environment with variables `base_url = http://localhost:8000` and `api_key = <YOUR_API_KEY>`.
2. On Collection → **Authorization** → Type: **API Key** → Key: `X-API-Key`, Value: `{{api_key}}`, Add to: **Header**.
3. Import OpenAPI schema directly from `http://localhost:8000/openapi.json` to auto-generate all endpoints and example payloads.

---

## 6. Error Handling Matrix for Backend Callers

| HTTP Status | Meaning | Recommended Backend Handling |
| :--- | :--- | :--- |
| `200 OK` | Request succeeded | Persist `prediction_id`, `model_version`, `prediction`, `probability`, `risk_category`. |
| `400 Bad Request` | Empty/malformed/non-CSV upload | Show file format error to uploading user. |
| `401 Unauthorized` | Missing or wrong `X-API-Key` | Verify server secret configuration; alert DevOps. |
| `413 Payload Too Large` | CSV exceeds `MAX_BATCH_ROWS` | Chunk CSV into batches <= `5000` rows. |
| `422 Unprocessable Entity` | Missing field, extra field, invalid category, or out-of-range lab measurement | Parse Pydantic error details and display field-level validation feedback. |
| `503 Service Unavailable` | Model artifact not loaded or `LIVER_API_KEY` not set on ML server | Trigger health alert and fallback/retry with exponential backoff. |

---

## 7. Medical Disclaimer Requirement

Any downstream UI consuming this API **must** clearly present the `disclaimer` string returned in responses: this is an academic research prototype and **not** a clinically validated diagnostic system.
