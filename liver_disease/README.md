# Liver Disease Prediction System

A modular, reproducible, backend-ready **Machine Learning Engineering** graduation project built on the **UCI Indian Liver Patient Dataset (ILPD)**.

> **ACADEMIC RESEARCH PROTOTYPE DISCLAIMER**: This system is an academic ML research prototype and is **NOT** a clinically validated medical device or diagnostic tool. It must never be used to diagnose patients or make clinical decisions. Predicted probabilities represent statistical model risk scores rather than clinically calibrated diagnostic probabilities.

---

## Key Capabilities

- **Verified Dataset & Zero-Leakage Pipeline**: Full data audit with SHA-256 provenance, detection and removal of 13 consecutive exact duplicate rows (`583 → 570`), strict stratified 80/20 splitting with verified zero train/test overlap, and in-fold preprocessing (`log1p`, median imputation, `RobustScaler`, binary OHE).
- **11-Algorithm Comprehensive Benchmark**: Systematic 5×3 Repeated Stratified CV comparing Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVM (RBF), KNN, and HistGradientBoosting — plus feature engineering, class imbalance (`class_weight`, `RandomOverSampler`, `SMOTE`), feature selection, tuning, ensemble, and calibration studies.
- **Selected Production Model (`v1.0.0`)**: `SoftVoting_ET_RF_LR` (weighted soft-voting ensemble of tuned ExtraTrees + RandomForest + LogisticRegression) with validation-selected screening threshold (`0.57`, min-specificity guardrail `>= 0.50`).
  - **5×3 Repeated CV (`n=456`)**: `ROC-AUC = 0.7417 ± 0.0469`, `PR-AUC = 0.8912 ± 0.0213`, `Brier = 0.1765`
  - **Locked Holdout Test (`n=114`, single-pass)**: `ROC-AUC = 0.8066` (95% CI: `0.723–0.880`), `PR-AUC = 0.9233` (95% CI: `0.877–0.958`), `Recall = 0.8148`, `Specificity = 0.5152`, `Precision = 0.8049`, `F1 = 0.8098`, `Balanced Accuracy = 0.6650`.
- **Explainable AI (SHAP)**: Global importance rankings, beeswarm/bar plots, and live individual patient SHAP explanations via `POST /explain`.
- **Backend-Ready FastAPI Service**: `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch` (CSV upload), `POST /explain`, Swagger UI (`/docs`), constant-time `X-API-Key` authentication from environment variables, strict Pydantic v2 validation, and zero retraining at startup.
- **Independent Streamlit Prototype**: Communicates exclusively with FastAPI over HTTP (zero duplicated inference code), supporting individual patient inputs, probability gauges, SHAP charts, batch CSV upload, and CSV download.
- **Automated QA & Deployment**: Comprehensive `pytest` suite (`tests/`), executed experimentation notebooks (`notebooks/01`–`05`), independent Dockerfiles, `docker-compose.yml`, and `render.yaml`.

---

## Repository Architecture

```text
.
├── data/
│   ├── raw/                  # Canonical ILPD CSV (immutable, SHA-256 verified)
│   ├── processed/            # Locked train.csv (456), test.csv (114), split_metadata.json
│   └── README.md             # Dataset provenance and audit summary
├── notebooks/                # Executed experimentation notebooks (01 to 05)
├── src/liver_disease/
│   ├── config.py             # Paths, feature schema, units, ranges, constants
│   ├── data/                 # loader.py, validation.py, preprocessing.py
│   ├── features/             # engineering.py (ClinicalFeatureEngineer)
│   ├── models/               # benchmark.py, train.py, evaluate.py, explain.py, predict.py
│   ├── api/                  # main.py, routes.py, schemas.py, security.py
│   └── utils/                # logging_config.py, io.py
├── artifacts/
│   ├── models/               # liver_disease_pipeline.joblib + model_metadata.json
│   ├── preprocessing/        # preprocessing_schema.json
│   └── reports/              # Audit/EDA/Benchmark/Evaluation JSONs, CSVs, and figures/
├── frontend/
│   ├── app.py                # Streamlit UI (HTTP-only client to FastAPI)
│   └── Dockerfile            # Independent UI container
├── tests/                    # Unit, preprocessing, model, prediction, API, security tests
├── scripts/                  # Reproducible CLI runners for every pipeline phase
├── docs/
│   ├── api/                  # REST API contract (API_DOCUMENTATION.md)
│   ├── architecture/         # Complete system architecture & code-verified audit (MD + PDF)
│   ├── deployment/           # Local, Docker, and cloud deployment guide (DEPLOYMENT_GUIDE.md)
│   ├── integration/          # Backend integration guide & Python client
│   ├── reports/              # DATA_DICTIONARY.md, EXPERIMENT_REPORT.md, MODEL_CARD.md
│   └── user_guides/          # End-user input & integration guide (MD + PDF)
├── Dockerfile                # Independent FastAPI backend container
├── docker-compose.yml        # Full-stack local orchestration
├── render.yaml               # Python hosting platform deployment blueprint
├── .env.example              # Environment variable template (no hardcoded secrets)
├── pyproject.toml            # Packaging metadata & pytest configuration
└── requirements.txt          # Pinned runtime & testing dependencies
```

---

## Architecture Layers & Team Navigation

| Team / Role | Primary Directory | Entry Point & Key Docs |
| :--- | :--- | :--- |
| **ML Engineer** | `src/liver_disease/{data,features,models}/`, `notebooks/`, `scripts/` | [EXPERIMENT_REPORT.md](docs/reports/EXPERIMENT_REPORT.md), [MODEL_CARD.md](docs/reports/MODEL_CARD.md), [DATA_DICTIONARY.md](docs/reports/DATA_DICTIONARY.md) |
| **Backend Developer** | `src/liver_disease/api/` | [API_DOCUMENTATION.md](docs/api/API_DOCUMENTATION.md), [BACKEND_INTEGRATION_GUIDE.md](docs/integration/BACKEND_INTEGRATION_GUIDE.md) |
| **Frontend Developer** | `frontend/` (`app.py`, zero ML deps) | [Liver_Disease_User_Input_Guide.md](docs/user_guides/Liver_Disease_User_Input_Guide.md) |
| **DevOps / MLOps** | `Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, `render.yaml` | [DEPLOYMENT_GUIDE.md](docs/deployment/DEPLOYMENT_GUIDE.md), [Complete Architecture Guide](docs/architecture/Liver_Disease_ML_Project_Complete_Explanation.md) |

---

## Quick Start

### 1. Environment Setup
```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
Copy-Item .env.example .env
# Generate and set a strong LIVER_API_KEY inside .env:
.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Run the FastAPI Backend
```powershell
$env:PYTHONPATH="src"
$env:LIVER_API_KEY="your-strong-random-api-key"
.venv\Scripts\uvicorn.exe liver_disease.api.main:app --host 0.0.0.0 --port 8000
```
Open Swagger UI at `http://localhost:8000/docs`.

### 3. Run the Streamlit Prototype Frontend
```powershell
$env:API_BASE_URL="http://localhost:8000"
$env:LIVER_API_KEY="your-strong-random-api-key"
.venv\Scripts\streamlit.exe run frontend\app.py --server.port 8501
```
Open `http://localhost:8501`.

### 4. Run Automated Tests & Live Smoke Check
```powershell
.venv\Scripts\pytest.exe tests -v
.venv\Scripts\python.exe scripts\verify_live_api.py
```

### 5. Reproduce Full Training Pipeline (Optional)
```powershell
.venv\Scripts\python.exe scripts\run_data_audit.py
.venv\Scripts\python.exe scripts\run_eda.py
.venv\Scripts\python.exe scripts\run_benchmark.py
.venv\Scripts\python.exe scripts\run_training.py
```

---

## Documentation Index

| Document | Purpose |
| :--- | :--- |
| [DATA_DICTIONARY.md](docs/reports/DATA_DICTIONARY.md) | ILPD provenance, medical units, clinical ranges, missing/duplicate audit |
| [EXPERIMENT_REPORT.md](docs/reports/EXPERIMENT_REPORT.md) | 11-model CV comparison, ablations, calibration, threshold & holdout metrics |
| [MODEL_CARD.md](docs/reports/MODEL_CARD.md) | Final model overview, 95% bootstrap CIs, limitations, bias & ethics |
| [API_DOCUMENTATION.md](docs/api/API_DOCUMENTATION.md) | Complete FastAPI endpoint reference, schemas, error codes |
| [BACKEND_INTEGRATION_GUIDE.md](docs/integration/BACKEND_INTEGRATION_GUIDE.md) | Production Python `requests` client, `curl`, Postman setup |
| [DEPLOYMENT_GUIDE.md](docs/deployment/DEPLOYMENT_GUIDE.md) | Local, Docker Compose, and cloud (Render/Railway/Fly) deployment |
| [Liver_Disease_User_Input_Guide.md](docs/user_guides/Liver_Disease_User_Input_Guide.md) ([PDF](docs/user_guides/Liver_Disease_User_Input_Guide.pdf)) | Exact 10 user inputs, units, validation rules, and UI/API mapping |
| [Liver_Disease_ML_Project_Complete_Explanation.md](docs/architecture/Liver_Disease_ML_Project_Complete_Explanation.md) ([PDF](docs/architecture/Liver_Disease_ML_Project_Complete_Explanation.pdf)) | End-to-end architecture, ML theory, API design, and walkthrough |
| [Liver_Disease_ML_Project_Code_Verified_Audit.md](docs/architecture/Liver_Disease_ML_Project_Code_Verified_Audit.md) ([PDF](docs/architecture/Liver_Disease_ML_Project_Code_Verified_Audit.pdf)) | Source-code-verified technical audit, handover matrix, and 35 Q&As |
