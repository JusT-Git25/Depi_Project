# Deployment Guide — Liver Disease Prediction System

This guide explains how to run, containerize, and deploy the **FastAPI ML Backend** and the **Streamlit Demonstration Frontend** independently or together.

> **Critical Engineering Guarantee**: The FastAPI service loads the pre-trained, versioned model bundle (`artifacts/models/liver_disease_pipeline.joblib`) once during application startup (`lifespan`). It **never** retrains the model automatically on startup.

---

## 1. Local Execution (Without Docker)

### Prerequisites
- Python 3.11+ (verified on Python 3.12.13)
- Virtual environment with `requirements.txt` installed

### Step 1: Configure Environment Variables
Copy `.env.example` to `.env` and generate a strong API key:

```powershell
Copy-Item .env.example .env
# Generate a cryptographically strong key:
.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

Update `LIVER_API_KEY` inside `.env` with the generated value. **Never commit `.env` to Git.**

### Step 2: (Optional) Reproduce Training Pipeline
Pre-trained artifacts already exist in `artifacts/models/`. To reproduce from scratch:

```powershell
.venv\Scripts\python.exe scripts\run_data_audit.py
.venv\Scripts\python.exe scripts\run_eda.py
.venv\Scripts\python.exe scripts\run_benchmark.py
.venv\Scripts\python.exe scripts\run_training.py
```

### Step 3: Start the FastAPI Backend (Independent Service)
```powershell
$env:PYTHONPATH="src"
$env:LIVER_API_KEY="your-strong-random-api-key"
.venv\Scripts\uvicorn.exe liver_disease.api.main:app --host 0.0.0.0 --port 8000
```

Verify locally:
- Health probe: `http://localhost:8000/health`
- Swagger UI: `http://localhost:8000/docs`

### Step 4: Start the Streamlit Frontend (Independent Service)
In a second terminal:
```powershell
$env:API_BASE_URL="http://localhost:8000"
$env:LIVER_API_KEY="your-strong-random-api-key"
.venv\Scripts\streamlit.exe run frontend\app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

---

## 2. Docker & Docker Compose Deployment

The repository ships with **two independent Dockerfiles** so the external Backend team can deploy the ML API without Streamlit, or run both together via Docker Compose:

- `Dockerfile` — Production FastAPI ML Backend (`python:3.12-slim`, non-root user, built-in `HEALTHCHECK` on `/health`, copies only `src/` and `artifacts/models/`).
- `frontend/Dockerfile` — Lightweight Streamlit prototype (`streamlit`, `requests`, `pandas`, `plotly` only — zero ML model libraries).
- `docker-compose.yml` — Runs both containers on an internal bridge network with `frontend` waiting for `api` health check.

### Run Both Services with Docker Compose
```bash
# Ensure .env contains LIVER_API_KEY=...
docker compose up --build -d

# Inspect health & logs
docker compose ps
docker compose logs -f api
```

### Run ONLY the FastAPI Backend Container (External Backend Integration)
```bash
docker build -t liver-disease-api:1.0.0 .

docker run -d \
  --name liver-disease-api \
  -p 8000:8000 \
  -e LIVER_API_KEY="replace-with-strong-secret" \
  -e CORS_ALLOWED_ORIGINS="https://your-backend-or-frontend.example.com" \
  liver-disease-api:1.0.0
```

---

## 3. Cloud Deployment (Python Hosting Platforms)

### Option A: Render (Blueprint Included — `render.yaml`)
1. Push this repository (including `artifacts/models/liver_disease_pipeline.joblib`, excluding `.env`) to GitHub.
2. In Render Dashboard → **New +** → **Blueprint** → select the repository.
3. Render reads `render.yaml` and creates:
   - `liver-disease-prediction-api` (Docker web service with `/health` check)
   - `liver-disease-prediction-ui` (Optional Streamlit demo service)
4. Set secret environment variables in the Render dashboard:
   - On API service: `LIVER_API_KEY` (strong 32+ char secret)
   - On UI service (if enabled): `API_BASE_URL` = `https://<your-api>.onrender.com`, `LIVER_API_KEY` = same secret.

### Option B: Railway / Fly.io / Azure Container Apps / AWS App Runner
- Point the platform to the root `Dockerfile` on port `8000`.
- Configure health check path: `GET /health`.
- Configure required secret environment variable: `LIVER_API_KEY`.
- Optional tuning variables: `LOG_LEVEL=INFO`, `MAX_BATCH_ROWS=5000`, `CORS_ALLOWED_ORIGINS`.

---

## 4. Production Operational Checklist

- [ ] Set a cryptographically random `LIVER_API_KEY` (at least 32 characters) in the host secret manager.
- [ ] Terminate TLS/HTTPS at the reverse proxy or hosting platform (never send `X-API-Key` over unencrypted HTTP in production).
- [ ] Restrict `CORS_ALLOWED_ORIGINS` to trusted application origins.
- [ ] Monitor `GET /health` for `status == "healthy"` and `model_loaded == true`.
- [ ] Pin and track `model_version` returned in every prediction response for auditability.
- [ ] Re-run `pytest` before promoting any new model artifact version.
