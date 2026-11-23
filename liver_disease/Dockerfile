# ==============================================================================
# Liver Disease Prediction FastAPI Backend — Production Container
# ==============================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    MODEL_ARTIFACT_PATH=/app/artifacts/models/liver_disease_pipeline.joblib

WORKDIR /app

# Install minimal system dependencies required by LightGBM / scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy production application source and serialized model artifacts ONLY
COPY src/ ./src/
COPY artifacts/models/ ./artifacts/models/

# Run as non-root user for container security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "liver_disease.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
