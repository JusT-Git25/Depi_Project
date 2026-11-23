# ==============================================================================
# Liver Disease Prediction Streamlit Prototype — Independent Frontend Container
# ==============================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_BASE_URL=http://api:8000

WORKDIR /app

# Lightweight UI dependencies only (zero ML inference libraries)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       "streamlit>=1.39.0,<2.0.0" \
       "requests>=2.32.0,<3.0.0" \
       "pandas>=2.2.0,<3.0.0" \
       "plotly>=5.24.0,<7.0.0"

COPY app.py ./app.py

RUN useradd --create-home --shell /bin/bash streamlituser \
    && chown -R streamlituser:streamlituser /app
USER streamlituser

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
