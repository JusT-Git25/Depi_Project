"""FastAPI application entrypoint with lifespan model loading and structured error handling."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from liver_disease import __version__ as API_VERSION
from liver_disease.api.routes import router
from liver_disease.config import DEFAULT_MODEL_ARTIFACT_PATH, MEDICAL_DISCLAIMER, PROJECT_ROOT
from liver_disease.models.predict import LiverDiseasePredictor
from liver_disease.utils.logging_config import setup_logging

load_dotenv(PROJECT_ROOT / ".env", override=False)

logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"))


def _resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_ARTIFACT_PATH", "").strip()
    if not env_path:
        return DEFAULT_MODEL_ARTIFACT_PATH
    p = Path(env_path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def create_app(model_artifact_path: Path | str | None = None) -> FastAPI:
    """Application factory enabling clean production startup and isolated pytest testing."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        artifact_path = Path(model_artifact_path) if model_artifact_path else _resolve_model_path()
        try:
            app.state.predictor = LiverDiseasePredictor(artifact_path=artifact_path)
            logger.info("FastAPI startup complete — model loaded from %s", artifact_path)
        except Exception as exc:
            app.state.predictor = None
            logger.error("Failed to load model artifact at startup (%s): %s", artifact_path, exc)
        yield
        logger.info("FastAPI shutting down cleanly.")

    app = FastAPI(
        title="Liver Disease Prediction API",
        version=API_VERSION,
        description=(
            "Backend-ready Machine Learning REST API built on the Indian Liver Patient Dataset (ILPD).\n\n"
            f"**{MEDICAL_DISCLAIMER}**\n\n"
            "### Authentication\n"
            "All inference and metadata endpoints require the `X-API-Key` HTTP header configured "
            "via the `LIVER_API_KEY` environment variable."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS configuration from environment
    raw_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type", "Accept"],
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error while processing request."},
        )

    app.include_router(router)
    return app


app = create_app()
