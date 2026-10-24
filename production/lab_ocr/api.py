"""FastAPI backend + static web UI (no Streamlit).

Run:  python -m lab_ocr serve   (or: uvicorn lab_ocr.api:app)
Set LAB_OCR_MODEL_SPEC to the model spec JSON to enable prediction.
"""
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import OCRConfig
from .exceptions import (FileValidationError, LabOCRError, OCREngineError,
                         PDFProcessingError, PredictionError, UnsupportedFileError)
from .exporters import report_to_csv, report_to_json
from .schemas import LabReportResult, LabResult
from .service import LabOCRService

logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"

CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


class ReviewedResults(BaseModel):
    results: list[LabResult] = Field(max_length=500)
    # True only when the user explicitly confirmed they checked every value against the original report.
    reviewed_all: bool = False


class AdviceRequest(BaseModel):
    prediction: dict
    language: str = "en"
    patient: dict | None = None


def create_app(service: LabOCRService | None = None, predictor=None, model_spec_path: str | None = None) -> FastAPI:
    app = FastAPI(title="Lab Report Analyzer", docs_url=None, redoc_url=None, openapi_url=None)
    service = service or LabOCRService(OCRConfig())
    state = {"predictor": predictor, "model_error": None}

    spec_path = model_spec_path or os.getenv("LAB_OCR_MODEL_SPEC")
    if state["predictor"] is None and spec_path:
        try:
            from .prediction import RiskPredictor
            state["predictor"] = RiskPredictor.from_spec_file(spec_path)
        except LabOCRError as exc:
            state["model_error"] = str(exc)
            logger.error("Model could not be loaded: %s", exc)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = CSP
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health")
    def health():
        p = state["predictor"]
        return {
            "model_loaded": p is not None,
            "model_name": p.spec.name if p else None,
            "demo": bool(p and p.spec.demo),
            "model_error": state["model_error"],
            "known_tests": sorted(p.spec.feature_mapping) if p else [],
            "max_file_size_mb": service.config.max_file_size_mb,
        }

    @app.post("/api/extract")
    def extract(file: UploadFile = File(...)):
        limit = service.config.max_file_size_bytes
        content = file.file.read(limit + 1)  # never read more than the limit
        try:
            report = service.process_bytes(file.filename or "upload", content)
        except (UnsupportedFileError, FileValidationError) as exc:
            raise HTTPException(400, str(exc))
        except PDFProcessingError as exc:
            raise HTTPException(422, str(exc))
        except OCREngineError as exc:
            raise HTTPException(503, str(exc))
        except Exception:
            logger.exception("Extraction failed")  # no report content is logged
            raise HTTPException(500, "The report could not be processed.")
        return report.model_dump(mode="json", exclude={"pages": {"__all__": {"words"}}})

    @app.post("/api/predict")
    def predict(body: ReviewedResults):
        predictor_ = state["predictor"]
        if predictor_ is None:
            raise HTTPException(503, state["model_error"] or "No prediction model is configured on this server.")
        results = [r.model_copy(update={"manually_reviewed": True}) if body.reviewed_all else r for r in body.results]
        report = LabReportResult(source_filename="reviewed", results=results)
        try:
            return predictor_.predict_report(report).model_dump(mode="json")
        except PredictionError as exc:
            raise HTTPException(422, str(exc))

    @app.post("/api/advice")
    def advice(body: AdviceRequest):
        """Optional AI advice for an existing prediction (works offline via rule-based fallback)."""
        try:
            from production.advisor import generate_advice
        except ImportError:
            raise HTTPException(503, "The advice module is not installed on this server.")
        try:
            return generate_advice(body.prediction, body.patient, body.language)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.post("/api/export/{fmt}")
    def export(fmt: str, body: ReviewedResults):
        report = LabReportResult(source_filename="reviewed", results=body.results)
        if fmt == "json":
            return Response(report_to_json(report), media_type="application/json",
                            headers={"Content-Disposition": 'attachment; filename="lab_results.json"'})
        if fmt == "csv":
            return Response(report_to_csv(report), media_type="text/csv",
                            headers={"Content-Disposition": 'attachment; filename="lab_results.csv"'})
        raise HTTPException(404, "Unknown export format.")

    return app


app = create_app()
