"""FastAPI route definitions for health, model metadata, predictions, batch CSV, and explainability."""

from __future__ import annotations

import io
import logging
import os
import uuid
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import ValidationError

from liver_disease import __version__ as API_VERSION
from liver_disease.api.schemas import (
    BatchInvalidRecord,
    BatchPredictionRecord,
    BatchPredictionResponse,
    ExplainResponse,
    FeatureMetadata,
    HealthResponse,
    ModelInfoResponse,
    PatientFeatures,
    PredictionResponse,
)
from liver_disease.api.security import verify_api_key
from liver_disease.config import (
    API_TO_RAW_COLUMN_MAP,
    FEATURE_SPECS,
    MEDICAL_DISCLAIMER,
    RAW_FEATURE_COLUMNS,
    RAW_TO_API_COLUMN_MAP,
)
from liver_disease.models.predict import LiverDiseasePredictor

logger = logging.getLogger(__name__)

HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)

router = APIRouter()


def get_predictor(request: Request) -> LiverDiseasePredictor:
    """Retrieve the pre-loaded LiverDiseasePredictor from FastAPI app state."""
    predictor: LiverDiseasePredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model artifact is not loaded. Service cannot process inference requests.",
        )
    return predictor


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health & model readiness probe",
)
async def health_check(request: Request) -> HealthResponse:
    """Return API availability and whether the serialized ML pipeline is loaded."""
    predictor: LiverDiseasePredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return HealthResponse(
            status="degraded",
            api_version=API_VERSION,
            model_loaded=False,
        )
    return HealthResponse(
        status="healthy",
        api_version=API_VERSION,
        model_loaded=True,
        model_version=predictor.model_version,
        model_name=predictor.model_name,
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Metadata"],
    summary="Inspect loaded model metadata, schema, units, and evaluation metrics",
    dependencies=[Depends(verify_api_key)],
)
async def model_info(
    predictor: LiverDiseasePredictor = Depends(get_predictor),
) -> ModelInfoResponse:
    """Return complete metadata required by external backend teams to integrate with the API."""
    info = predictor.get_model_info()
    required_features = [
        FeatureMetadata(
            api_name=spec.api_name,
            raw_csv_name=spec.raw_name,
            display_name=spec.display_name,
            dtype=spec.dtype,
            unit=spec.unit,
            min_value=spec.min_value,
            max_value=spec.max_value,
            normal_reference_range=spec.normal_reference_range,
            description=spec.description,
        )
        for spec in FEATURE_SPECS.values()
    ]
    return ModelInfoResponse(
        model_version=info["model_version"],
        model_name=info["model_name"],
        model_type=info["model_type"],
        calibration_method=info["calibration_method"],
        decision_threshold=info["decision_threshold"],
        default_threshold=info["default_threshold"],
        risk_category_bands=info["risk_category_bands"],
        target_encoding=info["target_encoding"],
        supported_input_formats={
            "single_prediction_json": "application/json with canonical snake_case feature keys",
            "batch_prediction_csv": {
                "content_type": "multipart/form-data (CSV file upload)",
                "accepted_column_schemas": [
                    list(API_TO_RAW_COLUMN_MAP.keys()),
                    list(RAW_FEATURE_COLUMNS),
                ],
                "max_batch_rows": int(os.getenv("MAX_BATCH_ROWS", "5000")),
            },
        },
        required_features=required_features,
        transformed_features=info["transformed_features"],
        explainability_supported=info["explainability_supported"],
        explainer_type=info["explainer_type"],
        trained_at_utc=info["trained_at_utc"],
        dataset_info=info["dataset_info"],
        evaluation_metrics=info["evaluation_metrics"],
        global_feature_importance=info["global_feature_importance"],
        disclaimer=info["disclaimer"],
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Predict liver disease risk score for an individual patient",
    dependencies=[Depends(verify_api_key)],
)
async def predict_single(
    patient: PatientFeatures,
    predictor: LiverDiseasePredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Accept a single validated patient record and return prediction, probability, and risk tier."""
    try:
        result = predictor.predict_one(patient.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_422, detail=str(exc)) from exc
    return PredictionResponse(**result)


@router.post(
    "/explain",
    response_model=ExplainResponse,
    tags=["Explainability"],
    summary="Predict and explain an individual prediction using SHAP",
    dependencies=[Depends(verify_api_key)],
)
async def explain_single(
    patient: PatientFeatures,
    predictor: LiverDiseasePredictor = Depends(get_predictor),
) -> ExplainResponse:
    """Return individual prediction along with SHAP local feature attributions."""
    try:
        result = predictor.explain_one(patient.model_dump(), top_k=6)
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
    return ExplainResponse(**result)


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Batch predictions via CSV file upload",
    dependencies=[Depends(verify_api_key)],
)
async def predict_batch(
    file: UploadFile = File(..., description="CSV file with ILPD or canonical API columns."),
    predictor: LiverDiseasePredictor = Depends(get_predictor),
) -> BatchPredictionResponse:
    """Validate uploaded CSV strictly and return predictions for all valid rows plus row-level error details."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a '.csv' file.",
        )

    raw_bytes = await file.read()
    if not raw_bytes or not raw_bytes.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV file is empty.",
        )

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV must be valid UTF-8 encoded text.",
        ) from exc

    try:
        df = pd.read_csv(io.StringIO(text), keep_default_na=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed CSV file: {exc}",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV contains headers but zero data rows.",
        )

    max_rows = int(os.getenv("MAX_BATCH_ROWS", "5000"))
    if len(df) > max_rows:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"CSV contains {len(df)} rows, exceeding MAX_BATCH_ROWS={max_rows}.",
        )

    # Validate exact column schema (either raw ILPD columns or canonical snake_case API columns)
    df.columns = [str(c).strip() for c in df.columns]
    col_set = set(df.columns)
    raw_set = set(RAW_FEATURE_COLUMNS)
    api_set = set(API_TO_RAW_COLUMN_MAP.keys())

    if col_set == raw_set:
        df_api = df.rename(columns=RAW_TO_API_COLUMN_MAP)
    elif col_set == api_set:
        df_api = df.copy()
    else:
        if len(col_set & api_set) > len(col_set & raw_set):
            missing = sorted(api_set - col_set)
            extra = sorted(col_set - api_set)
        else:
            missing = sorted(raw_set - col_set)
            extra = sorted(col_set - raw_set)
        raise HTTPException(
            status_code=HTTP_422,
            detail=(
                f"Invalid CSV columns. Missing: {missing}; Unexpected: {extra}. "
                f"Expected exact raw ILPD columns {RAW_FEATURE_COLUMNS} "
                f"or canonical API columns {list(API_TO_RAW_COLUMN_MAP.keys())}."
            ),
        )

    # Row-by-row strict Pydantic validation (never silently accept invalid medical data)
    valid_rows: List[Dict[str, Any]] = []
    valid_row_indices: List[int] = []
    invalid_records: List[BatchInvalidRecord] = []

    for row_idx, row in df_api.iterrows():
        raw_dict = {
            k: (None if pd.isna(v) else v)
            for k, v in row.to_dict().items()
        }
        try:
            validated = PatientFeatures(**raw_dict)
            valid_rows.append(validated.model_dump())
            valid_row_indices.append(int(row_idx))
        except ValidationError as exc:
            err_msgs = [
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            ]
            invalid_records.append(BatchInvalidRecord(row_index=int(row_idx), errors=err_msgs))

    if not valid_rows:
        raise HTTPException(
            status_code=HTTP_422,
            detail={
                "message": "All rows in the uploaded CSV failed medical input validation.",
                "invalid_count": len(invalid_records),
                "invalid_records": [r.model_dump() for r in invalid_records[:25]],
            },
        )

    valid_df = pd.DataFrame(valid_rows)
    raw_preds = predictor.predict_dataframe(valid_df, allow_missing=False)
    predictions: List[BatchPredictionRecord] = []
    for orig_idx, pred in zip(valid_row_indices, raw_preds):
        pred["row_index"] = orig_idx
        predictions.append(BatchPredictionRecord(**pred))

    return BatchPredictionResponse(
        batch_id=str(uuid.uuid4()),
        model_version=predictor.model_version,
        total_rows=int(len(df)),
        valid_count=len(predictions),
        invalid_count=len(invalid_records),
        predictions=predictions,
        invalid_records=invalid_records,
        disclaimer=MEDICAL_DISCLAIMER,
    )
