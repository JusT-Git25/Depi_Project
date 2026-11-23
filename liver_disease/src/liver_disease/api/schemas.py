"""Strict Pydantic v2 request and response schemas for the Liver Disease Prediction API."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from liver_disease.config import FEATURE_SPECS


def _reject_bool_and_nonfinite(value: Any, field_name: str) -> float:
    """Reject booleans, NaN, and Inf for numeric medical attributes."""
    if isinstance(value, bool):
        raise ValueError(f"'{field_name}' must be a numeric medical measurement, not a boolean.")
    try:
        num = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must be a valid number.") from exc
    if not math.isfinite(num):
        raise ValueError(f"'{field_name}' must be a finite numeric value.")
    return num


class PatientFeatures(BaseModel):
    """Input schema representing one patient's ILPD liver function test attributes.

    All fields are strictly required, range-checked against medically plausible bounds,
    and documented with their clinical measurement units. Extra/unexpected fields are forbidden.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 62,
                "gender": "Male",
                "total_bilirubin": 10.9,
                "direct_bilirubin": 5.5,
                "alkaline_phosphotase": 699,
                "alamine_aminotransferase": 64,
                "aspartate_aminotransferase": 100,
                "total_proteins": 7.5,
                "albumin": 3.2,
                "albumin_and_globulin_ratio": 0.74,
            }
        },
    )

    age: int = Field(
        ...,
        ge=int(FEATURE_SPECS["Age"].min_value or 1),
        le=int(FEATURE_SPECS["Age"].max_value or 120),
        description="Patient age in years (allowed: 1–120 years).",
    )
    gender: Literal["Male", "Female"] = Field(
        ...,
        description="Patient gender ('Male' or 'Female').",
    )
    total_bilirubin: float = Field(
        ...,
        ge=FEATURE_SPECS["TB"].min_value,
        le=FEATURE_SPECS["TB"].max_value,
        description="Total Bilirubin (TB) in mg/dL (normal reference: 0.1–1.2 mg/dL; allowed: 0.05–100.0).",
    )
    direct_bilirubin: float = Field(
        ...,
        ge=FEATURE_SPECS["DB"].min_value,
        le=FEATURE_SPECS["DB"].max_value,
        description="Direct Bilirubin (DB) in mg/dL (normal reference: 0.0–0.3 mg/dL; allowed: 0.0–50.0).",
    )
    alkaline_phosphotase: float = Field(
        ...,
        ge=FEATURE_SPECS["Alkphos"].min_value,
        le=FEATURE_SPECS["Alkphos"].max_value,
        description="Alkaline Phosphotase (ALP) in IU/L (normal reference: 44–147 IU/L; allowed: 10–5000).",
    )
    alamine_aminotransferase: float = Field(
        ...,
        ge=FEATURE_SPECS["Sgpt"].min_value,
        le=FEATURE_SPECS["Sgpt"].max_value,
        description="Alamine Aminotransferase (ALT / SGPT) in IU/L (normal reference: 7–56 IU/L; allowed: 1–5000).",
    )
    aspartate_aminotransferase: float = Field(
        ...,
        ge=FEATURE_SPECS["Sgot"].min_value,
        le=FEATURE_SPECS["Sgot"].max_value,
        description="Aspartate Aminotransferase (AST / SGOT) in IU/L (normal reference: 10–40 IU/L; allowed: 1–10000).",
    )
    total_proteins: float = Field(
        ...,
        ge=FEATURE_SPECS["TP"].min_value,
        le=FEATURE_SPECS["TP"].max_value,
        description="Total Serum Proteins (TP) in g/dL (normal reference: 6.0–8.3 g/dL; allowed: 1.0–20.0).",
    )
    albumin: float = Field(
        ...,
        ge=FEATURE_SPECS["ALB"].min_value,
        le=FEATURE_SPECS["ALB"].max_value,
        description="Serum Albumin (ALB) in g/dL (normal reference: 3.5–5.0 g/dL; allowed: 0.5–12.0).",
    )
    albumin_and_globulin_ratio: float = Field(
        ...,
        ge=FEATURE_SPECS["A/G Ratio"].min_value,
        le=FEATURE_SPECS["A/G Ratio"].max_value,
        description="Albumin and Globulin Ratio (A/G Ratio), dimensionless (normal reference: 1.0–2.5; allowed: 0.1–10.0).",
    )

    @field_validator("age", mode="before")
    @classmethod
    def _validate_age_not_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("'age' must be an integer, not a boolean.")
        if isinstance(v, float) and not v.is_integer():
            raise ValueError("'age' must be a whole number of years.")
        return int(v) if isinstance(v, float) else v

    @field_validator("gender", mode="before")
    @classmethod
    def _strip_gender(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator(
        "total_bilirubin",
        "direct_bilirubin",
        "alkaline_phosphotase",
        "alamine_aminotransferase",
        "aspartate_aminotransferase",
        "total_proteins",
        "albumin",
        "albumin_and_globulin_ratio",
        mode="before",
    )
    @classmethod
    def _validate_numeric(cls, v: Any, info: Any) -> float:
        return _reject_bool_and_nonfinite(v, info.field_name)


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: Literal["healthy", "degraded"]
    api_version: str
    model_loaded: bool
    model_version: Optional[str] = None
    model_name: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response schema for POST /predict."""

    prediction_id: str
    prediction: Literal[1, 2] = Field(..., description="1 = Liver Disease, 2 = No Liver Disease")
    prediction_label: Literal["Liver Disease", "No Liver Disease"]
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model score P(Liver Disease). Not a clinically calibrated diagnostic probability.",
    )
    risk_category: Literal["Low", "Moderate", "High"]
    decision_threshold: float
    model_version: str
    disclaimer: str


class BatchPredictionRecord(BaseModel):
    """Single valid row result inside POST /predict/batch."""

    row_index: int
    prediction_id: str
    prediction: Literal[1, 2]
    prediction_label: Literal["Liver Disease", "No Liver Disease"]
    probability: float
    risk_category: Literal["Low", "Moderate", "High"]
    decision_threshold: float
    model_version: str


class BatchInvalidRecord(BaseModel):
    """Validation failure report for an invalid row inside POST /predict/batch."""

    row_index: int
    errors: List[str]


class BatchPredictionResponse(BaseModel):
    """Response schema for POST /predict/batch."""

    batch_id: str
    model_version: str
    total_rows: int
    valid_count: int
    invalid_count: int
    predictions: List[BatchPredictionRecord]
    invalid_records: List[BatchInvalidRecord]
    disclaimer: str


class FeatureContribution(BaseModel):
    """Individual SHAP feature attribution entry."""

    feature: str
    raw_value: Any
    transformed_value: float
    shap_value: float
    abs_impact: float
    direction: Literal["increases_risk", "decreases_risk"]


class ExplanationDetail(BaseModel):
    """Nested SHAP explanation detail."""

    explainer_type: str
    base_value: float
    top_contributing_features: List[FeatureContribution]
    all_feature_contributions: List[FeatureContribution]
    limitations_note: str
    disclaimer: str


class ExplainResponse(PredictionResponse):
    """Response schema for POST /explain."""

    explanation: ExplanationDetail


class FeatureMetadata(BaseModel):
    """Feature definition returned inside GET /model-info."""

    api_name: str
    raw_csv_name: str
    display_name: str
    dtype: str
    unit: str
    min_value: Optional[float]
    max_value: Optional[float]
    normal_reference_range: str
    description: str


class ModelInfoResponse(BaseModel):
    """Response schema for GET /model-info."""

    model_version: str
    model_name: str
    model_type: str
    calibration_method: str
    decision_threshold: float
    default_threshold: float
    risk_category_bands: Dict[str, str]
    target_encoding: Dict[str, str]
    supported_input_formats: Dict[str, Any]
    required_features: List[FeatureMetadata]
    transformed_features: List[str]
    explainability_supported: bool
    explainer_type: Optional[str]
    trained_at_utc: Optional[str]
    dataset_info: Dict[str, Any]
    evaluation_metrics: Dict[str, Any]
    global_feature_importance: List[Dict[str, Any]]
    disclaimer: str
