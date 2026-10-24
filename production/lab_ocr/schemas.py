from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict

class ReferenceRange(BaseModel):
    low: float | None = None
    high: float | None = None
    operator: str | None = None
    raw: str | None = None

class LabResult(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    test_name: str | None = None
    original_test_name: str | None = None
    raw_result_text: str
    value: float | str | None = None
    value_operator: str | None = None
    unit: str | None = None
    reference_range: ReferenceRange | None = None
    abnormal_flag: str | None = None
    specimen_type: str | None = None
    page_number: int = Field(ge=1)
    confidence: float | None = Field(default=None, ge=0, le=100)
    validation_status: Literal["validated", "needs_review", "invalid"] = "needs_review"
    validation_warnings: list[str] = Field(default_factory=list)
    manually_reviewed: bool = False

class PageOCR(BaseModel):
    page_number: int = Field(ge=1)
    text: str
    method: Literal["pdf_text", "ocr", "empty"]
    average_confidence: float | None = None
    words: list[dict[str, Any]] = Field(default_factory=list)

class LabReportResult(BaseModel):
    source_filename: str
    report_date: str | None = None
    collection_date: str | None = None
    facility_name: str | None = None
    pages: list[PageOCR] = Field(default_factory=list)
    raw_text: str = ""
    results: list[LabResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    processing_status: Literal["success", "success_with_warnings", "review_required", "failed"] = "review_required"
