"""Medical laboratory OCR feature."""
from .service import LabOCRService
from .schemas import LabReportResult, LabResult

__all__ = ["LabOCRService", "LabReportResult", "LabResult"]
