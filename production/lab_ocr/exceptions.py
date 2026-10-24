class LabOCRError(Exception):
    """Base exception with a safe user-facing message."""

class UnsupportedFileError(LabOCRError):
    pass

class FileValidationError(LabOCRError):
    pass

class PDFProcessingError(LabOCRError):
    pass

class OCREngineError(LabOCRError):
    pass

class PredictionError(LabOCRError):
    """Model could not be loaded or run. Message is safe to show to the user."""
