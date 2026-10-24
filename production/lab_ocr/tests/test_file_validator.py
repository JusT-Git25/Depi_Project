import pytest
from lab_ocr.config import OCRConfig
from lab_ocr.file_validator import validate_upload
from lab_ocr.exceptions import FileValidationError, UnsupportedFileError

def test_valid_pdf():
    assert validate_upload("report.pdf", b"%PDF-1.7 fake", OCRConfig()) == ".pdf"

def test_reject_unsupported():
    with pytest.raises(UnsupportedFileError):
        validate_upload("report.exe", b"abc", OCRConfig())

def test_reject_empty():
    with pytest.raises(FileValidationError):
        validate_upload("report.png", b"", OCRConfig())

def test_reject_fake_pdf():
    with pytest.raises(FileValidationError):
        validate_upload("report.pdf", b"not a pdf", OCRConfig())
