from pathlib import Path
from .config import OCRConfig
from .exceptions import FileValidationError, UnsupportedFileError

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
ALLOWED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

def validate_upload(filename: str, content: bytes, config: OCRConfig) -> str:
    # Use only the basename; never use user-supplied path components for file access.
    suffix = Path(filename.replace("\\", "/")).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileError("Unsupported file type. Upload a PDF or supported image (PNG/JPG/TIFF/BMP/WEBP).")
    if not content:
        raise FileValidationError("The uploaded file is empty.")
    if len(content) > config.max_file_size_bytes:
        raise FileValidationError(f"File exceeds the configured {config.max_file_size_mb} MB limit.")
    if suffix == ".pdf" and not content.startswith(b"%PDF-"):
        raise FileValidationError("The file extension says PDF, but the file does not appear to be a valid PDF.")
    return suffix
