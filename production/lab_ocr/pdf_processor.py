import fitz
from .config import OCRConfig
from .exceptions import FileValidationError, PDFProcessingError
from .schemas import PageOCR

def extract_pdf_pages(content: bytes, config: OCRConfig) -> list[tuple[int, str, bytes | None, str]]:
    """Return (page number, selectable text, rendered PNG or None, method)."""
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError("The PDF is corrupted, unsupported, or password-protected.") from exc
    try:
        if doc.needs_pass:
            raise PDFProcessingError("Password-protected PDFs are not supported. Upload an unlocked copy.")
        if len(doc) == 0:
            raise FileValidationError("The PDF contains no pages.")
        if len(doc) > config.max_pdf_pages:
            raise FileValidationError(f"The PDF has {len(doc)} pages; the limit is {config.max_pdf_pages}.")
        pages = []
        scale = config.pdf_render_dpi / 72
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            if len("".join(text.split())) >= config.min_selectable_text_chars:
                pages.append((i + 1, text, None, "pdf_text"))
            else:
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                pages.append((i + 1, text, pix.tobytes("png"), "ocr"))
        return pages
    finally:
        doc.close()
