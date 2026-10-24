import logging
from io import BytesIO
from pathlib import Path
from PIL import Image
from .config import OCRConfig
from .exceptions import OCREngineError
from .file_validator import validate_upload
from .image_processor import load_image
from .medical_extractor import extract_results
from .ocr_engine import TesseractEngine
from .pdf_processor import extract_pdf_pages
from .schemas import LabReportResult, PageOCR
from .text_cleaner import clean_ocr_text
from .result_validator import validate_report

logger = logging.getLogger(__name__)

class LabOCRService:
    def __init__(self, config: OCRConfig | None = None, engine=None):
        self.config = config or OCRConfig()
        self._engine = engine

    def _get_engine(self):
        if self._engine is None:
            self._engine = TesseractEngine(self.config)
        return self._engine

    def process_bytes(self, filename: str, content: bytes) -> LabReportResult:
        suffix = validate_upload(filename, content, self.config)
        report = LabReportResult(source_filename=Path(filename.replace("\\", "/")).name)
        pages: list[PageOCR] = []
        if suffix == ".pdf":
            pdf_pages = extract_pdf_pages(content, self.config)
            for page_no, selectable_text, rendered_png, method in pdf_pages:
                if method == "pdf_text":
                    text, words, confidence = clean_ocr_text(selectable_text), [], None
                else:
                    try:
                        image = Image.open(BytesIO(rendered_png)).convert("RGB")
                        text, words, confidence = self._get_engine().recognize(image)
                        text = clean_ocr_text(text)
                    except OCREngineError:
                        raise
                    except Exception as exc:
                        logger.warning("Page processing failed (page=%d, type=%s)", page_no, type(exc).__name__)
                        text, words, confidence = "", [], None
                        report.warnings.append(f"Page {page_no} could not be processed.")
                pages.append(PageOCR(page_number=page_no, text=text, method=method if text else "empty",
                                     average_confidence=confidence, words=words))
        else:
            image = load_image(content)
            text, words, confidence = self._get_engine().recognize(image)
            pages.append(PageOCR(page_number=1, text=clean_ocr_text(text), method="ocr",
                                 average_confidence=confidence, words=words))
        report.pages = pages
        report.raw_text = "\n\n".join(f"[Page {p.page_number}]\n{p.text}" for p in pages)
        for page in pages:
            report.results.extend(extract_results(page.text, page.page_number, page.average_confidence))
        if any(not p.text for p in pages):
            report.warnings.append("One or more pages produced no readable text.")
        logger.info("Lab report processed: pages=%d results=%d", len(pages), len(report.results))
        return validate_report(report)

    def process_file(self, path: str | Path) -> LabReportResult:
        path = Path(path)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ValueError("The selected file could not be read.") from exc
        return self.process_bytes(path.name, content)
