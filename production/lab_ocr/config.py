from dataclasses import dataclass
import os
from pathlib import Path

@dataclass(frozen=True)
class OCRConfig:
    max_file_size_mb: int = 15
    max_pdf_pages: int = 40
    pdf_render_dpi: int = 250
    min_selectable_text_chars: int = 35
    tesseract_cmd: str | None = os.getenv("TESSERACT_CMD")
    language: str = os.getenv("LAB_OCR_LANGUAGE", "eng")
    timeout_seconds: int = 30
    temp_root: Path | None = None

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024
