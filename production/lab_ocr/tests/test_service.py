import fitz
from lab_ocr.service import LabOCRService
from lab_ocr.config import OCRConfig

class FakeEngine:
    def recognize(self, image):
        return "Glucose: 95 mg/dL (Reference Range: 70-100)", [], 93.0

def test_digital_pdf_avoids_ocr():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "CBC Report\\nHemoglobin: 12.5 g/dL (Reference Range: 12.0-16.0)\\nWBC: 7.2 x10^3/uL")
    content = doc.tobytes()
    doc.close()
    class ExplodingEngine:
        def recognize(self, image):
            raise AssertionError("OCR should not run on selectable text")
    report = LabOCRService(OCRConfig(min_selectable_text_chars=10), engine=ExplodingEngine()).process_bytes("sample.pdf", content)
    assert report.pages[0].method == "pdf_text"
    assert "Hemoglobin" in report.raw_text

def test_scanned_page_uses_fake_ocr():
    doc = fitz.open()
    doc.new_page()
    content = doc.tobytes()
    doc.close()
    report = LabOCRService(OCRConfig(min_selectable_text_chars=10), engine=FakeEngine()).process_bytes("scan.pdf", content)
    assert report.pages[0].method == "ocr"
    assert report.results
