import os
import logging
from PIL import Image
from .config import OCRConfig
from .exceptions import OCREngineError
from .image_processor import preprocess_image

logger = logging.getLogger(__name__)

class TesseractEngine:
    def __init__(self, config: OCRConfig):
        self.config = config
        try:
            import pytesseract
            from pytesseract import Output
            self._pytesseract = pytesseract
            self._output = Output
            if config.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
            # Fail early with a safe, actionable message.
            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise OCREngineError(
                "Tesseract OCR is unavailable. Install Tesseract and its language data, "
                "or set TESSERACT_CMD to the executable path."
            ) from exc

    def recognize(self, image: Image.Image) -> tuple[str, list[dict], float | None]:
        try:
            prepared = preprocess_image(image)
            data = self._pytesseract.image_to_data(
                prepared, lang=self.config.language, output_type=self._output.DICT,
                timeout=self.config.timeout_seconds
            )
            words = []
            confidences = []
            for i, token in enumerate(data.get("text", [])):
                token = (token or "").strip()
                try:
                    conf = float(data["conf"][i])
                except (ValueError, TypeError, IndexError):
                    conf = -1
                if token:
                    item = {
                        "text": token,
                        "confidence": conf if conf >= 0 else None,
                        "bbox": {
                            "x": int(data["left"][i]), "y": int(data["top"][i]),
                            "width": int(data["width"][i]), "height": int(data["height"][i])
                        }
                    }
                    words.append(item)
                    if conf >= 0:
                        confidences.append(conf)
            text = self._pytesseract.image_to_string(
                prepared, lang=self.config.language, timeout=self.config.timeout_seconds
            )
            return text.strip(), words, (sum(confidences) / len(confidences) if confidences else None)
        except Exception as exc:
            logger.warning("OCR execution failed (%s)", type(exc).__name__)
            raise OCREngineError("OCR could not process this page. Try a clearer image or another file.") from exc
