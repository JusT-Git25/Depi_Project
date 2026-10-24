from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
from .exceptions import FileValidationError

def load_image(content: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(content))
        image.load()
        if getattr(image, "n_frames", 1) > 1:
            image.seek(0)
        return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise FileValidationError("The uploaded image is corrupted or cannot be decoded.") from exc

def preprocess_image(image: Image.Image) -> Image.Image:
    image = ImageOps.grayscale(image)
    # Upscale small scans for better OCR without unbounded growth.
    max_dimension = max(image.size)
    if max_dimension < 1800:
        scale = min(2.0, 1800 / max_dimension)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    # Avoid aggressive thresholding by default; it can destroy faint text.
    return image
