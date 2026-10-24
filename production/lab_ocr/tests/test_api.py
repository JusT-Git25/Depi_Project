import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lab_ocr.api import create_app
from lab_ocr.config import OCRConfig
from lab_ocr.service import LabOCRService


class FakeEngine:
    def recognize(self, image):
        return "Glucose: 150 mg/dL (Reference Range: 70-100)\nHbA1c: 7.0 %", [], 93.0


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (60, 30), "white").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(predictor):
    return TestClient(create_app(LabOCRService(OCRConfig(), engine=FakeEngine()), predictor=predictor))


def test_index_and_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "Lab report analyzer" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert client.get("/api/health").headers["cache-control"] == "no-store"


def test_no_streamlit_dependency():
    import sys
    import lab_ocr.api  # noqa: F401
    assert "streamlit" not in sys.modules


def test_health_reports_model(client):
    h = client.get("/api/health").json()
    assert h["model_loaded"] and "Glucose" in h["known_tests"]


def test_extract_then_predict(client):
    r = client.post("/api/extract", files={"file": ("r.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert all("words" not in p for p in body["pages"])
    names = {x["test_name"] for x in body["results"]}
    assert {"Glucose", "HbA1c"} <= names

    unconfirmed = client.post("/api/predict", json={"results": body["results"]}).json()
    assert unconfirmed["status"] == "insufficient_data"  # nothing reviewed yet

    ok = client.post("/api/predict", json={"results": body["results"], "reviewed_all": True}).json()
    assert ok["status"] == "ok" and ok["prediction"] in {"low", "high"}


def test_extract_rejects_bad_file_type(client):
    r = client.post("/api/extract", files={"file": ("evil.exe", b"abc", "application/octet-stream")})
    assert r.status_code == 400


def test_extract_rejects_oversize(predictor):
    c = TestClient(create_app(LabOCRService(OCRConfig(max_file_size_mb=1), engine=FakeEngine()), predictor=predictor))
    r = c.post("/api/extract", files={"file": ("big.png", b"0" * (1024 * 1024 + 10), "image/png")})
    assert r.status_code == 400


def test_predict_without_model_is_503():
    c = TestClient(create_app(LabOCRService(OCRConfig(), engine=FakeEngine())))
    assert c.post("/api/predict", json={"results": []}).status_code == 503


def test_export_csv(client):
    body = client.post("/api/extract", files={"file": ("r.png", png_bytes(), "image/png")}).json()
    r = client.post("/api/export/csv", json={"results": body["results"]})
    assert r.status_code == 200 and "Glucose" in r.text
    assert client.post("/api/export/xml", json={"results": []}).status_code == 404
