"""Shared pytest fixtures for the Liver Disease Prediction System test suite."""

from __future__ import annotations

from typing import Any, Dict, Iterator

import pytest
from fastapi.testclient import TestClient

from liver_disease.api.main import create_app
from liver_disease.config import DEFAULT_MODEL_ARTIFACT_PATH

TEST_API_KEY = "test-secure-api-key-for-automated-qa-only-9f8e7d6c"


@pytest.fixture()
def sample_high_risk_patient() -> Dict[str, Any]:
    """Patient profile with markedly elevated bilirubin and liver enzymes."""
    return {
        "age": 62,
        "gender": "Male",
        "total_bilirubin": 10.9,
        "direct_bilirubin": 5.5,
        "alkaline_phosphotase": 699.0,
        "alamine_aminotransferase": 64.0,
        "aspartate_aminotransferase": 100.0,
        "total_proteins": 7.5,
        "albumin": 3.2,
        "albumin_and_globulin_ratio": 0.74,
    }


@pytest.fixture()
def sample_low_risk_patient() -> Dict[str, Any]:
    """Younger patient profile with normal-range liver function markers."""
    return {
        "age": 24,
        "gender": "Female",
        "total_bilirubin": 0.6,
        "direct_bilirubin": 0.1,
        "alkaline_phosphotase": 140.0,
        "alamine_aminotransferase": 18.0,
        "aspartate_aminotransferase": 19.0,
        "total_proteins": 7.2,
        "albumin": 4.2,
        "albumin_and_globulin_ratio": 1.4,
    }


@pytest.fixture()
def auth_headers() -> Dict[str, str]:
    """Valid API authentication headers matching the test environment key."""
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture()
def api_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """FastAPI TestClient with configured test API key and loaded production artifact."""
    monkeypatch.setenv("LIVER_API_KEY", TEST_API_KEY)
    app = create_app(model_artifact_path=DEFAULT_MODEL_ARTIFACT_PATH)
    with TestClient(app) as client:
        yield client
