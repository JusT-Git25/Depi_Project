"""Security tests for API key authentication and fail-closed behavior."""

from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from liver_disease.api.main import create_app
from liver_disease.config import DEFAULT_MODEL_ARTIFACT_PATH


def test_missing_api_key_returns_401(
    api_client: TestClient,
    sample_high_risk_patient: Dict[str, Any],
) -> None:
    assert api_client.get("/model-info").status_code == 401
    assert api_client.post("/predict", json=sample_high_risk_patient).status_code == 401
    assert api_client.post("/explain", json=sample_high_risk_patient).status_code == 401
    assert (
        api_client.post(
            "/predict/batch",
            files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")},
        ).status_code
        == 401
    )


def test_invalid_api_key_returns_401(
    api_client: TestClient,
    sample_high_risk_patient: Dict[str, Any],
) -> None:
    bad_headers = {"X-API-Key": "definitely-wrong-key"}
    assert api_client.get("/model-info", headers=bad_headers).status_code == 401
    assert api_client.post("/predict", json=sample_high_risk_patient, headers=bad_headers).status_code == 401
    assert api_client.post("/explain", json=sample_high_risk_patient, headers=bad_headers).status_code == 401


def test_unset_server_api_key_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    sample_high_risk_patient: Dict[str, Any],
) -> None:
    monkeypatch.delenv("LIVER_API_KEY", raising=False)
    app = create_app(model_artifact_path=DEFAULT_MODEL_ARTIFACT_PATH)
    with TestClient(app) as client:
        resp = client.post(
            "/predict",
            json=sample_high_risk_patient,
            headers={"X-API-Key": "anything"},
        )
        assert resp.status_code == 503
