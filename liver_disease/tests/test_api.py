"""End-to-end integration tests for FastAPI endpoints, Pydantic validation, and batch CSV uploads."""

from __future__ import annotations

from typing import Any, Dict

from fastapi.testclient import TestClient


def test_health_and_swagger_docs(api_client: TestClient) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_version"] == "1.0.0"

    docs_resp = api_client.get("/docs")
    assert docs_resp.status_code == 200
    assert "swagger" in docs_resp.text.lower()


def test_model_info_endpoint(api_client: TestClient, auth_headers: Dict[str, str]) -> None:
    resp = api_client.get("/model-info", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_version"] == "1.0.0"
    assert len(data["required_features"]) == 10
    assert data["explainability_supported"] is True
    assert "disclaimer" in data


def test_predict_valid_patient(
    api_client: TestClient,
    auth_headers: Dict[str, str],
    sample_high_risk_patient: Dict[str, Any],
) -> None:
    resp = api_client.post("/predict", json=sample_high_risk_patient, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in (1, 2)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_category"] in ("Low", "Moderate", "High")
    assert data["model_version"] == "1.0.0"
    assert data["prediction_id"]


def test_predict_rejects_invalid_and_unexpected_inputs(
    api_client: TestClient,
    auth_headers: Dict[str, str],
    sample_high_risk_patient: Dict[str, Any],
) -> None:
    # Missing required field
    missing_payload = dict(sample_high_risk_patient)
    missing_payload.pop("albumin")
    assert api_client.post("/predict", json=missing_payload, headers=auth_headers).status_code == 422

    # Unexpected extra field
    extra_payload = {**sample_high_risk_patient, "extra_field": 42}
    assert api_client.post("/predict", json=extra_payload, headers=auth_headers).status_code == 422

    # Negative bilirubin
    neg_payload = {**sample_high_risk_patient, "total_bilirubin": -1.5}
    assert api_client.post("/predict", json=neg_payload, headers=auth_headers).status_code == 422

    # Invalid gender
    bad_gender = {**sample_high_risk_patient, "gender": "Other"}
    assert api_client.post("/predict", json=bad_gender, headers=auth_headers).status_code == 422

    # Boolean passed for numeric field
    bool_payload = {**sample_high_risk_patient, "age": True}
    assert api_client.post("/predict", json=bool_payload, headers=auth_headers).status_code == 422


def test_explain_endpoint(
    api_client: TestClient,
    auth_headers: Dict[str, str],
    sample_high_risk_patient: Dict[str, Any],
) -> None:
    resp = api_client.post("/explain", json=sample_high_risk_patient, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "explanation" in data
    assert len(data["explanation"]["top_contributing_features"]) > 0
    assert "limitations_note" in data["explanation"]


def test_batch_csv_valid_and_mixed_invalid_rows(
    api_client: TestClient,
    auth_headers: Dict[str, str],
) -> None:
    # Canonical API CSV with 2 valid rows + 1 invalid row (negative TB)
    csv_content = (
        "age,gender,total_bilirubin,direct_bilirubin,alkaline_phosphotase,"
        "alamine_aminotransferase,aspartate_aminotransferase,total_proteins,albumin,albumin_and_globulin_ratio\n"
        "62,Male,10.9,5.5,699,64,100,7.5,3.2,0.74\n"
        "24,Female,0.6,0.1,140,18,19,7.2,4.2,1.40\n"
        "45,Male,-5.0,0.2,190,25,30,6.8,3.4,1.00\n"
    )
    resp = api_client.post(
        "/predict/batch",
        files={"file": ("patients.csv", csv_content.encode("utf-8"), "text/csv")},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_rows"] == 3
    assert data["valid_count"] == 2
    assert data["invalid_count"] == 1
    assert len(data["predictions"]) == 2
    assert data["invalid_records"][0]["row_index"] == 2

    # Raw ILPD CSV schema also supported
    raw_csv = (
        "Age,Gender,TB,DB,Alkphos,Sgpt,Sgot,TP,ALB,A/G Ratio\n"
        "62,Male,10.9,5.5,699,64,100,7.5,3.2,0.74\n"
    )
    resp_raw = api_client.post(
        "/predict/batch",
        files={"file": ("ilpd_raw.csv", raw_csv.encode("utf-8"), "text/csv")},
        headers=auth_headers,
    )
    assert resp_raw.status_code == 200
    assert resp_raw.json()["valid_count"] == 1


def test_batch_csv_error_handling(
    api_client: TestClient,
    auth_headers: Dict[str, str],
) -> None:
    # Non-CSV extension
    resp_ext = api_client.post(
        "/predict/batch",
        files={"file": ("data.txt", b"hello", "text/plain")},
        headers=auth_headers,
    )
    assert resp_ext.status_code == 400

    # Empty CSV
    resp_empty = api_client.post(
        "/predict/batch",
        files={"file": ("empty.csv", b"", "text/csv")},
        headers=auth_headers,
    )
    assert resp_empty.status_code == 400

    # Unexpected columns
    bad_cols = "age,gender,foo,bar\n45,Male,1,2\n"
    resp_cols = api_client.post(
        "/predict/batch",
        files={"file": ("bad_cols.csv", bad_cols.encode("utf-8"), "text/csv")},
        headers=auth_headers,
    )
    assert resp_cols.status_code == 422

    # All rows invalid
    all_bad = (
        "age,gender,total_bilirubin,direct_bilirubin,alkaline_phosphotase,"
        "alamine_aminotransferase,aspartate_aminotransferase,total_proteins,albumin,albumin_and_globulin_ratio\n"
        "250,Unknown,-1,-2,-3,-4,-5,-6,-7,-8\n"
    )
    resp_all_bad = api_client.post(
        "/predict/batch",
        files={"file": ("all_bad.csv", all_bad.encode("utf-8"), "text/csv")},
        headers=auth_headers,
    )
    assert resp_all_bad.status_code == 422
