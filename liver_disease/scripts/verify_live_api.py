"""Live HTTP smoke-test script for Phase 7 & 8: starts real Uvicorn FastAPI server,
probes every endpoint over TCP/HTTP, verifies auth (401) and validation (422), and shuts down cleanly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORT = 8017
BASE_URL = f"http://127.0.0.1:{PORT}"
SMOKE_KEY = "live-smoke-test-key-32-chars-minimum-8a7b6c5d"


def wait_for_server(url: str, timeout_s: float = 25.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"FastAPI server did not become ready within {timeout_s}s.")


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["LIVER_API_KEY"] = SMOKE_KEY

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "liver_disease.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        wait_for_server(BASE_URL)
        headers = {"X-API-Key": SMOKE_KEY}

        # 1. GET /health
        r_health = requests.get(f"{BASE_URL}/health", timeout=5)
        assert r_health.status_code == 200, r_health.text
        h = r_health.json()
        assert h["status"] == "healthy" and h["model_loaded"] is True
        print(f"[PASS] GET /health -> {h}")

        # 2. GET /docs
        r_docs = requests.get(f"{BASE_URL}/docs", timeout=5)
        assert r_docs.status_code == 200 and "swagger" in r_docs.text.lower()
        print("[PASS] GET /docs -> Swagger UI served (HTTP 200)")

        # 3. Auth check -> 401 without key
        r_unauth = requests.get(f"{BASE_URL}/model-info", timeout=5)
        assert r_unauth.status_code == 401, f"Expected 401, got {r_unauth.status_code}"
        print("[PASS] GET /model-info without key -> HTTP 401")

        # 4. GET /model-info with valid key
        r_info = requests.get(f"{BASE_URL}/model-info", headers=headers, timeout=10)
        assert r_info.status_code == 200, r_info.text
        info = r_info.json()
        print(
            f"[PASS] GET /model-info -> model={info['model_name']} v{info['model_version']} "
            f"threshold={info['decision_threshold']} explainer={info['explainer_type']}"
        )

        # 5. POST /predict
        patient = {
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
        r_pred = requests.post(f"{BASE_URL}/predict", json=patient, headers=headers, timeout=10)
        assert r_pred.status_code == 200, r_pred.text
        pred = r_pred.json()
        print(
            f"[PASS] POST /predict -> class={pred['prediction']} ({pred['prediction_label']}), "
            f"prob={pred['probability']}, risk={pred['risk_category']}"
        )

        # 6. POST /predict invalid input -> 422
        r_bad = requests.post(
            f"{BASE_URL}/predict",
            json={**patient, "total_bilirubin": -3.0},
            headers=headers,
            timeout=10,
        )
        assert r_bad.status_code == 422, f"Expected 422, got {r_bad.status_code}"
        print("[PASS] POST /predict negative bilirubin -> HTTP 422")

        # 7. POST /explain
        r_exp = requests.post(f"{BASE_URL}/explain", json=patient, headers=headers, timeout=25)
        assert r_exp.status_code == 200, r_exp.text
        exp = r_exp.json()
        top_feat = exp["explanation"]["top_contributing_features"][0]
        print(
            f"[PASS] POST /explain -> explainer={exp['explanation']['explainer_type']}, "
            f"top_driver={top_feat['feature']} (SHAP={top_feat['shap_value']:+.4f})"
        )

        # 8. POST /predict/batch
        csv_body = (
            "age,gender,total_bilirubin,direct_bilirubin,alkaline_phosphotase,"
            "alamine_aminotransferase,aspartate_aminotransferase,total_proteins,albumin,albumin_and_globulin_ratio\n"
            "62,Male,10.9,5.5,699,64,100,7.5,3.2,0.74\n"
            "24,Female,0.6,0.1,140,18,19,7.2,4.2,1.40\n"
        )
        r_batch = requests.post(
            f"{BASE_URL}/predict/batch",
            files={"file": ("smoke.csv", csv_body.encode("utf-8"), "text/csv")},
            headers=headers,
            timeout=20,
        )
        assert r_batch.status_code == 200, r_batch.text
        b = r_batch.json()
        assert b["valid_count"] == 2 and b["invalid_count"] == 0
        print(f"[PASS] POST /predict/batch -> valid={b['valid_count']}, invalid={b['invalid_count']}")

        print("\nALL LIVE FASTAPI HTTP CHECKS PASSED!")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
