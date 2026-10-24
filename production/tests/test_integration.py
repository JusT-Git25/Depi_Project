"""Integration tests: OCR model-spec -> production bundle -> calibrated risk -> advice.

Skipped automatically if the production model artifact has not been trained yet.
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "production" / "lab_ocr" / "models" / "diabetes_spec.json"
MODEL = REPO_ROOT / "artifacts" / "diabetes_risk_model.joblib"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="production model not trained")

from production.lab_ocr.prediction import RiskPredictor  # noqa: E402
from production.lab_ocr.schemas import LabReportResult, LabResult  # noqa: E402


def _result(name, value, unit, reviewed=True):
    return LabResult(test_name=name, original_test_name=name, raw_result_text=f"{name} {value}",
                     value=value, unit=unit, page_number=1,
                     validation_status="validated", manually_reviewed=reviewed)


def _report(*results):
    return LabReportResult(source_filename="r.pdf", results=list(results))


@pytest.fixture
def predictor():
    return RiskPredictor.from_spec_file(SPEC)


def test_bundle_is_wrapped_and_predicts(predictor):
    from production.integrations.production_bundle import ProductionRiskModel

    assert isinstance(predictor._model, ProductionRiskModel)
    out = predictor.predict_report(_report(_result("Total Cholesterol", 240, "mg/dL"),
                                           _result("HbA1c", 7.2, "%")))
    assert out.status == "ok"
    assert out.prediction in {"Low", "Elevated", "High"}
    assert abs(sum(out.probabilities.values()) - 1) < 0.01
    assert max(out.probabilities, key=out.probabilities.get) == out.prediction
    # AHA cutoff: total cholesterol 240 -> HighChol = 1
    assert out.used_features["HighChol"] == 1.0
    # HbA1c has no CDC slot: surfaced as a lab signal, not a model input
    assert out.lab_signals == {"hba1c": 7.2}


def test_high_cholesterol_flags_higher_risk_than_normal(predictor):
    high = predictor.predict_report(_report(_result("Total Cholesterol", 260, "mg/dL")))
    normal = predictor.predict_report(_report(_result("Total Cholesterol", 150, "mg/dL")))
    assert high.risk_score > normal.risk_score
    assert high.used_features["HighChol"] == 1.0
    assert normal.used_features["HighChol"] == 0.0


def test_unreviewed_values_are_not_used(predictor):
    rep = _report(_result("Total Cholesterol", 240, "mg/dL", reviewed=False))
    assert predictor.predict_report(rep).status == "insufficient_data"
    assert predictor.predict_report(rep, allow_unreviewed=True).status == "ok"


def test_unit_mismatch_is_excluded(predictor):
    rep = _report(_result("Total Cholesterol", 5.2, "mmol/L"))
    out = predictor.predict_report(rep)
    assert out.status == "insufficient_data"
    assert any("mmol/L" in w for w in out.warnings)


def test_pipeline_json_entrypoint_adds_provenance_and_advice_fallback():
    from production.pipeline import predict_from_lab_json

    lab = json.loads((REPO_ROOT / "lab_ocr_sample.json").read_text())
    out = predict_from_lab_json(lab, advice=True)
    assert out["risk_level"] in {"Low", "Elevated", "High"}
    assert out["partial_input"] is True
    assert "HighBP" in out["provenance"]
    # No API key configured -> rule-based fallback, never an exception
    assert out["advice"]["source"] in {"ai", "fallback"}
    assert out["advice"]["disclaimer"]


def test_glycemic_alert_elevated_override_in_pipeline():
    from production.pipeline import predict_from_lab_json

    # Report with normal blood pressure (115/75) and normal cholesterol (160)
    # but elevated HbA1c (8.5%)
    lab_elevated = [
        {"test": "Blood Pressure", "value": "115/75", "unit": "mmHg"},
        {"test": "Total Cholesterol", "value": 160, "unit": "mg/dL"},
        {"test": "HbA1c", "value": 8.5, "unit": "%"},
    ]
    out = predict_from_lab_json(lab_elevated, advice=True)
    assert out["clinical_assessment"]["status"] == "elevated_range"
    assert out["clinical_assessment"]["flagged"] is True
    # Crucial safety feature: risk level overridden to High despite normal BP/chol
    assert out["final_level"] == "High"
    assert out["risk_level"] == "High"
    assert out["flagged_for_screening"] is True
    assert "reference threshold" in out["recommendation"].lower()
    # Advice should include HbA1c finding and non-contradictory summary
    assert any("HbA1c" in f["factor"] for f in out["advice"]["key_factors"])
    # Independent assessments: survey_risk contains original score, top-level avoids contradiction
    assert out["survey_risk"]["risk_score"] is not None
    assert out["risk_score"] is None  # nulled to prevent "5.3% is High" contradiction


def test_random_glucose_is_not_treated_as_fasting():
    from production.pipeline import predict_from_lab_json

    # Random glucose 130 mg/dL is normal casual/postprandial (cutoff is >= 200 for elevated)
    lab_random = [
        {"test": "Random Blood Sugar", "value": 130, "unit": "mg/dL"},
    ]
    out = predict_from_lab_json(lab_random, advice=True)
    assert out["clinical_assessment"]["status"] == "normal"
    assert out["clinical_assessment"]["flagged"] is False
    assert out["final_level"] == "Low"


def test_hba1c_ifcc_mmol_mol_is_converted():
    from production.pipeline import predict_from_lab_json

    # HbA1c 30 mmol/mol (= 4.9% NGSP, completely normal)
    lab_ifcc = [
        {"test": "HbA1c", "value": 30, "unit": "mmol/mol"},
    ]
    out = predict_from_lab_json(lab_ifcc, advice=True)
    # Must NOT flag normal HbA1c as elevated!
    assert out["clinical_assessment"]["status"] == "normal"
    assert out["clinical_assessment"]["flagged"] is False
    assert out["final_level"] == "Low"
    assert out["lab_signals"]["hba1c"] == 4.9


def test_implausible_ocr_reads_routed_to_needs_recheck():
    from production.pipeline import predict_from_lab_json

    # HbA1c 74% is physically impossible (likely 7.4% OCR misread)
    lab_implausible = [
        {"test": "HbA1c", "value": 74.0, "unit": "%"},
    ]
    out = predict_from_lab_json(lab_implausible, advice=True)
    assert out["clinical_assessment"]["status"] == "needs_recheck"
    assert out["clinical_assessment"]["flagged"] is False
    assert any("outside physiological bounds" in w for w in out["clinical_assessment"]["warnings"])


def test_lab_only_ocr_report_with_advice_does_not_crash(predictor):
    from production.pipeline import predict_from_report

    # Report with only reviewed Glucose
    rep = _report(_result("Glucose", 140, "mg/dL"))
    out = predict_from_report(rep, advice=True)
    assert out["status"] == "ok"
    assert out["final_level"] in {"Low", "Elevated", "High"}
    assert out["risk_score"] is None  # no fabricated score
    assert "advice" in out
    assert out["advice"]["source"] in {"ai", "fallback"}


