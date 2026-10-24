import json
import pytest
from lab_ocr.exceptions import PredictionError
from lab_ocr.prediction import ModelSpec, RiskPredictor
from conftest import make_result


def test_predicts_with_reviewed_values(predictor, make_report):
    rep = make_report(make_result("Glucose", 150, "mg/dL"), make_result("HbA1c", 7.0, "%"))
    out = predictor.predict_report(rep)
    assert out.status == "ok"
    assert out.prediction in {"low", "high"}
    assert abs(sum(out.probabilities.values()) - 1) < 0.01
    assert out.used_features == {"glucose": 150.0, "hba1c": 7.0}


def test_unreviewed_values_are_not_used(predictor, make_report):
    rep = make_report(make_result("Glucose", 150, "mg/dL", reviewed=False), make_result("HbA1c", 7.0, "%", reviewed=False))
    assert predictor.predict_report(rep).status == "insufficient_data"
    assert predictor.predict_report(rep, allow_unreviewed=True).status == "ok"


def test_unit_mismatch_is_excluded_not_converted(predictor, make_report):
    rep = make_report(make_result("Glucose", 8.3, "mmol/L"), make_result("HbA1c", 7.0, "%"))
    out = predictor.predict_report(rep)
    assert out.status == "insufficient_data"
    assert out.missing_features == ["glucose"]
    assert any("mmol/L" in w for w in out.warnings)


def test_missing_unit_is_excluded_when_model_expects_one(predictor, make_report):
    rep = make_report(make_result("Glucose", 100, None), make_result("HbA1c", 5.5, "%"))
    assert predictor.predict_report(rep).status == "insufficient_data"


def test_inexact_values_are_excluded(predictor, make_report):
    rep = make_report(make_result("Glucose", 100, "mg/dL"), make_result("HbA1c", 5.0, "%", operator="<"))
    out = predictor.predict_report(rep)
    assert out.status == "insufficient_data" and out.missing_features == ["hba1c"]


def test_non_numeric_value_is_excluded(predictor, make_report):
    rep = make_report(make_result("Glucose", "Positive", "mg/dL"), make_result("HbA1c", 5.0, "%"))
    out = predictor.predict_report(rep)
    assert out.status == "insufficient_data"
    assert any("not numeric" in w for w in out.warnings)


def test_fill_values_are_used_and_flagged(spec_path, make_report):
    data = json.loads(spec_path.read_text())
    data["fill_values"] = {"hba1c": 5.5}
    spec_path.write_text(json.dumps(data))
    out = RiskPredictor.from_spec_file(spec_path).predict_report(make_report(make_result("Glucose", 100, "mg/dL")))
    assert out.status == "ok"
    assert any("default" in w for w in out.warnings)


def test_spec_rejects_unknown_feature(tmp_path):
    bad = {"model_path": "m.joblib", "feature_order": ["a"], "feature_mapping": {"A": "b"}}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(PredictionError):
        ModelSpec.load(p)


def test_missing_model_file_gives_safe_error(tmp_path):
    spec = {"model_path": "nope.joblib", "feature_order": ["a"], "feature_mapping": {"A": "a"}}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(spec))
    with pytest.raises(PredictionError):
        RiskPredictor.from_spec_file(p)
