import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from lab_ocr.prediction import RiskPredictor, ModelSpec
from lab_ocr.schemas import LabReportResult, LabResult


@pytest.fixture
def spec_path(tmp_path):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"glucose": rng.normal(100, 25, 300), "hba1c": rng.normal(5.6, 0.8, 300)})
    y = ((X["glucose"] > 110) & (X["hba1c"] > 5.7)).astype(int)
    joblib.dump(LogisticRegression().fit(X, y), tmp_path / "m.joblib")
    spec = {
        "name": "test-model", "version": "t1", "model_path": "m.joblib",
        "feature_order": ["glucose", "hba1c"],
        "feature_mapping": {"Glucose": "glucose", "HbA1c": "hba1c"},
        "expected_units": {"glucose": "mg/dL", "hba1c": "%"},
        "labels": {"0": "low", "1": "high"},
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


@pytest.fixture
def predictor(spec_path):
    return RiskPredictor.from_spec_file(spec_path)


def make_result(name, value, unit, *, reviewed=True, operator=None):
    return LabResult(test_name=name, original_test_name=name, raw_result_text=f"{name} {value}",
                     value=value, unit=unit, value_operator=operator, page_number=1,
                     validation_status="validated", manually_reviewed=reviewed)


@pytest.fixture
def make_report():
    def _make(*results):
        return LabReportResult(source_filename="x.pdf", results=list(results))
    return _make
