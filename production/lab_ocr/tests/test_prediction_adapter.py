from lab_ocr.schemas import LabReportResult, LabResult
from lab_ocr.integrations.lab_results_adapter import prepare_prediction_features

def test_adapter_excludes_unreviewed_by_default():
    report = LabReportResult(source_filename="x.pdf", results=[
        LabResult(test_name="Glucose", original_test_name="Glucose", raw_result_text="Glucose 90",
                  value=90, unit="mg/dL", page_number=1, validation_status="validated")
    ])
    out = prepare_prediction_features(report, {"Glucose": "glucose"})
    assert not out.features
    assert out.warnings

def test_adapter_allows_reviewed_result():
    report = LabReportResult(source_filename="x.pdf", results=[
        LabResult(test_name="Glucose", original_test_name="Glucose", raw_result_text="Glucose 90",
                  value=90, unit="mg/dL", page_number=1, validation_status="validated", manually_reviewed=True)
    ])
    out = prepare_prediction_features(report, {"Glucose": "glucose"})
    assert out.features == {"glucose": 90}
