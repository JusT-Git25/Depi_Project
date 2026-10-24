import csv, io
from lab_ocr.schemas import LabReportResult, LabResult
from lab_ocr.exporters import report_to_csv, report_to_json

def test_json_csv_exports():
    report = LabReportResult(source_filename="sample.pdf", results=[
        LabResult(test_name="Glucose", original_test_name="Glucose", raw_result_text="Glucose: 5 mmol/L",
                  value=5, unit="mmol/L", page_number=1, validation_status="needs_review")
    ])
    assert '"Glucose"' in report_to_json(report)
    rows = list(csv.DictReader(io.StringIO(report_to_csv(report))))
    assert rows[0]["test_name"] == "Glucose"
    assert rows[0]["value"] == "5.0"
