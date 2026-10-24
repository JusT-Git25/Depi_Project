import csv
import io
import json
from .schemas import LabReportResult

CSV_FIELDS = [
    "test_name", "original_test_name", "value", "value_operator", "unit",
    "reference_range", "abnormal_flag", "specimen_type", "page_number",
    "confidence", "validation_status", "validation_warnings", "raw_result_text",
    "manually_reviewed"
]

def report_to_json(report: LabReportResult) -> str:
    return report.model_dump_json(indent=2)

def report_to_csv(report: LabReportResult) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for result in report.results:
        row = result.model_dump()
        rr = row.get("reference_range")
        row["reference_range"] = json.dumps(rr, ensure_ascii=False) if rr else ""
        row["validation_warnings"] = " | ".join(row.get("validation_warnings") or [])
        writer.writerow(row)
    return buffer.getvalue()
