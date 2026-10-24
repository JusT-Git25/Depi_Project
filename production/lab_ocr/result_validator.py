from .schemas import LabReportResult, LabResult

def validate_result(result: LabResult) -> LabResult:
    warnings = list(result.validation_warnings)
    if result.page_number < 1:
        warnings.append("Invalid page number.")
    if result.value is None:
        warnings.append("No unambiguous value parsed.")
        result.validation_status = "needs_review"
    if result.reference_range and result.reference_range.low is not None and result.reference_range.high is not None:
        if result.reference_range.low > result.reference_range.high:
            warnings.append("Reference range lower bound exceeds upper bound.")
            result.validation_status = "invalid"
    if result.confidence is not None and result.confidence < 55:
        warnings.append("OCR confidence is below the review threshold.")
        result.validation_status = "needs_review"
    # OCR extraction must not decide medical normality.
    result.validation_warnings = sorted(set(warnings))
    if not result.manually_reviewed and result.validation_status == "validated":
        # "validated" means technical parse checks only, not clinical validation.
        pass
    return result

def validate_report(report: LabReportResult) -> LabReportResult:
    report.results = [validate_result(r) for r in report.results]
    if not report.pages:
        report.warnings.append("No pages were processed.")
    if not report.results:
        report.warnings.append("No structured laboratory results were confidently detected; inspect raw text.")
    if any(r.validation_status != "validated" for r in report.results):
        report.processing_status = "review_required"
    elif report.warnings:
        report.processing_status = "success_with_warnings"
    else:
        report.processing_status = "success"
    if report.results and any(p.method == "ocr" for p in report.pages):
        report.warnings.append("OCR-derived values require comparison with the original report.")
        if report.processing_status == "success":
            report.processing_status = "success_with_warnings"
    return report
