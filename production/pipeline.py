"""End-to-end pipeline: lab report (or OCR JSON) -> risk prediction -> advice.

Wires the three modules together:

    lab_ocr (extract) -> lab_adapter (normalize to CDC) -> predict (calibrated risk)
                      -> advisor (AI advice, with rule-based fallback)

Each stage is optional: the model is missing-robust, so a labs-only input still
produces a (wider-uncertainty) prediction, and advice falls back when no API key
is configured.

CLI:
    python -m production.pipeline --file report.pdf --lang ar
    python -m production.pipeline --json lab_ocr_sample.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from production.predict import load_model, predict_risk

__all__ = ["predict_from_lab_json", "predict_from_report", "predict_from_file", "advise"]


def advise(prediction: dict, patient: dict | None = None, language: str = "en") -> dict:
    """AI advice with a rule-based fallback. Imported lazily (needs the anthropic extra)."""
    from production.advisor import generate_advice

    return generate_advice(prediction, patient, language)


def _with_advice(result: dict, language: str, want_advice: bool) -> dict:
    if want_advice and result.get("status") == "ok" and (result.get("final_level") or result.get("risk_level")):
        patient_ctx = result.get("used_features") or {}
        result["advice"] = advise(result, patient=patient_ctx, language=language)
    return result


def predict_from_lab_json(lab_json: Any, *, language: str = "en", advice: bool = False) -> dict:
    """OCR JSON (any lab naming/shape) -> prediction (+ optional advice)."""
    from production.lab_adapter import labs_to_cdc
    from production.clinical_rules import evaluate_glycemic_signals

    adapted = labs_to_cdc(lab_json)
    signals = adapted["lab_signals"]
    clinical = adapted.get("clinical_assessment") or evaluate_glycemic_signals(signals)

    # If no CDC survey features could be extracted, but direct lab signals exist,
    # evaluate directly from laboratory findings without crashing the 21-feature model.
    if not adapted["filled"] and signals:
        status = "needs_recheck" if (clinical and clinical.get("status") == "needs_recheck") else "ok"
        final_level = clinical.get("clinical_level") if clinical else None
        flagged = clinical.get("flagged", False) if clinical else False
        rec = (clinical["recommendation_ar"] if language == "ar" else clinical["recommendation"]) if clinical else "Consult your physician to review these lab results."
        sum_text = (clinical["summary_ar"] if language == "ar" else clinical["summary"]) if clinical else "Laboratory findings evaluated directly."
        result = {
            "status": status,
            "final_level": final_level,
            "risk_level": final_level,
            "risk_score": None,
            "risk_percent": None,
            "flagged_for_screening": flagged,
            "recommendation": f"{sum_text} {rec}".strip(),
            "survey_risk": None,
            "clinical_assessment": clinical,
            "glycemic_alert": clinical,
            "top_risk_factors": [],
            "partial_input": True,
            "missing_count": len(adapted.get("missing", [])),
            "lab_signals": signals,
            "provenance": adapted["provenance"],
            "unknown_fields": adapted["unknown_fields"],
            "adapter_warnings": adapted["warnings"] + ["No CDC survey features found; evaluated directly from laboratory glycemic signals."],
        }
        return _with_advice(result, language, advice)

    survey_result = predict_risk(adapted["cdc_partial"])
    survey_risk = {
        "risk_score": survey_result["risk_score"],
        "risk_percent": survey_result["risk_percent"],
        "risk_level": survey_result["risk_level"],
        "flagged_for_screening": survey_result["flagged_for_screening"],
        "recommendation": survey_result["recommendation"],
        "top_risk_factors": survey_result["top_risk_factors"],
    }

    model_level = survey_result["risk_level"]
    clinical_level = clinical.get("clinical_level") if clinical else None

    if clinical and clinical.get("flagged"):
        final_level = "High" if (clinical_level == "High" or model_level == "High") else "Elevated"
        flagged = True
        rec = clinical["recommendation_ar"] if language == "ar" else clinical["recommendation"]
        sum_text = clinical["summary_ar"] if language == "ar" else clinical["summary"]
        recommendation = f"{sum_text} {rec} (Survey model estimated risk: {survey_result['risk_percent']})"
        top_risk_score = None
        top_risk_percent = None
    else:
        final_level = model_level
        flagged = survey_result["flagged_for_screening"]
        recommendation = survey_result["recommendation"]
        top_risk_score = survey_result["risk_score"]
        top_risk_percent = survey_result["risk_percent"]

    result = {
        "status": "ok",
        "final_level": final_level,
        "risk_level": final_level,
        "risk_score": top_risk_score,
        "risk_percent": top_risk_percent,
        "flagged_for_screening": flagged,
        "recommendation": recommendation,
        "survey_risk": survey_risk,
        "clinical_assessment": clinical,
        "glycemic_alert": clinical,
        "top_risk_factors": survey_result["top_risk_factors"],
        "screening_threshold": survey_result.get("screening_threshold"),
        "partial_input": survey_result.get("partial_input", True),
        "missing_features": survey_result.get("missing_features", []),
        "missing_count": len(survey_result.get("missing_features", [])),
        "disclaimer": survey_result.get("disclaimer"),
        "lab_signals": signals,
        "provenance": adapted["provenance"],
        "unknown_fields": adapted["unknown_fields"],
        "adapter_warnings": adapted["warnings"],
    }
    return _with_advice(result, language, advice)


def predict_from_report(report, *, language: str = "en", advice: bool = False,
                        allow_unreviewed: bool = False) -> dict:
    """An extracted LabReportResult (from lab_ocr) -> prediction (+ optional advice)."""
    from production.lab_ocr.prediction.spec import ModelSpec
    from production.lab_ocr.prediction.predictor import RiskPredictor

    spec_path = Path(__file__).resolve().parent / "lab_ocr" / "models" / "diabetes_spec.json"
    predictor = RiskPredictor(ModelSpec.load(spec_path), model=load_model())
    out = predictor.predict_report(report, allow_unreviewed=allow_unreviewed)
    result = out.model_dump(mode="json")
    return _with_advice(result, language, advice)


def predict_from_file(path: str | Path, *, language: str = "en", advice: bool = False) -> dict:
    """A PDF/image lab report -> OCR -> prediction (+ optional advice)."""
    from production.lab_ocr import LabOCRService

    report = LabOCRService().process_file(path)
    return predict_from_report(report, language=language, advice=advice)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab report -> diabetes risk prediction pipeline")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="PDF or image lab report to OCR")
    src.add_argument("--json", help="OCR JSON file (lab_adapter format)")
    parser.add_argument("--lang", default="en", choices=["en", "ar"])
    parser.add_argument("--advice", action="store_true", help="also generate AI advice")
    args = parser.parse_args()

    if args.json:
        result = predict_from_lab_json(json.loads(Path(args.json).read_text()),
                                       language=args.lang, advice=args.advice)
    else:
        result = predict_from_file(args.file, language=args.lang, advice=args.advice)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
