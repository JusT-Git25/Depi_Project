"""Bridge between the OCR model-spec predictor and the production model bundle.

The lab_ocr predictor expects a scikit-learn-style estimator that takes a DataFrame
of model features and returns class labels. Our production artifact is instead a
dict bundle: a missing-robust LightGBM + calibrator + clinical threshold + metadata,
and it expects CDC feature names, not raw lab values.

`ProductionRiskModel` wraps the bundle and:
  * accepts the OCR feature frame (e.g. Glucose, HbA1c, Total Cholesterol),
  * routes it through the same `lab_adapter.labs_to_cdc()` normalization used
    everywhere else (AHA cutoffs, unit conversion, age bands, ...),
  * calls the calibrated `predict_risk()` and exposes the result as
    Low / Elevated / High with a probability distribution for the UI.

It also carries a marker (`__depi_bundle__`) so `lab_ocr` can recognise the bundle
and use the NaN-tolerant path instead of requiring every feature to be present.
"""
from __future__ import annotations

import math

import pandas as pd

BUNDLE_MARKER = "__depi_bundle__"

# OCR canonical test name (from lab_ocr/mappings/test_names.json) -> lab_adapter alias.
# Only tests that actually map onto the CDC-21 feature set are useful to the model;
# glucose/HbA1c are carried through as lab_signals, not model inputs.
OCR_TO_LAB_ALIAS = {
    "Glucose": "glucose",
    "Fasting Blood Sugar": "glucose_fasting",
    "Fasting Glucose": "glucose_fasting",
    "Random Blood Sugar": "glucose_random",
    "Random Glucose": "glucose_random",
    "HbA1c": "hba1c",
    "Total Cholesterol": "total_chol",
    "HDL Cholesterol": "hdl",
    "LDL Cholesterol": "ldl",
    "Triglycerides": "triglycerides",
}


def is_production_bundle(model) -> bool:
    """True when `model` is one of our packaged bundles (not a plain estimator)."""
    return isinstance(model, ProductionRiskModel) or (
        getattr(model, BUNDLE_MARKER, False) and hasattr(model, "predict_risk")
    )


def _frame_to_lab_json(frame: pd.DataFrame, feature_order: list[str]) -> dict:
    """Turn one OCR feature row into a lab_adapter-style {test: {value, unit}} dict."""
    row = frame.iloc[0]
    lab: dict[str, dict] = {}
    for name in feature_order:
        alias = OCR_TO_LAB_ALIAS.get(name)
        if alias is None or name not in frame.columns:
            continue
        value = row[name]
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            continue
        lab[alias] = {"value": float(value)}
    return lab


_LEVEL_ORDER = ["Low", "Elevated", "High"]
# Confidence shape by distance from the predicted class: peak on the prediction,
# half-weight on the adjacent level, small remainder two levels away.
_LEVEL_SHAPE = {0: 0.85, 1: 0.12, 2: 0.03}


def _probability_distribution(level: str) -> dict[str, float]:
    """Deterministic distribution that peaks on the predicted risk level.

    The model is a calibrated *screening* score, so we report the class confidence
    rather than pretending the raw probability is a 3-class softmax.
    """
    idx = _LEVEL_ORDER.index(level) if level in _LEVEL_ORDER else 0
    weights = {}
    for name, pos in zip(_LEVEL_ORDER, range(3)):
        weights[name] = _LEVEL_SHAPE[abs(pos - idx)]
    total = sum(weights.values())
    return {name: round(w / total, 4) for name, w in weights.items()}


def run_bundle(bundle: dict, frame: pd.DataFrame, feature_order: list[str],
               allow_unreviewed: bool = False, language: str = "en") -> dict:
    """Core prediction: OCR feature frame -> calibrated risk via the production model.

    Returns a plain dict; the caller wraps it in a PredictionResult.
    """
    from production.lab_adapter import labs_to_cdc

    lab_json = _frame_to_lab_json(frame, feature_order)
    if not lab_json:
        return {"status": "insufficient_data", "used_features": {},
                "missing_features": [], "warnings": ["No usable lab values were confirmed."],
                "probabilities": {}, "prediction": None, "top_risk_factors": []}

    adapted = labs_to_cdc(lab_json)
    partial = adapted["cdc_partial"]
    warnings = list(adapted["warnings"])
    if adapted["unknown_fields"]:
        warnings.append(f"Unmapped lab tests ignored by the model: {adapted['unknown_fields']}")
    missing = [k for k, v in partial.items() if isinstance(v, float) and math.isnan(v)]

    clinical = adapted.get("clinical_assessment") or adapted.get("glycemic_alert")
    signals = adapted["lab_signals"]

    # Lab-only path (no CDC survey features present)
    if not adapted["filled"] and signals:
        status = "needs_recheck" if (clinical and clinical.get("status") == "needs_recheck") else "ok"
        final_level = clinical.get("clinical_level") if (clinical and status == "needs_recheck") else (clinical.get("clinical_level", "Low") if clinical else "Low")
        flagged = clinical.get("flagged", False) if clinical else False
        rec = (clinical["recommendation_ar"] if language == "ar" else clinical["recommendation"]) if clinical else ""
        sum_text = (clinical["summary_ar"] if language == "ar" else clinical["summary"]) if clinical else ""
        return {
            "status": status,
            "prediction": final_level,
            "final_level": final_level,
            "risk_level": final_level,
            "probabilities": _probability_distribution(final_level) if final_level else {},
            "used_features": adapted["filled"],
            "lab_signals": signals,
            "clinical_assessment": clinical,
            "glycemic_alert": clinical,
            "survey_risk": None,
            "missing_features": missing,
            "warnings": warnings,
            "risk_score": None,
            "risk_percent": None,
            "flagged_for_screening": flagged,
            "recommendation": f"{sum_text} {rec}".strip(),
            "top_risk_factors": [],
            "provenance": adapted["provenance"],
            "allow_unreviewed": allow_unreviewed,
        }

    from production.predict import predict_risk

    survey_result = predict_risk(partial)
    survey_risk = {
        "risk_score": survey_result["risk_score"],
        "risk_percent": survey_result["risk_percent"],
        "risk_level": survey_result["risk_level"],
        "flagged_for_screening": survey_result["flagged_for_screening"],
        "top_risk_factors": survey_result["top_risk_factors"],
    }
    model_level = survey_result["risk_level"]
    clinical_level = clinical.get("clinical_level") if clinical else None

    if clinical and clinical.get("flagged"):
        final_level = "High" if (clinical_level == "High" or model_level == "High") else "Elevated"
        flagged = True
        rec = clinical["recommendation_ar"] if language == "ar" else clinical["recommendation"]
        sum_text = clinical["summary_ar"] if language == "ar" else clinical["summary"]
        recommendation = f"{sum_text} {rec} (Survey model risk: {survey_result['risk_percent']})"
        top_risk_score = None
        top_risk_percent = None
    else:
        final_level = model_level
        flagged = survey_result["flagged_for_screening"]
        recommendation = survey_result["recommendation"]
        top_risk_score = survey_result["risk_score"]
        top_risk_percent = survey_result["risk_percent"]

    return {
        "status": "ok",
        "prediction": final_level,
        "final_level": final_level,
        "risk_level": final_level,
        "probabilities": _probability_distribution(final_level),
        "used_features": adapted["filled"],
        "lab_signals": signals,
        "clinical_assessment": clinical,
        "glycemic_alert": clinical,
        "survey_risk": survey_risk,
        "missing_features": missing,
        "warnings": warnings,
        "risk_score": top_risk_score,
        "risk_percent": top_risk_percent,
        "flagged_for_screening": flagged,
        "recommendation": recommendation,
        "top_risk_factors": survey_result["top_risk_factors"],
        "provenance": adapted["provenance"],
        "allow_unreviewed": allow_unreviewed,
    }


class ProductionRiskModel:
    """scikit-learn-ish adapter so `lab_ocr` can run our calibrated bundle."""

    __depi_bundle__ = True

    def __init__(self, bundle: dict):
        if not isinstance(bundle, dict) or "model" not in bundle:
            raise ValueError("ProductionRiskModel expects a production model bundle dict.")
        self.bundle = bundle
        self.classes_ = ["Low", "Elevated", "High"]
        self._last: dict = {}

    @classmethod
    def load(cls, path) -> "ProductionRiskModel":
        import joblib

        return cls(joblib.load(path))

    def predict_risk(self, frame: pd.DataFrame, feature_order: list[str], *,
                     allow_unreviewed: bool = False) -> dict:
        self._last = run_bundle(self.bundle, frame, feature_order, allow_unreviewed)
        return self._last

    # --- scikit-learn surface used by RiskPredictor when it treats us as an estimator
    def predict(self, frame: pd.DataFrame):
        order = list(frame.columns)
        self.predict_risk(frame, order)
        return [self._last.get("prediction")]

    def predict_proba(self, frame: pd.DataFrame):
        order = list(frame.columns)
        if not self._last:
            self.predict_risk(frame, order)
        dist = self._last.get("probabilities") or {}
        return [[dist.get(c, 0.0) for c in self.classes_]]


def load_production_model(path) -> ProductionRiskModel:
    """Load a production bundle and wrap it for the OCR predictor."""
    return ProductionRiskModel.load(path)
