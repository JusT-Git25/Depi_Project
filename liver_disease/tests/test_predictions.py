"""Tests for prediction consistency, clinical monotonicity, batch schemas, and SHAP explanations."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from liver_disease.config import DEFAULT_MODEL_ARTIFACT_PATH
from liver_disease.models.predict import LiverDiseasePredictor, categorize_risk


def test_single_prediction_consistency_and_schema(sample_high_risk_patient: Dict[str, Any]) -> None:
    predictor = LiverDiseasePredictor(DEFAULT_MODEL_ARTIFACT_PATH)

    res1 = predictor.predict_one(sample_high_risk_patient)
    res2 = predictor.predict_one(sample_high_risk_patient)

    assert res1["prediction"] in (1, 2)
    assert res1["prediction_label"] in ("Liver Disease", "No Liver Disease")
    assert 0.0 <= res1["probability"] <= 1.0
    assert res1["risk_category"] in ("Low", "Moderate", "High")
    assert res1["probability"] == res2["probability"]
    assert res1["prediction"] == res2["prediction"]
    assert res1["prediction_id"] != res2["prediction_id"]


def test_clinical_risk_ordering_sanity(
    sample_high_risk_patient: Dict[str, Any],
    sample_low_risk_patient: Dict[str, Any],
) -> None:
    """Markedly abnormal bilirubin/enzymes should score higher than normal-range markers."""
    predictor = LiverDiseasePredictor(DEFAULT_MODEL_ARTIFACT_PATH)
    high = predictor.predict_one(sample_high_risk_patient)
    low = predictor.predict_one(sample_low_risk_patient)
    assert high["probability"] > low["probability"]


def test_batch_prediction_supports_both_schemas(
    sample_high_risk_patient: Dict[str, Any],
    sample_low_risk_patient: Dict[str, Any],
) -> None:
    predictor = LiverDiseasePredictor(DEFAULT_MODEL_ARTIFACT_PATH)

    api_df = pd.DataFrame([sample_high_risk_patient, sample_low_risk_patient])
    raw_df = pd.DataFrame(
        [
            {
                "Age": 62,
                "Gender": "Male",
                "TB": 10.9,
                "DB": 5.5,
                "Alkphos": 699.0,
                "Sgpt": 64.0,
                "Sgot": 100.0,
                "TP": 7.5,
                "ALB": 3.2,
                "A/G Ratio": 0.74,
            },
            {
                "Age": 24,
                "Gender": "Female",
                "TB": 0.6,
                "DB": 0.1,
                "Alkphos": 140.0,
                "Sgpt": 18.0,
                "Sgot": 19.0,
                "TP": 7.2,
                "ALB": 4.2,
                "A/G Ratio": 1.4,
            },
        ]
    )

    preds_api = predictor.predict_dataframe(api_df)
    preds_raw = predictor.predict_dataframe(raw_df)

    assert len(preds_api) == len(preds_raw) == 2
    assert preds_api[0]["probability"] == preds_raw[0]["probability"]
    assert preds_api[1]["probability"] == preds_raw[1]["probability"]


def test_shap_explanation_output(sample_high_risk_patient: Dict[str, Any]) -> None:
    predictor = LiverDiseasePredictor(DEFAULT_MODEL_ARTIFACT_PATH)
    res = predictor.explain_one(sample_high_risk_patient, top_k=5)

    assert "explanation" in res
    exp = res["explanation"]
    assert len(exp["top_contributing_features"]) == 5
    assert len(exp["all_feature_contributions"]) >= 10
    assert "causation" in exp["limitations_note"].lower()
    # Sorted descending by absolute impact
    impacts = [c["abs_impact"] for c in exp["top_contributing_features"]]
    assert impacts == sorted(impacts, reverse=True)


def test_risk_categorization_boundaries() -> None:
    assert categorize_risk(0.10) == "Low"
    assert categorize_risk(0.349) == "Low"
    assert categorize_risk(0.35) == "Moderate"
    assert categorize_risk(0.649) == "Moderate"
    assert categorize_risk(0.65) == "High"
    assert categorize_risk(0.99) == "High"
