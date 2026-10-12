"""Inference API for the packaged diabetes risk model.

    from predict import predict_risk
    result = predict_risk({"HighBP": 1, "HighChol": 1, ...})   # all 21 columns

Accepts one patient (dict) or several (list of dicts / DataFrame). Returns a dict, or a
list of dicts for several patients, with a calibrated risk score and a recommendation.

Input coding follows the CDC (BRFSS) dataset:
    binary columns   0 = no, 1 = yes (Sex: 0 = female, 1 = male)
    BMI              number (12-98 in the training data)
    GenHlth          1 = excellent ... 5 = poor
    MentHlth/PhysHlth  days of poor health in the past 30 (0-30)
    Age              13 age bands: 1 = 18-24, 2 = 25-29, ... 9 = 55-59, ... 13 = 80+
    Education        1-6      Income  1-8
Run `python predict.py` for a demo with three sample patients.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import sklearn

from common import MODEL_PATH, apply_calibrator, engineer_features

DISCLAIMER = "Screening aid only - not a diagnosis."
MIN_FACTOR_IMPACT = 0.10   # log-odds; smaller contributions are noise, not worth showing as "risk factors"
RECOMMENDATIONS = {
    "Low": "Risk is below the screening threshold. Maintain a healthy lifestyle and keep up routine check-ups.",
    "Elevated": ("Risk is above the screening threshold. Recommend a confirmatory blood test "
                 "(fasting glucose or HbA1c) and a follow-up with a healthcare provider."),
    "High": ("Risk is high. Recommend arranging a confirmatory blood test (fasting glucose or HbA1c) "
             "and seeing a healthcare provider soon."),
}


@lru_cache(maxsize=None)
def load_model(path: str | Path = MODEL_PATH) -> dict:
    bundle = joblib.load(path)
    now = {"scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__}
    for lib, ver in now.items():
        if bundle["versions"].get(lib) != ver:
            warnings.warn(f"Model was trained with {lib} {bundle['versions'].get(lib)} but {ver} is installed; "
                          "re-run train_final.py if predictions look wrong.")
    return bundle


def _to_frame(patient_data, bundle: dict) -> pd.DataFrame:
    if isinstance(patient_data, Mapping):
        df = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.DataFrame):
        df = patient_data.copy()
    else:
        df = pd.DataFrame(list(patient_data))

    missing = [c for c in bundle["raw_features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing patient fields: {missing}")
    df = df[bundle["raw_features"]]
    try:
        df = df.apply(pd.to_numeric)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"All patient fields must be numeric: {exc}") from exc
    if df.isnull().any().any():
        raise ValueError(f"Null values in fields: {list(df.columns[df.isnull().any()])}")

    for col, (lo, hi) in bundle["input_ranges"].items():
        bad = (df[col] < lo) | (df[col] > hi)
        if bad.any():
            raise ValueError(f"{col}={df.loc[bad, col].iloc[0]} is outside the valid range [{lo:g}, {hi:g}]")
    return df.reset_index(drop=True)


def predict_risk(patient_data, model_path: str | Path = MODEL_PATH):
    """Return risk score (calibrated probability), level, recommendation and top risk factors."""
    bundle = load_model(model_path)
    df = _to_frame(patient_data, bundle)
    X = engineer_features(df)[bundle["model_features"]]

    model = bundle["model"]
    raw = model.predict_proba(X)[:, 1]
    scores = apply_calibrator(bundle["calibration_method"], bundle["calibrator"], raw)
    # Per-patient feature contributions (log-odds) straight from LightGBM; last column is the bias.
    contrib = model.predict(X, pred_contrib=True)[:, :-1]

    results = []
    for i, score in enumerate(scores):
        if score >= bundle["high_risk_cutoff"]:
            level = "High"
        elif score >= bundle["threshold"]:
            level = "Elevated"
        else:
            level = "Low"
        order = contrib[i].argsort()[::-1][:3]
        factors = [
            {"feature": X.columns[j], "value": float(X.iloc[i, j]), "contribution": round(float(contrib[i, j]), 3)}
            for j in order if contrib[i, j] >= MIN_FACTOR_IMPACT
        ]
        results.append({
            "risk_score": round(float(score), 4),
            "risk_percent": f"{score:.1%}",
            "risk_level": level,
            "flagged_for_screening": bool(score >= bundle["threshold"]),
            "screening_threshold": round(bundle["threshold"], 4),
            "recommendation": RECOMMENDATIONS[level],
            "top_risk_factors": factors,
            "disclaimer": DISCLAIMER,
        })
    return results[0] if isinstance(patient_data, Mapping) else results


if __name__ == "__main__":
    import json

    samples = {
        "Young, healthy": dict(HighBP=0, HighChol=0, CholCheck=1, BMI=22, Smoker=0, Stroke=0, HeartDiseaseorAttack=0,
                               PhysActivity=1, Fruits=1, Veggies=1, HvyAlcoholConsump=0, AnyHealthcare=1,
                               NoDocbcCost=0, GenHlth=1, MentHlth=0, PhysHlth=0, DiffWalk=0, Sex=0, Age=3,
                               Education=6, Income=8),
        "Middle-aged, overweight": dict(HighBP=1, HighChol=0, CholCheck=1, BMI=29, Smoker=0, Stroke=0,
                                        HeartDiseaseorAttack=0, PhysActivity=1, Fruits=1, Veggies=1,
                                        HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0, GenHlth=3, MentHlth=2,
                                        PhysHlth=3, DiffWalk=0, Sex=1, Age=8, Education=5, Income=6),
        "Older, many risk factors": dict(HighBP=1, HighChol=1, CholCheck=1, BMI=38, Smoker=1, Stroke=0,
                                         HeartDiseaseorAttack=1, PhysActivity=0, Fruits=0, Veggies=0,
                                         HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=1, GenHlth=5, MentHlth=10,
                                         PhysHlth=25, DiffWalk=1, Sex=1, Age=11, Education=3, Income=2),
    }
    for name, patient in samples.items():
        print(f"--- {name}")
        print(json.dumps(predict_risk(patient), indent=2))
