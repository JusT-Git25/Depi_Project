"""Train, calibrate and package the final COPD risk model.

    python COPD_Packaging.py            # train + save artifacts/copd_risk_model.joblib + demo
    from COPD_Packaging import predict_risk

Order of the whole COPD project:
    1. COPD_EDA_Cleaning.ipynb   EDA, missing values, outliers, correlation, imbalance -> COPD_clean.csv
    2. COPD_LazyPredict.ipynb    which model is best (LightGBM)
    3. COPD_GridSearch.py        best LightGBM parameters -> artifacts/copd_best_params.json
    4. COPD_Packaging.py         this file

Pipeline (same idea as production/train_final.py for diabetes):
    balanced fit split -> LightGBM -> calibrator (natural cal split) -> screening threshold (cal split)
Balancing makes the raw probabilities too high, so the calibrator is fitted on a split that keeps the
natural 8% COPD rate; that brings the probabilities back to real-world values.
The test split (same 80/20 stratified split, seed 42, as every other script) is only used for final metrics.

Input coding for predict_risk (CDC BRFSS):
    _AGE80    age in years (18-80, 80 means 80+)
    SEXVAR    1 = male, 2 = female
    BMI       body mass index, e.g. 27.5 (NOT multiplied by 100); clipped to 14-60
    SMOKE100  smoked at least 100 cigarettes in life: 1 = yes, 2 = no (only used to fill SMOKDAY2)
    SMOKDAY2  current smoking: 1 = every day, 2 = some days, 3 = not at all (former smoker).
              Leave out for people with SMOKE100 = 2 (it is set to 0 = never smoked automatically).
    EXERANY2  any exercise in the past 30 days: 1 = yes, 2 = no
    GENHLTH   general health: 1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor
A missing field is filled with the same value the EDA notebook used (median / most frequent) and is
listed in "missing_features" in the result.
"""
import json
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from copd_data import balance, cleaning_settings, get_train_test, prepare_patient

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "copd_risk_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "copd_metrics.json"
BEST_PARAMS_PATH = ARTIFACT_DIR / "copd_best_params.json"

RANDOM_STATE = 42
TARGET_RECALL = 0.80       # screening goal (same as the diabetes model)
HIGH_RISK_CUTOFF = 0.30    # calibrated probability above which risk is "High" (~4x the 8% base rate)
MIN_FACTOR_IMPACT = 0.10   # log-odds; smaller contributions are noise


def lgbm_params() -> dict:
    """Best params from COPD_GridSearch.py (falls back to sensible defaults if it was not run)."""
    best = {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 15,
            "min_child_samples": 200, "colsample_bytree": 0.8}
    if BEST_PARAMS_PATH.exists():
        best = json.loads(BEST_PARAMS_PATH.read_text())["best_params"]
    return dict(**best, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)


VALID_VALUES = {
    "SEXVAR": {1, 2}, "SMOKE100": {1, 2}, "SMOKDAY2": {0, 1, 2, 3},
    "EXERANY2": {1, 2}, "GENHLTH": {1, 2, 3, 4, 5},
}
VALID_RANGES = {"_AGE80": (18, 80), "BMI": (10, 80)}

DISCLAIMER = "Screening aid only - not a diagnosis. COPD is confirmed with spirometry."
RECOMMENDATIONS = {
    "Low Risk": "Risk is below the screening threshold. Avoid smoking and keep up routine check-ups.",
    "Moderate Risk": ("Risk is above the screening threshold. Consider a breathing test (spirometry) "
                      "and a follow-up with a healthcare provider."),
    "High Risk": ("Risk is high. Recommend arranging a breathing test (spirometry) "
                  "and seeing a healthcare provider soon."),
}


# ---------------------------------------------------------------------- data
def get_splits():
    """fit (balanced) / cal (natural) / test (natural). Cleaning is done in the EDA notebook."""
    X_tv, X_test, y_tv, y_test = get_train_test()
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv)
    X_fit_bal, y_fit_bal = balance(X_fit, y_fit, random_state=RANDOM_STATE)
    return (X_fit_bal, y_fit_bal.to_numpy()), (X_cal, y_cal.to_numpy()), (X_test, y_test.to_numpy())


# ---------------------------------------------------------------- calibration
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(method, p_raw, y):
    if method == "none":
        return None
    if method == "sigmoid":
        return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw), y)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_raw, y)
    raise ValueError(method)


def apply_calibrator(method, calibrator, p_raw):
    p_raw = np.asarray(p_raw, dtype=float)
    if method == "none":
        return p_raw
    if method == "sigmoid":
        return calibrator.predict_proba(_logit(p_raw))[:, 1]
    return calibrator.predict(p_raw)


def select_method(p_raw, y, methods=("none", "sigmoid", "isotonic"), n_splits=5):
    """Lowest cross-validated Brier score on the calibration split (test is never used)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = {m: [] for m in methods}
    for tr, va in skf.split(p_raw, y):
        for m in methods:
            cal = fit_calibrator(m, p_raw[tr], y[tr])
            scores[m].append(brier_score_loss(y[va], apply_calibrator(m, cal, p_raw[va])))
    mean_scores = {m: float(np.mean(v)) for m, v in scores.items()}
    return min(mean_scores, key=mean_scores.get), mean_scores


def pick_threshold(y, p, target_recall=TARGET_RECALL):
    """Highest threshold (best precision) that still reaches the target recall."""
    _, recalls, thresholds = precision_recall_curve(y, p)
    ok = np.where(recalls[:-1] >= target_recall)[0]
    return float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])


def threshold_report(y, p, threshold):
    pred = np.asarray(p) >= threshold
    y = np.asarray(y)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    return {"Threshold": threshold, "Recall": recall, "Precision": precision,
            "Specificity": tn / (tn + fp), "FlaggedShare": float(pred.mean()),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


# --------------------------------------------------------------------- training
def train_and_save():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    (X_fit, y_fit), (X_cal, y_cal), (X_test, y_test) = get_splits()
    params = lgbm_params()

    print(f"Training the Final Tuned LightGBM on {len(X_fit)} balanced rows "
          f"(COPD share {np.mean(y_fit):.0%})...")
    model = lgb.LGBMClassifier(**params).fit(X_fit, y_fit)

    raw_cal = model.predict_proba(X_cal)[:, 1]
    print(f"Raw mean probability on the natural cal split: {raw_cal.mean():.3f} "
          f"(real COPD rate {np.mean(y_cal):.3f}) -> needs calibration after balancing")
    method, cv_scores = select_method(raw_cal, y_cal)
    calibrator = fit_calibrator(method, raw_cal, y_cal)
    threshold = pick_threshold(y_cal, apply_calibrator(method, calibrator, raw_cal))
    print(f"Calibration: {method} (CV Brier { {k: round(v, 4) for k, v in cv_scores.items()} }) "
          f"| screening threshold = {threshold:.4f}")

    p_test = apply_calibrator(method, calibrator, model.predict_proba(X_test)[:, 1])
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, p_test),
        "PR-AUC": average_precision_score(y_test, p_test),
        "Brier": brier_score_loss(y_test, p_test),
        "MeanPredicted": float(p_test.mean()),
        "ObservedRate": float(np.mean(y_test)),
        "screening_threshold": threshold_report(y_test, p_test, threshold),
        "high_risk_threshold": threshold_report(y_test, p_test, HIGH_RISK_CUTOFF),
    }

    bundle = {
        "model": model,
        "calibration_method": method,
        "calibrator": calibrator,
        "threshold": threshold,
        "high_risk_cutoff": HIGH_RISK_CUTOFF,
        "target_recall": TARGET_RECALL,
        "features": cleaning_settings()["features"],
        "lgbm_params": params,
        "balancing": "random under-sampling of the fit split",
        "test_metrics": metrics,
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                     "lightgbm": lgb.__version__, "pandas": pd.__version__},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    s, h = metrics["screening_threshold"], metrics["high_risk_threshold"]
    print(f"\nTest ROC-AUC {metrics['ROC-AUC']:.4f} | PR-AUC {metrics['PR-AUC']:.4f} | Brier {metrics['Brier']:.4f} "
          f"| mean predicted {metrics['MeanPredicted']:.3f} vs observed {metrics['ObservedRate']:.3f}")
    print(f"Screening threshold {threshold:.3f}: Recall {s['Recall']:.3f} | Precision {s['Precision']:.3f} "
          f"| flags {s['FlaggedShare']:.1%} of people")
    print(f"High-risk cutoff {HIGH_RISK_CUTOFF:.2f}: Recall {h['Recall']:.3f} | Precision {h['Precision']:.3f} "
          f"| flags {h['FlaggedShare']:.1%} of people")
    print(f"Saved {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024:.0f} KB) and {METRICS_PATH.name}")


# -------------------------------------------------------------------- inference
@lru_cache(maxsize=None)
def load_model(path=MODEL_PATH):
    return joblib.load(path)


def _validate(record: dict):
    for column, allowed in VALID_VALUES.items():
        v = record.get(column)
        if v is not None and not pd.isna(v) and v not in allowed:
            raise ValueError(f"{column}={v} is invalid; allowed: {sorted(allowed)}")
    for column, (lo, hi) in VALID_RANGES.items():
        v = record.get(column)
        if v is not None and not pd.isna(v) and not lo <= v <= hi:
            raise ValueError(f"{column}={v} is outside the valid range [{lo}, {hi}]")


def _prepare(patient_data):
    """dict / list of dicts / DataFrame -> (clean model frame, list of missing-field lists)."""
    if isinstance(patient_data, dict):
        records = [patient_data]
    elif isinstance(patient_data, pd.DataFrame):
        records = patient_data.to_dict("records")
    else:
        records = list(patient_data)
    if not records:
        raise ValueError("No patient records provided.")
    frames, missing = [], []
    for r in records:
        if all(v is None or pd.isna(v) for v in r.values()):
            raise ValueError("At least one feature value is required.")
        _validate(r)
        f, m = prepare_patient(r)
        frames.append(f)
        missing.append(m)
    return pd.concat(frames, ignore_index=True), missing


def predict_risk(patient_data, model_path=MODEL_PATH):
    """Takes patient data (dict, list of dicts or DataFrame); returns risk score and category."""
    if not Path(model_path).exists():
        train_and_save()
    bundle = load_model(model_path)
    X, missing = _prepare(patient_data)
    X = X[bundle["features"]]

    model = bundle["model"]
    scores = apply_calibrator(bundle["calibration_method"], bundle["calibrator"],
                              model.predict_proba(X)[:, 1])
    # Per-patient feature contributions in log-odds (last column is the bias)
    contrib = model.predict(X, pred_contrib=True)[:, :-1]

    results = []
    for i, score in enumerate(scores):
        if score >= bundle["high_risk_cutoff"]:
            category = "High Risk"
        elif score >= bundle["threshold"]:
            category = "Moderate Risk"
        else:
            category = "Low Risk"
        row = X.iloc[i]
        order = contrib[i].argsort()[::-1][:3]
        factors = [{"feature": X.columns[j], "value": float(row.iloc[j]),
                    "contribution": round(float(contrib[i, j]), 3)}
                   for j in order if contrib[i, j] >= MIN_FACTOR_IMPACT and X.columns[j] not in missing[i]]
        results.append({
            "Risk Score (%)": round(float(score) * 100, 2),
            "Category": category,
            "flagged_for_screening": bool(score >= bundle["threshold"]),
            "screening_threshold": round(bundle["threshold"], 4),
            "recommendation": RECOMMENDATIONS[category],
            "top_risk_factors": factors,
            "missing_features": missing[i],
            "disclaimer": DISCLAIMER,
        })
    return results[0] if isinstance(patient_data, dict) else results


if __name__ == "__main__":
    train_and_save()

    samples = {
        "Young, healthy, never smoked": {
            "_AGE80": 28, "SEXVAR": 2, "BMI": 22.5, "SMOKE100": 2,
            "EXERANY2": 1, "GENHLTH": 1},
        "Middle-aged former smoker": {
            "_AGE80": 55, "SEXVAR": 1, "BMI": 29.0, "SMOKE100": 1, "SMOKDAY2": 3,
            "EXERANY2": 1, "GENHLTH": 3},
        "Older daily smoker, poor health (High Risk Profile)": {
            "_AGE80": 68, "SEXVAR": 2, "BMI": 24.0, "SMOKE100": 1, "SMOKDAY2": 1,
            "EXERANY2": 2, "GENHLTH": 5},
    }
    print("\n" + "=" * 50)
    print("Testing predict_risk function on sample patients...")
    print("=" * 50)
    for name, patient in samples.items():
        print(f"\n--- {name}")
        print(json.dumps(predict_risk(patient), indent=2))
