"""Shared configuration, data loading and calibration helpers.

calibration.py (analysis) and train_final.py (packaging) both import from here,
so there are exactly two places to touch when the rest of the team's work lands:

    * LGBM_PARAMS          -> paste the best Optuna / RandomizedSearchCV params
    * engineer_features()  -> add the engineered columns

After changing either one, re-run `python calibration.py` and `python train_final.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "diabetes_data.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "diabetes_risk_model.joblib"

RANDOM_STATE = 42
TARGET_RECALL = 0.80      # screening goal from analysis.py
HIGH_RISK_CUTOFF = 0.50   # calibrated probability above which risk is labelled "High"

# Currently the LightGBM settings from analysis.py (no class weights, so the
# probabilities start out close to calibrated). Replace with the tuned values.
LGBM_PARAMS = dict(
    n_estimators=250,
    max_depth=6,
    num_leaves=31,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hook for the feature-engineering work. Must work on raw patient columns only.

    It is applied during training AND inside predict_risk(), so anything added here
    reaches the packaged model automatically. Currently a no-op.
    """
    return df


# --------------------------------------------------------------------------- data
def load_data(path: Path = CSV_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the CDC diabetes CSV and return (raw features, binary target)."""
    df = pd.read_csv(path)
    if "Diabetes_binary" in df:
        y = df.pop("Diabetes_binary")
    elif "Diabetes" in df:
        y = df.pop("Diabetes")
    elif "Diabetes_012" in df:
        # The committed CSV is the 3-class file: 0 = none, 1 = prediabetes, 2 = diabetes.
        # Class 2 has 35,346 rows (13.9%), the same as the positive class in the binary
        # target the team's scripts download from UCI, so positives = diabetes only.
        y = df.pop("Diabetes_012") == 2
    else:
        raise ValueError(f"No diabetes target column found in {path}")
    return df, y.astype(int)


def get_splits() -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Return fit / cal / test splits.

    Test is the same 80/20 stratified split (seed 42) the team already uses.
    The remaining 80% is split again 75/25: "fit" trains the model, "cal" is held out
    to fit the calibrator and to choose the clinical threshold. Test is never used
    for any fitting or selection.
    """
    X, y = load_data()
    X = engineer_features(X)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv
    )
    return {
        "fit": (X_fit, y_fit.to_numpy()),
        "cal": (X_cal, y_cal.to_numpy()),
        "test": (X_test, y_test.to_numpy()),
    }


# ---------------------------------------------------------------- calibrators
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(method: str, p_raw: np.ndarray, y: np.ndarray):
    """method: 'none' | 'sigmoid' (Platt scaling) | 'isotonic'."""
    if method == "none":
        return None
    if method == "sigmoid":
        # Platt scaling: logistic regression on the model's log-odds.
        return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw), y)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_raw, y)
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(method: str, calibrator, p_raw: np.ndarray) -> np.ndarray:
    p_raw = np.asarray(p_raw, dtype=float)
    if method == "none":
        return p_raw
    if method == "sigmoid":
        return calibrator.predict_proba(_logit(p_raw))[:, 1]
    if method == "isotonic":
        return calibrator.predict(p_raw)
    raise ValueError(f"Unknown calibration method: {method}")


def select_method(p_raw: np.ndarray, y: np.ndarray, methods=("sigmoid", "isotonic"), n_splits: int = 5):
    """Pick the calibrator with the lowest cross-validated Brier score on the calibration set.

    Only the calibration split is used, so the test set is never consulted.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = {m: [] for m in methods}
    for tr, va in skf.split(p_raw, y):
        for m in methods:
            cal = fit_calibrator(m, p_raw[tr], y[tr])
            scores[m].append(brier_score_loss(y[va], apply_calibrator(m, cal, p_raw[va])))
    mean_scores = {m: float(np.mean(v)) for m, v in scores.items()}
    return min(mean_scores, key=mean_scores.get), mean_scores


# -------------------------------------------------------------------- metrics
def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Weighted average gap between predicted probability and observed rate per bin."""
    bins = np.clip((np.asarray(p) * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "ECE": expected_calibration_error(y, p),
        "MeanPredicted": float(np.mean(p)),
        "ObservedRate": float(np.mean(y)),
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
    }


def pick_clinical_threshold(y: np.ndarray, p: np.ndarray, target_recall: float = TARGET_RECALL) -> float:
    """Highest threshold (so best precision) that still reaches the target recall."""
    _, recalls, thresholds = precision_recall_curve(y, p)
    ok = np.where(recalls[:-1] >= target_recall)[0]
    return float(thresholds[ok[-1]])


def threshold_report(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "Threshold": threshold,
        "Recall": recall,
        "Precision": precision,
        "Specificity": tn / (tn + fp),
        "F1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "FlaggedShare": float(pred.mean()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }
