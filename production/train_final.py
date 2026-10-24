"""Train, calibrate and package the final diabetes risk model.

Pipeline:  LightGBM (fit split) -> calibrator (cal split) -> clinical threshold (cal split)
Test split is only used to report final metrics.

Output:  artifacts/diabetes_risk_model.joblib   (everything predict.py needs)
         artifacts/metrics.json
Run from repo root:     python -m production.train_final
"""
import json
import sys
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from production.common import (
    ARTIFACT_DIR,
    HIGH_RISK_CUTOFF,
    LAB_MAPPABLE,
    LGBM_PARAMS,
    MASK_MAPPABLE_RATE,
    MASK_SURVEY_RATE,
    MODEL_PATH,
    SURVEY_ONLY,
    TARGET_RECALL,
    apply_calibrator,
    apply_lab_missingness,
    calibration_metrics,
    fit_calibrator,
    get_splits,
    labs_only_frame,
    load_data,
    pick_clinical_threshold,
    select_method,
    threshold_report,
)


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    splits = get_splits()
    X_fit, y_fit = splits["fit"]
    X_cal, y_cal = splits["cal"]
    X_test, y_test = splits["test"]

    print("Training LightGBM (missing-robust: full + 2 lab-masked copies)...")
    X_aug = pd.concat([
        X_fit,
        apply_lab_missingness(X_fit, seed=1),
        apply_lab_missingness(X_fit, seed=2),
    ], ignore_index=True)
    y_aug = np.concatenate([y_fit, y_fit, y_fit])
    model = lgb.LGBMClassifier(**LGBM_PARAMS).fit(X_aug, y_aug)

    raw_cal = model.predict_proba(X_cal)[:, 1]
    method, cv_scores = select_method(raw_cal, y_cal)
    calibrator = fit_calibrator(method, raw_cal, y_cal)
    threshold = pick_clinical_threshold(y_cal, apply_calibrator(method, calibrator, raw_cal))
    print(f"Calibration: {method} (CV Brier {cv_scores}) | clinical threshold = {threshold:.4f}")

    p_test = apply_calibrator(method, calibrator, model.predict_proba(X_test)[:, 1])
    p_labs = apply_calibrator(method, calibrator, model.predict_proba(labs_only_frame(X_test))[:, 1])
    metrics = {
        "calibration": calibration_metrics(y_test, p_test),
        "clinical_threshold": threshold_report(y_test, p_test, threshold),
        "labs_only": {  # worst-case stress: survey fields masked, same threshold
            **calibration_metrics(y_test, p_labs),
            **{"threshold_eval": threshold_report(y_test, p_labs, threshold)},
        },
    }

    # Valid input ranges of the RAW patient columns, used by predict_risk() for validation.
    raw_X, _ = load_data()
    bundle = {
        "model": model,
        "calibration_method": method,
        "calibrator": calibrator,
        "threshold": threshold,
        "high_risk_cutoff": HIGH_RISK_CUTOFF,
        "target_recall": TARGET_RECALL,
        "raw_features": list(raw_X.columns),
        "model_features": list(X_fit.columns),
        "input_ranges": {c: (float(raw_X[c].min()), float(raw_X[c].max())) for c in raw_X.columns},
        "lgbm_params": LGBM_PARAMS,
        "missing_robust": True,
        "lab_mappable": LAB_MAPPABLE,
        "survey_only": SURVEY_ONLY,
        "mask_rates": {"survey": MASK_SURVEY_RATE, "mappable": MASK_MAPPABLE_RATE},
        "test_metrics": metrics,
        "versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    c, t = metrics["calibration"], metrics["clinical_threshold"]
    print(f"\nTest ROC-AUC {c['ROC-AUC']:.4f} | PR-AUC {c['PR-AUC']:.4f} | Brier {c['Brier']:.4f} | ECE {c['ECE']:.4f}")
    print(f"At threshold {threshold:.3f}: Recall {t['Recall']:.3f} | Precision {t['Precision']:.3f} | "
          f"flags {t['FlaggedShare']:.1%} of patients")
    lb = metrics["labs_only"]
    print(f"Labs-only stress ROC-AUC {lb['ROC-AUC']:.4f} | PR-AUC {lb['PR-AUC']:.4f} | "
          f"Recall {lb['threshold_eval']['Recall']:.3f} | Precision {lb['threshold_eval']['Precision']:.3f}")
    print(f"Saved {MODEL_PATH.relative_to(MODEL_PATH.parents[1])} "
          f"({MODEL_PATH.stat().st_size / 1024:.0f} KB) and artifacts/metrics.json")


if __name__ == "__main__":
    main()
