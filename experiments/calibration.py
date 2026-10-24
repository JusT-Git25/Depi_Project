"""Model calibration analysis for the LightGBM diabetes risk model.

Question answered here: do the predicted probabilities mean what they say?
(i.e. of the patients given ~30% risk, do ~30% actually have diabetes?)

For two LightGBM variants we compare three versions of the probabilities:
    * Uncalibrated
    * Platt scaling (sigmoid)
    * Isotonic regression
using Brier score, log loss and expected calibration error (ECE), and plot reliability
curves. Calibrators are fitted on a held-out calibration split, never on the test set.

Outputs:  experiments/calibration_curves.png   artifacts/calibration_results.csv
Run from repo root:  python -m experiments.calibration
"""
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

from production.common import (
    ARTIFACT_DIR,
    LGBM_PARAMS,
    TARGET_RECALL,
    apply_calibrator,
    calibration_metrics,
    fit_calibrator,
    get_splits,
    pick_clinical_threshold,
    select_method,
    threshold_report,
)

OUT_DIR = Path(__file__).resolve().parent

METHODS = ["none", "sigmoid", "isotonic"]
LABELS = {"none": "Uncalibrated", "sigmoid": "Platt (sigmoid)", "isotonic": "Isotonic"}
COLORS = {"none": "#d62728", "sigmoid": "#1f77b4", "isotonic": "#2ca02c"}
MAIN_VARIANT = "LightGBM (unweighted)"


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    splits = get_splits()
    X_fit, y_fit = splits["fit"]
    X_cal, y_cal = splits["cal"]
    X_test, y_test = splits["test"]
    print(f"fit={len(y_fit):,}  cal={len(y_cal):,}  test={len(y_test):,}  "
          f"(positive rate {y_test.mean():.3%})\n")

    # The unweighted model is what analysis.py uses. The scale_pos_weight variant is what
    # TreeBoosts.py benchmarked; it is included to show how class weighting distorts probabilities.
    spw = (y_fit == 0).sum() / (y_fit == 1).sum()
    variants = {
        MAIN_VARIANT: LGBM_PARAMS,
        "LightGBM (scale_pos_weight)": {**LGBM_PARAMS, "scale_pos_weight": spw},
    }

    rows, test_probs, main_raw_cal = [], {}, None
    for name, params in variants.items():
        print(f"--> Training {name}...")
        model = lgb.LGBMClassifier(**params).fit(X_fit, y_fit)
        raw_cal = model.predict_proba(X_cal)[:, 1]
        raw_test = model.predict_proba(X_test)[:, 1]
        if name == MAIN_VARIANT:
            main_raw_cal = raw_cal
        test_probs[name] = {}
        for method in METHODS:
            calibrator = fit_calibrator(method, raw_cal, y_cal)
            p_test = apply_calibrator(method, calibrator, raw_test)
            test_probs[name][method] = p_test
            rows.append({"Model": name, "Calibration": LABELS[method], **calibration_metrics(y_test, p_test)})

    results = pd.DataFrame(rows)
    results.to_csv(ARTIFACT_DIR / "calibration_results.csv", index=False)
    print("\n" + "=" * 100)
    print("CALIBRATION COMPARISON ON THE TEST SET (lower Brier / LogLoss / ECE is better)")
    print("=" * 100)
    print(results.set_index(["Model", "Calibration"]).round(4).to_string())

    # ---- calibrator choice and clinical threshold for the main model (no test data used)
    method, cv_scores = select_method(main_raw_cal, y_cal)
    print(f"\nCalibrator chosen by 5-fold CV Brier on the calibration split: {LABELS[method]}  "
          f"({', '.join(f'{LABELS[k]}={v:.5f}' for k, v in cv_scores.items())})")

    calibrator = fit_calibrator(method, main_raw_cal, y_cal)
    p_cal = apply_calibrator(method, calibrator, main_raw_cal)
    threshold = pick_clinical_threshold(y_cal, p_cal)
    p_test = test_probs[MAIN_VARIANT][method]

    print("\n" + "=" * 100)
    print(f"THRESHOLDS ON CALIBRATED PROBABILITIES (clinical = Recall >= {TARGET_RECALL:.0%}, picked on the cal split)")
    print("=" * 100)
    threshold_rows = {
        "Default 0.50": threshold_report(y_test, p_test, 0.50),
        f"Clinical {threshold:.3f}": threshold_report(y_test, p_test, threshold),
    }
    print(pd.DataFrame(threshold_rows).T.round(4).to_string())

    # ---- reliability diagrams
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, name in zip(axes, variants):
        top = 0.0
        for m in METHODS:
            frac_pos, mean_pred = calibration_curve(y_test, test_probs[name][m], n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", lw=2, color=COLORS[m], label=LABELS[m])
            top = max(top, mean_pred.max(), frac_pos.max())
        top = min(1.0, top * 1.08)
        ax.plot([0, top], [0, top], "k--", lw=1, label="Perfect calibration")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction with diabetes")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    fig.suptitle("Reliability curves on the test set (10 equal-size bins)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_curves.png", dpi=200)
    print("\nSaved calibration_curves.png and artifacts/calibration_results.csv")


if __name__ == "__main__":
    main()
