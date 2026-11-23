"""Comprehensive binary classification evaluation metrics, confidence intervals, and plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from liver_disease.config import RANDOM_STATE


def compute_classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute complete clinical-screening binary classification metrics.

    Positive class (1) = Liver Disease.
    Negative class (0) = No Liver Disease.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true_arr, y_pred)), 4),
        "precision": round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 4),
        "recall_sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "npv": round(npv, 4),
        "f1_score": round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4),
        "f2_score": round(float(fbeta_score(y_true_arr, y_pred, beta=2.0, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true_arr, y_proba_arr)), 4),
        "pr_auc": round(float(average_precision_score(y_true_arr, y_proba_arr)), 4),
        "brier_score": round(float(brier_score_loss(y_true_arr, y_proba_arr)), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "classification_report": classification_report(
            y_true_arr,
            y_pred,
            target_names=["No Liver Disease (2)", "Liver Disease (1)"],
            output_dict=True,
            zero_division=0,
        ),
    }


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, float]]:
    """Compute percentile bootstrap confidence intervals for key test-set metrics."""
    rng = np.random.default_rng(random_state)
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    n = len(y_true_arr)

    tracked: Dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "accuracy": [],
        "balanced_accuracy": [],
        "precision": [],
        "recall_sensitivity": [],
        "specificity": [],
        "f1_score": [],
    }

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx]
        if len(np.unique(yt)) < 2:
            continue
        yp = y_proba_arr[idx]
        m = compute_classification_metrics(yt, yp, threshold=threshold)
        for k in tracked:
            tracked[k].append(float(m[k]))

    alpha = (1.0 - confidence_level) / 2.0
    low_q = 100.0 * alpha
    high_q = 100.0 * (1.0 - alpha)

    cis: Dict[str, Dict[str, float]] = {}
    for k, vals in tracked.items():
        arr = np.asarray(vals, dtype=float)
        cis[k] = {
            "mean": round(float(np.mean(arr)), 4),
            "ci_lower": round(float(np.percentile(arr, low_q)), 4),
            "ci_upper": round(float(np.percentile(arr, high_q)), 4),
            "confidence_level": confidence_level,
            "n_valid_bootstraps": int(len(arr)),
        }
    return cis


def optimize_validation_threshold(
    y_true_oof: np.ndarray,
    y_proba_oof: np.ndarray,
    min_specificity: float = 0.50,
) -> Tuple[float, Dict[str, Any]]:
    """Select decision threshold strictly on out-of-fold validation predictions.

    Strategy:
    Evaluate candidate thresholds across [0.15, 0.85] (step 0.01).
    Among thresholds meeting `specificity >= min_specificity` (preventing trivial
    all-positive predictions), maximize balanced screening utility:
    `0.5 * recall_sensitivity + 0.5 * f1_score` (rewarding high disease capture
    while preserving precision and specificity).
    """
    y_true_arr = np.asarray(y_true_oof, dtype=int)
    y_proba_arr = np.asarray(y_proba_oof, dtype=float)

    candidates = []
    for thr in np.arange(0.15, 0.86, 0.01):
        m = compute_classification_metrics(y_true_arr, y_proba_arr, threshold=float(thr))
        utility = 0.5 * m["recall_sensitivity"] + 0.5 * m["f1_score"]
        candidates.append(
            {
                "threshold": round(float(thr), 2),
                "recall_sensitivity": m["recall_sensitivity"],
                "specificity": m["specificity"],
                "precision": m["precision"],
                "f1_score": m["f1_score"],
                "balanced_accuracy": m["balanced_accuracy"],
                "fn": m["confusion_matrix"]["fn"],
                "fp": m["confusion_matrix"]["fp"],
                "utility": round(float(utility), 4),
            }
        )

    feasible = [c for c in candidates if c["specificity"] >= min_specificity]
    if not feasible:
        # Fallback to maximizing Youden's J (sensitivity + specificity - 1)
        best = max(candidates, key=lambda c: c["recall_sensitivity"] + c["specificity"] - 1.0)
        rule_used = "fallback_youden_j"
    else:
        best = max(feasible, key=lambda c: (c["utility"], c["balanced_accuracy"], -c["fn"]))
        rule_used = f"max_screening_utility_with_specificity_ge_{min_specificity:.2f}"

    return float(best["threshold"]), {
        "selected_threshold": float(best["threshold"]),
        "default_threshold": 0.50,
        "selection_rule": rule_used,
        "min_specificity_constraint": min_specificity,
        "selected_oof_metrics": best,
        "default_oof_metrics": compute_classification_metrics(y_true_arr, y_proba_arr, threshold=0.50),
        "threshold_sweep": candidates,
    }


def save_evaluation_plots(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str,
    figures_dir: Path,
    prefix: str = "final_test",
) -> Dict[str, str]:
    """Generate and save ROC, PR, Confusion Matrix, and Calibration curves."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba_arr >= threshold).astype(int)

    paths: Dict[str, str] = {}

    # 1. ROC & PR Curve side-by-side
    fpr, tpr, _ = roc_curve(y_true_arr, y_proba_arr)
    roc_auc = roc_auc_score(y_true_arr, y_proba_arr)
    prec, rec, _ = precision_recall_curve(y_true_arr, y_proba_arr)
    pr_auc = average_precision_score(y_true_arr, y_proba_arr)
    baseline_prev = float(np.mean(y_true_arr))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    axes[0].plot(fpr, tpr, color="#2C7BB6", lw=2.2, label=f"{model_name} (ROC-AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Random Classifier (0.500)")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(rec, prec, color="#D7191C", lw=2.2, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
    axes[1].axhline(baseline_prev, color="gray", lw=1.2, linestyle="--", label=f"Prevalence Baseline ({baseline_prev:.3f})")
    axes[1].set_title("Precision-Recall (PR) Curve", fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (PPV)")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    roc_pr_path = figures_dir / f"{prefix}_roc_pr_curves.png"
    plt.savefig(roc_pr_path, dpi=180, bbox_inches="tight")
    plt.close()
    paths["roc_pr_curves"] = str(roc_pr_path)

    # 2. Confusion Matrices (Default 0.50 vs Selected Threshold)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, thr, label_title in [
        (axes[0], 0.50, "Default Threshold (0.50)"),
        (axes[1], threshold, f"Validation-Selected Threshold ({threshold:.2f})"),
    ]:
        yp_t = (y_proba_arr >= thr).astype(int)
        cm = confusion_matrix(y_true_arr, yp_t, labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred: No Disease (2)", "Pred: Disease (1)"],
            yticklabels=["True: No Disease (2)", "True: Disease (1)"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(label_title, fontweight="bold")
    plt.suptitle(f"Holdout Test Confusion Matrices — {model_name}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    cm_path = figures_dir / f"{prefix}_confusion_matrices.png"
    plt.savefig(cm_path, dpi=180, bbox_inches="tight")
    plt.close()
    paths["confusion_matrices"] = str(cm_path)

    # 3. Calibration / Reliability Curve
    prob_true, prob_pred = calibration_curve(y_true_arr, y_proba_arr, n_bins=8, strategy="quantile")
    brier = brier_score_loss(y_true_arr, y_proba_arr)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.plot(prob_pred, prob_true, marker="o", lw=2, color="#1B9E77", label=f"{model_name} (Brier = {brier:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    ax.set_title("Probability Calibration (Reliability) Diagram", fontweight="bold")
    ax.set_xlabel("Mean Predicted Score")
    ax.set_ylabel("Observed Disease Frequency")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    cal_path = figures_dir / f"{prefix}_calibration_curve.png"
    plt.savefig(cal_path, dpi=180, bbox_inches="tight")
    plt.close()
    paths["calibration_curve"] = str(cal_path)

    return paths
