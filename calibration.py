import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)
import lightgbm as lgb

# ==========================================================
# 1. Fetch Dataset
# ==========================================================
print("Loading CDC Diabetes dataset...")
cdc_data = fetch_ucirepo(id=891)
X = cdc_data.data.features.copy()
y = cdc_data.data.targets.values.ravel()

pos_ratio = (y == 0).sum() / (y == 1).sum()

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================================
# 2. Baseline LightGBM (No Calibration)
# ==========================================================
print("\n" + "="*60)
print("BASELINE LIGHTGBM (No Calibration)")
print("="*60)

base_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    scale_pos_weight=pos_ratio,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
base_model.fit(X_train, y_train)

y_proba_base = base_model.predict_proba(X_test)[:, 1]

print(f"ROC-AUC:     {roc_auc_score(y_test, y_proba_base):.4f}")
print(f"PR-AUC:      {average_precision_score(y_test, y_proba_base):.4f}")
print(f"Brier Score: {brier_score_loss(y_test, y_proba_base):.4f}")

# ==========================================================
# 3. Calibrated LightGBM (Isotonic)
# ==========================================================
print("\n" + "="*60)
print("CALIBRATED LIGHTGBM (Isotonic)")
print("="*60)

calibrated_model = CalibratedClassifierCV(
    base_model,
    method='isotonic',
    cv=5
)
calibrated_model.fit(X_train, y_train)

y_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]

print(f"ROC-AUC:     {roc_auc_score(y_test, y_proba_cal):.4f}")
print(f"PR-AUC:      {average_precision_score(y_test, y_proba_cal):.4f}")
print(f"Brier Score: {brier_score_loss(y_test, y_proba_cal):.4f}")


# ==========================================================
# 3b. Calibrated LightGBM (Sigmoid)
# ==========================================================
print("\n" + "="*60)
print("CALIBRATED LIGHTGBM (Sigmoid)")
print("="*60)

calibrated_model_sigmoid = CalibratedClassifierCV(
    base_model,
    method='sigmoid',
    cv=5
)
calibrated_model_sigmoid.fit(X_train, y_train)

y_proba_sigmoid = calibrated_model_sigmoid.predict_proba(X_test)[:, 1]

print(f"ROC-AUC:     {roc_auc_score(y_test, y_proba_sigmoid):.4f}")
print(f"PR-AUC:      {average_precision_score(y_test, y_proba_sigmoid):.4f}")
print(f"Brier Score: {brier_score_loss(y_test, y_proba_sigmoid):.4f}")

# ==========================================================
# 4. Calibration Curve Plot (3 Models)
# ==========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Baseline
prob_true_base, prob_pred_base = calibration_curve(y_test, y_proba_base, n_bins=10)
axes[0].plot(prob_pred_base, prob_true_base, marker='o', label="Baseline", color="#1f77b4", lw=2)
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
axes[0].set_title("Calibration Curve: Baseline", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Mean Predicted Probability")
axes[0].set_ylabel("Fraction of Positives")
axes[0].legend(loc="best")
axes[0].grid(True, alpha=0.3)

# Isotonic
prob_true_iso, prob_pred_iso = calibration_curve(y_test, y_proba_cal, n_bins=10)
axes[1].plot(prob_pred_iso, prob_true_iso, marker='o', label="Isotonic", color="#d62728", lw=2)
axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
axes[1].set_title("Calibration Curve: Isotonic", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Mean Predicted Probability")
axes[1].set_ylabel("Fraction of Positives")
axes[1].legend(loc="best")
axes[1].grid(True, alpha=0.3)

# Sigmoid
prob_true_sig, prob_pred_sig = calibration_curve(y_test, y_proba_sigmoid, n_bins=10)
axes[2].plot(prob_pred_sig, prob_true_sig, marker='o', label="Sigmoid", color="#2ca02c", lw=2)
axes[2].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
axes[2].set_title("Calibration Curve: Sigmoid", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Mean Predicted Probability")
axes[2].set_ylabel("Fraction of Positives")
axes[2].legend(loc="best")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("calibration_curve.png", dpi=300)
print("\nSaved calibration curve to: calibration_curve.png")

# ==========================================================
# 5. Summary Table (3 Models)
# ==========================================================
print("\n" + "="*60)
print("CALIBRATION IMPACT SUMMARY")
print("="*60)

summary = pd.DataFrame({
    "Model": ["Baseline", "Isotonic", "Sigmoid"],
    "ROC-AUC": [
        roc_auc_score(y_test, y_proba_base),
        roc_auc_score(y_test, y_proba_cal),
        roc_auc_score(y_test, y_proba_sigmoid)
    ],
    "PR-AUC": [
        average_precision_score(y_test, y_proba_base),
        average_precision_score(y_test, y_proba_cal),
        average_precision_score(y_test, y_proba_sigmoid)
    ],
    "Brier Score": [
        brier_score_loss(y_test, y_proba_base),
        brier_score_loss(y_test, y_proba_cal),
        brier_score_loss(y_test, y_proba_sigmoid)
    ]
}).set_index("Model")

print(summary.round(4).to_string())
print("\nCalibration complete!")