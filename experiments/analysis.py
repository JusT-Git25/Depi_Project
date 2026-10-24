import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import lightgbm as lgb
import shap

# 1. Fetch Dataset
print("Loading CDC Diabetes dataset...")
cdc_data = fetch_ucirepo(id=891)
X = cdc_data.data.features
y = cdc_data.data.targets.values.ravel()

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Train LightGBM (natural calibration)
print("Training LightGBM model...")
model = lgb.LGBMClassifier(
    n_estimators=250,
    max_depth=6,
    num_leaves=31,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
model.fit(X_train, y_train)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
print(f"Test ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}\n")

# 3. Threshold Analysis
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)

best_f1_idx = np.argmax(f1_scores)
best_f1_thresh = thresholds[best_f1_idx]
best_f1 = f1_scores[best_f1_idx]

# Clinical screening threshold: Target >= 80% Recall with highest possible Precision
clinical_indices = np.where(recalls[:-1] >= 0.80)[0]
if len(clinical_indices) == 0:
    raise ValueError("No threshold reaches 80% recall; cannot pick a clinical threshold.")
clinical_idx = clinical_indices[-1]
clinical_thresh = thresholds[clinical_idx]

print("="*65)
print("                    THRESHOLD STRATEGY COMPARISON")
print("="*65)
for name, thresh in [("Default (0.50)", 0.50),
                     ("Max F1 Threshold", best_f1_thresh),
                     ("High Sensitivity Clinical (>=80% Recall)", clinical_thresh)]:
    y_pred = (y_proba >= thresh).astype(int)
    rep = classification_report(y_test, y_pred, output_dict=True)
    pos_key = "1.0" if "1.0" in rep else "1"
    rec = rep[pos_key]["recall"]
    prec = rep[pos_key]["precision"]
    f1 = rep[pos_key]["f1-score"]
    acc = rep["accuracy"]
    print(f"[{name}] (Threshold = {thresh:.3f})")
    print(f"   Accuracy: {acc:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}\n")

# 4. Plot 1: Threshold Curves & Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Precision, Recall, F1 vs Threshold
axes[0].plot(thresholds, precisions[:-1], label="Precision", color="#1f77b4", lw=2)
axes[0].plot(thresholds, recalls[:-1], label="Recall (Sensitivity)", color="#d62728", lw=2)
axes[0].plot(thresholds, f1_scores, label="F1 Score", color="#2ca02c", lw=2, linestyle="--")
axes[0].axvline(best_f1_thresh, color="green", linestyle=":", label=f"Max F1 ({best_f1_thresh:.2f})")
axes[0].axvline(clinical_thresh, color="red", linestyle=":", label=f"Clinical Rec>=80% ({clinical_thresh:.2f})")
axes[0].set_title("Precision, Recall & F1 vs Decision Threshold", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Decision Threshold")
axes[0].set_ylabel("Score")
axes[0].set_xlim([0.05, 0.85])
axes[0].legend(loc="best")
axes[0].grid(True, alpha=0.3)

# Confusion Matrix at Clinical Threshold
cm_clinical = confusion_matrix(y_test, (y_proba >= clinical_thresh).astype(int))
sns.heatmap(cm_clinical, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1],
            xticklabels=["Predicted Healthy", "Predicted Diabetic"],
            yticklabels=["Actual Healthy", "Actual Diabetic"])
axes[1].set_title(f"Confusion Matrix (Clinical Threshold = {clinical_thresh:.2f})", fontsize=12, fontweight="bold")

plt.tight_layout()
OUT_DIR = Path(__file__).resolve().parent
plt.savefig(OUT_DIR / "threshold_curves.png", dpi=300)
print("Saved threshold curves plot to: threshold_curves.png")

# 5. SHAP Feature Importance Analysis
print("\nComputing SHAP values (using TreeExplainer)...")
explainer = shap.TreeExplainer(model)
X_sample = X_test.sample(n=min(5000, len(X_test)), random_state=42)
try:
    # New SHAP API (callable explainer) preferred; fall back to deprecated shap_values.
    shap_values = explainer(X_sample).values
except (TypeError, AttributeError):
    shap_values = explainer.shap_values(X_sample)

shap_vals_target = shap_values[1] if isinstance(shap_values, list) else shap_values

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals_target, X_sample, show=False)
plt.title("SHAP Summary Plot: Feature Impact on Diabetes Risk", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_summary.png", dpi=300)
print("Saved SHAP summary plot to: shap_summary.png")
print("\nAnalysis complete!")
