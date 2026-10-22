import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from catboost import CatBoostClassifier

# 1. Load Data
print("Loading Heart Disease dataset...")
df = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")

# 2. Features and Target
X = df.drop("HeartDiseaseorAttack", axis=1)
y = df["HeartDiseaseorAttack"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Train Baseline CatBoost with Best Hyperparameters (from Tuning)
print("\nTraining Baseline CatBoost...")
base_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=False,
    thread_count=-1,
    auto_class_weights="Balanced"
)
base_model.fit(X_train, y_train)

y_proba_base = base_model.predict_proba(X_test)[:, 1]

# 5. Apply Isotonic Calibration
print("\nApplying Isotonic Calibration...")
cal_iso = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
cal_iso.fit(X_train, y_train)
y_proba_iso = cal_iso.predict_proba(X_test)[:, 1]

# 6. Apply Sigmoid Calibration
print("\nApplying Sigmoid Calibration...")
cal_sig = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
cal_sig.fit(X_train, y_train)
y_proba_sig = cal_sig.predict_proba(X_test)[:, 1]

# 7. Summary Table
summary = pd.DataFrame({
    "Model": ["Baseline", "Isotonic", "Sigmoid"],
    "Brier Score": [
        brier_score_loss(y_test, y_proba_base),
        brier_score_loss(y_test, y_proba_iso),
        brier_score_loss(y_test, y_proba_sig)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_proba_base),
        roc_auc_score(y_test, y_proba_iso),
        roc_auc_score(y_test, y_proba_sig)
    ],
    "PR-AUC": [
        average_precision_score(y_test, y_proba_base),
        average_precision_score(y_test, y_proba_iso),
        average_precision_score(y_test, y_proba_sig)
    ]
}).set_index("Model")

print("\n" + "="*60)
print("CALIBRATION IMPACT SUMMARY")
print("="*60)
print(summary.round(4).to_string())

# 8. Calibration Curve Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, y_proba) in enumerate([
    ("Baseline", y_proba_base),
    ("Isotonic", y_proba_iso),
    ("Sigmoid", y_proba_sig)
]):
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    axes[i].plot(prob_pred, prob_true, marker='o', label=name)
    axes[i].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    axes[i].set_title(f"Calibration Curve: {name}")
    axes[i].set_xlabel("Mean Predicted Probability")
    axes[i].set_ylabel("Fraction of Positives")
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("heart_calibration_curve.png", dpi=300)
print("\nSaved calibration curve to: heart_calibration_curve.png")