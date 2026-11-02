import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# 1. Load Data
print("Loading COPD dataset (CDC BRFSS)...")
df = pd.read_csv("COPD_dataset.csv")

# 2. Basic Cleaning
df = df[df["CHCCOPD3"].isin([1.0, 2.0])].copy()
df["COPD"] = (df["CHCCOPD3"] == 1.0).astype(int)
df["BMI"] = df["_BMI5"] / 100
df.loc[(df["BMI"] < 10) | (df["BMI"] > 80), "BMI"] = np.nan
for column in ["SMOKE100", "SMOKDAY2", "EXERANY2", "GENHLTH"]:
    df[column] = df[column].replace({7.0: np.nan, 9.0: np.nan})
df.loc[df["SMOKE100"] == 2.0, "SMOKDAY2"] = 0.0

# 3. Features and Target
X = df[["_AGE80", "SEXVAR", "BMI", "SMOKE100", "SMOKDAY2", "EXERANY2", "GENHLTH"]]
y = df["COPD"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Train Tuned LightGBM (Best Params from COPD_HyperparameterTuning.py)
print("\nTraining Tuned LightGBM...")
base_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=100,
    reg_lambda=0,
    colsample_bytree=0.85,
    subsample=0.85,
    subsample_freq=1,
    random_state=42,
    verbosity=-1,
    n_jobs=2
)
base_model.fit(X_train, y_train)
y_proba_base = base_model.predict_proba(X_test)[:, 1]

# 6. Apply Isotonic and Sigmoid Calibration
print("\nApplying Isotonic Calibration...")
cal_iso = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
cal_iso.fit(X_train, y_train)
y_proba_iso = cal_iso.predict_proba(X_test)[:, 1]

print("\nApplying Sigmoid Calibration...")
cal_sig = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
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

print("\n" + "=" * 60)
print("CALIBRATION IMPACT SUMMARY")
print("=" * 60)
print(summary.round(4).to_string())

# 8. Calibration Curve Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, y_proba) in enumerate([
    ("Baseline", y_proba_base), ("Isotonic", y_proba_iso), ("Sigmoid", y_proba_sig)
]):
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
    axes[i].plot(prob_pred, prob_true, marker="o", label=name)
    axes[i].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    axes[i].set_title(f"Calibration Curve: {name}")
    axes[i].set_xlabel("Mean Predicted Probability")
    axes[i].set_ylabel("Fraction of Positives")
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("copd_calibration_curve.png", dpi=300)
print("\nSaved calibration curve to: copd_calibration_curve.png")
