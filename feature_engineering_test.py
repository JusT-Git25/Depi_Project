import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# ==========================================================
# 1. Load Data
# ==========================================================
print("Loading CDC Diabetes dataset...")
cdc_data = fetch_ucirepo(id=891)
X = cdc_data.data.features.copy()
y = cdc_data.data.targets.values.ravel()

pos_ratio = (y == 0).sum() / (y == 1).sum()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================================
# 2. Baseline (No FE)
# ==========================================================
print("\n" + "="*60)
print("BASELINE (No Feature Engineering)")
print("="*60)

baseline = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.08,
    scale_pos_weight=pos_ratio, random_state=42, n_jobs=-1, verbose=-1
)
baseline.fit(X_train, y_train)

y_pred_base = baseline.predict(X_test)
y_proba_base = baseline.predict_proba(X_test)[:, 1]

report_base = classification_report(y_test, y_pred_base, output_dict=True)
key = '1.0' if '1.0' in report_base else '1'

print(f"Recall:    {report_base[key]['recall']:.4f}")
print(f"Precision: {report_base[key]['precision']:.4f}")
print(f"F1:        {report_base[key]['f1-score']:.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_base):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, y_proba_base):.4f}")

# ==========================================================
# 3. Feature Engineering (New Features)
# ==========================================================
def apply_fe(df):
    df = df.copy()
    
    # New Feature 1: HighBP * Age
    df["HighBP_Age"] = df["HighBP"] * df["Age"]
    
    # New Feature 2: Smoker * BMI
    df["Smoker_BMI"] = df["Smoker"] * df["BMI"]
    
    # New Feature 3: PhysActivity * BMI
    df["PhysActivity_BMI"] = df["PhysActivity"] * df["BMI"]
    
    # New Feature 4: Fruits + Veggies
    df["Fruits_Veggies"] = df["Fruits"] + df["Veggies"]
    
    return df

X_train_fe = apply_fe(X_train)
X_test_fe = apply_fe(X_test)

# ==========================================================
# 4. LightGBM with New FE
# ==========================================================
print("\n" + "="*60)
print("LIGHTGBM WITH NEW FEATURE ENGINEERING")
print("="*60)

fe_model = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.08,
    scale_pos_weight=pos_ratio, random_state=42, n_jobs=-1, verbose=-1
)
fe_model.fit(X_train_fe, y_train)

y_pred_fe = fe_model.predict(X_test_fe)
y_proba_fe = fe_model.predict_proba(X_test_fe)[:, 1]

report_fe = classification_report(y_test, y_pred_fe, output_dict=True)
key = '1.0' if '1.0' in report_fe else '1'

print(f"Recall:    {report_fe[key]['recall']:.4f}")
print(f"Precision: {report_fe[key]['precision']:.4f}")
print(f"F1:        {report_fe[key]['f1-score']:.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_fe):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, y_proba_fe):.4f}")

# ==========================================================
# 5. Feature Importance
# ==========================================================
print("\n" + "="*60)
print("TOP 15 FEATURES (with New FE)")
print("="*60)

importance = pd.DataFrame({
    "Feature": X_train_fe.columns,
    "Importance": fe_model.feature_importances_
}).sort_values("Importance", ascending=False).head(15)

print(importance.to_string(index=False))

# ==========================================================
# 6. Comparison Summary
# ==========================================================
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)

comparison = pd.DataFrame({
    "Model": ["Baseline", "New FE"],
    "Recall": [report_base[key]['recall'], report_fe[key]['recall']],
    "Precision": [report_base[key]['precision'], report_fe[key]['precision']],
    "F1": [report_base[key]['f1-score'], report_fe[key]['f1-score']],
    "ROC-AUC": [roc_auc_score(y_test, y_proba_base), roc_auc_score(y_test, y_proba_fe)],
    "PR-AUC": [average_precision_score(y_test, y_proba_base), average_precision_score(y_test, y_proba_fe)]
}).set_index("Model")

print(comparison.round(4).to_string())