import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# 1. Load Data
print("Loading Cardiovascular Disease dataset...")
df = pd.read_csv("cardio_train.csv", sep=";")

# 2. Basic Cleaning (Remove impossible values)
print("\nCleaning impossible values...")
df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 220)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 140)]
df = df[(df['height'] >= 120) & (df['height'] <= 220)]
df = df[(df['weight'] >= 40) & (df['weight'] <= 180)]
print(f"Data shape after cleaning: {df.shape}")

# 3. Preprocessing
df["age"] = df["age"] / 365.25
X = df.drop(columns=["cardio", "id"])
y = df["cardio"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================================
# 5. Baseline (No Feature Engineering)
# ==========================================================
print("\n" + "="*60)
print("BASELINE (No Feature Engineering)")
print("="*60)

baseline_model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=-1,
    random_state=42, verbose=-1
)
baseline_model.fit(X_train, y_train)
y_pred_base = baseline_model.predict(X_test)
y_proba_base = baseline_model.predict_proba(X_test)[:, 1]
report_base = classification_report(y_test, y_pred_base, output_dict=True)
key = '1.0' if '1.0' in report_base else '1'

print(f"Recall:    {report_base[key]['recall']:.4f}")
print(f"Precision: {report_base[key]['precision']:.4f}")
print(f"F1:        {report_base[key]['f1-score']:.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_base):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, y_proba_base):.4f}")

# ==========================================================
# 6. Feature Engineering
# ==========================================================
def apply_fe(df):
    df = df.copy()
    
    # 1. Body Mass Index (BMI)
    df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
    
    # 2. Pulse Pressure (Systolic - Diastolic)
    df["Pulse_Pressure"] = df["ap_hi"] - df["ap_lo"]
    
    # 3. Mean Arterial Pressure (MAP)
    df["MAP"] = df["ap_lo"] + ((df["ap_hi"] - df["ap_lo"]) / 3)
    
    # 4. Blood Pressure Ratio
    df["BP_Ratio"] = df["ap_hi"] / df["ap_lo"]
    
    # 5. Lifestyle Risk Score (Smoke + Alcohol + Inactive)
    df["Lifestyle_Risk"] = df["smoke"] + df["alco"] + (1 - df["active"])
    
    return df

X_train_fe = apply_fe(X_train)
X_test_fe = apply_fe(X_test)

# ==========================================================
# 7. LightGBM with Feature Engineering
# ==========================================================
print("\n" + "="*60)
print("LIGHTGBM WITH FEATURE ENGINEERING")
print("="*60)

fe_model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=-1,
    random_state=42, verbose=-1
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

# 8. Feature Importance
print("\n" + "="*60)
print("TOP 10 FEATURES (with FE)")
print("="*60)
importance = pd.DataFrame({
    "Feature": X_train_fe.columns,
    "Importance": fe_model.feature_importances_
}).sort_values("Importance", ascending=False).head(10)
print(importance.to_string(index=False))