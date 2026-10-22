import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

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

pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()

# ==========================================================
# 4. Baseline (No Feature Engineering)
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
# 5. Feature Engineering
# ==========================================================
def apply_fe(df):
    df = df.copy()
    
    # 1. Cardio-Metabolic Risk 
    df["CardioRisk"] = df["HighBP"] + df["HighChol"] + df["Diabetes"]
    
    # 2. Age & BMI Interaction 
    df["Age_BMI"] = df["Age"] * df["BMI"]
    
    # 3. Lifestyle Risk 
    df["LifestyleRisk"] = df["Smoker"] + (1 - df["PhysActivity"])
    
    # 4. General Health & Age
    df["GenHlth_Age"] = df["GenHlth"] * df["Age"]
    
    # 5. Mental & Physical Health combined
    df["TotalUnhealthyDays"] = df["MentHlth"] + df["PhysHlth"]
    
    return df

X_train_fe = apply_fe(X_train)
X_test_fe = apply_fe(X_test)

# ==========================================================
# 6. LightGBM with Feature Engineering
# ==========================================================
print("\n" + "="*60)
print("LIGHTGBM WITH FEATURE ENGINEERING")
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
# 7. Feature Importance
# ==========================================================
print("\n" + "="*60)
print("TOP 15 FEATURES (with FE)")
print("="*60)

importance = pd.DataFrame({
    "Feature": X_train_fe.columns,
    "Importance": fe_model.feature_importances_
}).sort_values("Importance", ascending=False).head(15)

print(importance.to_string(index=False))