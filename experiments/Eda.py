import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

TARGET_COL = "Diabetes"


def _normalize_target(data: pd.DataFrame) -> pd.DataFrame:
    """Unify the target column to binary TARGET_COL (1 = diabetes).

    The committed CSV uses the 3-class `Diabetes_012` (0 = none, 1 = prediabetes,
    2 = diabetes); positives are diabetes only (class 2), matching common.load_data().
    A freshly fetched UCI copy may already be binary.
    """
    data = data.copy()
    if TARGET_COL in data.columns:
        return data
    if "Diabetes_binary" in data.columns:
        data[TARGET_COL] = data.pop("Diabetes_binary").astype(int)
    elif "Diabetes_012" in data.columns:
        data[TARGET_COL] = (data.pop("Diabetes_012") == 2).astype(int)
    else:
        raise ValueError(f"No diabetes target column found (columns: {list(data.columns)})")
    return data


# Load from fast local CSV (or fetch once if missing)
csv_file = REPO_ROOT / "diabetes_data.csv"
if not os.path.exists(csv_file):
    print("Local CSV not found. Fetching from UCI repository...")
    from ucimlrepo import fetch_ucirepo
    cdc = fetch_ucirepo(id=891)
    data = pd.DataFrame(cdc.data.features)
    data[TARGET_COL] = np.asarray(cdc.data.targets.iloc[:, 0].values).ravel().astype(int)
    # A fresh UCI fetch is binary (0/1); keep the on-disk copy consistent with it.
    data.to_csv(csv_file, index=False)
    print("Saved local copy to diabetes_data.csv!")
else:
    print("Loading data from local diabetes_data.csv...")
    data = pd.read_csv(csv_file)
data = _normalize_target(data)

# 2. Data Shape & Types
print("\n" + "="*50)
print(f"Data Shape: {data.shape[0]} rows, {data.shape[1]} columns")
print("="*50)

# Missing values check
null_counts = data.isnull().sum().sum()
print(f"Total Missing / Null Values: {null_counts}")

# Target class distribution
class_counts = data["Diabetes"].value_counts()
class_pcts = data["Diabetes"].value_counts(normalize=True) * 100
print("\n--- Target Class Distribution ---")
for cls in [0, 1]:
    name = "Non-Diabetic" if cls == 0 else "Diabetic"
    print(f"  {name} ({cls}): {class_counts.get(cls, 0):,} ({class_pcts.get(cls, 0.0):.2f}%)")

# 3. Quick Summary Statistics of Key Numerical & Ordinal Features
key_features = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Income"]
print("\n--- Key Features Summary Statistics ---")
print(data[key_features].describe().round(2).to_string())

# 4. Correlation with Target (Top Risk Drivers)
print("\n--- Top Features Correlated with Diabetes ---")
correlations = data.corr(numeric_only=True)["Diabetes"].drop("Diabetes").sort_values(ascending=False)
print(correlations.round(4).to_string())

# ==========================================================
# 5. FEATURE ENGINEERING INVESTIGATION
# ==========================================================
print("\n" + "="*50)
print("FEATURE ENGINEERING EXPLORATION")
print("="*50)

# Idea 1: Cardio-Metabolic Risk Index (Sum of co-morbidities)
# Combines HighBP, HighChol, Stroke, HeartDiseaseorAttack
data["CardioRiskIndex"] = (
    data["HighBP"] + 
    data["HighChol"] + 
    data["Stroke"] + 
    data["HeartDiseaseorAttack"]
)

# Idea 2: BMI Categories (CDC Standard: Underweight, Normal, Overweight, Obese)
# BMI < 18.5 -> 0, 18.5-24.9 -> 1, 25-29.9 -> 2, >= 30 -> 3
data["BMICategory"] = pd.cut(
    data["BMI"],
    bins=[0, 18.5, 24.9, 29.9, 100],
    labels=[0, 1, 2, 3]
).astype(float)

# Idea 3: Metabolic Syndrome Proxy (HighBP + HighChol + Obese BMI)
data["MetabolicSyndrome"] = (
    (data["HighBP"] == 1) & 
    (data["HighChol"] == 1) & 
    (data["BMI"] >= 30)
).astype(int)

# Idea 4: Total Unhealthy Days (Physical + Mental health days past 30 days)
data["TotalUnhealthyDays"] = data["PhysHlth"] + data["MentHlth"]

# Check if engineered features have stronger correlation with Diabetes
new_features = ["CardioRiskIndex", "BMICategory", "MetabolicSyndrome", "TotalUnhealthyDays"]
print("Correlations of New Engineered Features:")
print(data[new_features + ["Diabetes"]].corr(numeric_only=True)["Diabetes"].drop("Diabetes").round(4))
