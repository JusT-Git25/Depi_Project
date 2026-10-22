import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

import os
import pandas as pd
import numpy as np

# Load from fast local CSV (or fetch once if missing)
csv_file = "diabetes_data.csv"
if not os.path.exists(csv_file):
    print("Local CSV not found. Fetching from UCI repository...")
    from ucimlrepo import fetch_ucirepo
    cdc = fetch_ucirepo(id=891)
    data = pd.DataFrame(cdc.data.features)
    data["Diabetes"] = cdc.data.targets.iloc[:, 0].values
    data.to_csv(csv_file, index=False)
    print("Saved local copy to diabetes_data.csv!")
else:
    print("Loading data from local diabetes_data.csv...")
    data = pd.read_csv(csv_file)

# 2. Data Shape & Types
print("\n" + "="*50)
print(f"Data Shape: {data.shape[0]} rows, {data.shape[1]} columns")
print("="*50)

# Missing values check
null_counts = data.isnull().sum().sum()
print(f"Total Missing / Null Values: {null_counts}")

# Target class distribution
class_counts = data["Diabetes_012"].value_counts()
class_pcts = data["Diabetes_012"].value_counts(normalize=True) * 100
print("\n--- Target Class Distribution ---")
for cls in [0, 1]:
    name = "Non-Diabetic" if cls == 0 else "Diabetic/Pre"
    print(f"  {name} ({cls}): {class_counts[cls]:,} ({class_pcts[cls]:.2f}%)")

# 3. Quick Summary Statistics of Key Numerical & Ordinal Features
key_features = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Income"]
print("\n--- Key Features Summary Statistics ---")
print(data[key_features].describe().round(2).to_string())

# 4. Correlation with Target (Top Risk Drivers)
print("\n--- Top Features Correlated with Diabetes ---")
correlations = data.corr()["Diabetes_012"].drop("Diabetes_012").sort_values(ascending=False)
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
print(data[new_features + ["Diabetes_012"]].corr()["Diabetes_012"].drop("Diabetes_012").round(4))
