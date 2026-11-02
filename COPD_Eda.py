import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load Dataset
print("Loading COPD dataset (CDC BRFSS)...")
df = pd.read_csv("COPD_dataset.csv")
print("Data loaded successfully.")

# Basic Information
print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Info:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Survey Codes: 7 = Don't know, 9 = Refused (both mean "no usable answer")
print("\nUnusable Survey Codes (7 = Don't know, 9 = Refused):")
for column in ["CHCCOPD3", "SMOKE100", "SMOKDAY2", "EXERANY2", "GENHLTH"]:
    print(f"{column}: 7 -> {(df[column] == 7).sum()}, 9 -> {(df[column] == 9).sum()}")

# Suspicious Values
print("\nSuspicious BMI Values (_BMI5 is stored as BMI x 100):")
print("BMI < 10:", (df["_BMI5"] / 100 < 10).sum())
print("BMI > 60:", (df["_BMI5"] / 100 > 60).sum())

# Structural Missingness
print("\nSMOKDAY2 is only asked of people who smoked 100+ cigarettes.")
print("SMOKDAY2 missing among never-smokers (SMOKE100 = 2):",
      df.loc[df["SMOKE100"] == 2, "SMOKDAY2"].isnull().mean().round(4))

# Cleaning (same rules used in every other COPD script)
df = df[df["CHCCOPD3"].isin([1.0, 2.0])].copy()
df["COPD"] = (df["CHCCOPD3"] == 1.0).astype(int)
df["BMI"] = df["_BMI5"] / 100
df.loc[(df["BMI"] < 10) | (df["BMI"] > 80), "BMI"] = np.nan
for column in ["SMOKE100", "SMOKDAY2", "EXERANY2", "GENHLTH"]:
    df[column] = df[column].replace({7.0: np.nan, 9.0: np.nan})
df.loc[df["SMOKE100"] == 2.0, "SMOKDAY2"] = 0.0

# Target Distribution
print("\nTarget Distribution:")
print(df["COPD"].value_counts())

print("\nTarget Distribution (%):")
print(df["COPD"].value_counts(normalize=True) * 100)

# COPD Rate per Category
for column in ["SEXVAR", "SMOKE100", "SMOKDAY2", "EXERANY2", "GENHLTH"]:
    print(f"\nCOPD rate by {column}:")
    print((df.groupby(column)["COPD"].mean() * 100).round(2))

# Correlation with Target
features = ["_AGE80", "SEXVAR", "BMI", "SMOKE100", "SMOKDAY2", "EXERANY2", "GENHLTH", "COPD"]
print("\nCorrelation with Target (COPD):")
correlation = df[features].corr()["COPD"].sort_values(ascending=False)
print(correlation)

# Create plots folder inside the project folder
plots_folder = "plots"
os.makedirs(plots_folder, exist_ok=True)

# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("COPD Correlation Heatmap")
plt.tight_layout()
heatmap_path = os.path.join(plots_folder, "copd_correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nCorrelation heatmap saved to: {heatmap_path}")

# 2. Target Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="COPD")
plt.title("COPD Distribution")
plt.xlabel("COPD")
plt.ylabel("Count")
plt.xticks([0, 1], ["No COPD", "COPD"])
plt.tight_layout()
target_path = os.path.join(plots_folder, "copd_target_distribution.png")
plt.savefig(target_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Target distribution saved to: {target_path}")

# 3. COPD Rate by Age
plt.figure(figsize=(9, 5))
age_rate = df.assign(AgeBand=pd.cut(df["_AGE80"], range(15, 90, 5))).groupby(
    "AgeBand", observed=True)["COPD"].mean() * 100
age_rate.plot(kind="bar")
plt.title("COPD Rate by Age Group")
plt.xlabel("Age Group (Years)")
plt.ylabel("COPD Rate (%)")
plt.tight_layout()
age_path = os.path.join(plots_folder, "copd_rate_by_age.png")
plt.savefig(age_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Age plot saved to: {age_path}")

# 4. General Health vs COPD
plt.figure(figsize=(8, 5))
sns.countplot(data=df.dropna(subset=["GENHLTH"]), x="GENHLTH", hue="COPD")
plt.title("General Health (1 = Excellent ... 5 = Poor) vs COPD")
plt.xlabel("General Health")
plt.ylabel("Count")
plt.tight_layout()
genhlth_path = os.path.join(plots_folder, "copd_genhlth_vs_copd.png")
plt.savefig(genhlth_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"General health plot saved to: {genhlth_path}")
