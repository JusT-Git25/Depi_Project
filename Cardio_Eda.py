import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Dataset
print("Loading Cardiovascular Disease dataset...")

df = pd.read_csv("cardio_train.csv", sep=";")
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

# Convert Age from Days to Years
df["age"] = df["age"] / 365.25
print("\nAge converted from days to years.")

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Check Suspicious Values
print("\nSuspicious Blood Pressure Values:")

print("ap_hi < 50:", (df["ap_hi"] < 50).sum())
print("ap_hi > 250:", (df["ap_hi"] > 250).sum())

print("ap_lo < 30:", (df["ap_lo"] < 30).sum())
print("ap_lo > 200:", (df["ap_lo"] > 200).sum())

print("\nSuspicious Height Values:")
print("height < 100:", (df["height"] < 100).sum())
print("height > 220:", (df["height"] > 220).sum())

print("\nSuspicious Weight Values:")
print("weight < 30:", (df["weight"] < 30).sum())
print("weight > 150:", (df["weight"] > 150).sum())

# Target Distribution
print("\nTarget Distribution:")
print(df["cardio"].value_counts())

print("\nTarget Distribution (%):")
print(df["cardio"].value_counts(normalize=True) * 100)


# Categorical Features Distribution
categorical_features = [
    "gender",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active"
]

for column in categorical_features:
    print(f"\n{column} Distribution:")
    print(df[column].value_counts())

# Correlation with Target
print("\nCorrelation with Target (cardio):")

correlation = df.corr(numeric_only=True)["cardio"].sort_values(ascending=False)
print(correlation)

# Create plots folder inside the project folder
plots_folder = "plots"
os.makedirs(plots_folder, exist_ok=True)

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()

# Save the figure inside the project folder
heatmap_path = os.path.join(plots_folder, "correlation_heatmap.png")

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nCorrelation heatmap saved to: {heatmap_path}")

os.startfile(os.path.abspath(heatmap_path))

# Make sure plots folder exists
os.makedirs("plots", exist_ok=True)

# 1. Target Distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="cardio"
)

plt.title("Cardiovascular Disease Distribution")
plt.xlabel("Cardiovascular Disease")
plt.ylabel("Count")
plt.xticks([0, 1], ["No Disease", "Disease"])

plt.tight_layout()

target_path = os.path.join("plots", "target_distribution.png")

plt.savefig(
    target_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Target distribution saved to: {target_path}")

os.startfile(os.path.abspath(target_path))

# 2. Age Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age (Years)")
plt.ylabel("Count")

plt.tight_layout()

age_path = os.path.join("plots", "age_distribution.png")

plt.savefig(
    age_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Age distribution saved to: {age_path}")

os.startfile(os.path.abspath(age_path))

# 3. Cholesterol vs Cardiovascular Disease
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="cholesterol",
    hue="cardio"
)

plt.title("Cholesterol Level vs Cardiovascular Disease")
plt.xlabel("Cholesterol Level")
plt.ylabel("Count")

plt.tight_layout()

cholesterol_path = os.path.join(
    "plots",
    "cholesterol_vs_cardio.png"
)

plt.savefig(
    cholesterol_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Cholesterol plot saved to: {cholesterol_path}")

os.startfile(os.path.abspath(cholesterol_path))