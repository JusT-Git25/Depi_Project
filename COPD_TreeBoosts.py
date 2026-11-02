import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split

# Load Data
print("Loading COPD dataset (CDC BRFSS)...")
df = pd.read_csv("COPD_dataset.csv")
print("Data loaded successfully.")

# Basic Cleaning
df = df[df["CHCCOPD3"].isin([1.0, 2.0])].copy()
df["COPD"] = (df["CHCCOPD3"] == 1.0).astype(int)
df["BMI"] = df["_BMI5"] / 100
df.loc[(df["BMI"] < 10) | (df["BMI"] > 80), "BMI"] = np.nan
for column in ["SMOKE100", "SMOKDAY2", "EXERANY2", "GENHLTH"]:
    df[column] = df[column].replace({7.0: np.nan, 9.0: np.nan})
df.loc[df["SMOKE100"] == 2.0, "SMOKDAY2"] = 0.0
print("Data cleaned. Rows:", len(df))

# Define Features and Target
X = df[["_AGE80", "SEXVAR", "BMI", "SMOKE100", "SMOKDAY2", "EXERANY2", "GENHLTH"]]
y = df["COPD"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData Split:")
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

# Train LightGBM (handles missing values natively)
print("\nTraining LightGBM...")

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    verbosity=-1
)

model.fit(X_train, y_train)

print("Model trained successfully.")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", roc_auc)

# PR-AUC
pr_auc = average_precision_score(y_test, y_prob)
print("PR-AUC:", pr_auc)
