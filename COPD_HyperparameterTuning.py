import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

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

# Define LightGBM Model
model = lgb.LGBMClassifier(
    random_state=42,
    verbosity=-1,
    n_jobs=2
)

# Define Hyperparameter Search Space
param_distributions = {
    "n_estimators": [150, 200, 300, 400, 500],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [20, 50, 100, 200],
    "reg_lambda": [0, 1, 5, 10],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "subsample": [0.7, 0.85, 1.0],
    "subsample_freq": [1],
}

# Randomized Search
print("\nStarting Hyperparameter Tuning...")

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="average_precision",
    cv=3,
    random_state=42,
    n_jobs=1,
    verbose=1
)

random_search.fit(X_train, y_train)

# Best Parameters
print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validation PR-AUC:")
print(random_search.best_score_)

# Evaluate Best Model on Test Set
best_model = random_search.best_estimator_

y_prob = best_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("\nTest Set Results:")
print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)
