import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import average_precision_score, roc_auc_score
from catboost import CatBoostClassifier

# Load Data
print("Loading Cardiovascular Disease dataset...")
df = pd.read_csv("cardio_train.csv", sep=";")
print("Data loaded successfully.")

# Convert Age
df["age"] = df["age"] / 365.25
print("Age converted from days to years.")

# Define Features and Target
X = df.drop(columns=["cardio", "id"])
y = df["cardio"]

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

# Define CatBoost Model
model = CatBoostClassifier(
    random_seed=42,
    verbose=False
)

# Define Hyperparameter Search Space
param_distributions = {
    "iterations": [100, 150, 200, 250, 300],
    "learning_rate": [0.05, 0.06, 0.07, 0.08, 0.09],
    "depth": [5, 6, 7],
    "l2_leaf_reg": [7, 9, 11, 13, 15],
    "random_strength": [0, 0.5, 1],
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
    n_jobs=-1,
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