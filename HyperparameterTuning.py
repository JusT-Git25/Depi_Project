import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load Data
print("Loading CDC Diabetes dataset...")
cdc_data = fetch_ucirepo(id=891)
X = cdc_data.data.features
y = cdc_data.data.targets.values.ravel()

# Handle class imbalance
pos_ratio = (y == 0).sum() / (y == 1).sum()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Data loaded successfully.")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Positive class ratio: {pos_ratio:.2f}")

# 2. Define LightGBM baseline model
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    scale_pos_weight=pos_ratio,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)


# 3. Define fine-tuning search space (round 3)
param_grid = {
    "n_estimators": [250, 300, 350, 400],
    "learning_rate": [0.04, 0.05, 0.06],
    "max_depth": [3, 4, 5],
    "num_leaves": [23, 31, 39],
    "min_child_samples": [20, 30, 40],
    "subsample": [0.75, 0.8, 0.85],
    "subsample_freq": [1, 5],
    "colsample_bytree": [0.9, 1.0],
    "reg_alpha": [0, 0.1, 0.2],
    "reg_lambda": [0, 0.1, 0.2]
}


# 4. Randomized Hyperparameter Search
random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid,
    n_iter=30,
    scoring="average_precision",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nStarting Hyperparameter Tuning...")
random_search.fit(X_train, y_train)
print("\nBest Parameters:")
print(random_search.best_params_)
print("\nBest CV PR-AUC:")
print(random_search.best_score_)

# 5. Evaluate the Best Model on the Test Set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
test_pr_auc = average_precision_score(y_test, y_proba)

print("\nTest PR-AUC:")
print(test_pr_auc)





# 6. Compare Baseline vs Tuned Model
report = classification_report(y_test, y_pred, output_dict=True)

tuned_results = {
    "Accuracy": report["accuracy"],
    "Recall (Diabetic)": report["1"]["recall"],
    "Precision (Diabetic)": report["1"]["precision"],
    "F1 (Diabetic)": report["1"]["f1-score"],
    "ROC-AUC": roc_auc_score(y_test, y_proba),
    "PR-AUC": test_pr_auc
}

baseline_results = {
    "Accuracy": 0.7209,
    "Recall (Diabetic)": 0.7933,
    "Precision (Diabetic)": 0.3064,
    "F1 (Diabetic)": 0.4420,
    "ROC-AUC": 0.8269,
    "PR-AUC": 0.4242
}

comparison_df = pd.DataFrame({
    "Baseline LightGBM": baseline_results,
    "Tuned LightGBM": tuned_results
})

comparison_df["Difference"] = (
    comparison_df["Tuned LightGBM"]
    - comparison_df["Baseline LightGBM"]
)


# 7. Final Results
print("\n" + "=" * 75)
print("             BASELINE VS TUNED LIGHTGBM")
print("=" * 75)

print(comparison_df.round(4).to_string())
