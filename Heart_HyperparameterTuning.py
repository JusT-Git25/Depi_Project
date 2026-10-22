import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# 1. Load Dataset
print("Loading Heart Disease dataset...")
df = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")
print("Data loaded successfully.")

# 2. Separate Features and Target
X = df.drop("HeartDiseaseorAttack", axis=1)
y = df["HeartDiseaseorAttack"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData Split:")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# 4. Calculate Class Weight
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
scale_pos_weight = float(negative_count / positive_count)

print(f"\nScale Pos Weight: {scale_pos_weight:.2f}")

# 5. Define CatBoost Model
catboost_model = CatBoostClassifier(
    random_seed=42,
    verbose=False,
    thread_count=-1,
    auto_class_weights="Balanced"
)

# 6. Define Hyperparameter Search Space
param_grid = {
    "iterations": [200, 300, 500],
    "depth": [4, 5, 6, 7],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "l2_leaf_reg": [1, 3, 5, 7, 10]
}

# 7. Randomized Search
print("\nStarting Hyperparameter Tuning...")

random_search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=param_grid,
    n_iter=15,
    scoring="average_precision",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

# 8. Best Parameters
print("\nBest Parameters:")
print(random_search.best_params_)

print(f"\nBest Cross-Validation PR-AUC: "
      f"{random_search.best_score_:.4f}")

# 9. Evaluate Best Model on Test Set
best_model = random_search.best_estimator_

y_test_prob = best_model.predict_proba(X_test)[:, 1]

test_roc_auc = roc_auc_score(y_test, y_test_prob)
test_pr_auc = average_precision_score(y_test, y_test_prob)

print("\nTuned CatBoost Test Results:")
print(f"ROC-AUC: {test_roc_auc:.4f}")
print(f"PR-AUC: {test_pr_auc:.4f}")
# 10. Full Evaluation of Tuned Model
y_test_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\nTuned CatBoost Classification Report:")
print(classification_report(y_test, y_test_pred))

print("\nTuned CatBoost Full Evaluation:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"ROC-AUC: {test_roc_auc:.4f}")
print(f"PR-AUC: {test_pr_auc:.4f}")