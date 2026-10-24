import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# 1. Fetch Dataset
print("Loading CDC Diabetes dataset...")
cdc_data = fetch_ucirepo(id=891)
X = cdc_data.data.features
y = cdc_data.data.targets.values.ravel()

# Check class distribution
positive_count = np.sum(y == 1)
negative_count = np.sum(y == 0)
print(f"Total samples: {len(y)}")
print(f"Non-diabetic (0): {negative_count}, Diabetic/Pre-diabetic (1): {positive_count}")
print(f"Imbalance ratio (neg/pos): {negative_count / positive_count:.2f} "
      f"(handled via class_weight='balanced')\n")


def _positive_metrics(report: dict) -> tuple[float, float, float]:
    """Fetch precision/recall/F1 for the positive class regardless of label dtype.

    classification_report(output_dict=True) keys are str(labels), i.e. "1" for int
    labels but "1.0" for float labels.
    """
    key = "1.0" if "1.0" in report else "1"
    return report[key]["precision"], report[key]["recall"], report[key]["f1-score"]

# 2. Train-Test Split (80/20 with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Define the 3 Models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),
    
    "Gradient Boosting (GBDT)": HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=42
    )

}

# 4. Train, Predict, and Evaluate
results = []

for name, model in models.items():
    print(f"--> Training {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    # Extract report as dict
    report = classification_report(y_test, y_pred, output_dict=True)
    precision, recall, f1 = _positive_metrics(report)

    results.append({
        "Model": name,
        "Accuracy": report["accuracy"],
        "Diabetic Recall": recall,
        "Diabetic Precision": precision,
        "Diabetic F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

# 5. Summary Table
comparison_df = pd.DataFrame(results).set_index("Model")
print("\n" + "="*70)
print("                   MODEL PERFORMANCE SUMMARY")
print("="*70)
print(comparison_df.round(4).to_string())
