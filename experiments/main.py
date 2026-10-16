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
scale_pos_weight = negative_count / positive_count
print(f"Total samples: {len(y)}")
print(f"Non-diabetic (0): {negative_count}, Diabetic/Pre-diabetic (1): {positive_count}")
print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}\n")

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
    
    results.append({
        "Model": name,
        "Accuracy": report["accuracy"],
        "Diabetic Recall": report["1.0"]["recall"] if "1.0" in report else report["1"]["recall"],
        "Diabetic Precision": report["1.0"]["precision"] if "1.0" in report else report["1"]["precision"],
        "Diabetic F1": report["1.0"]["f1-score"] if "1.0" in report else report["1"]["f1-score"],
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

# 5. Summary Table
comparison_df = pd.DataFrame(results).set_index("Model")
print("\n" + "="*70)
print("                   MODEL PERFORMANCE SUMMARY")
print("="*70)
print(comparison_df.round(4).to_string())
