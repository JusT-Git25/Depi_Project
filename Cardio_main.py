import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load Dataset
print("Loading Cardiovascular Disease dataset...")

df = pd.read_csv("cardio_train.csv", sep=";")
print("Data loaded successfully.")

# Convert Age from Days to Years
df["age"] = df["age"] / 365.25
print("Age converted from days to years.")

# Prepare Features and Target
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

# Train Logistic Regression
print("\nTraining Logistic Regression...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

print("Model trained successfully.")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", roc_auc)

# PR-AUC
pr_auc = average_precision_score(y_test, y_prob)
print("PR-AUC:", pr_auc)