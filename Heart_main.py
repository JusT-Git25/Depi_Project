import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Load the dataset
print("Loading Heart Disease dataset...")

df = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")

print("Data loaded successfully.")

# Separate features and target
X = df.drop("HeartDiseaseorAttack", axis=1)
y = df["HeartDiseaseorAttack"]

# Split the data into training and testing sets
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

print("\nTarget Distribution:")
print(f"Training positive ratio: {y_train.mean():.4f}")
print(f"Test positive ratio: {y_test.mean():.4f}")

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

print("Logistic Regression trained successfully.")

# Make predictions
y_pred = logistic_model.predict(X_test_scaled)
y_prob = logistic_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate Logistic Regression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score
)

print("\nLogistic Regression Results:")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# Train Random Forest
print("\nTraining Random Forest...")

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(X_train, y_train)

print("Random Forest trained successfully.")

# Make predictions
y_pred_rf = random_forest_model.predict(X_test)
y_prob_rf = random_forest_model.predict_proba(X_test)[:, 1]

# Evaluate Random Forest
print("\nRandom Forest Results:")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
pr_auc_rf = average_precision_score(y_test, y_prob_rf)

print(f"ROC-AUC: {roc_auc_rf:.4f}")
print(f"PR-AUC: {pr_auc_rf:.4f}")

# Train HistGradientBoosting
print("\nTraining HistGradientBoosting...")

gradient_boosting_model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42
)

gradient_boosting_model.fit(X_train, y_train)

print("HistGradientBoosting trained successfully.")

# Make predictions
y_pred_gb = gradient_boosting_model.predict(X_test)
y_prob_gb = gradient_boosting_model.predict_proba(X_test)[:, 1]

# Evaluate HistGradientBoosting
print("\nHistGradientBoosting Results:")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb))

roc_auc_gb = roc_auc_score(y_test, y_prob_gb)
pr_auc_gb = average_precision_score(y_test, y_prob_gb)

print(f"ROC-AUC: {roc_auc_gb:.4f}")
print(f"PR-AUC: {pr_auc_gb:.4f}")