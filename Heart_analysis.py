import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score

# Load dataset
print("Loading Heart Disease dataset...")
df = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")
print("Data loaded successfully.")

# Separate features and target
X = df.drop("HeartDiseaseorAttack", axis=1)
y = df["HeartDiseaseorAttack"]

# First split: Training and Test
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

# Second split: Training and Validation
X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\nValidation Split:")
print(f"Sub-training samples: {len(X_subtrain)}")
print(f"Validation samples: {len(X_val)}")

# Calculate class weight using sub-training data
negative_count = (y_subtrain == 0).sum()
positive_count = (y_subtrain == 1).sum()
scale_pos_weight = negative_count / positive_count

print(f"\nScale Pos Weight: {scale_pos_weight:.2f}")

# Train CatBoost on sub-training data
print("\nTraining CatBoost on sub-training data...")

catboost_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=5,  
    random_seed=42,
    verbose=False,
    thread_count=-1,
    class_weights=[1, scale_pos_weight]
)

catboost_model.fit(X_subtrain, y_subtrain)

print("CatBoost trained successfully.")

# Predict probabilities on validation data
y_val_prob = catboost_model.predict_proba(X_val)[:, 1]

print("\nValidation probabilities calculated successfully.")

# Search for the best threshold
thresholds = np.arange(0.10, 0.91, 0.01)

best_threshold = None
best_f1 = -1

for threshold in thresholds:
    y_val_pred = (y_val_prob >= threshold).astype(int)

    recall = recall_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred)

    if recall >= 0.80 and f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest Threshold on Validation Set:")
print(f"Threshold: {best_threshold:.2f}")
print(f"F1 Score: {best_f1:.4f}")

# Evaluate the selected threshold on validation data
y_val_pred_best = (y_val_prob >= best_threshold).astype(int)

best_recall = recall_score(y_val, y_val_pred_best)
best_precision = precision_score(y_val, y_val_pred_best, zero_division=0)
best_f1 = f1_score(y_val, y_val_pred_best)

print("\nValidation Results at Best Threshold:")
print(f"Threshold: {best_threshold:.2f}")
print(f"Recall: {best_recall:.4f}")
print(f"Precision: {best_precision:.4f}")
print(f"F1 Score: {best_f1:.4f}")

# Calculate class weight using full training data
negative_count_full = (y_train == 0).sum()
positive_count_full = (y_train == 1).sum()
scale_pos_weight_full = negative_count_full / positive_count_full

print(f"\nFull Training Scale Pos Weight: {scale_pos_weight_full:.2f}")

# Train final CatBoost on the full training data
print("\nTraining Final CatBoost on full training data...")

final_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=5, 
    random_seed=42,
    verbose=False,
    thread_count=-1,
    class_weights=[1, scale_pos_weight_full]
)

final_model.fit(X_train, y_train)

print("Final CatBoost trained successfully.")

# Predict probabilities on the test set
y_test_prob = final_model.predict_proba(X_test)[:, 1]

# Apply the threshold selected using the validation set
y_test_pred = (y_test_prob >= best_threshold).astype(int)

# Calculate final metrics
test_recall = recall_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_prob)
test_pr_auc = average_precision_score(y_test, y_test_prob)

print("\nFinal Test Results:")
print(f"Threshold: {best_threshold:.2f}")
print(f"Recall: {test_recall:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"ROC-AUC: {test_roc_auc:.4f}")
print(f"PR-AUC: {test_pr_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# SHAP Analysis
print("\nStarting SHAP Analysis...")

# Use a sample of the test set for explanation
X_shap = X_test.sample(
    n=min(2000, len(X_test)),
    random_state=42
)

# Create SHAP Tree Explainer
explainer = shap.TreeExplainer(final_model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_shap)

print("SHAP analysis completed successfully.")

# Calculate mean absolute SHAP values
shap_importance = pd.DataFrame({
    "Feature": X_shap.columns,
    "Mean_ABS_SHAP": np.abs(shap_values).mean(axis=0)
})

# Sort features by importance
shap_importance = shap_importance.sort_values(
    by="Mean_ABS_SHAP",
    ascending=False
)

print("\nSHAP Feature Importance:")
print(shap_importance.to_string(index=False))