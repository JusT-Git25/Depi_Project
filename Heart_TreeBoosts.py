import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from catboost import CatBoostClassifier

# Load the dataset
print("Loading Heart Disease dataset...")

df = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")

print("Data loaded successfully.")

# Separate features and target
X = df.drop("HeartDiseaseorAttack", axis=1)
y = df["HeartDiseaseorAttack"]

# Split the data
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

# Calculate class weight for the positive class
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print(f"\nScale Pos Weight: {scale_pos_weight:.2f}")

# Train LightGBM
print("\nTraining LightGBM...")

lightgbm_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

lightgbm_model.fit(X_train, y_train)

print("LightGBM trained successfully.")

# Make predictions
y_pred_lgb = lightgbm_model.predict(X_test)
y_prob_lgb = lightgbm_model.predict_proba(X_test)[:, 1]

# Evaluate LightGBM
print("\nLightGBM Results:")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lgb))

roc_auc_lgb = roc_auc_score(y_test, y_prob_lgb)
pr_auc_lgb = average_precision_score(y_test, y_prob_lgb)

print(f"ROC-AUC: {roc_auc_lgb:.4f}")
print(f"PR-AUC: {pr_auc_lgb:.4f}")

# Train CatBoost
print("\nTraining CatBoost...")

catboost_model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.08,
    random_seed=42,
    verbose=False,
    thread_count=-1,
    class_weights=[1, scale_pos_weight]
)

catboost_model.fit(X_train, y_train)

print("CatBoost trained successfully.")

# Make predictions
y_pred_cb = catboost_model.predict(X_test)
y_prob_cb = catboost_model.predict_proba(X_test)[:, 1]

# Evaluate CatBoost
print("\nCatBoost Results:")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_cb))

roc_auc_cb = roc_auc_score(y_test, y_prob_cb)
pr_auc_cb = average_precision_score(y_test, y_prob_cb)

print(f"ROC-AUC: {roc_auc_cb:.4f}")
print(f"PR-AUC: {pr_auc_cb:.4f}")