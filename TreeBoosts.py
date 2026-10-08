import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# 1. Load Data
print("Loading CDC Diabetes dataset...")
cdc_data = fetch_ucirepo(id=891)
X = cdc_data.data.features
y = cdc_data.data.targets.values.ravel()

# Scale pos weight ratio
pos_ratio = (y == 0).sum() / (y == 1).sum()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Define Modern Boosters
booster_models = {

    ##"HistGradientBoosting": HistGradientBoostingClassifier(  ##lowest overall
    ##    max_iter=200,
    ##    max_depth=6,
    ##    learning_rate=0.08,
    ##    class_weight="balanced",
    ##    random_state=42
    ##),
    
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=pos_ratio,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    
    "CatBoost": CatBoostClassifier(
        iterations=200,
        depth=6,
        thread_count=-1,
        learning_rate=0.08,
        scale_pos_weight=pos_ratio,
        random_seed=42,
        verbose=50
    )
}

# 3. Benchmark
results = []
for name, model in booster_models.items():
    print(f"--> Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        "Model": name,
        "Accuracy": report["accuracy"],
        "Recall (Diabetic)": report["1.0"]["recall"] if "1.0" in report else report["1"]["recall"],
        "Precision (Diabetic)": report["1.0"]["precision"] if "1.0" in report else report["1"]["precision"],
        "F1 (Diabetic)": report["1.0"]["f1-score"] if "1.0" in report else report["1"]["f1-score"],
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": average_precision_score(y_test, y_proba)
    })

df_res = pd.DataFrame(results).set_index("Model")
print("\n" + "="*70)
print("             BOOSTING FAMILY BENCHMARK SUMMARY")
print("="*70)
print(df_res.round(4).to_string())
