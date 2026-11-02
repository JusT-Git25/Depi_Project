import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# 1. Load Data
print("Loading COPD dataset (CDC BRFSS)...")
df = pd.read_csv("COPD_dataset.csv")

# 2. Basic Cleaning
print("\nCleaning data...")
df = df[df["CHCCOPD3"].isin([1.0, 2.0])].copy()
df["COPD"] = (df["CHCCOPD3"] == 1.0).astype(int)
df["BMI"] = df["_BMI5"] / 100
df.loc[(df["BMI"] < 10) | (df["BMI"] > 80), "BMI"] = np.nan
for column in ["SMOKE100", "SMOKDAY2", "EXERANY2", "GENHLTH"]:
    df[column] = df[column].replace({7.0: np.nan, 9.0: np.nan})
df.loc[df["SMOKE100"] == 2.0, "SMOKDAY2"] = 0.0
print(f"Data shape after cleaning: {df.shape}")

# 3. Features and Target
X = df[["_AGE80", "SEXVAR", "BMI", "SMOKE100", "SMOKDAY2", "EXERANY2", "GENHLTH"]]
y = df["COPD"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


def evaluate(title, model, X_tr, X_te):
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    key = "1.0" if "1.0" in report else "1"
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Recall:    {report[key]['recall']:.4f}")
    print(f"Precision: {report[key]['precision']:.4f}")
    print(f"F1:        {report[key]['f1-score']:.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_test, y_proba):.4f}")
    return model


def new_model():
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=-1,
        random_state=42, verbose=-1
    )


# ==========================================================
# 5. Baseline (No Feature Engineering)
# ==========================================================
evaluate("BASELINE (No Feature Engineering)", new_model(), X_train, X_test)


# ==========================================================
# 6. Feature Engineering
# ==========================================================
def apply_fe(df):
    df = df.copy()

    # SMOKE100: 1 = yes, 2 = no.  SMOKDAY2: 0 = never smoked, 1 = every day,
    # 2 = some days, 3 = not at all (former smoker)

    # 1. Smoking status (0 never, 1 former, 2 current)
    df["Smoking_Status"] = np.select(
        [df["SMOKDAY2"] == 0, df["SMOKDAY2"] == 3, df["SMOKDAY2"].isin([1, 2])],
        [0, 1, 2], default=np.nan
    )

    # 2. Current daily smoker
    df["Daily_Smoker"] = (df["SMOKDAY2"] == 1).astype(int)

    # 3. Age x Smoking interaction (COPD builds up over years of exposure)
    df["Age_x_Smoking"] = df["_AGE80"] * df["Smoking_Status"]

    # 4. Poor general health flag (4 = fair, 5 = poor)
    df["Poor_Health"] = (df["GENHLTH"] >= 4).astype(float).where(df["GENHLTH"].notna())

    # 5. Underweight flag (low BMI is a known COPD marker)
    df["Underweight"] = (df["BMI"] < 18.5).astype(float).where(df["BMI"].notna())

    # 6. Age x General Health interaction
    df["Age_x_GenHlth"] = df["_AGE80"] * df["GENHLTH"]

    # 7. Risk score (current/former smoker + inactive + poor health)
    df["Risk_Count"] = (
        (df["Smoking_Status"] > 0).astype(int)
        + (df["EXERANY2"] == 2).astype(int)
        + (df["GENHLTH"] >= 4).astype(int)
    )

    return df


X_train_fe = apply_fe(X_train)
X_test_fe = apply_fe(X_test)

# ==========================================================
# 7. LightGBM with Feature Engineering
# ==========================================================
fe_model = evaluate("LIGHTGBM WITH FEATURE ENGINEERING", new_model(), X_train_fe, X_test_fe)

# 8. Feature Importance
print("\n" + "=" * 60)
print("TOP 10 FEATURES (with FE)")
print("=" * 60)
importance = pd.DataFrame({
    "Feature": X_train_fe.columns,
    "Importance": fe_model.feature_importances_
}).sort_values("Importance", ascending=False).head(10)
print(importance.to_string(index=False))
