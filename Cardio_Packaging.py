import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

# 1. Load Dataset
print("Loading Cardiovascular Disease dataset...")
df = pd.read_csv("cardio_train.csv", sep=";")

# 2. Basic Cleaning (Remove impossible values)
df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 220)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 140)]
df = df[(df['height'] >= 120) & (df['height'] <= 220)]
df = df[(df['weight'] >= 40) & (df['weight'] <= 180)]

# 3. Preprocessing
df["age"] = df["age"] / 365.25
X = df.drop(columns=["cardio", "id"])
y = df["cardio"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Train Final Model (Best Params from Tuning)
print("\nTraining the Final Tuned CatBoost...")
final_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=13,
    random_strength=0,
    random_seed=42,
    verbose=False
)
final_model.fit(X_train, y_train)

# 6. predict_risk Function
def predict_risk(patient_data):
    """
    Takes patient data (dict or DataFrame) and returns the risk score and category.
    """
    if isinstance(patient_data, dict):
        patient_df = pd.DataFrame([patient_data])
    else:
        patient_df = patient_data

    # Calculate probability
    risk_score = final_model.predict_proba(patient_df)[:, 1][0]
    
    # Determine risk category
    if risk_score < 0.33:
        risk_category = "Low Risk"
    elif risk_score < 0.66:
        risk_category = "Moderate Risk"
    else:
        risk_category = "High Risk"
        
    return {
        "Risk Score (%)": round(risk_score * 100, 2),
        "Category": risk_category
    }

# 7. Test the Function
print("\n" + "="*50)
print("Testing predict_risk function on a sample patient...")
print("="*50)

# Sample patient data (High Risk Profile: older, high BP, high cholesterol, smoker)
sample_patient = {
    'gender': 2,          # 2 = Male
    'height': 175.0,      # cm
    'weight': 90.0,       # kg
    'ap_hi': 150,         # Systolic BP
    'ap_lo': 95,          # Diastolic BP
    'cholesterol': 3,     # 3 = Well above normal
    'gluc': 1,            # 1 = Normal
    'smoke': 1,           # 1 = Yes
    'alco': 0,            # 0 = No
    'active': 0,          # 0 = Inactive
    'age': 55.0           # Age in years
}

result = predict_risk(sample_patient)
print("Patient Data (High Risk Profile):")
print(f"Prediction Result: {result}")
print("="*50)