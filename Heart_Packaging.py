import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

# 1. Load Dataset
print("Loading Heart Disease dataset...")
df = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")

# 2. Separate Features and Target
X = df.drop("HeartDiseaseorAttack", axis=1)
y = df["HeartDiseaseorAttack"]

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Train the Final Tuned Model (In-Memory, without saving to disk)
print("\nTraining the Final Tuned Model with Best Hyperparameters...")
final_model = CatBoostClassifier(
    learning_rate=0.03,   # Updated from tuning
    l2_leaf_reg=5,        # Updated from tuning
    iterations=500,       # Updated from tuning
    depth=6,              # Updated from tuning
    random_seed=42,
    verbose=False,
    thread_count=-1,
    auto_class_weights="Balanced" 
)
final_model.fit(X_train, y_train)

# 5. Define the predict_risk Function
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

# 6. Test the Function
print("\n" + "="*50)
print("Testing predict_risk function on a sample patient...")
print("="*50)

# Sample patient data (High Risk Profile)
sample_patient = {
    'HighBP': 1.0, 'HighChol': 1.0, 'CholCheck': 1.0, 'BMI': 30.0, 
    'Smoker': 1.0, 'Stroke': 0.0, 'Diabetes': 1.0, 'PhysActivity': 0.0, 
    'Fruits': 0.0, 'Veggies': 1.0, 'HvyAlcoholConsump': 0.0, 
    'AnyHealthcare': 1.0, 'NoDocbcCost': 0.0, 'GenHlth': 4.0, 
    'MentHlth': 10.0, 'PhysHlth': 15.0, 'DiffWalk': 1.0, 
    'Sex': 1.0, 'Age': 10.0, 'Education': 4.0, 'Income': 5.0
}

result = predict_risk(sample_patient)
print(f"Patient Data (High Risk Profile):")
print(f"Prediction Result: {result}")
print("="*50)