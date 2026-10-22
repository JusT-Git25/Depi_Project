import pandas as pd

# Load the dataset
print("Loading Heart Disease dataset...")

df = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")

print("Data loaded successfully.")

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Target distribution
print("\nTarget Distribution:")
print(df["HeartDiseaseorAttack"].value_counts())

# Target percentage
print("\nTarget Percentage:")
print(df["HeartDiseaseorAttack"].value_counts(normalize=True) * 100)

# Calculate scale_pos_weight
negative_count = (df["HeartDiseaseorAttack"] == 0).sum()
positive_count = (df["HeartDiseaseorAttack"] == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nClass Weight:")
print(f"Negative samples: {negative_count}")
print(f"Positive samples: {positive_count}")
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Correlation with the target
print("\nCorrelation with HeartDiseaseorAttack:")

correlations = df.corr()["HeartDiseaseorAttack"].sort_values(ascending=False)

print(correlations)

# Top 10 features correlated with the target
top_features = correlations.drop("HeartDiseaseorAttack").abs().sort_values(ascending=False).head(10)

print("\nTop 10 Features by Absolute Correlation:")
print(top_features)