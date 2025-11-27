# create_basic_model.py - Creates a basic model without training
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

print("Creating basic compatible model...")

# Create a very simple model that will definitely work
model = RandomForestClassifier(
    n_estimators=10,
    max_depth=5,
    random_state=42
)

# Create minimal training data
X = np.array([[25, 20, 80, 120, 98, 37.0, 15, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]])
y = np.array([0])  # GREEN

# Fit the model
model.fit(X, y)

# Save it
model_path = "models/triage_model_basic.pkl"
joblib.dump(model, model_path)

# Basic feature info
feature_info = {
    'feature_names': [
        'age', 'rr', 'hr', 'sbp', 'spo2', 'temp_c', 'gcs', 'comorbid_count', 
        'on_oxygen', 'sex_M', 'avpu_ord', 'case_type_cardiac', 'case_type_maternal',
        'case_type_sepsis', 'case_type_stroke', 'case_type_trauma', 'case_type_other'
    ],
    'sklearn_version': '1.2.2',
    'note': 'Basic compatibility model'
}

joblib.dump(feature_info, "models/feature_info.pkl")

print(f"✅ Basic model created at: {model_path}")
print("This model will work with scikit-learn 1.2.2")
