# train_simple_model.py - ULTRA-COMPATIBLE VERSION
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import sklearn

print("=== Training Ultra-Compatible AI Triage Model ===")
print(f"scikit-learn version: {sklearn.__version__}")

# Create simple training data (no advanced features)
np.random.seed(42)
n_samples = 2000

data = []
for i in range(n_samples):
    # Basic features only
    age = np.random.randint(18, 85)
    rr = np.random.randint(12, 30)
    hr = np.random.randint(60, 140)
    sbp = np.random.randint(90, 180)
    spo2 = np.random.randint(88, 100)
    temp_c = np.random.uniform(36.0, 39.0)
    gcs = np.random.randint(3, 16)
    comorbid_count = np.random.randint(0, 3)
    on_oxygen = np.random.choice([0, 1])
    sex_M = np.random.choice([0, 1])
    avpu_ord = np.random.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05])
    
    # Case types
    case_types = ['cardiac', 'maternal', 'sepsis', 'stroke', 'trauma', 'other']
    case_type = np.random.choice(case_types)
    
    # Simple triage logic based on vital thresholds
    if (sbp < 90 or hr > 130 or rr > 25 or spo2 < 90 or gcs < 9):
        triage = 3  # RED
    elif (sbp < 100 or hr > 120 or rr > 22 or spo2 < 94 or gcs < 13):
        triage = 2  # ORANGE  
    elif (sbp < 110 or hr > 110 or rr > 20 or spo2 < 96):
        triage = 1  # YELLOW
    else:
        triage = 0  # GREEN
    
    row = {
        'age': age, 'rr': rr, 'hr': hr, 'sbp': sbp, 'spo2': spo2, 
        'temp_c': temp_c, 'gcs': gcs, 'comorbid_count': comorbid_count,
        'on_oxygen': on_oxygen, 'sex_M': sex_M, 'avpu_ord': avpu_ord,
        'case_type_cardiac': 1 if case_type == 'cardiac' else 0,
        'case_type_maternal': 1 if case_type == 'maternal' else 0,
        'case_type_sepsis': 1 if case_type == 'sepsis' else 0,
        'case_type_stroke': 1 if case_type == 'stroke' else 0,
        'case_type_trauma': 1 if case_type == 'trauma' else 0,
        'case_type_other': 1 if case_type == 'other' else 0,
        'triage': triage
    }
    data.append(row)

df = pd.DataFrame(data)
X = df.drop('triage', axis=1)
y = df['triage']

print(f"Training set: {X.shape}")
print("Triage distribution:")
for i, label in enumerate(['GREEN', 'YELLOW', 'ORANGE', 'RED']):
    count = (y == i).sum()
    print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use simple RandomForest without advanced parameters
model = RandomForestClassifier(
    n_estimators=50,  # Smaller for compatibility
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")

# Save model
model_path = "models/triage_model_simple.pkl"
joblib.dump(model, model_path)

# Save feature info
feature_info = {
    'feature_names': X.columns.tolist(),
    'sklearn_version': sklearn.__version__,
    'training_data_info': 'Simple compatible training data',
    'clinical_basis': 'Basic vital sign thresholds'
}

joblib.dump(feature_info, "models/feature_info_simple.pkl")

print(f"✅ Simple compatible model saved to: {model_path}")
