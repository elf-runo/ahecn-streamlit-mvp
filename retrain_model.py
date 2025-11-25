# retrain_model.py - MEDICALLY REALISTIC VERSION
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import sklearn

def create_medically_realistic_training_data():
    """Create medically accurate synthetic training data based on clinical patterns"""
    np.random.seed(42)
    n_samples = 5000
    
    # Initialize arrays for each feature
    data = {
        'age': [],
        'rr': [],      # Respiratory Rate
        'hr': [],      # Heart Rate  
        'sbp': [],     # Systolic BP
        'spo2': [],    # Oxygen Saturation
        'temp_c': [],  # Temperature
        'gcs': [],     # Glasgow Coma Scale
        'comorbid_count': [],
        'on_oxygen': [],
        'sex_M': [],
        'avpu_ord': [],  # A=0, V=1, P=2, U=3
    }
    
    # Case type distributions based on real emergency department patterns
    case_types = ['cardiac', 'maternal', 'sepsis', 'stroke', 'trauma', 'other']
    
    for i in range(n_samples):
        # Assign case type with realistic distribution
        case_weights = [0.15, 0.12, 0.18, 0.15, 0.20, 0.20]  # Based on ED prevalence
        case_type = np.random.choice(case_types, p=case_weights)
        
        # Age distribution based on case type
        if case_type == 'maternal':
            age = np.random.randint(18, 45)
        elif case_type == 'trauma':
            age = np.random.randint(15, 70)
        else:
            age = np.random.randint(30, 85)
        
        # Generate medically realistic vitals based on case type and severity
        severity = np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05])  # GREEN, YELLOW, ORANGE, RED
        
        if severity == 0:  # GREEN - Normal ranges
            rr = np.random.randint(12, 20)
            hr = np.random.randint(60, 100)
            sbp = np.random.randint(110, 140)
            spo2 = np.random.randint(96, 100)
            temp_c = np.random.uniform(36.5, 37.5)
            gcs = 15
            avpu = 0  # Alert
            
        elif severity == 1:  # YELLOW - Mild abnormalities
            rr = np.random.randint(20, 24)
            hr = np.random.randint(100, 120)
            sbp = np.random.randint(100, 110)
            spo2 = np.random.randint(94, 96)
            temp_c = np.random.uniform(37.6, 38.5)
            gcs = np.random.choice([14, 15])
            avpu = np.random.choice([0, 1], p=[0.8, 0.2])  # Mostly alert, some verbal
            
        elif severity == 2:  # ORANGE - Moderate abnormalities
            rr = np.random.randint(24, 30)
            hr = np.random.randint(120, 140)
            sbp = np.random.randint(90, 100)
            spo2 = np.random.randint(90, 94)
            temp_c = np.random.uniform(38.6, 39.5)
            gcs = np.random.choice([9, 10, 11, 12, 13])
            avpu = np.random.choice([1, 2], p=[0.6, 0.4])  # Verbal or pain
            
        else:  # RED - Critical abnormalities
            rr = np.random.choice([8, 9, 10, 30, 35, 40])  # Bradypnea or tachypnea
            hr = np.random.choice([40, 45, 50, 140, 150, 160])  # Bradycardia or tachycardia
            sbp = np.random.randint(70, 90)
            spo2 = np.random.randint(85, 90)
            temp_c = np.random.choice([35.0, 35.5, 39.5, 40.0])  # Hypo/hyperthermia
            gcs = np.random.randint(3, 9)
            avpu = np.random.choice([2, 3], p=[0.3, 0.7])  # Pain or unresponsive
        
        # Case-specific adjustments based on medical knowledge
        if case_type == 'cardiac':
            hr = min(hr + np.random.randint(10, 20), 200)  # Tachycardia more common
            if severity > 1:
                sbp = max(sbp - np.random.randint(10, 20), 60)  # Hypotension in severe cardiac
                
        elif case_type == 'sepsis':
            hr = min(hr + np.random.randint(15, 25), 180)  # Tachycardia characteristic
            if severity > 0:
                sbp = max(sbp - np.random.randint(5, 15), 70)  # Hypotension common
            temp_c = temp_c + np.random.uniform(0.5, 1.5)  # Fever more pronounced
            
        elif case_type == 'stroke':
            if severity > 1:
                gcs = max(gcs - np.random.randint(2, 5), 3)  # Reduced consciousness
                avpu = min(avpu + 1, 3)  # Worse AVPU
                
        elif case_type == 'trauma':
            if severity > 1:
                hr = min(hr + np.random.randint(20, 30), 160)  Tachycardia from blood loss
                sbp = max(sbp - np.random.randint(15, 25), 60)  # Hypotension from hemorrhage
        
        # Add small random variations while keeping medically realistic
        rr = max(8, min(rr + np.random.randint(-2, 3), 40))
        hr = max(40, min(hr + np.random.randint(-5, 6), 200))
        sbp = max(60, min(sbp + np.random.randint(-5, 6), 200))
        spo2 = max(85, min(spo2 + np.random.randint(-1, 2), 100))
        temp_c = max(35.0, min(temp_c + np.random.uniform(-0.2, 0.2), 41.0))
        gcs = max(3, min(gcs, 15))
        
        # Store the data
        data['age'].append(age)
        data['rr'].append(rr)
        data['hr'].append(hr)
        data['sbp'].append(sbp)
        data['spo2'].append(spo2)
        data['temp_c'].append(round(temp_c, 1))
        data['gcs'].append(gcs)
        data['comorbid_count'].append(np.random.randint(0, 4))
        data['on_oxygen'].append(1 if spo2 < 94 else np.random.choice([0, 1], p=[0.8, 0.2]))
        data['sex_M'].append(np.random.choice([0, 1]))
        data['avpu_ord'].append(avpu)
        
        # Add case type one-hot encoding
        for ct in case_types:
            key = f'case_type_{ct}'
            if key not in data:
                data[key] = []
            data[key].append(1 if ct == case_type else 0)
    
    return pd.DataFrame(data)

def create_clinically_accurate_labels(df):
    """Create triage labels based on established clinical scoring systems"""
    labels = []
    
    for _, row in df.iterrows():
        # Calculate NEWS2 score (National Early Warning Score)
        news2_score = 0
        
        # Respiratory Rate scoring
        if row['rr'] <= 8: news2_score += 3
        elif 9 <= row['rr'] <= 11: news2_score += 1
        elif 21 <= row['rr'] <= 24: news2_score += 2
        elif row['rr'] >= 25: news2_score += 3
        
        # Oxygen Saturation scoring
        if row['spo2'] <= 91: news2_score += 3
        elif 92 <= row['spo2'] <= 93: news2_score += 2
        elif 94 <= row['spo2'] <= 95: news2_score += 1
        if row['on_oxygen']: news2_score += 2
        
        # Systolic BP scoring
        if row['sbp'] <= 90: news2_score += 3
        elif 91 <= row['sbp'] <= 100: news2_score += 2
        elif 101 <= row['sbp'] <= 110: news2_score += 1
        elif row['sbp'] >= 220: news2_score += 3
        
        # Heart Rate scoring
        if row['hr'] <= 40: news2_score += 3
        elif 41 <= row['hr'] <= 50: news2_score += 1
        elif 111 <= row['hr'] <= 130: news2_score += 1
        elif row['hr'] >= 131: news2_score += 3
        
        # Temperature scoring
        if row['temp_c'] <= 35.0: news2_score += 3
        elif 35.1 <= row['temp_c'] <= 36.0: news2_score += 1
        elif 38.1 <= row['temp_c'] <= 39.0: news2_score += 1
        elif row['temp_c'] >= 39.1: news2_score += 2
        
        # AVPU scoring (GCS proxy)
        if row['avpu_ord'] >= 2:  # Pain or Unresponsive
            news2_score += 3
        
        # Additional clinical factors
        mews_score = 0
        if row['sbp'] < 90: mews_score += 3
        elif row['sbp'] < 100: mews_score += 2
        elif row['sbp'] > 160: mews_score += 2
        
        if row['hr'] < 40: mews_score += 2
        elif row['hr'] > 130: mews_score += 3
        elif row['hr'] > 110: mews_score += 2
        
        if row['rr'] < 10: mews_score += 2
        elif row['rr'] > 30: mews_score += 2
        
        if row['temp_c'] < 35.0: mews_score += 2
        elif row['temp_c'] > 38.5: mews_score += 2
        
        # Combined clinical judgment
        combined_score = news2_score + (mews_score * 0.5)
        
        # Age adjustment
        if row['age'] > 65: combined_score += 1
        if row['comorbid_count'] >= 2: combined_score += 1
        
        # Assign triage category based on clinical thresholds
        if combined_score >= 7 or row['gcs'] < 9:
            labels.append(3)  # RED - Critical
        elif combined_score >= 5 or row['gcs'] < 13:
            labels.append(2)  # ORANGE - Urgent
        elif combined_score >= 3:
            labels.append(1)  # YELLOW - Semi-urgent
        else:
            labels.append(0)  # GREEN - Non-urgent
            
    return np.array(labels)

def train_model():
    print("Creating medically realistic training data...")
    X = create_medically_realistic_training_data()
    y = create_clinically_accurate_labels(X)
    
    print(f"Training set: {X.shape[0]} samples, {X.shape[1]} features")
    print("Triage distribution:")
    triage_labels = ['GREEN', 'YELLOW', 'ORANGE', 'RED']
    for i, label in enumerate(triage_labels):
        count = np.sum(y == i)
        print(f"  {label}: {count} cases ({count/len(y)*100:.1f}%)")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest with medical context
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'  # Important for imbalanced medical data
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Feature importance for medical validation
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    return model, X.columns.tolist()

# Main execution
if __name__ == "__main__":
    print("=== Training Medically Accurate AI Triage Model ===")
    
    model, feature_names = train_model()
    
    # Save model
    model_path = "models/triage_model_medically_accurate.pkl"
    joblib.dump(model, model_path)
    
    # Save feature info
    feature_info = {
        'feature_names': feature_names,
        'sklearn_version': sklearn.__version__,
        'training_data_info': 'Medically realistic synthetic data based on clinical patterns',
        'clinical_basis': 'NEWS2 + MEWS scoring with clinical adjustments'
    }
    
    joblib.dump(feature_info, "models/feature_info.pkl")
    
    print(f"✅ Medically accurate model saved to: {model_path}")
    print(f"✅ Clinical basis: NEWS2 + MEWS scoring with realistic value ranges")
