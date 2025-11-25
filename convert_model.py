import joblib
import sklearn
import numpy as np
from pathlib import Path

def convert_model():
    """Convert old scikit-learn model to be compatible with newer versions"""
    BASE_DIR = Path(__file__).parent
    OLD_MODEL_PATH = BASE_DIR / "models" / "triage_model_rf_v1 (1).pkl"
    NEW_MODEL_PATH = BASE_DIR / "models" / "triage_model_rf_v1_converted.pkl"
    
    try:
        print(f"Current scikit-learn version: {sklearn.__version__}")
        print("Loading old model...")
        
        # Try to load with compatible settings
        with open(OLD_MODEL_PATH, 'rb') as f:
            model = joblib.load(f)
            
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Save with current sklearn version
        joblib.dump(model, NEW_MODEL_PATH)
        print(f"Converted model saved to: {NEW_MODEL_PATH}")
        
        return True
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

if __name__ == "__main__":
    convert_model()
