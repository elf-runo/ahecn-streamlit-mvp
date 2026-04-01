# analytics_engine.py
import math

def mortality_risk(severity_index: float, eta_min: float, pathology: str = "General") -> float:
    sev = max(0.0, min(1.0, float(severity_index or 0.0)))
    eta = max(0.0, float(eta_min or 0.0))
    
    base_risk = sev * 50.0
    
    # Exponential "Golden Hour" penalty for time-critical pathologies
    time_critical = ["Trauma", "Stroke", "Cardiac", "Maternal"]
    
    if pathology in time_critical and eta > 30.0:
        # Risk spikes exponentially after 30 minutes
        delay_penalty = 15.0 * math.exp(0.02 * (eta - 30.0))
    else:
        # Linear degradation for standard/sepsis cases
        delay_penalty = (eta / 90.0) * 30.0
        
    total_risk = base_risk + min(50.0, delay_penalty)
    return round(min(99.9, total_risk), 1)
