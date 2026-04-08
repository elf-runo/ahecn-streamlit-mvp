# clinical_engine.py
from __future__ import annotations
import math
from typing import Dict, Any, Tuple, Optional, List

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sigmoid_severity(score: float, midpoint: float, steepness: float) -> float:
    """Applies a logistic curve to physiological scores to mimic biological shock."""
    return 1 / (1 + math.exp(-steepness * (score - midpoint)))

def calc_qsofa(rr: Optional[int], sbp: Optional[int], avpu: str = "A") -> Dict[str, Any]:
    score = 0
    if rr is not None and rr >= 22: score += 1
    if sbp is not None and sbp <= 100: score += 1
    if (avpu or "A").upper() != "A": score += 1
    return {"score": score, "high_risk": score >= 2}

def calc_news2(
    rr: Optional[int], spo2: Optional[int], sbp: Optional[int], 
    hr: Optional[int], temp_c: Optional[float], avpu: str = "A", 
    o2_device: str = "Air", spo2_scale: int = 1
) -> Dict[str, Any]:
    score = 0
    missing_count = 0
    
    # [AUDIT FIX]: Track missing vitals. A missing vital is a clinical risk.
    for vital in [rr, spo2, sbp, hr, temp_c]:
        if vital is None: missing_count += 1
        
    if rr is not None:
        if rr <= 8 or rr >= 25: score += 3
        elif rr >= 21: score += 2
        elif rr <= 11: score += 1
        
    if spo2 is not None:
        # [AUDIT FIX]: RCP NEWS2 SpO2 Scale 2 Implementation (for COPD / Hypercapnic patients)
        if spo2_scale == 2:
            if spo2 <= 83: score += 3
            elif 84 <= spo2 <= 85: score += 2
            elif 86 <= spo2 <= 87: score += 1
            elif 88 <= spo2 <= 92: score += 0
            elif spo2 >= 93 and (o2_device or "Air").lower() != "air":
                if 93 <= spo2 <= 94: score += 1
                elif 95 <= spo2 <= 96: score += 2
                elif spo2 >= 97: score += 3
        else:
            # Standard Scale 1
            if spo2 <= 91: score += 3
            elif 92 <= spo2 <= 93: score += 2
            elif 94 <= spo2 <= 95: score += 1
            
    if (o2_device or "Air").lower() != "air": score += 2
    
    if temp_c is not None:
        if temp_c <= 35.0: score += 3
        elif temp_c >= 39.1: score += 2
        elif temp_c <= 36.0 or temp_c >= 38.1: score += 1
        
    if sbp is not None:
        if sbp <= 90 or sbp >= 220: score += 3
        elif sbp <= 100: score += 2
        elif sbp <= 110: score += 1
        
    if hr is not None:
        if hr <= 40 or hr >= 131: score += 3
        elif hr >= 111: score += 2
        elif hr <= 50 or hr >= 91: score += 1
        
    if (avpu or "A").upper() != "A": score += 3

    # [AUDIT FIX]: Missing Data Escalation. If >=2 vitals are missing, force a clinical review.
    if missing_count >= 2: score += 2

    emergency = score >= 7 or (avpu or "A").upper() != "A"
    review = (score >= 5) or (score == 3) or (missing_count > 0)
    return {"score": int(score), "emergency": emergency, "review": review, "incomplete": missing_count > 0}

def calc_meows(hr: Optional[int], rr: Optional[int], sbp: Optional[int], temp_c: Optional[float], spo2: Optional[int]) -> Dict[str, Any]:
    red: List[str] = []
    yellow: List[str] = []
    if hr is not None:
        if hr >= 120 or hr <= 50: red.append("HR extreme")
        elif hr >= 110: yellow.append("HR abnormal")
    if sbp is not None:
        if sbp <= 90 or sbp >= 160: red.append("SBP extreme")
        elif sbp >= 150: yellow.append("SBP abnormal")
    if rr is not None:
        if rr >= 25 or rr < 12: red.append("RR extreme")
        elif rr >= 21: yellow.append("RR abnormal")
    if temp_c is not None:
        if temp_c >= 38.0 or temp_c <= 36.0: red.append("Temp extreme")
        elif temp_c >= 37.5: yellow.append("Temp abnormal")
    if spo2 is not None:
        if spo2 < 95: red.append("SpO2 critical")
    return {"red": red, "yellow": yellow}

def calc_pews(age_years: float, rr: Optional[int], hr: Optional[int], spo2: Optional[int], behavior: str = "Normal") -> Dict[str, Any]:
    score = 0
    if hr is not None:
        if age_years < 1 and (hr < 100 or hr > 180): score += 3
        elif 1 <= age_years < 5 and (hr < 80 or hr > 160): score += 3
        elif age_years >= 5 and (hr < 60 or hr > 140): score += 3
    if rr is not None:
        if age_years < 1 and (rr < 20 or rr > 60): score += 3
        elif 1 <= age_years < 5 and (rr < 15 or rr > 50): score += 3
        elif age_years >= 5 and (rr < 12 or rr > 40): score += 3
    if spo2 is not None:
        if spo2 < 90: score += 3
        elif spo2 < 94: score += 1
    b = (behavior or "Normal").lower()
    if b in ["lethargic", "unresponsive"]: score += 3
    elif b == "irritable": score += 1
    urgent = score >= 5 or b in ["lethargic", "unresponsive"]
    return {"score": int(score), "urgent": urgent}

COMPLETE_AUTO_RED_CODES = {
    'I21.9', 'I46.9', 'I71.0', 'I26.9', 'I40.9', 'R57.0', 'I71.3', 
    'I63.9', 'I61.9', 'I60.9', 'G41.9', 'G82.2', 'I62.9', 'G93.6', 'G06.0', 
    'J93.9', 'J96.0', 'J80', 'R04.2', 'T78.2', 'J05.1', 
    'S06.5', 'S36.1', 'S27.0', 'T07', 'T31.5', 'S14.1', 'T79.4', 'S26.9', 'T71', 'S36.0', 'T79.0', 'S32.8', 
    'O72.0', 'O15.0', 'O44.1', 'O71.1', 'O00.9', 'O68', 'O88.1', 'O45.9', 'O81.9', 'O14.2', 'O73.0', 
    'A41.9', 'R57.2', 'A01.0', 'A20.0', 'B50.0', 'A32.1', 'M72.6', 
    'K25.4', 'K56.6', 'K40.3', 'K65.0', 'K92.2', 'K55.0', 'I85.0', 'K57.2', 'K59.3', 
    'N44.0', 'N49.8', 
    'P22.0', 'P29.0', 'P21.9', 'P77', 'Q20.3', 'P07.3' 
}

CRITICAL_INTERVENTION_KEYWORDS = [
    "Defibrillation", "Thrombolysis", "Cardioversion", "Massive transfusion", 
    "Intubation", "Laparotomy", "Emergency C-Section", "Cath Lab Activation",
    "Hemodialysis", "Surgical debridement", "Neuro surgery", "Chest tube"
]

def validated_triage_decision(vitals: Dict[str, Any], icd_row: Dict[str, Any], context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    icd10 = (icd_row.get("icd10") or icd_row.get("code") or "").strip()
    label = icd_row.get("label", "Unknown diagnosis")
    bundle = icd_row.get("bundle", icd_row.get("case_type", ""))

    default_interventions = str(icd_row.get("default_interventions", ""))
    is_time_critical = any(k.lower() in default_interventions.lower() for k in [kw.lower() for kw in CRITICAL_INTERVENTION_KEYWORDS])

    if icd10 in COMPLETE_AUTO_RED_CODES or is_time_critical:
        meta = {
            "primary_driver": "Pathological Risk",
            "reason": f"Absolute clinical emergency: {label}",
            "ews_type": "Bypassed",
            "ews_score": 0,
            "severity_index": 1.0,
            "score_details": {"auto_red": True, "icd10": icd10, "bundle": bundle, "severity_index": 1.0},
        }
        return "RED", meta

    age = float(context.get("age", 30) or 30)
    pregnant = bool(context.get("pregnant", False) or (bundle == "Maternal"))

    rr, hr, sbp = vitals.get("rr"), vitals.get("hr"), vitals.get("sbp")
    spo2, temp, avpu = vitals.get("spo2"), vitals.get("temp"), vitals.get("avpu", "A")
    o2_device = context.get("o2_device", "Air")
    spo2_scale = int(context.get("spo2_scale", 1) or 1)
    behavior = context.get("behavior", "Normal")

    is_red, is_yellow = False, False

    if age < 18:
        pews = calc_pews(age, rr, hr, spo2, behavior=behavior)
        ews_type = "PEWS"
        score = pews["score"]
        severity_index = clamp(sigmoid_severity(score, midpoint=4.0, steepness=1.0), 0.0, 1.0)
        score_details = {"PEWS": pews}
        if score >= 5 or pews.get("urgent", False): is_red = True
        elif score >= 3: is_yellow = True

    elif pregnant:
        meows = calc_meows(hr, rr, sbp, temp, spo2)
        ews_type = "MEOWS"
        score = len(meows["red"]) * 2 + len(meows["yellow"]) 
        if len(meows["red"]) >= 1: 
            is_red = True
            severity_index = clamp(0.7 + (len(meows["red"]) * 0.1), 0.0, 1.0)
        elif len(meows["yellow"]) >= 1: 
            is_yellow = True
            severity_index = clamp(0.4 + (len(meows["yellow"]) * 0.1), 0.0, 0.6)
        else: severity_index = 0.05
        score_details = {"MEOWS": meows}

    else:
        news = calc_news2(rr, spo2, sbp, hr, temp, avpu=avpu, o2_device=o2_device, spo2_scale=spo2_scale)
        ews_type = "NEWS2"
        score = news["score"]
        severity_index = clamp(sigmoid_severity(score, midpoint=5.0, steepness=0.8), 0.0, 1.0)
        score_details = {"NEWS2": news}
        if score >= 7 or news.get("emergency", False): is_red = True
        elif score >= 5 or news.get("review", False): is_yellow = True

    qsofa = calc_qsofa(rr, sbp, avpu=avpu)
    score_details["qSOFA"] = qsofa

    if is_red:
        color = "RED"
        reason = f"Critical physiological instability ({ews_type} threshold breached)"
    elif is_yellow:
        color = "YELLOW"
        reason = f"Elevated clinical risk detected ({ews_type} trigger)"
    else:
        if bundle in ['Maternal', 'Pediatric', 'GI', 'Renal', 'Toxicology']:
            color = "YELLOW"
            reason = f"Stable vitals, but elevated baseline pathological risk ({bundle})"
            severity_index = max(0.4, severity_index)
        else:
            color = "GREEN"
            reason = f"Stable physiology ({ews_type} score {score})"
            
    # Add flag to UI if vitals were missing
    if score_details.get(ews_type, {}).get("incomplete", False):
        reason += " [WARNING: Incomplete Vitals]"

    meta = {
        "primary_driver": "Physiological Instability" if is_red or is_yellow else "Baseline Assessment",
        "reason": reason,
        "ews_type": ews_type,
        "ews_score": int(score),
        "severity_index": float(severity_index),
        "score_details": {**score_details, "severity_index": float(severity_index)},
    }
    return color, meta
