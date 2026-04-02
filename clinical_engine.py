# clinical_engine.py
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

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
    if rr is not None:
        if rr <= 8 or rr >= 25: score += 3
        elif rr >= 21: score += 2
        elif rr <= 11: score += 1
    if spo2 is not None:
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

    emergency = score >= 7 or (avpu or "A").upper() != "A"
    review = (score >= 5) or (score == 3)
    return {"score": int(score), "emergency": emergency, "review": review}

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
    # Medically accurate age-stratified thresholds
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

# Auto-RED ICD set aligned with the new 150-row CSV
COMPLETE_AUTO_RED_CODES = {
    'I21.9', 'I46.9', 'I71.0', 'I26.9', 'I40.9', 'R57.0', 'I71.3', # Cardiac Reds
    'I63.9', 'I61.9', 'I60.9', 'G41.9', 'G82.2', 'I62.9', 'G93.6', 'G06.0', # Neuro Reds
    'J93.9', 'J96.0', 'J80', 'R04.2', 'T78.2', 'J05.1', # Resp Reds
    'S06.5', 'S36.1', 'S27.0', 'T07', 'T31.5', 'S14.1', 'T79.4', 'S26.9', 'T71', 'S36.0', 'T79.0', 'S32.8', # Trauma Reds
    'O72.0', 'O15.0', 'O44.1', 'O71.1', 'O00.9', 'O68', 'O88.1', 'O45.9', 'O81.9', 'O14.2', 'O73.0', # Maternal Reds
    'A41.9', 'R57.2', 'A01.0', 'A20.0', 'B50.0', 'A32.1', 'M72.6', # Sepsis Reds
    'K25.4', 'K56.6', 'K40.3', 'K65.0', 'K92.2', 'K55.0', 'I85.0', 'K57.2', 'K59.3', # GI Reds
    'N44.0', 'N49.8', # Renal/Uro Reds
    'P22.0', 'P29.0', 'P21.9', 'P77', 'Q20.3', 'P07.3' # Neonatal Reds
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

    default_interventions = icd_row.get("default_interventions", "")
    if isinstance(default_interventions, list):
        default_interventions_text = " ".join(default_interventions)
    else:
        default_interventions_text = str(default_interventions or "")

    is_time_critical = any(k.lower() in default_interventions_text.lower() for k in [kw.lower() for kw in CRITICAL_INTERVENTION_KEYWORDS])

    # --- VECTOR 1: PATHOLOGY OVERRIDE (Absolute Emergencies) ---
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

    # --- VECTOR 2: PHYSIOLOGY SAFETY NET (Catches non-critical pathologies with bad vitals) ---
    age = float(context.get("age", 30) or 30)
    pregnant = bool(context.get("pregnant", False) or (bundle == "Maternal"))

    rr, hr, sbp = vitals.get("rr"), vitals.get("hr"), vitals.get("sbp")
    spo2, temp, avpu = vitals.get("spo2"), vitals.get("temp"), vitals.get("avpu", "A")
    o2_device = context.get("o2_device", "Air")
    spo2_scale = int(context.get("spo2_scale", 1) or 1)
    behavior = context.get("behavior", "Normal")

    is_red = False
    is_yellow = False

    if age < 18:
        pews = calc_pews(age, rr, hr, spo2, behavior=behavior)
        ews_type = "PEWS"
        score = pews["score"]
        severity_index = clamp(score / 10.0, 0.0, 1.0)
        score_details = {"PEWS": pews}
        
        # PERFECTED PEWS THRESHOLDS (Unconditional Override)
        if score >= 5 or pews.get("urgent", False): is_red = True
        elif score >= 3: is_yellow = True

    elif pregnant:
        meows = calc_meows(hr, rr, sbp, temp, spo2)
        ews_type = "MEOWS"
        score = len(meows["red"]) * 2 + len(meows["yellow"]) 
        
        # PERFECTED MEOWS THRESHOLDS (Any single Red parameter = RED Triage)
        if len(meows["red"]) >= 1: 
            is_red = True
            severity_index = clamp(0.6 + (len(meows["red"]) * 0.15), 0.0, 1.0)
        elif len(meows["yellow"]) >= 1: 
            is_yellow = True
            severity_index = clamp(len(meows["yellow"]) * 0.1, 0.0, 0.5)
        else:
            severity_index = 0.0
            
        score_details = {"MEOWS": meows}

    else:
        news = calc_news2(rr, spo2, sbp, hr, temp, avpu=avpu, o2_device=o2_device, spo2_scale=spo2_scale)
        ews_type = "NEWS2"
        score = news["score"]
        severity_index = clamp(score / 12.0, 0.0, 1.0)
        score_details = {"NEWS2": news}
        
        # PERFECTED NEWS2 THRESHOLDS
        if score >= 7 or news.get("emergency", False): is_red = True
        elif score >= 5 or news.get("review", False): is_yellow = True

    qsofa = calc_qsofa(rr, sbp, avpu=avpu)
    score_details["qSOFA"] = qsofa

    # --- DUAL-VECTOR RESOLUTION ---
    if is_red:
        color = "RED"
        reason = f"Critical physiological instability ({ews_type} threshold breached)"
    elif is_yellow:
        color = "YELLOW"
        reason = f"Elevated clinical risk detected ({ews_type} trigger)"
    else:
        # Baseline Pathology check for non-critical cases (Gastroenteritis, etc.)
        # If vitals are perfect, but the bundle inherently carries some risk, default to YELLOW
        if bundle in ['Maternal', 'Pediatric', 'GI', 'Renal', 'Toxicology']:
            color = "YELLOW"
            reason = f"Stable vitals, but elevated baseline pathological risk ({bundle})"
            severity_index = max(0.3, severity_index)
        else:
            color = "GREEN"
            reason = f"Stable physiology ({ews_type} score {score})"

    meta = {
        "primary_driver": "Physiological Instability" if is_red or is_yellow else "Baseline Assessment",
        "reason": reason,
        "ews_type": ews_type,
        "ews_score": int(score),
        "severity_index": float(severity_index),
        "score_details": {**score_details, "severity_index": float(severity_index)},
    }
    return color, meta
