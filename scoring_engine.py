# scoring_engine.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import math

def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False

def _to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or _is_nan(x):
            return default
        if isinstance(x, bool):
            return int(x)
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default

def _normalize_required_caps(required_caps: Any) -> List[str]:
    if required_caps is None:
        return []
    if isinstance(required_caps, (list, tuple, set)):
        return [str(x).strip().lower() for x in required_caps if str(x).strip()]
    if isinstance(required_caps, str):
        return [p.strip().lower() for p in required_caps.replace(",", ";").split(";") if p.strip()]
    s = str(required_caps).strip().lower()
    return [s] if s else []

def _parse_caps_kv_string(caps_str: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not caps_str:
        return out
    for token in caps_str.split(";"):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            k, v = token.split("=", 1)
            out[k.strip().lower()] = _to_int(v, 0)
        else:
            out[token.lower()] = 1
    return out

def _normalize_caps(facility: Dict[str, Any]) -> Dict[str, int]:
    raw = facility.get("caps", facility.get("capabilities", None))
    if isinstance(raw, dict):
        return {str(k).strip().lower(): _to_int(v, 0) for k, v in raw.items()}
    if isinstance(raw, str):
        s = raw.strip()
        if "=" in s:
            return _parse_caps_kv_string(s)
        parts = [p.strip().lower() for p in s.replace(",", ";").split(";") if p.strip()]
        return {p: 1 for p in parts}
    if isinstance(raw, (list, tuple, set)):
        return {str(x).strip().lower(): 1 for x in raw if str(x).strip()}
    return {}

def calculate_facility_score(
    facility: Dict[str, Any],
    required_caps: Any,
    eta: Any,
    triage_color: str,
    severity_index: Any,
    case_type: Optional[str] = None,
    **kwargs: Any,  
) -> Tuple[float, Dict[str, Any]]:
    
    details: Dict[str, Any] = {}
    base_score = 100.0

    req = _normalize_required_caps(required_caps)
    caps = _normalize_caps(facility)

    try: eta_min = float(eta)
    except Exception: eta_min = 999.0

    triage = (triage_color or "GREEN").upper()
    try: sev = float(severity_index or 0.0)
    except Exception: sev = 0.0
    sev = max(0.0, min(1.0, sev))

    # ---------------- GATE 1: GRACEFUL DEGRADATION (Soft Gating) ----------------
    missing_caps = []
    if req:
        missing_caps = [c for c in req if _to_int(caps.get(c, 0), 0) != 1 and c not in [k for k,v in caps.items() if v == 1]]
        match_ratio = (len(req) - len(missing_caps)) / len(req) if req else 1.0

        if match_ratio == 1.0:
            details['gate_capability'] = "PASSED"
        elif match_ratio > 0:
            details['gate_capability'] = "PARTIAL_MATCH"
            base_score -= (len(missing_caps) * 25) # Penalty, but NOT instant death (0.0)
        else:
            details['gate_capability'] = "FAILED"
            base_score -= 75 # Major penalty, moves to bottom of list
    else:
        details['gate_capability'] = "PASSED"
        
    details['missing_capabilities'] = missing_caps

    # ---------------- GATE 2: CAPACITY & ED OVERRIDE ----------------
    icu_open = _to_int(facility.get("ICU_open", 0), 0)
    requires_bed = ("icu" in req) or ("ventilator" in req) or (triage == "RED")
    has_ed = _to_int(caps.get("ed", 1), 1)

    icu_score = 0
    if requires_bed and icu_open < 1:
        if triage == "RED" and has_ed >= 1:
            details["gate_capacity"] = "WARNING_ED_STABILIZATION_ONLY"
            base_score -= 40 # Heavy penalty, but still an option for immediate stabilization
        else:
            details["gate_capacity"] = "FAILED"
            return 0.0, details # True hard stop: No ICU, no ED, not a critical priority.
    else:
        details["gate_capacity"] = "PASSED"
        if icu_open >= 3: icu_score = 15
        elif icu_open == 2: icu_score = 10
        elif icu_open == 1: icu_score = 5

    # ---------------- SCORING METRICS ----------------
    adjusted_eta = eta_min * (1.0 + sev)
    if adjusted_eta <= 30: prox = 50
    elif adjusted_eta <= 60: prox = 35
    elif adjusted_eta <= 90: prox = 15
    else: prox = 0

    ownership = str(facility.get("ownership", "Private") or "Private").strip()
    fiscal_score = 20 if ownership.lower() == "government" else 0
    severity_bonus = int(round(sev * 5))

    # Calculate final score based on degradation base + modifiers
    total = base_score - 100 + prox + icu_score + fiscal_score + severity_bonus
    total = min(100.0, max(0.1, total)) # 0.1 minimum ensures it still shows in the UI

    details.update({
        "eta_minutes": round(eta_min, 1),
        "adjusted_eta_minutes": round(adjusted_eta, 1),
        "severity_index": sev,
        "proximity_score": prox,
        "icu_score": icu_score,
        "fiscal_score": fiscal_score,
        "severity_bonus": severity_bonus,
        "icu_beds": icu_open,
        "ownership": ownership,
        "case_type": case_type,
        "total_score": round(total, 1),
    })

    return float(total), details
