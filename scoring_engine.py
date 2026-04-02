# scoring_engine.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import math

# --- THE GLOBALLY ACCEPTABLE DEGRADATION MATRIX ---
# Maps a missing Tier 1 capability to its required Tier 2 medical fallback
FALLBACK_MATRIX = {
    "neurosurgeon": ["icu", "ct"],           # Medical management of ICP
    "cardiologist": ["icu", "thrombolysis"], # Medical management of STEMI
    "cathlab": ["icu", "thrombolysis"],
    "neonatologist": ["icu", "ventilator"],  # General intensivist bridge
    "dialysis": ["icu", "crrt"],             # Continuous renal bridge
    "surgeon": ["icu", "bloodbank"]          # Permissive hypotension bridge
}

def _is_nan(x: Any) -> bool:
    try: return isinstance(x, float) and math.isnan(x)
    except: return False

def _to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or _is_nan(x): return default
        if isinstance(x, bool): return int(x)
        s = str(x).strip()
        if s == "": return default
        return int(float(s))
    except: return default

def _normalize_required_caps(required_caps: Any) -> List[str]:
    if not required_caps: return []
    if isinstance(required_caps, (list, tuple, set)): return [str(x).strip().lower() for x in required_caps if str(x).strip()]
    if isinstance(required_caps, str): return [p.strip().lower() for p in required_caps.replace(",", ";").split(";") if p.strip()]
    return [str(required_caps).strip().lower()]

def _parse_caps_kv_string(caps_str: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not caps_str: return out
    for token in caps_str.split(";"):
        token = token.strip()
        if not token: continue
        if "=" in token:
            k, v = token.split("=", 1)
            out[k.strip().lower()] = _to_int(v, 0)
        else: out[token.lower()] = 1
    return out

def _normalize_caps(facility: Dict[str, Any]) -> Dict[str, int]:
    raw = facility.get("caps", facility.get("capabilities", None))
    if isinstance(raw, dict): return {str(k).strip().lower(): _to_int(v, 0) for k, v in raw.items()}
    if isinstance(raw, str):
        s = raw.strip()
        if "=" in s: return _parse_caps_kv_string(s)
        parts = [p.strip().lower() for p in s.replace(",", ";").split(";") if p.strip()]
        return {p: 1 for p in parts}
    if isinstance(raw, (list, tuple, set)): return {str(x).strip().lower(): 1 for x in raw if str(x).strip()}
    return {}

def calculate_facility_score(
    facility: Dict[str, Any], required_caps: Any, eta: Any, triage_color: str, severity_index: Any, case_type: Optional[str] = None, **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
    
    details: Dict[str, Any] = {}
    base_score = 100.0
    req = _normalize_required_caps(required_caps)
    caps = _normalize_caps(facility)

    try: eta_min = float(eta)
    except: eta_min = 999.0
    triage = (triage_color or "GREEN").upper()
    try: sev = max(0.0, min(1.0, float(severity_index or 0.0)))
    except: sev = 0.0

    # ---------------- GATE 1: THE TIERED DEGRADATION MATRIX ----------------
    missing_caps = []
    bridged_caps = []
    failed_caps = []
    tier = "Tier 1: Definitive Care"

    if req:
        # What is the hospital explicitly missing?
        missing_caps = [c for c in req if _to_int(caps.get(c, 0), 0) != 1]
        
        if missing_caps:
            # Attempt to bridge the missing capabilities using the Fallback Matrix
            for mc in missing_caps:
                fallbacks_needed = FALLBACK_MATRIX.get(mc, [])
                # Does the hospital have ALL the fallbacks for this missing specialist?
                if fallbacks_needed and all(_to_int(caps.get(fb, 0), 0) == 1 for fb in fallbacks_needed):
                    bridged_caps.append({"missing": mc, "bridged_with": fallbacks_needed})
                else:
                    failed_caps.append(mc)

            # Determine the Globally Acceptable Tier
            if not failed_caps and bridged_caps:
                tier = "Tier 2: Advanced Medical Bridging"
                base_score -= 15  # Small penalty, highly acceptable route
            else:
                tier = "Tier 3: Basic Stabilization"
                base_score -= 60  # Severe penalty, critical hardware/personnel missing
                
    details['gate_capability'] = tier
    details['missing_capabilities'] = missing_caps
    details['bridged_capabilities'] = bridged_caps
    details['failed_capabilities'] = failed_caps

    # ---------------- GATE 2: CAPACITY & ED OVERRIDE ----------------
    icu_open = _to_int(facility.get("ICU_open", 0), 0)
    requires_bed = ("icu" in req) or ("ventilator" in req) or (triage == "RED")
    has_ed = _to_int(caps.get("ed", 1), 1)

    icu_score = 0
    if requires_bed and icu_open < 1:
        if triage == "RED" and has_ed >= 1:
            details["gate_capacity"] = "WARNING_ED_STABILIZATION_ONLY"
            base_score -= 40
            tier = "Tier 3: Basic Stabilization" # Force tier drop due to no ICU
        else:
            details["gate_capacity"] = "FAILED"
            return 0.0, details
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

    total = base_score - 100 + prox + icu_score + fiscal_score + severity_bonus
    total = min(100.0, max(0.1, total))

    details.update({
        "clinical_tier": tier,
        "eta_minutes": round(eta_min, 1),
        "proximity_score": prox,
        "icu_score": icu_score,
        "fiscal_score": fiscal_score,
        "severity_bonus": severity_bonus,
        "icu_beds": icu_open,
        "ownership": ownership,
        "total_score": round(total, 1),
    })

    return float(total), details
