# synthetic_cases.py
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ---- Helper distributions (demo-friendly, clinically plausible) ----

@dataclass
class CaseProfile:
    bundle: str
    icd10_pool: List[str]
    # Vitals ranges are baseline; severity sampling shifts toward extremes
    hr_base: Tuple[int, int]
    sbp_base: Tuple[int, int]
    rr_base: Tuple[int, int]
    spo2_base: Tuple[int, int]
    temp_base: Tuple[float, float]
    age_range: Tuple[int, int]
    # probability of pathology "auto-red" for this bundle in synthetic stream
    auto_red_weight: float


CASE_PROFILES: Dict[str, CaseProfile] = {
    "Maternal": CaseProfile(
        bundle="Maternal",
        icd10_pool=["O72.0", "O72.1", "O14.1", "O15.0", "O44.1", "O85"],
        hr_base=(80, 130),
        sbp_base=(80, 160),
        rr_base=(16, 30),
        spo2_base=(90, 99),
        temp_base=(36.5, 38.8),
        age_range=(15, 45),
        auto_red_weight=0.45,
    ),
    "Trauma": CaseProfile(
        bundle="Trauma",
        icd10_pool=["S06.0", "S06.4", "S06.5", "S36.0", "S36.1", "T31.0"],
        hr_base=(70, 140),
        sbp_base=(70, 170),
        rr_base=(14, 34),
        spo2_base=(88, 99),
        temp_base=(36.0, 38.2),
        age_range=(1, 85),
        auto_red_weight=0.40,
    ),
    "Stroke": CaseProfile(
        bundle="Stroke",
        icd10_pool=["I63.9", "I61.9", "I60.9", "I64", "G45.9"],
        hr_base=(55, 120),
        sbp_base=(110, 210),
        rr_base=(12, 26),
        spo2_base=(90, 99),
        temp_base=(36.0, 37.8),
        age_range=(30, 90),
        auto_red_weight=0.35,
    ),
    "Cardiac": CaseProfile(
        bundle="Cardiac",
        icd10_pool=["I21.9", "I21.0", "I46.9", "I44.2", "I47.2", "I50.9"],
        hr_base=(45, 135),
        sbp_base=(80, 170),
        rr_base=(14, 28),
        spo2_base=(88, 98),
        temp_base=(36.0, 37.8),
        age_range=(18, 95),
        auto_red_weight=0.40,
    ),
    "Sepsis": CaseProfile(
        bundle="Sepsis",
        icd10_pool=["A41.9", "A41.0", "A41.5"],
        hr_base=(90, 160),
        sbp_base=(70, 130),
        rr_base=(18, 36),
        spo2_base=(85, 96),
        temp_base=(37.2, 40.2),
        age_range=(0, 90),
        auto_red_weight=0.30,
    ),
    "Other": CaseProfile(
        bundle="Other",
        icd10_pool=["J96.0", "K92.2", "E10.1", "N17.9", "T78.2"],
        hr_base=(70, 150),
        sbp_base=(75, 160),
        rr_base=(16, 34),
        spo2_base=(80, 96),
        temp_base=(36.2, 39.5),
        age_range=(0, 90),
        auto_red_weight=0.25,
    ),
}

# Severity mixture: more moderate, fewer extremes, but enough REDs for demo charts
SEVERITY_MIX = [
    ("GREEN", 0.40),
    ("YELLOW", 0.35),
    ("RED", 0.25),
]

SEX_OPTIONS = ["Male", "Female", "Other"]


def _weighted_choice(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    r = rng.random()
    cum = 0.0
    for name, w in items:
        cum += w
        if r <= cum:
            return name
    return items[-1][0]


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _severity_to_shift(color: str) -> float:
    """
    Map triage class to a "physiology shift" factor.
    RED pushes vitals toward instability; GREEN stays near baseline.
    """
    return {"GREEN": 0.10, "YELLOW": 0.35, "RED": 0.75}.get(color, 0.25)


def _gen_vitals(rng: random.Random, prof: CaseProfile, target_color: str) -> Dict:
    shift = _severity_to_shift(target_color)

    # Baseline sample
    hr = rng.randint(*prof.hr_base)
    sbp = rng.randint(*prof.sbp_base)
    rr = rng.randint(*prof.rr_base)
    spo2 = rng.randint(*prof.spo2_base)
    temp = round(rng.uniform(*prof.temp_base), 1)

    # Push toward instability depending on target
    # (simple + demo-safe; triage engine remains source of truth)
    if shift > 0:
        # HR extremes
        if rng.random() < shift:
            hr = hr + rng.randint(20, 45) if rng.random() < 0.7 else hr - rng.randint(10, 25)
        # SBP drop for shock-like patterns
        if rng.random() < shift:
            sbp = sbp - rng.randint(15, 55)
        # RR rise for distress
        if rng.random() < shift:
            rr = rr + rng.randint(6, 16)
        # SpO2 drop
        if rng.random() < shift:
            spo2 = spo2 - rng.randint(4, 18)
        # Temp rise for sepsis
        if rng.random() < (shift * 0.8):
            temp = round(temp + rng.uniform(0.4, 1.4), 1)

    # Clamp physiological bounds
    hr = int(_clamp(hr, 30, 220))
    sbp = int(_clamp(sbp, 50, 240))
    rr = int(_clamp(rr, 8, 60))
    spo2 = int(_clamp(spo2, 60, 100))
    temp = float(_clamp(temp, 33.0, 42.0))

    # AVPU worsens with severity sometimes
    avpu = "A"
    if target_color == "RED" and rng.random() < 0.25:
        avpu = rng.choice(["V", "P", "U"])
    elif target_color == "YELLOW" and rng.random() < 0.08:
        avpu = "V"

    return {"hr": hr, "sbp": sbp, "rr": rr, "spo2": spo2, "temp": temp, "avpu": avpu}


def seed_synthetic_referrals_v2(
    *,
    n: int,
    facilities: List[Dict],
    icd_df: "object",  # pandas df
    validated_triage_decision_fn,
    now_ts_fn,
    rand_geo_fn,
    dist_km_fn,
    interpolate_route_fn,
    traffic_factor_fn,
    rng_seed: int = 42,
    append: bool = True,
) -> List[Dict]:
    """
    Generates schema-compatible referrals with:
    - vitals
    - ICD code
    - validated triage output (color + meta incl severity_index)
    - basic ETA + traffic
    - destination facility + ownership for analytics

    Returns the newly generated referrals list (caller decides insert/append).
    """
    if not facilities:
        return []

    import pandas as pd  # local import to keep module lightweight

    rng = random.Random(rng_seed)

    # Ensure ownership exists on facilities
    for f in facilities:
        f.setdefault("ownership", "Private")

    out: List[Dict] = []

    # last 30d distribution
    base = time.time() - 30 * 24 * 3600

    bundles = list(CASE_PROFILES.keys())
    bundle_weights = [0.20, 0.22, 0.18, 0.18, 0.14, 0.08]  # maternal, trauma, stroke, cardiac, sepsis, other

    for i in range(n):
        bundle = rng.choices(bundles, weights=bundle_weights, k=1)[0]
        prof = CASE_PROFILES[bundle]

        # Choose "intended" acuity; final acuity comes from validated triage
        target_color = _weighted_choice(rng, SEVERITY_MIX)

        age = rng.randint(*prof.age_range)
        sex = rng.choice(SEX_OPTIONS)

        vitals = _gen_vitals(rng, prof, target_color)

        # Choose ICD and row
        icd10 = rng.choice(prof.icd10_pool)
        row = None
        try:
            rows = icd_df[icd_df["icd10"] == icd10]
            if not rows.empty:
                row = rows.iloc[0].to_dict()
        except Exception:
            pass
            
        # BULLETPROOF FALLBACK: If the CSV lookup failed for any reason, force a default dictionary
        if row is None:
            row = {"icd10": icd10, "label": f"{bundle} condition", "bundle": bundle, "default_interventions": ""}

        # Location
        lat, lon = rand_geo_fn(rng)

        # Choose a destination: prefer facilities that have ICU if RED-ish, otherwise any
        # (facility matching logic can override later; this is synthetic seeding only)
        if target_color == "RED":
            candidates = [f for f in facilities if int(f.get("ICU_open", 0)) > 0]
            dest = rng.choice(candidates or facilities)
        else:
            dest = rng.choice(facilities)

        # ETA model
        dkm = dist_km_fn(lat, lon, float(dest["lat"]), float(dest["lon"]))
        ts_first = base + rng.randint(0, 30 * 24 * 3600)
        hr_of_day = datetime.fromtimestamp(ts_first).hour
        traffic_mult = traffic_factor_fn(hr_of_day)
        speed_kmh = rng.choice([28, 34, 40, 48, 55])
        eta_min = max(6, int(dkm / speed_kmh * 60 * traffic_mult))
        route = interpolate_route_fn(lat, lon, float(dest["lat"]), float(dest["lon"]), n=24)

        # Validated triage decision (source of truth)
        context = {
            "age": age,
            "pregnant": (bundle == "Maternal"),
            "infection": (bundle in ["Sepsis", "Other"]),
            "o2_device": "Air",
            "spo2_scale": 1,
            "behavior": "Normal",
        }
        color, meta = validated_triage_decision_fn(vitals, row, context)

        # Minimal interventions stub (optional)
        interventions = []
        default_interventions = row.get("default_interventions") or ""
        if isinstance(default_interventions, str) and default_interventions.strip():
            # if CSV uses semicolon separated
            for iv in [x.strip() for x in default_interventions.replace("\n", ";").split(";") if x.strip()]:
                interventions.append({
                    "name": iv,
                    "type": "diagnosis_default",
                    "timestamp": ts_first,
                    "performed_by": "referrer",
                    "status": "completed",
                })

        ref = {
            "id": f"SYN{i:05d}",
            "patient": {
                "name": f"Pt{i:05d}",
                "age": age,
                "sex": sex,
                "id": "",
                "location": {"lat": lat, "lon": lon},
            },
            "referrer": {
                "name": rng.choice(["Dr. Rai", "Dr. Khonglah", "Dr. Sharma", "Dr. Singh", "ANM Pynsuk"]),
                "facility": rng.choice(["PHC Mawlai", "CHC Smit", "CHC Pynursla", "District Hospital Shillong"]),
                "role": rng.choice(["Doctor/Physician", "ANM/ASHA/EMT"]),
                "cadre": rng.choice(["Doctor", "ANM", "ASHA", "Nurse", "EMT"]),
            },
            "provisionalDx": {
                "code": row.get("icd10", icd10),
                "label": row.get("label", f"{bundle} condition"),
                "case_type": bundle,
            },
            "interventions": interventions,
            "resuscitation": [],
            "triage": {
                "complaint": bundle,
                "decision": {
                    "color": color,
                    "overridden": False,
                    "base_color": color,
                    "triage_mode": "validated_rules",
                    "score_details": meta,  # meta must include severity_index
                },
                "hr": vitals["hr"],
                "sbp": vitals["sbp"],
                "rr": vitals["rr"],
                "temp": vitals["temp"],
                "spo2": vitals["spo2"],
                "avpu": vitals["avpu"],
            },
            "clinical": {"summary": f"Synthetic {bundle.lower()} case"},
            "reasons": {
                "severity": (color == "RED"),
                "bedOrICUUnavailable": (rng.random() < 0.12),
                "specialTest": (rng.random() < 0.18),
                "requiredCapabilities": [],  # facility matching fills later
            },
            "dest": dest.get("name", "Unknown Facility"),
            "alternates": [],
            "transport": {
                "eta_min": eta_min,
                "traffic": traffic_mult,
                "speed_kmh": speed_kmh,
                "ambulance": rng.choice(["BLS", "ALS", "ALS + Vent", "Neonatal"]),
                "priority": rng.choice(["Routine", "Urgent", "STAT"]),
                "consent": True,
            },
            "route": route,
            "times": {
                "first_contact_ts": ts_first,
                "decision_ts": ts_first + rng.randint(60, 10 * 60),
                "dispatch_ts": ts_first + rng.randint(10 * 60, 25 * 60),
                "arrive_dest_ts": ts_first + rng.randint(30 * 60, 120 * 60),
            },
            "status": rng.choice(["PREALERT", "DISPATCHED", "ARRIVE_DEST", "HANDOVER"]),
            "ambulance_available": (rng.random() > 0.2),
            "audit_log": [],
            # convenience fields for analytics flattening (optional)
            "facility_ownership": dest.get("ownership", "Private"),
        }

        out.append(ref)

    return out
