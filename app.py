# AHECN – Streamlit MVP v1.8 (Score-Only + Clinician Override)
import math
import json
import time
import random
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import os

# === PAGE CONFIG MUST BE FIRST STREAMLIT COMMAND ===
st.set_page_config(
    page_title="AHECN MVP v1.8",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === LOAD ICD CATALOG FROM CSV ===
def load_icd_catalogue():
    """Load ICD catalog from CSV file with robust error handling."""
    try:
        df = pd.read_csv('icd_catalogue.csv', encoding='utf-8')
        icd_lut = []
        for _, row in df.iterrows():
            default_caps = []
            if 'default_caps' in df.columns and pd.notna(row.get('default_caps')):
                default_caps = [cap.strip() for cap in str(row['default_caps']).split(';') if cap.strip()]
            
            icd_lut.append({
                "icd_code": row['icd10'],
                "label": row['label'],
                "case_type": row['bundle'],
                "age_min": int(row['age_min']),
                "age_max": int(row['age_max']),
                "default_interventions": "",
                "default_caps": default_caps
            })
        return icd_lut
    except Exception as e:
        st.error(f"Error loading ICD catalog: {str(e)}")
        return get_fallback_icd_catalog()

def get_fallback_icd_catalog():
    """Provide a fallback ICD catalog if CSV loading fails."""
    return [
        # Maternal
        {"icd_code": "O72.0", "label": "Third-stage haemorrhage", "case_type": "Maternal", "age_min": 12, "age_max": 55,
         "default_interventions": "IV fluids;Uterotonics;TXA", "default_caps": ["ICU", "BloodBank", "OBGYN_OT", "OR", "Ventilator"]},
        {"icd_code": "O72.1", "label": "Immediate postpartum haemorrhage", "case_type": "Maternal", "age_min": 12, "age_max": 55,
         "default_interventions": "IV fluids;Uterotonics;TXA", "default_caps": ["ICU", "BloodBank", "OBGYN_OT", "OR", "Ventilator"]},
        {"icd_code": "O14.1", "label": "Severe pre-eclampsia", "case_type": "Maternal", "age_min": 12, "age_max": 55,
         "default_interventions": "Magnesium sulfate;BP control", "default_caps": ["ICU", "OBGYN_OT"]},
        
        # Trauma
        {"icd_code": "S06.0", "label": "Concussion", "case_type": "Trauma", "age_min": 0, "age_max": 120,
         "default_interventions": "Neuro checks;Immobilization", "default_caps": ["CT"]},
        {"icd_code": "S06.5", "label": "Traumatic subdural haemorrhage", "case_type": "Trauma", "age_min": 0, "age_max": 120,
         "default_interventions": "Airway protection;IV access", "default_caps": ["CT", "Neurosurgery", "ICU", "OR"]},
        
        # Stroke
        {"icd_code": "I63.9", "label": "Cerebral infarction unspecified", "case_type": "Stroke", "age_min": 18, "age_max": 120,
         "default_interventions": "BP control;Glucose check", "default_caps": ["CT", "Thrombolysis", "ICU"]},
        
        # Cardiac
        {"icd_code": "I21.9", "label": "Acute myocardial infarction unspecified", "case_type": "Cardiac", "age_min": 18, "age_max": 120,
         "default_interventions": "Aspirin;Oxygen;IV access", "default_caps": ["CathLab", "ICU"]},
        
        # Sepsis
        {"icd_code": "A41.9", "label": "Sepsis unspecified organism", "case_type": "Sepsis", "age_min": 0, "age_max": 120,
         "default_interventions": "Antibiotics;IV fluids;Oxygen", "default_caps": ["ICU"]},
        
        # Other
        {"icd_code": "J96.0", "label": "Acute respiratory failure", "case_type": "Other", "age_min": 0, "age_max": 120,
         "default_interventions": "Oxygen;Nebulization", "default_caps": ["Ventilator", "ICU"]},
    ]

# Load ICD catalog
ICD_LUT = load_icd_catalogue()

def icd_options_for(case_type: str, age_years: float):
    """Return (choices, filtered_df) for the given case type + age."""
    try:
        a = float(age_years)
    except Exception:
        a = None
    df = pd.DataFrame(ICD_LUT)
    if case_type:
        df = df[df["case_type"].str.lower() == str(case_type).lower()]
    if a is not None:
        df = df[(df["age_min"] <= a) & (a <= df["age_max"])]
    if df.empty:
        return [], df
    df = df.copy()
    df["display"] = df["label"] + "  ·  " + df["icd_code"]
    return df["display"].tolist(), df

# === CSS AND STYLING ===
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --ok:#10b981; --warn:#f59e0b; --bad:#ef4444; --muted:#9ca3af; --chip:#1f2937; --card:#0f172a;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container{ padding-top:1.2rem; padding-bottom:3rem; }
h1,h2,h3{ letter-spacing:0.2px; }
.badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:0.78rem;
       background:#1f2937;color:#e5e7eb;margin-right:6px;margin-bottom:6px}
.badge.ok{ background:rgba(16,185,129,.15); color:#34d399; border:1px solid rgba(16,185,129,.35)}
.badge.warn{background:rgba(245,158,11,.15); color:#fbbf24; border:1px solid rgba(245,158,11,.35)}
.badge.bad{ background:rgba(239,68,68,.15); color:#f87171; border:1px solid rgba(239,68,68,.35)}
.pill{display:inline-flex;align-items:center;gap:.5rem;padding:.35rem .7rem;border-radius:999px;
      font-weight:600; font-size:.85rem}
.pill.red{ background:rgba(239,68,68,.18); color:#fca5a5; border:1px solid rgba(239,68,68,.35)}
.pill.yellow{background:rgba(245,158,11,.18); color:#fcd34d; border:1px solid rgba(245,158,11,.35)}
.pill.green{background:rgba(16,185,129,.18); color:#6ee7b7; border:1px solid rgba(16,185,129,.35)}
.card{ background:var(--card); border:1px solid #1f2937; border-radius:16px; padding:14px 16px;
       box-shadow:0 6px 16px rgba(0,0,0,.25); margin-bottom:12px}
.card h4{ margin:0 0 6px 0; }
.kpi{ background:#0d1b2a; border:1px solid #1f2937; border-radius:14px; padding:14px; }
.kpi .label{ color:#9ca3af; font-size:.8rem; }
.kpi .value{ font-size:1.6rem; font-weight:700; margin-top:4px}
hr.soft{ border:none; height:1px; background:#1f2937; margin:10px 0 14px }
.btnline > div > button{ width:100% }
.small{ color:#9ca3af; font-size:.85rem }
.required{ color:#ef4444; }
.override-badge { background: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.35); }
.audit-log { background: #1e293b; padding: 8px 12px; border-radius: 8px; border-left: 4px solid #8b5cf6; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)

# === CORE HELPERS ===
def _num(x):
    """Convert to float or return None if blank/invalid."""
    if x is None: return None
    s = str(x).strip()
    if s == "": return None
    try: return float(s)
    except Exception: return None

def _int(x, default=1):
    try: return int(str(x).strip())
    except Exception: return default

def _clip(v, lo, hi):
    x = _num(v)
    if x is None: return None
    return max(lo, min(hi, x))

def validate_vitals(hr, rr, sbp, temp, spo2):
    return dict(
        hr   = _clip(hr,   20, 240),
        rr   = _clip(rr,    5,  60),
        sbp  = _clip(sbp,  50, 260),
        temp = _clip(temp, 32,  42),
        spo2 = _clip(spo2, 50, 100),
    )

# === SCORING ENGINES ===
def calc_NEWS2(rr, spo2, sbp, hr, temp, avpu, o2_device="Air", spo2_scale=1):
    rr, spo2, sbp, hr, temp = (_num(rr), _num(spo2), _num(sbp), _num(hr), _num(temp))
    avpu = "A" if avpu is None else str(avpu).strip().upper()
    spo2_scale = _int(spo2_scale, 1)
    o2_device = "Air" if not o2_device else str(o2_device).strip()

    hits, score = [], 0

    # RR
    if rr is None: pass
    elif rr <= 8:      score += 3; hits.append("NEWS2 RR ≤8 =3")
    elif 9 <= rr <=11: score += 1; hits.append("NEWS2 RR 9–11 =1")
    elif 12 <= rr <=20:                 hits.append("NEWS2 RR 12–20 =0")
    elif 21 <= rr <=24: score += 2; hits.append("NEWS2 RR 21–24 =2")
    else:               score += 3; hits.append("NEWS2 RR ≥25 =3")

    # SpO2 scale 1/2
    def spo2_s1(s): return 3 if s<=91 else 2 if s<=93 else 1 if s<=95 else 0
    def spo2_s2(s): return 3 if s<=83 else 2 if s<=85 else 1 if s<=90 else 0 if s<=92 else 0
    if spo2 is not None:
        pts = spo2_s1(spo2) if int(spo2_scale)==1 else spo2_s2(spo2)
        score += pts; hits.append(f"NEWS2 SpO₂ (scale {spo2_scale}) +{pts}")
    if str(o2_device).lower() != "air":
        score += 2; hits.append("NEWS2 Supplemental O₂ +2")

    # SBP
    if sbp is not None:
        if sbp <= 90:        score += 3; hits.append("NEWS2 SBP ≤90 =3")
        elif sbp <=100:      score += 2; hits.append("NEWS2 SBP 91–100 =2")
        elif sbp <=110:      score += 1; hits.append("NEWS2 SBP 101–110 =1")
        elif sbp <=219:                     hits.append("NEWS2 SBP 111–219 =0")
        else:                score += 3; hits.append("NEWS2 SBP ≥220 =3")

    # HR
    if hr is not None:
        if hr <= 40:         score += 3; hits.append("NEWS2 HR ≤40 =3")
        elif hr <= 50:       score += 1; hits.append("NEWS2 HR 41–50 =1")
        elif hr <= 90:                      hits.append("NEWS2 HR 51–90 =0")
        elif hr <=110:       score += 1; hits.append("NEWS2 HR 91–110 =1")
        elif hr <=130:       score += 2; hits.append("NEWS2 HR 111–130 =2")
        else:                score += 3; hits.append("NEWS2 HR ≥131 =3")

    # Temp
    if temp is not None:
        if temp <= 35.0:         score += 3; hits.append("NEWS2 Temp ≤35.0 =3")
        elif temp <= 36.0:       score += 1; hits.append("NEWS2 Temp 35.1–36.0 =1")
        elif temp <= 38.0:                        hits.append("NEWS2 Temp 36.1–38.0 =0")
        elif temp <= 39.0:       score += 1; hits.append("NEWS2 Temp 38.1–39.0 =1")
        else:                    score += 2; hits.append("NEWS2 Temp ≥39.1 =2")

    # AVPU
    if avpu != "A":
        score += 3; hits.append("NEWS2 AVPU ≠ A =3")

    return score, hits, (5 <= int(score or 0) < 7), (int(score or 0) >= 7)

def _to4(out):
    """Normalize any NEWS2 output to (score, hits, review, urgent)."""
    try:
        if isinstance(out, (list, tuple)):
            n = len(out)
            if n == 4:
                s, h, r, u = out; return int(s or 0), list(h or []), bool(r), bool(u)
            if n == 3:
                s, h, u = out; s = int(s or 0); return s, list(h or []), (5 <= s < 7), bool(u)
            if n == 2:
                s, h = out; s = int(s or 0); return s, list(h or []), (5 <= s < 7), (s >= 7)
            if n == 1:
                s = int(out[0] or 0); return s, [], (5 <= s < 7), (s >= 7)
    except Exception:
        pass
    return 0, ["NEWS2 malformed return"], False, False

def safe_calc_NEWS2(rr, spo2, sbp, hr, temp, avpu, o2_device="Air", spo2_scale=1):
    """Always returns (score, hits, review, urgent). Never raises."""
    try:
        raw = calc_NEWS2(rr, spo2, sbp, hr, temp, avpu, o2_device, spo2_scale)
    except Exception as e:
        return 0, [f"NEWS2 error: {type(e).__name__}"], False, False
    return _to4(raw)

def calc_qSOFA(rr, sbp, avpu):
    rr, sbp = _num(rr), _num(sbp)
    avpu = "A" if avpu is None else str(avpu).strip().upper()
    hits, score = [], 0
    if rr is not None and rr >= 22: hits.append("RR ≥22"); score += 1
    if sbp is not None and sbp <= 100: hits.append("SBP ≤100"); score += 1
    if avpu != "A": hits.append("Altered mentation"); score += 1
    return score, hits, (score >= 2)

def calc_MEOWS(hr, rr, sbp, temp, spo2):
    hr, rr, sbp, temp, spo2 = _num(hr), _num(rr), _num(sbp), _num(temp), _num(spo2)
    red, yellow = [], []
    if sbp is not None:
        if sbp < 90 or sbp > 160: red.append("SBP critical")
        elif sbp < 100 or sbp > 150: yellow.append("SBP borderline")
    if hr is not None:
        if hr > 120 or hr < 50: red.append("HR critical")
        elif hr > 100: yellow.append("HR high")
    if rr is not None:
        if rr > 30 or rr < 10: red.append("RR critical")
        elif rr > 21: yellow.append("RR high")
    if temp is not None:
        if temp >= 38.0 or temp < 35.0: red.append("Temp critical")
        elif temp >= 37.6: yellow.append("Temp high")
    if spo2 is not None:
        if spo2 < 94: red.append("SpO₂ <94%")
        elif spo2 < 96: yellow.append("SpO₂ 94–95%")
    return dict(red=red, yellow=yellow)

def _band(x, ylo, yhi, rlo, rhi):
    """Return 2 if in red range, 1 if in yellow range, else 0."""
    x = _num(x)
    if x is None: return 0
    if x >= rhi or x <= rlo: return 2
    if x >= yhi or x <= ylo: return 1
    return 0

def calc_PEWS(age, rr, hr, behavior="Normal", spo2=None):
    age  = _num(age); rr = _num(rr); hr = _num(hr); spo2 = _num(spo2)
    if age is None: return 0, {"detail": "age missing"}, False, False

    if age < 1:         rr_y, rr_r = (40, 50), (50, 60); hr_y, hr_r = (140, 160), (160, 200)
    elif age < 5:       rr_y, rr_r = (30, 40), (40, 60); hr_y, hr_r = (130, 150), (150, 200)
    elif age < 12:      rr_y, rr_r = (24, 30), (30, 60); hr_y, hr_r = (120, 140), (140, 200)
    else:               rr_y, rr_r = (20, 24), (24, 60); hr_y, hr_r = (110, 130), (130, 200)

    sc = 0
    sc += _band(rr, rr_y[0], rr_y[1], rr_r[0], rr_r[1])
    sc += _band(hr, hr_y[0], hr_y[1], hr_r[0], hr_r[1])
    if spo2 is not None: sc += 2 if spo2 < 92 else (1 if spo2 < 95 else 0)

    beh = str(behavior or "Normal").lower()
    if beh == "lethargic": sc += 2
    elif beh == "irritable": sc += 1

    return sc, {"age": age}, (sc >= 6), (sc >= 4)

def triage_decision(vitals, context):
    """
    NEW: Score-only triage decision without ad-hoc flags
    vitals: dict(hr, rr, sbp, temp, spo2, avpu)
    context: dict(age, pregnant, infection, o2_device, spo2_scale, behavior)
    """
    v = validate_vitals(vitals.get("hr"), vitals.get("rr"), vitals.get("sbp"),
                        vitals.get("temp"), vitals.get("spo2"))
    avpu = vitals.get("avpu","A")
    reasons = []

    # Scores only - no ad-hoc flags
    news2_score, news2_hits, news2_review, news2_urgent = _to4(
        safe_calc_NEWS2(
            v["rr"], v["spo2"], v["sbp"], v["hr"], v["temp"], avpu,
            context.get("o2_device", "Air"), context.get("spo2_scale", 1)
        )
    )
    q_score, q_hits, q_high = (
        calc_qSOFA(v["rr"], v["sbp"], avpu) if context.get("infection") else (0, [], False)
    )
    meows = (
        calc_MEOWS(v["hr"], v["rr"], v["sbp"], v["temp"], v["spo2"])
        if context.get("pregnant") else dict(red=[], yellow=[])
    )
    pews_sc, pews_meta, pews_high, pews_watch = (
        calc_PEWS(context.get("age"), v["rr"], v["hr"], context.get("behavior","Normal"), v["spo2"])
        if (context.get("age") is not None and context.get("age") < 18)
        else (0, {}, False, False)
    )

    # Colour determination based purely on scores
    colour = "GREEN"
    
    # RED criteria
    if (news2_urgent or 
        q_high or 
        (context.get("pregnant") and len(meows["red"]) > 0) or 
        (context.get("age") is not None and context.get("age") < 18 and pews_high)):
        colour = "RED"
    
    # YELLOW criteria (only if not RED)
    elif colour == "GREEN" and (
        news2_review or 
        (context.get("pregnant") and len(meows["yellow"]) > 0) or 
        (context.get("age") is not None and context.get("age") < 18 and pews_watch)):
        colour = "YELLOW"

    # Build reasons for display
    if news2_urgent: reasons.append(f"NEWS2 {news2_score} (≥7)")
    elif news2_review: reasons.append(f"NEWS2 {news2_score} (≥5)")
    if q_high: reasons.append(f"qSOFA {q_score} (≥2)")
    if context.get("pregnant") and meows["red"]: reasons.append("MEOWS red band")
    if context.get("pregnant") and meows["yellow"] and colour == "YELLOW": reasons.append("MEOWS yellow band")
    if (context.get("age") is not None and context.get("age") < 18 and pews_high): reasons.append(f"PEWS {pews_sc} (≥6)")
    if (context.get("age") is not None and context.get("age") < 18 and pews_watch and colour == "YELLOW"): reasons.append(f"PEWS {pews_sc} (≥4)")

    details = {
        "NEWS2": dict(score=news2_score, hits=news2_hits, review=news2_review, urgent=news2_urgent),
        "qSOFA": dict(score=q_score, hits=q_hits, high=q_high),
        "MEOWS": meows,
        "PEWS": dict(score=pews_sc, high=pews_high, watch=pews_watch),
        "reasons": reasons
    }
    return colour, details

def tri_color(vit):
    """Used by seeding; reuses the same rule engine for consistency."""
    v = dict(
        hr=vit.get("hr"), rr=vit.get("rr"), sbp=vit.get("sbp"),
        temp=vit.get("temp"), spo2=vit.get("spo2"), avpu=vit.get("avpu","A")
    )
    context = dict(
        age=30,
        pregnant=(vit.get("complaint") == "Maternal"),
        infection=(vit.get("complaint") in ["Sepsis","Other"]),
        o2_device="Air", spo2_scale=1, behavior="Normal"
    )
    colour, _ = triage_decision(v, context)
    return colour

# === UI HELPERS ===
def triage_pill(color:str, overridden=False):
    c = (color or "").upper()
    cls = "red" if c=="RED" else "yellow" if c=="YELLOW" else "green"
    if overridden:
        st.markdown(f'<span class="pill {cls} override-badge">{c} (OVERRIDDEN)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="pill {cls}">{c}</span>', unsafe_allow_html=True)

def kpi_tile(label, value, help_text=None):
    st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>',
                unsafe_allow_html=True)
    if help_text: st.caption(help_text)

def cap_badges(list_or_csv):
    if isinstance(list_or_csv, str):
        items = [x.strip() for x in list_or_csv.split(",") if x.strip() and x.strip()!="—"]
    else:
        items = list_or_csv or []
    if not items:
        st.markdown('<span class="badge">—</span>', unsafe_allow_html=True); return
    cols = st.columns(min(4, max(1, len(items))))
    for i,cap in enumerate(items[:12]):
        cols[i%len(cols)].markdown(f'<span class="badge">{cap}</span>', unsafe_allow_html=True)

def render_triage_banner(hr, rr, sbp, temp, spo2, avpu, complaint, override_applied=False):
    vitals = dict(
        hr=_num(hr), rr=_num(rr), sbp=_num(sbp), temp=_num(temp), spo2=_num(spo2),
        avpu=(str(avpu).strip().upper() if avpu is not None else "A")
    )
    age = _num(st.session_state.get("patient_age", None))
    o2_device = st.session_state.get("o2_device", "Air")
    spo2_scale = _int(st.session_state.get("spo2_scale", 1), 1)
    behavior = st.session_state.get("pews_behavior", "Normal")
    context = dict(
        age=age,
        pregnant=(complaint == "Maternal"),
        infection=(complaint in ["Sepsis", "Other"]),
        o2_device=o2_device,
        spo2_scale=spo2_scale,
        behavior=behavior
    )
    
    # Calculate base color from scores
    base_colour, details = triage_decision(vitals, context)
    
    # Apply override if present
    final_colour = base_colour
    override_reason = ""
    if override_applied and st.session_state.get("triage_override_active", False):
        final_colour = st.session_state.get("triage_override_color", base_colour)
        override_reason = st.session_state.get("triage_override_reason", "")

    st.markdown("### Triage decision")
    triage_pill(final_colour, overridden=(final_colour != base_colour))

    # Show override info if applied
    if final_colour != base_colour:
        st.warning(f"**Override applied**: {override_reason}")
        st.info(f"Original score-based triage: **{base_colour}**")

    why = details["reasons"]
    st.caption("Why: " + (", ".join(why) if why else "All scores within normal thresholds"))

    with st.expander("Score details"):
        st.write(details)

def facility_card(row):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"#### 🏥 {row['name']}  ", unsafe_allow_html=True)
        sub = f"ETA ~ {row['eta_min']} min • {row['km']} km • ICU open: {row['ICU_open']} • Acceptance: {row['accept']}%"
        st.markdown(f'<div class="small">{sub}</div>', unsafe_allow_html=True)
        st.markdown("**Specialties**"); cap_badges(row.get("specialties",""))
        st.markdown("**High-end equipment**"); cap_badges(row.get("highend",""))
        st.markdown('<hr class="soft" />', unsafe_allow_html=True)
        cta1, cta2 = st.columns(2)
        pick = cta1.button("Select as destination", key=f"pick_{row['name']}")
        alt  = cta2.button("Add as alternate", key=f"alt_{row['name']}")
        st.markdown('</div>', unsafe_allow_html=True)
        return pick, alt

# === GEOMETRY & UTILITIES ===
def interpolate_route(lat1, lon1, lat2, lon2, n=20):
    return [[lat1 + (lat2-lat1)*i/(n-1), lon1 + (lon2-lon1)*i/(n-1)] for i in range(n)]

def traffic_factor_for_hour(hr):
    if 8 <= hr <= 10 or 17 <= hr <= 20: return 1.5
    if 7 <= hr < 8 or 10 < hr < 12 or 15 <= hr < 17: return 1.2
    return 1.0

def dist_km(lat1, lon1, lat2, lon2):
    R=6371
    dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def eta_minutes_for(km, traffic_mult, speed_kmh=36):
    """Simple ETA model (rural speeds) with traffic multiplier."""
    if km is None:
        return None
    return max(5, int(km / max(speed_kmh, 1e-6) * 60 * float(traffic_mult)))

def score_facility_for_case(f, origin_lat, origin_lon, need_caps, traffic_mult):
    """Multi-factor score: capability coverage, readiness, proximity, traffic."""
    try:
        origin_lat = float(origin_lat)
        origin_lon = float(origin_lon)
        traffic_mult = float(traffic_mult)
        
        km = dist_km(origin_lat, origin_lon, float(f["lat"]), float(f["lon"]))
        
        # Safe capability calculation
        if not need_caps:
            coverage = 1.0
        else:
            coverage = sum(int(f["caps"].get(c, 0)) for c in need_caps) / len(need_caps)
        
        # Safe numeric conversions
        icu_open = int(f.get("ICU_open", 0))
        acceptance_rate = float(f.get("acceptanceRate", 0.75))
        
        readiness = 0.5 * (min(icu_open, 3) / 3.0) + 0.5 * acceptance_rate
        proximity = max(0.0, 1.0 - km / 60.0)
        penalty = 0.0 if icu_open > 0 else -0.20
        traffic_term = 1 - (traffic_mult - 1.0) / 0.5
        
        score = 0.55 * coverage + 0.25 * readiness + 0.15 * proximity + 0.05 * traffic_term + penalty
        score = max(0.0, min(1.0, score))

        eta_min = eta_minutes_for(km, traffic_mult)
        route = interpolate_route(origin_lat, origin_lon, float(f["lat"]), float(f["lon"]), n=24)

        return dict(
            name=str(f["name"]),
            km=round(km, 1),
            eta_min=int(eta_min) if eta_min else 0,
            ICU_open=icu_open,
            accept=int(round(acceptance_rate * 100, 0)),
            specialties=", ".join([str(s) for s, v in f["specialties"].items() if v]) or "—",
            highend=", ".join([str(i) for i, v in f["highend"].items() if v]) or "—",
            score=int(round(score * 100, 0)),
            route=route,
            lat=float(f["lat"]),
            lon=float(f["lon"]),
        )
    except Exception as e:
        st.error(f"Error scoring facility {f.get('name', 'unknown')}: {str(e)}")
        return None

def rank_facilities_for_case(origin, need_caps, traffic_mult=1.0, topk=10):
    """Strict capability fit + score, then return best N with routes."""
    rows = []
    for f in st.session_state.facilities:
        if need_caps and not all(f["caps"].get(c, 0) == 1 for c in need_caps):
            continue
        scored = score_facility_for_case(f, origin[0], origin[1], need_caps, traffic_mult)
        if scored is not None:
            rows.append(scored)
    
    if not rows:
        return []
        
    rows = sorted(rows, key=lambda r: (-r["score"], r["km"]))
    return rows[:topk]

def now_ts(): return time.time()
def minutes(a,b):
    if not a or not b: return None
    return int((b-a)/60)

# === DEMO FACILITIES (East Khasi Hills) ===
EH_BASE = dict(lat_min=25.45, lat_max=25.65, lon_min=91.80, lon_max=91.95)
def rand_geo(rng):
    return (EH_BASE["lat_min"]+rng.random()*(EH_BASE["lat_max"]-EH_BASE["lat_min"]),
            EH_BASE["lon_min"]+rng.random()*(EH_BASE["lon_max"]-EH_BASE["lon_min"]))

SPECIALTIES = ["Obstetrics","Paediatrics","Cardiology","Neurology","Orthopaedics","General Surgery","Anaesthesia","ICU"]
INTERVENTIONS = ["CathLab","OBGYN_OT","CT","MRI","Dialysis","Thrombolysis","Ventilator","BloodBank","OR","Neurosurgery"]

def default_facilities(count=15):
    rng = random.Random(17)
    base_names = [
        "Civil Hospital Shillong","NEIGRIHMS","Nazareth Hospital","Ganesh Das Maternal & Child",
        "Shillong Polyclinic & Trauma Center","Smit CHC","Pynursla CHC",
        "Mawsynram PHC","Sohra Civil Hospital","Madansynram CHC","Jowai (ref) Hub","Mawlai CHC"
    ]
    names = (base_names * ((count // len(base_names)) + 1))[:count]
    fac=[]
    for idx, n in enumerate(names):
        lat, lon = rand_geo(rng)
        caps = {c:int(rng.random()<0.7) for c in ["ICU","Ventilator","BloodBank","OR","CT","Thrombolysis","OBGYN_OT","CathLab","Dialysis","Neurosurgery"]}
        spec = {s:int(rng.random()<0.6) for s in SPECIALTIES}
        hi   = {i:int(rng.random()<0.5) for i in INTERVENTIONS}
        fac.append(dict(
            name=f"{n} #{idx+1}" if names.count(n)>1 else n,
            lat=lat, lon=lon, ICU_open=rng.randint(0,4),
            acceptanceRate=round(0.7+rng.random()*0.25,2),
            caps=caps, specialties=spec, highend=hi,
            type=rng.choice(["PHC","CHC","District Hospital","Tertiary"])
        ))
    return fac

# === Schema safety helpers ===
REQ_CAPS = ["ICU","Ventilator","BloodBank","OR","CT","Thrombolysis","OBGYN_OT","CathLab","Dialysis","Neurosurgery"]

def normalize_facility(f):
    f = dict(f)
    f.setdefault("name", "Unknown Facility")
    f.setdefault("type", "PHC")
    f.setdefault("ICU_open", 0)
    f.setdefault("acceptanceRate", 0.75)
    f.setdefault("lat", 25.58)
    f.setdefault("lon", 91.89)
    caps = f.get("caps", {}) or {}
    f["caps"] = {k: int(bool(caps.get(k, 0))) for k in REQ_CAPS}
    specs = f.get("specialties", {}) or {}
    f["specialties"] = {s: int(bool(specs.get(s, 0))) for s in SPECIALTIES}
    hi = f.get("highend", {}) or {}
    f["highend"] = {i: int(bool(hi.get(i, 0))) for i in INTERVENTIONS}
    return f

def facilities_df():
    fac = [normalize_facility(x) for x in st.session_state.facilities]
    rows = [{"name": x["name"], "type": x["type"], "ICU_open": x["ICU_open"], "acceptanceRate": x["acceptanceRate"]} for x in fac]
    return pd.DataFrame(rows)

# === Synthetic data seeding ===
RESUS = ["Airway positioning","Oxygen","IV fluids","Uterotonics","TXA","Bleeding control","Antibiotics","Nebulization","Immobilization","AED/CPR"]

def seed_referrals(n=300, rng_seed=42):
    st.session_state.referrals.clear()
    rng = random.Random(rng_seed)
    conds = ["Maternal","Trauma","Stroke","Cardiac","Sepsis","Other"]
    facs  = st.session_state.facilities
    base  = time.time() - 7*24*3600  # last 7 days

    for i in range(n):
        cond = rng.choices(conds, weights=[0.22,0.23,0.18,0.18,0.14,0.05])[0]
        hr   = rng.randint(80, 145)
        sbp  = rng.randint(85, 140)
        rr   = rng.randint(14, 32)
        spo2 = rng.randint(88, 98)
        temp = round(36 + rng.random()*3, 1)
        avpu = "A"
        vit = dict(hr=hr, rr=rr, sbp=sbp, temp=temp, spo2=spo2, avpu=avpu, complaint=cond)
        color = tri_color(vit)
        severity = {"RED":"Critical", "YELLOW":"Moderate", "GREEN":"Non-critical"}[color]

        lat, lon = rand_geo(rng)
        dest = rng.choice(facs)
        dkm  = dist_km(lat, lon, dest["lat"], dest["lon"])
        ts_first = base + rng.randint(0, 7*24*3600)
        hr_of_day = datetime.fromtimestamp(ts_first).hour
        traffic_mult = traffic_factor_for_hour(hr_of_day)
        speed_kmh = rng.choice([30, 36, 45])
        eta_min = max(5, int(dkm / speed_kmh * 60 * traffic_mult))
        route = interpolate_route(lat, lon, dest["lat"], dest["lon"], n=24)

        amb_avail = (rng.random() > 0.25)
        t_dec = ts_first + rng.randint(60, 6*60)
        t_disp = t_dec + (rng.randint(2*60, 10*60) if amb_avail else rng.randint(15*60, 45*60))
        t_arr  = t_disp + (eta_min*60)
        t_hov  = t_arr + rng.randint(5*60, 20*60)

        # Provisional diagnosis
        provisional_dx = dict(
            code="", 
            label=("PPH" if cond=="Maternal" else rng.choice(["Sepsis","Head injury","STEMI","Stroke?","—"])),
            case_type=cond
        )

        st.session_state.referrals.append(dict(
            id=f"S{i:04d}",
            patient=dict(name=f"Pt{i:04d}", age=rng.randint(1,85), sex=("Female" if rng.random()<0.5 else "Male"),
                         id="", location=dict(lat=lat,lon=lon)),
            referrer=dict(name=rng.choice(["Dr. Rai","Dr. Khonglah","ANM Pynsuk"]), facility=rng.choice(
                ["PHC Mawlai","CHC Smit","CHC Pynursla","District Hospital Shillong","Tertiary Shillong Hub"]
            )),
            provisionalDx=provisional_dx,
            resuscitation=rng.sample(RESUS, rng.randint(0,3)),
            triage=dict(complaint=cond, decision=dict(color=color), hr=hr, sbp=sbp, rr=rr, temp=temp, spo2=spo2, avpu=avpu),
            clinical=dict(summary="Auto-seeded"),
            severity=severity,
            reasons=dict(severity=True, bedOrICUUnavailable=(rng.random()<0.2), specialTest=(rng.random()<0.3), requiredCapabilities=[]),
            dest=dest["name"],
            transport=dict(eta_min=eta_min, traffic=traffic_mult, speed_kmh=speed_kmh, ambulance=rng.choice(["BLS","ALS","ALS + Vent"])),
            route=route,
            times=dict(first_contact_ts=ts_first, decision_ts=t_dec, dispatch_ts=t_disp, arrive_dest_ts=t_arr, handover_ts=t_hov),
            status=rng.choice(["HANDOVER","ARRIVE_DEST","DEPART_SCENE"]),
            ambulance_available=amb_avail,
            audit_log=[]
        ))

# === SESSION STATE INITIALIZATION ===
if "facilities" not in st.session_state:
    st.session_state.facilities = default_facilities(count=15)

# Initialize session state variables
if "patient_age" not in st.session_state: 
    st.session_state.patient_age = 30
if "o2_device" not in st.session_state: 
    st.session_state.o2_device = "Air"
if "spo2_scale" not in st.session_state: 
    st.session_state.spo2_scale = 1
if "pews_behavior" not in st.session_state: 
    st.session_state.pews_behavior = "Normal"

# Triage override state
if "triage_override_active" not in st.session_state:
    st.session_state.triage_override_active = False
if "triage_override_color" not in st.session_state:
    st.session_state.triage_override_color = None
if "triage_override_reason" not in st.session_state:
    st.session_state.triage_override_reason = ""

if "referrals" not in st.session_state: 
    st.session_state.referrals = []
if "active_fac" not in st.session_state: 
    st.session_state.active_fac = st.session_state.facilities[0]["name"]

# Initialize facility matching session state
if "matched_primary" not in st.session_state:
    st.session_state.matched_primary = None
if "matched_alts" not in st.session_state:
    st.session_state.matched_alts = set()

# Normalize schema
st.session_state.facilities = [normalize_facility(x) for x in st.session_state.facilities]

# Auto-seed on first run (ensures ≥100)
if len(st.session_state.referrals) < 100:
    seed_referrals(n=300)

# === MAIN APP UI ===
st.title("AHECN – Streamlit MVP v1.8 (Score-Only Triage + Clinician Override)")
tabs = st.tabs(["Referrer","Ambulance / EMT","Receiving Hospital","Government","Data / Admin","Facility Admin"])

# ======== Referrer Tab ========
with tabs[0]:
    st.subheader("Patient & Referrer")
    
    # Patient and Referrer Details
    c1, c2, c3 = st.columns(3)
    with c1:
        p_name = st.text_input("Patient name", "John Doe")
        p_age = st.number_input("Age", 0, 120, 35)
        p_sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    with c2:
        p_id = st.text_input("Patient ID", "PT-001")
        p_lat = st.number_input("Latitude", value=25.58, format="%.6f")
        p_lon = st.number_input("Longitude", value=91.89, format="%.6f")
    with c3:
        r_name = st.text_input("Referrer name", "Dr. Smith")
        r_fac = st.text_input("Referrer facility", "PHC Mawlai")
    
    # NEW: Referrer Role Selector
    st.subheader("Referrer Role & Diagnosis")
    referrer_role = st.radio("Referrer role", ["Doctor/Physician", "ANM/ASHA/EMT"], horizontal=True)
    
    ocr = st.text_area("Clinical Notes / OCR (paste)", height=100, placeholder="Paste clinical notes, observations, or free-text assessment here...")

    # Vitals Section - REMOVED AD-HOC RED FLAGS
    st.subheader("Vitals + Scores")
    v1, v2, v3 = st.columns(3)
    with v1:
        hr = st.number_input("HR", 0, 250, 118)
        sbp = st.number_input("SBP", 0, 300, 92)
        rr = st.number_input("RR", 0, 80, 26)
        temp = st.number_input("Temp °C", 30.0, 43.0, 38.4, step=0.1)
    with v2:
        spo2 = st.number_input("SpO₂ %", 50, 100, 92)
        avpu = st.selectbox("AVPU", ["A", "V", "P", "U"], index=0)
        complaint = st.selectbox("Chief complaint", ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"], index=0)
    with v3:
        # REMOVED: rf_sbp, rf_spo2, rf_avpu, rf_seizure, rf_pph checkboxes
        st.info("**Score-based triage**\n\nTriage color determined by NEWS2/MEOWS/PEWS thresholds only")

    # Additional scoring parameters
    o2_col, scale_col, beh_col = st.columns(3)
    with o2_col:
        o2_device = st.selectbox("O₂ device", ["Air", "O2"])
        st.session_state.o2_device = o2_device
    with scale_col:
        spo2_scale = st.selectbox("SpO₂ scale (NEWS2)", [1, 2], index=0)
        st.session_state.spo2_scale = spo2_scale
    with beh_col:
        pews_beh = st.selectbox("PEWS behavior", ["Normal", "Irritable", "Lethargic"], index=0)
        st.session_state.pews_behavior = pews_beh
        st.session_state.patient_age = p_age

    # Calculate and display scores
    n_score, n_hits, n_review, n_emerg = safe_calc_NEWS2(rr, spo2, sbp, hr, temp, avpu, o2_device, spo2_scale)
    st.write(f"NEWS2: **{n_score}** {'• EMERGENCY' if n_emerg else '• review' if n_review else ''}")

    q_score, q_hits, q_high = calc_qSOFA(rr, sbp, avpu)
    if complaint in ["Sepsis", "Other"]:
        st.write(f"qSOFA: **{q_score}** {'• ≥2 high risk' if q_high else ''}")
    else:
        st.caption("qSOFA applies to sepsis/infection context")

    meows = calc_MEOWS(hr, rr, sbp, temp, spo2)
    m_band = "Red" if meows["red"] else ("Yellow" if meows["yellow"] else "Green")
    m_trig = bool(meows["red"] or meows["yellow"])
    if complaint == "Maternal":
        st.write(f"MEOWS: **{m_band}** {'• trigger' if m_trig else ''}")
    else:
        st.caption("MEOWS applies to maternal cases only")

    if p_age < 18:
        pews_sc, pews_meta, pews_high, pews_watch = calc_PEWS(p_age, rr, hr, pews_beh, spo2)
        st.write(f"PEWS: **{pews_sc}** {'• ≥6 high risk' if pews_high else ('• watch' if pews_watch else '')}")
    else:
        st.caption("PEWS disabled for ≥18y")

    # Triage decision banner
    render_triage_banner(hr, rr, sbp, temp, spo2, avpu, complaint)

    # === CLINICIAN OVERRIDE CONTROL ===
    st.subheader("Clinician Triage Override")
    
    override_col1, override_col2 = st.columns([1, 2])
    with override_col1:
        override_active = st.checkbox("Override triage decision", 
                                    value=st.session_state.triage_override_active,
                                    help="Override the score-based triage decision")
        
        if override_active:
            override_color = st.selectbox("Override to", 
                                        ["RED", "YELLOW", "GREEN"],
                                        index=0)
            st.session_state.triage_override_color = override_color
        else:
            st.session_state.triage_override_color = None
            
    with override_col2:
        if override_active:
            override_reason = st.text_area("Override reason (required)", 
                                         value=st.session_state.triage_override_reason,
                                         placeholder="Document clinical justification for override...",
                                         height=80)
            st.session_state.triage_override_reason = override_reason
            
            if not override_reason.strip():
                st.error("Please provide a reason for the triage override")
            else:
                st.success("Override will be logged in referral audit trail")
        else:
            st.info("Check to override score-based triage decision")
    
    st.session_state.triage_override_active = override_active

    # ========== STEP 1: ICD-Coded Diagnosis (Role-based) ==========
    st.subheader("Provisional Diagnosis")
    
    if referrer_role == "Doctor/Physician":
        # Doctor/Physician: Mandatory ICD selection
        st.markdown("**ICD-coded Diagnosis** <span class='required'>*</span>", unsafe_allow_html=True)
        
        # Search and filter functionality
        search_col, filter_col = st.columns([2, 1])
        with search_col:
            icd_search = st.text_input("Search ICD codes", placeholder="Type to search diagnoses...")
        with filter_col:
            show_all = st.checkbox("Show all diagnoses", value=False)
        
        # Get filtered ICD options
        icd_choices, icd_df_filt = icd_options_for(complaint if not show_all else None, p_age)
        
        # Apply search filter if provided
        if icd_search:
            icd_choices = [choice for choice in icd_choices if icd_search.lower() in choice.lower()]
            icd_df_filt = icd_df_filt[icd_df_filt["display"].str.lower().str.contains(icd_search.lower())]
        
        if icd_choices:
            chosen_icd = st.selectbox("Select ICD diagnosis", icd_choices, index=0, 
                                    help="Filtered by age and case type. Use search for more options.")
            row = icd_df_filt[icd_df_filt["display"] == chosen_icd].iloc[0]
            default_iv = [x.strip() for x in str(row.get("default_interventions", "")).split(";") if x.strip()]
            
            # Display ICD details
            st.info(f"**Selected:** {row['label']} ({row['icd_code']}) • Age range: {row['age_min']}-{row['age_max']} years")
        else:
            st.warning("No ICD codes match your search/filters. Try different criteria or check 'Show all diagnoses'.")
            chosen_icd = None
            row = None
            default_iv = []

        # Interventions checklist (auto-suggested from ICD)
        if chosen_icd and default_iv:
            st.markdown("**Suggested Interventions**")
            iv_cols = st.columns(3)
            iv_selected = []
            for i, item in enumerate(default_iv):
                if iv_cols[i % 3].checkbox(item, value=True, key=f"iv_{i}"):
                    iv_selected.append(item)
        else:
            iv_selected = []

        # Additional notes (optional for doctors)
        dx_free = st.text_input("Additional clinical notes (optional)", "")
        
        # Diagnosis payload for doctors
        if chosen_icd and row is not None:
            dx_payload = dict(code=row["icd_code"], label=row["label"], case_type=row["case_type"])
        else:
            st.error("Please select an ICD diagnosis to proceed")
            dx_payload = None

    else:
        # ANM/ASHA/EMT: Optional ICD with prominent free-text
        st.markdown("**Reason for Referral** <span class='required'>*</span>", unsafe_allow_html=True)
        
        # Free-text reason (primary for non-doctors)
        referral_reason = st.text_area("Describe the reason for referral", 
                                     placeholder="Describe symptoms, observations, and reason for transfer...",
                                     height=80)
        
        # Optional ICD selection
        with st.expander("Optional: Select ICD diagnosis (if known)"):
            icd_choices, icd_df_filt = icd_options_for(complaint, p_age)
            if icd_choices:
                chosen_icd = st.selectbox("ICD diagnosis (optional)", [""] + icd_choices, index=0)
                if chosen_icd:
                    row = icd_df_filt[icd_df_filt["display"] == chosen_icd].iloc[0]
                    default_iv = [x.strip() for x in str(row.get("default_interventions", "")).split(";") if x.strip()]
                    
                    # Optional interventions
                    st.markdown("**Suggested interventions**")
                    for i, item in enumerate(default_iv):
                        if st.checkbox(item, value=False, key=f"iv_{i}"):
                            iv_selected.append(item) if 'iv_selected' in locals() else iv_selected.extend([item])
                else:
                    row = None
                    default_iv = []
                    iv_selected = []
            else:
                st.info("No ICD suggestions for this age & case type")
                chosen_icd = None
                row = None
                default_iv = []
                iv_selected = []
        
        # Additional notes
        dx_free = st.text_input("Additional notes (optional)", "")
        
        # Diagnosis payload for non-doctors
        dx_payload = dict(code=row["icd_code"] if chosen_icd else "", 
                         label=referral_reason or (row["label"] if chosen_icd else ""), 
                         case_type=str(complaint))
        
        if not referral_reason and not chosen_icd:
            st.error("Please provide a reason for referral or select an ICD diagnosis")

    # Resuscitation interventions (common to both roles)
    st.subheader("Resuscitation / Stabilization done (tick all applied)")
    RESUS_LIST = ["Airway positioning", "Oxygen", "IV fluids", "Uterotonics", "TXA", "Bleeding control", 
                  "Antibiotics", "Nebulization", "Immobilization", "AED/CPR"]
    cols = st.columns(5)
    resus_done = []
    for i, item in enumerate(RESUS_LIST):
        if cols[i % 5].checkbox(item, value=False, key=f"resus_{i}"):
            resus_done.append(item)

    # Referral reasons and capabilities
    st.subheader("Reason(s) for referral + capabilities needed")
    c1, c2 = st.columns(2)
    with c1:
        ref_beds = st.checkbox("No ICU/bed available", False)
        ref_tests = st.checkbox("Special intervention/test required", True)
        ref_severity = True
    need_caps = []
    if ref_tests:
        st.caption("Select required capabilities for this case")
        cap_cols = st.columns(5)
        CAP_LIST = ["ICU", "Ventilator", "BloodBank", "OR", "CT", "Thrombolysis", "OBGYN_OT", "CathLab", "Dialysis", "Neurosurgery"]
        for i, cap in enumerate(CAP_LIST):
            pre = (cap in ["ICU", "BloodBank", "OBGYN_OT"]) if complaint == "Maternal" else False
            if cap_cols[i % 5].checkbox(cap, value=pre, key=f"cap_{cap}"):
                need_caps.append(cap)

    # Facility matching
    st.markdown("### Facility matching")

    if st.button("Find matched facilities"):
        # Validate diagnosis before proceeding
        if referrer_role == "Doctor/Physician" and dx_payload is None:
            st.error("Please select an ICD diagnosis to find matching facilities")
        elif referrer_role == "ANM/ASHA/EMT" and not dx_payload.get("label"):
            st.error("Please provide a reason for referral to find matching facilities")
        else:
            traffic_mult = traffic_factor_for_hour(datetime.now().hour)
            rows = rank_facilities_for_case(
                origin=(p_lat, p_lon),
                need_caps=need_caps,
                traffic_mult=traffic_mult,
                topk=10
            )

            if not rows:
                st.warning("No capability-fit facilities. Try relaxing requirements.")
            else:
                df = (
                    pd.DataFrame(rows)
                    .sort_values(["score", "km"], ascending=[False, True])
                    .head(10)
                )

                st.dataframe(
                    df[["name", "km", "eta_min", "ICU_open", "accept", "score"]]
                    .rename(columns={
                        "name": "Facility",
                        "km": "km",
                        "eta_min": "ETA (min)",
                        "ICU_open": "ICU",
                        "accept": "Accept %",
                        "score": "Score",
                    }),
                    use_container_width=True,
                )

                st.markdown("### Suggested destinations")
                st.session_state.matched_primary = None
                st.session_state.matched_alts = set()

                for _, r in df.iterrows():
                    pick, alt = facility_card(r)
                    if pick:
                        st.session_state.matched_primary = r["name"]
                    if alt:
                        st.session_state.matched_alts.add(r["name"])

                if not st.session_state.matched_primary:
                    st.session_state.matched_primary = df.iloc[0]["name"]

                show_map = st.checkbox("Show routes to suggestions", value=True)
                if show_map and st.session_state.matched_primary:
                    try:
                        primary_name = st.session_state.matched_primary
                        dest_fac = next((f for f in st.session_state.facilities if f["name"] == primary_name), None)
                        
                        if dest_fac and p_lat and p_lon:
                            origin_layer = pdk.Layer(
                                "ScatterplotLayer",
                                data=[{"lon": p_lon, "lat": p_lat}],
                                get_position="[lon, lat]",
                                get_radius=200,
                                get_fill_color=[66, 133, 244, 180],
                            )
                            dest_layer = pdk.Layer(
                                "ScatterplotLayer",
                                data=[{"lon": float(dest_fac["lon"]), "lat": float(dest_fac["lat"])}],
                                get_position="[lon, lat]",
                                get_radius=220,
                                get_fill_color=[239, 68, 68, 200],
                            )
                            path_layer = pdk.Layer(
                                "PathLayer",
                                data=[{"path": [[p_lon, p_lat], [float(dest_fac["lon"]), float(dest_fac["lat"])]]}],
                                get_path="path",
                                get_color=[16, 185, 129, 200],
                                width_scale=6,
                                width_min_pixels=3,
                            )
                            st.pydeck_chart(pdk.Deck(
                                layers=[origin_layer, dest_layer, path_layer],
                                initial_view_state=pdk.ViewState(latitude=p_lat, longitude=p_lon, zoom=9),
                                map_style="mapbox://styles/mapbox/dark-v10",
                            ))
                        else:
                            st.warning("Could not render map: missing location data")
                    except Exception as e:
                        st.error(f"Map rendering error: {str(e)}")

                st.info(
                    f"Primary: {st.session_state.matched_primary} • "
                    f"Alternates: {', '.join(sorted(st.session_state.matched_alts)) or '—'}"
                )

    # Final referral details
    st.markdown("### Referral details")
    colA, colB, colC = st.columns(3)
    with colA:
        priority = st.selectbox("Transport priority", ["Routine", "Urgent", "STAT"], index=1)
    with colB:
        amb_type = st.selectbox("Ambulance type", ["BLS", "ALS", "ALS + Vent", "Neonatal"], index=1)
    with colC:
        consent = st.checkbox("Patient/family consent obtained", value=True)

    primary = st.session_state.get("matched_primary")
    alternates = sorted(list(st.session_state.get("matched_alts", [])))

    def _save_referral(dispatch=False):
        # Validate based on role
        if referrer_role == "Doctor/Physician" and dx_payload is None:
            st.error("Please select an ICD diagnosis to create referral")
            return False
        elif referrer_role == "ANM/ASHA/EMT" and not dx_payload.get("label"):
            st.error("Please provide a reason for referral")
            return False
            
        if not primary:
            st.error("Select a primary destination from 'Find matched facilities' above.")
            return False
            
        vit = dict(hr=hr, rr=rr, sbp=sbp, temp=temp, spo2=spo2, avpu=avpu, complaint=complaint)
        
        # Calculate base triage color
        age = _num(p_age)
        context = dict(
            age=age,
            pregnant=(complaint == "Maternal"),
            infection=(complaint in ["Sepsis", "Other"]),
            o2_device=st.session_state.o2_device,
            spo2_scale=st.session_state.spo2_scale,
            behavior=st.session_state.pews_behavior
        )
        base_colour, score_details = triage_decision(vit, context)
        
        # Apply override if active
        final_colour = base_colour
        audit_log = []
        
        if st.session_state.triage_override_active and st.session_state.triage_override_color:
            final_colour = st.session_state.triage_override_color
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": "TRIAGE_OVERRIDE",
                "user": r_name,
                "details": {
                    "from": base_colour,
                    "to": final_colour,
                    "reason": st.session_state.triage_override_reason,
                    "scores": {
                        "NEWS2": score_details["NEWS2"]["score"],
                        "qSOFA": score_details["qSOFA"]["score"],
                        "MEOWS_red": len(score_details["MEOWS"]["red"]),
                        "PEWS": score_details["PEWS"]["score"]
                    }
                }
            }
            audit_log.append(audit_entry)

        # Combine interventions
        all_interventions = (iv_selected if 'iv_selected' in locals() else []) + ([dx_free] if dx_free else [])
        
        ref = dict(
            id="R" + str(int(time.time()))[-6:],
            patient=dict(name=p_name, age=int(p_age), sex=p_sex, id=p_id, location=dict(lat=float(p_lat), lon=float(p_lon))),
            referrer=dict(name=r_name, facility=r_fac, role=referrer_role),
            provisionalDx=dx_payload,
            interventions=all_interventions,
            resuscitation=resus_done,
            triage=dict(
                complaint=complaint, 
                decision=dict(
                    color=final_colour,
                    base_color=base_colour,
                    overridden=(final_colour != base_colour)
                ), 
                hr=hr, sbp=sbp, rr=rr, temp=temp, spo2=spo2, avpu=avpu,
                scores=score_details
            ),
            clinical=dict(summary=" ".join(ocr.split()[:60])),
            reasons=dict(severity=True, bedOrICUUnavailable=ref_beds, specialTest=ref_tests, requiredCapabilities=need_caps),
            dest=primary,
            alternates=alternates,
            transport=dict(priority=priority, ambulance=amb_type, consent=bool(consent)),
            times=dict(first_contact_ts=now_ts(), decision_ts=now_ts()),
            status="PREALERT",
            ambulance_available=None,
            audit_log=audit_log
        )
        if dispatch:
            ref["times"]["dispatch_ts"] = now_ts()
            ref["status"] = "DISPATCHED"
            ref["ambulance_available"] = True
            
        st.session_state.referrals.insert(0, ref)
        return True

    col1, col2 = st.columns(2)
    if col1.button("Create referral"):
        if _save_referral(dispatch=False):
            st.success(f"Referral created → {primary}")
            # Reset override after successful referral
            st.session_state.triage_override_active = False
            st.session_state.triage_override_color = None
            st.session_state.triage_override_reason = ""
    if col2.button("Create & dispatch now"):
        if _save_referral(dispatch=True):
            st.success(f"Referral created and DISPATCHED → {primary}")
            # Reset override after successful referral
            st.session_state.triage_override_active = False
            st.session_state.triage_override_color = None
            st.session_state.triage_override_reason = ""

# ======== Ambulance / EMT Tab ========
with tabs[1]:
    st.subheader("Active jobs (availability • route • live ETA)")
    avail = st.radio("Ambulance availability", ["Available", "Unavailable"], horizontal=True)

    active = [r for r in st.session_state.referrals if r["status"] in
              ["PREALERT", "DISPATCHED", "ARRIVE_SCENE", "DEPART_SCENE", "ARRIVE_DEST"]]
    if not active:
        st.info("No active jobs")
    else:
        ids = [f"{r['id']} • {r['patient']['name']} • {r['triage']['complaint']} • {r['triage']['decision']['color']}" for r in active]
        pick = st.selectbox("Select case", ids, index=0)
        r = active[ids.index(pick)]

        c1, c2, c3, c4, c5 = st.columns(5)
        if c1.button("Dispatch"):     
            r["times"]["dispatch_ts"] = now_ts()
            r["status"] = "DISPATCHED"
            r["ambulance_available"] = (avail == "Available")
            st.rerun()
        if c2.button("Arrive scene"): 
            r["times"]["arrive_scene_ts"] = now_ts()
            r["status"] = "ARRIVE_SCENE"
            st.rerun()
        if c3.button("Depart scene"): 
            r["times"]["depart_scene_ts"] = now_ts()
            r["status"] = "DEPART_SCENE"
            st.rerun()
        if c4.button("Arrive dest"):  
            r["times"]["arrive_dest_ts"] = now_ts()
            r["status"] = "ARRIVE_DEST"
            st.rerun()
        if c5.button("Handover"):     
            r["times"]["handover_ts"] = now_ts()
            r["status"] = "HANDOVER"
            st.rerun()

        st.markdown("### Route & live traffic")
        current_traffic = r["transport"].get("traffic", 1.0)
        traffic_idx = 0 if current_traffic == 1.0 else 1 if current_traffic <= 1.2 else 2
        traffic_state = st.radio("Traffic", ["Free", "Moderate", "Heavy"], index=traffic_idx, horizontal=True)
        tf = {"Free": 1.0, "Moderate": 1.2, "Heavy": 1.5}[traffic_state]
        r["transport"]["traffic"] = tf
        
        if r.get("route"):
            p1, p2 = r["route"][0], r["route"][-1]
            dkm = dist_km(p1[0], p1[1], p2[0], p2[1])
            speed = r["transport"].get("speed_kmh", 36)
            eta_min = max(5, int(dkm / speed * 60 * tf))
            r["transport"]["eta_min"] = eta_min

        left, right = st.columns([1, 3])
        with left:
            st.write(f"**ETA:** {r['transport'].get('eta_min', '—')} min")
            st.write(f"**Ambulance:** {r['transport'].get('ambulance', '—')}")
            st.write("**Triage:**")
            decision = r['triage']['decision']
            if decision.get('overridden'):
                st.markdown(f'<span class="pill {decision["color"].lower()} override-badge">{decision["color"]} (OVERRIDDEN)</span>', unsafe_allow_html=True)
                st.caption(f"Original: {decision.get('base_color', 'Unknown')}")
            else:
                triage_pill(decision['color'])

        if r.get("route"):
            try:
                path = [dict(path=[[pt[1], pt[0]] for pt in r["route"]])]
                layer = pdk.Layer("PathLayer", data=path, get_path="path", get_color=[16, 185, 129, 200], width_scale=5, width_min_pixels=3)
                v = pdk.ViewState(latitude=r["route"][0][0], longitude=r["route"][0][1], zoom=10)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=v, map_style="mapbox://styles/mapbox/dark-v10"))
            except Exception as e:
                st.error(f"Error rendering map: {str(e)}")
        else:
            st.caption("No route saved in this record.")

# ======== Receiving Hospital Tab ========
with tabs[2]:
    st.subheader("Incoming referrals & case actions")
    fac_names = [f["name"] for f in st.session_state.facilities]
    current_idx = fac_names.index(st.session_state.active_fac) if st.session_state.active_fac in fac_names else 0
    st.session_state.active_fac = st.selectbox("Facility", fac_names, index=current_idx)

    incoming = [
        r for r in st.session_state.referrals
        if r["dest"] == st.session_state.active_fac
        and r["status"] in ["PREALERT", "DISPATCHED", "ARRIVE_SCENE", "DEPART_SCENE", "ARRIVE_DEST"]
    ]

    if not incoming:
        st.info("No incoming referrals")
    else:
        for r in incoming:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{r['patient']['name']}** — {r['triage']['complaint']} ")
                    
                    # Handle both string and dictionary provisionalDx formats
                    dx = r.get("provisionalDx", {})
                    if isinstance(dx, dict):
                        dx_txt = (dx.get("code", "") + " " + dx.get("label", "")).strip()
                    else:
                        # Handle legacy string format
                        dx_txt = str(dx)
                    
                    st.write(f"| Dx: **{dx_txt or '—'}**")
                    
                    # Show referrer role if available
                    referrer_info = r.get('referrer', {})
                    if referrer_info.get('role'):
                        st.caption(f"Referrer: {referrer_info.get('name', '')} ({referrer_info.get('role', '')})")
                    
                with col2:
                    decision = r['triage']['decision']
                    if decision.get('overridden'):
                        st.markdown(f'<span class="pill {decision["color"].lower()} override-badge">{decision["color"]} (OVERRIDDEN)</span>', unsafe_allow_html=True)
                    else:
                        triage_pill(decision['color'])

                open_key = f"open_{r['id']}"
                if st.button("Open case", key=open_key):
                    # Handle both string and dictionary provisionalDx formats in ISBAR
                    dx = r.get("provisionalDx", {})
                    if isinstance(dx, dict):
                        dx_txt = (dx.get("code", "") + " " + dx.get("label", "")).strip()
                    else:
                        dx_txt = str(dx)
                    
                    # Show audit log if exists
                    if r.get('audit_log'):
                        st.markdown("#### Audit Trail")
                        for audit in r['audit_log']:
                            if audit['action'] == 'TRIAGE_OVERRIDE':
                                st.markdown(f"""
                                <div class="audit-log">
                                    <strong>🔧 Triage Override</strong><br>
                                    <small>{audit['timestamp']} by {audit['user']}</small><br>
                                    {audit['details']['from']} → {audit['details']['to']}<br>
                                    <em>Reason: {audit['details']['reason']}</em>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    isbar = f"""I: {r['patient']['name']}, {r['patient']['age']} {r['patient']['sex']}
    Dx (provisional): {dx_txt}
    S: {r['triage']['complaint']}; triage {r['triage']['decision']['color']}
    B: HR {r['triage']['hr']}, SBP {r['triage']['sbp']}, RR {r['triage']['rr']}, Temp {r['triage']['temp']}, SpO2 {r['triage']['spo2']}, AVPU {r['triage']['avpu']}
    A: Resus: {", ".join(r.get('resuscitation', [])) or "—"}
    R: {"Bed/ICU unavailable; " if r['reasons'].get('bedOrICUUnavailable') else ""}{"Special test; " if r['reasons'].get('specialTest') else ""}Severity
    Notes: {r['clinical'].get('summary', "—")}
    """
                    st.code(isbar)

                    c1, c2, c3 = st.columns(3)
                    if c1.button("Accept", key=f"acc_{r['id']}"):
                        r["status"] = "ARRIVE_DEST"
                        r["times"]["arrive_dest_ts"] = now_ts()
                        st.success("Accepted")
                        st.rerun()

                    reject_reason = c2.selectbox(
                        "Reject reason",
                        ["—", "No ICU bed", "No specialist", "Equipment down", "Over capacity", "Outside scope"],
                        key=f"rejrs_{r['id']}",
                    )
                    if c3.button("Reject", key=f"rej_{r['id']}") and reject_reason != "—":
                        r["status"] = "PREALERT"
                        r["reasons"]["rejected"] = True
                        r["reasons"]["reject_reason"] = reject_reason
                        st.warning(f"Requested divert / rejected: {reject_reason}")
                        st.rerun()
                
                st.markdown("---")

# ======== Government Tab ========
with tabs[3]:
    st.subheader("Government – Master Dashboard (SLA • Severity • Flow • Supply)")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        tri_filter = st.selectbox("Triage", ["All", "RED", "YELLOW", "GREEN"], index=0)
    with col2:
        sev_filter = st.selectbox("Severity", ["All", "Critical", "Moderate", "Non-critical"], index=0)
    with col3:
        cond_filter = st.selectbox("Condition", ["All", "Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"], index=0)

    data = st.session_state.referrals.copy()
    if tri_filter != "All": 
        data = [r for r in data if r["triage"]["decision"]["color"] == tri_filter]
    if sev_filter != "All": 
        data = [r for r in data if r.get("severity") == sev_filter]
    if cond_filter != "All": 
        data = [r for r in data if r["triage"]["complaint"] == cond_filter]

    def minutes_safe(a, b):
        return None if not a or not b else int((b - a) / 60)

    def pct(vals, p):
        arr = [v for v in vals if isinstance(v, (int, float)) and v is not None]
        return int(np.percentile(arr, p)) if arr else None

    total = len(data) or 1
    reds = [r for r in data if r["triage"]["decision"]["color"] == "RED"]
    with_disp = [r for r in data if r["times"].get("dispatch_ts")]

    # NEW: Override statistics
    overrides = [r for r in data if r["triage"]["decision"].get("overridden")]
    override_rate = len(overrides) / total * 100 if total > 0 else 0

    # KPIs
    pct_red_60 = int(100 * len([r for r in reds if r["times"].get("arrive_dest_ts")
                                and minutes_safe(r["times"]["first_contact_ts"], r["times"]["arrive_dest_ts"]) <= 60]) / (len(reds) or 1))
    pct_disp_10 = int(100 * len([r for r in with_disp if minutes_safe(r["times"]["first_contact_ts"], r["times"]["dispatch_ts"]) <= 10]) / (len(with_disp) or 1))
    
    dispatch_delays = [minutes_safe(r["times"].get("decision_ts"), r["times"].get("dispatch_ts")) for r in data]
    travel_times = [minutes_safe(r["times"].get("dispatch_ts"), r["times"].get("arrive_dest_ts")) for r in data]

    d_p50, d_p90 = pct(dispatch_delays, 50), pct(dispatch_delays, 90)
    t_p50, t_p90 = pct(travel_times, 50), pct(travel_times, 90)

    k1, k2, k3, k4 = st.columns(4)
    with k1: 
        kpi_tile("% RED ≤60m", f"{pct_red_60}%", "Critical cases arriving within 60min")
    with k2: 
        kpi_tile("% Dispatch ≤10m", f"{pct_disp_10}%", "Cases dispatched within 10min")
    with k3: 
        kpi_tile("Override Rate", f"{override_rate:.1f}%", "Triage decisions overridden")
    with k4: 
        kpi_tile("Travel P50 / P90", f"{t_p50 or '—'} / {t_p90 or '—'} min", "Travel time percentiles")

    # Charts
    st.markdown("### Severity mix")
    sev_series = pd.Series([r.get("severity", "—") for r in data]).value_counts()
    st.bar_chart(sev_series, use_container_width=True)

    st.markdown("### Triage Override Analysis")
    override_df = pd.DataFrame([{
        'Original': r['triage']['decision'].get('base_color', r['triage']['decision']['color']),
        'Final': r['triage']['decision']['color'],
        'Overridden': r['triage']['decision'].get('overridden', False)
    } for r in data])
    
    if not override_df.empty:
        override_counts = override_df['Overridden'].value_counts()
        st.bar_chart(override_counts, use_container_width=True)

    st.markdown("### Referral funnel")
    s1 = len(data)
    s2 = len([r for r in data if r["times"].get("dispatch_ts")])
    s3 = len([r for r in data if r["status"] in ["ARRIVE_DEST", "HANDOVER"]])
    funnel_df = pd.DataFrame({"stage": ["Referrals", "Dispatched", "Arrived/Handover"], "count": [s1, s2, s3]}).set_index("stage")
    st.bar_chart(funnel_df, use_container_width=True)

    st.markdown("### Case density heat-map (Hexagon)")
    if data:
        try:
            mdf = pd.DataFrame([dict(lat=r["patient"]["location"]["lat"], lon=r["patient"]["location"]["lon"]) for r in data])
            layer = pdk.Layer(
                "HexagonLayer",
                data=mdf,
                get_position='[lon, lat]',
                radius=1500, elevation_scale=4,
                elevation_range=[0, 3000],
                pickable=True, extruded=True,
            )
            v = pdk.ViewState(latitude=mdf["lat"].mean(), longitude=mdf["lon"].mean(), zoom=9)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=v, map_style="mapbox://styles/mapbox/dark-v10"))
        except Exception as e:
            st.error(f"Map error: {str(e)}")
    else:
        st.caption("No data for map.")

# ======== Data / Admin Tab ========
with tabs[4]:
    st.subheader("Seed / Import / Export (JSON & CSV)")
    
    st.markdown("#### Generate Synthetic Data")
    seed_n = st.slider("Seed referrals (synthetic)", 100, 1000, 300, step=50)
    if st.button("Seed synthetic data"):
        seed_referrals(n=seed_n)
        st.success(f"Seeded {seed_n} referrals")
        st.rerun()

    st.markdown("#### Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "Export JSON",
            data=json.dumps(dict(referrals=st.session_state.referrals, facilities=st.session_state.facilities), indent=2),
            file_name="ahecn_data.json", 
            mime="application/json"
        )
    
    with col2:
        if st.session_state.referrals:
            csv_data = pd.DataFrame(st.session_state.referrals).to_csv(index=False)
            st.download_button(
                "Export referrals (CSV)", 
                data=csv_data, 
                file_name="ahecn_referrals.csv", 
                mime="text/csv"
            )

    st.markdown("#### Import Data")
    upload = st.file_uploader("Import JSON", type=["json"])
    if upload:
        try:
            data = json.load(upload)
            st.session_state.referrals = data.get("referrals", [])
            imported_fac = data.get("facilities", st.session_state.facilities)
            st.session_state.facilities = [normalize_facility(x) for x in imported_fac]
            st.success("Imported (schema normalized)")
            st.rerun()
        except Exception as e:
            st.error(f"Import failed: {str(e)}")

# ======== Facility Admin Tab ========
with tabs[5]:
    st.subheader("Facility capabilities & readiness (edit live)")

    st.markdown("**Generate more demo facilities**")
    new_n = st.slider("Number of facilities", 10, 30, len(st.session_state.facilities), step=1)
    if st.button("Regenerate facilities"):
        st.session_state.facilities = [normalize_facility(x) for x in default_facilities(count=new_n)]
        st.success(f"Generated {new_n} facilities")
        st.rerun()

    fac_df = facilities_df()
    st.dataframe(fac_df, use_container_width=True)

    st.markdown("**Edit Facility Details**")
    target = st.selectbox("Select facility", [f["name"] for f in st.session_state.facilities])
    F = next(f for f in st.session_state.facilities if f["name"] == target)

    c1, c2 = st.columns(2)
    with c1:
        new_icu = st.number_input("ICU_open", 0, 30, value=int(F["ICU_open"]))
        new_acc = st.slider("Acceptance rate", 0.0, 1.0, value=float(F["acceptanceRate"]), step=0.01)
    with c2:
        st.caption("Toggle key specialties")
        for s in SPECIALTIES:
            F["specialties"][s] = st.checkbox(s, value=bool(F["specialties"].get(s, 0)), key=f"spec_{s}")

    st.caption("High-end interventional equipment")
    hi_cols = st.columns(5)
    for i, cap in enumerate(INTERVENTIONS):
        F["highend"][cap] = hi_cols[i % 5].checkbox(cap, value=bool(F["highend"].get(cap, 0)), key=f"hi_{cap}")

    if st.button("Update facility"):
        F["ICU_open"] = int(new_icu)
        F["acceptanceRate"] = float(new_acc)
        st.success("Facility updated")
