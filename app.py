# AHECN – Streamlit MVP v1.8 (Investor Demo + UI/UX)
# Roles: Referrer • Ambulance/EMT • Receiving Hospital • Government • Data/Admin • Facility Admin
# Region: East Khasi Hills, Meghalaya (synthetic geo + facilities)

import math, json, time, random, statistics
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="AHECN – Streamlit MVP v1.8", layout="wide")

# ---------------------- THEME & GLOBAL CSS ----------------------
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
</style>
""", unsafe_allow_html=True)

# ---------------------- SCORING ENGINES (rules, MVP-safe) ----------------------

# 0) Basic validation (keeps inputs plausible)
def _clip(v, lo, hi):
    try: 
        x = float(v)
    except Exception:
        return None
    return max(lo, min(hi, x))

def validate_vitals(hr, rr, sbp, temp, spo2):
    return dict(
        hr   = _clip(hr,   20, 240),
        rr   = _clip(rr,    5,  60),
        sbp  = _clip(sbp,  50, 260),
        temp = _clip(temp, 32,  42),
        spo2 = _clip(spo2, 50, 100),
    )

# 1) NEWS2 (simplified, consistent with RCP tables incl. O2 device & scale)
    # Coerce everything to numbers / safe strings
    rr, spo2, sbp, hr, temp = (_num(rr), _num(spo2), _num(sbp), _num(hr), _num(temp))
    avpu = "A" if avpu is None else str(avpu).strip().upper()
    spo2_scale = _int(spo2_scale, 1)
    o2_device = "Air" if not o2_device else str(o2_device).strip()

def _num(x):
    """Convert to float or return None if blank/invalid."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None

def _int(x, default=1):
    try:
        return int(str(x).strip())
    except Exception:
        return default

def calc_NEWS2(rr, spo2, sbp, hr, temp, avpu, o2_device="Air", spo2_scale=1):
    hits = []
    score = 0

    # RR
    if rr is None: pass
    elif rr <= 8:     score += 3; hits.append("NEWS2 RR ≤8 =3")
    elif 9 <= rr <=11:score += 1; hits.append("NEWS2 RR 9–11 =1")
    elif 12 <= rr <=20:           hits.append("NEWS2 RR 12–20 =0")
    elif 21 <= rr <=24:score += 2; hits.append("NEWS2 RR 21–24 =2")
    else:             score += 3; hits.append("NEWS2 RR ≥25 =3")

    # SpO2 scale 1 (default)
    def spo2_s1(s):
        if s <= 91: return 3
        if 92 <= s <= 93: return 2
        if 94 <= s <= 95: return 1
        return 0
    # SpO2 scale 2 (COPD/CRF)
    def spo2_s2(s):
        if s <= 83: return 3
        if 84 <= s <= 85: return 2
        if 86 <= s <= 90: return 1
        if 91 <= s <= 92: return 0
        return 0
    if spo2 is not None:
        pts = spo2_s1(spo2) if int(spo2_scale)==1 else spo2_s2(spo2)
        score += pts; 
        hits.append(f"NEWS2 SpO₂ (scale {spo2_scale}) +{pts}")
    if str(o2_device).lower() != "air":
        score += 2; hits.append("NEWS2 Supplemental O₂ +2")

    # SBP
    if sbp is not None:
        if sbp <= 90:       score += 3; hits.append("NEWS2 SBP ≤90 =3")
        elif 91 <= sbp <=100:score += 2; hits.append("NEWS2 SBP 91–100 =2")
        elif 101 <= sbp <=110:score += 1; hits.append("NEWS2 SBP 101–110 =1")
        elif 111 <= sbp <=219:                     hits.append("NEWS2 SBP 111–219 =0")
        else:               score += 3; hits.append("NEWS2 SBP ≥220 =3")

    # HR
    if hr is not None:
        if hr <= 40:        score += 3; hits.append("NEWS2 HR ≤40 =3")
        elif 41 <= hr <= 50:score += 1; hits.append("NEWS2 HR 41–50 =1")
        elif 51 <= hr <= 90:                    hits.append("NEWS2 HR 51–90 =0")
        elif 91 <= hr <=110:score += 1; hits.append("NEWS2 HR 91–110 =1")
        elif 111 <= hr <=130:score += 2; hits.append("NEWS2 HR 111–130 =2")
        else:               score += 3; hits.append("NEWS2 HR ≥131 =3")

    # Temp
    if temp is not None:
        if temp <= 35.0:        score += 3; hits.append("NEWS2 Temp ≤35.0 =3")
        elif 35.1 <= temp <= 36.0:score += 1; hits.append("NEWS2 Temp 35.1–36.0 =1")
        elif 36.1 <= temp <= 38.0:                    hits.append("NEWS2 Temp 36.1–38.0 =0")
        elif 38.1 <= temp <= 39.0:score += 1; hits.append("NEWS2 Temp 38.1–39.0 =1")
        else:                    score += 2; hits.append("NEWS2 Temp ≥39.1 =2")

    # AVPU
    if str(avpu).upper() != "A":
        score += 3; hits.append("NEWS2 AVPU ≠ A =3")

    return score, hits, (score>=5), (score>=7)  # (score, explainers, review, urgent)

# 2) qSOFA
    rr, sbp = _num(rr), _num(sbp)
    avpu = "A" if avpu is None else str(avpu).strip().upper()

def calc_qSOFA(rr, sbp, avpu):
    hits = []
    score = 0
    if rr is not None and rr >= 22: score += 1; hits.append("qSOFA RR ≥22")
    if sbp is not None and sbp <= 100: score += 1; hits.append("qSOFA SBP ≤100")
    if str(avpu).upper() != "A": score += 1; hits.append("qSOFA altered mentation")
    return score, hits, (score>=2)

# 3) MEOWS (very simplified, colour-bands; any RED ⇒ escalate)
    hr, rr, sbp, temp, spo2 = _num(hr), _num(rr), _num(sbp), _num(temp), _num(spo2)

def calc_MEOWS(hr, rr, sbp, temp, spo2, red_flags=None):
    red = []; yellow = []
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
    if red_flags:
        red += red_flags
    return dict(red=red, yellow=yellow)

# 4) PEWS (age-banded, simplified 0–6)
def _band(x, ylo, yhi, rlo, rhi):
    """Return 2 if in red range, 1 if in yellow range, else 0."""
    x = _num(x)
    if x is None:
        return 0
    if x >= rhi or x <= rlo:
        return 2
    if x >= yhi or x <= ylo:
        return 1
    return 0

def calc_PEWS(age, rr, hr, behavior="Normal", spo2=None):
    # Coerce safely
    age  = _num(age)
    rr   = _num(rr)
    hr   = _num(hr)
    spo2 = _num(spo2)

    if age is None:
        return 0, {"detail": "age missing"}, False, False

    # Default bands by broad age category
    if age < 1:         # Infant
        rr_y, rr_r = (40, 50), (50, 60)
        hr_y, hr_r = (140, 160), (160, 200)
    elif age < 5:       # Toddler
        rr_y, rr_r = (30, 40), (40, 60)
        hr_y, hr_r = (130, 150), (150, 200)
    elif age < 12:      # Child
        rr_y, rr_r = (24, 30), (30, 60)
        hr_y, hr_r = (120, 140), (140, 200)
    else:               # Adolescent
        rr_y, rr_r = (20, 24), (24, 60)
        hr_y, hr_r = (110, 130), (130, 200)

    sc = 0
    sc += _band(rr, rr_y[0], rr_y[1], rr_r[0], rr_r[1])
    sc += _band(hr, hr_y[0], hr_y[1], hr_r[0], hr_r[1])

    if spo2 is not None:
        sc += 2 if spo2 < 92 else (1 if spo2 < 95 else 0)

    beh = str(behavior or "Normal").lower()
    if beh == "lethargic":
        sc += 2
    elif beh == "irritable":
        sc += 1

    return sc, {"age": age}, (sc >= 6), (sc >= 4)


# 5) Master decision with fail-safe bias (explainable)
def triage_decision(vitals, flags, context):
    """
    vitals: dict(hr, rr, sbp, temp, spo2, avpu)
    flags : dict(seizure, pph)
    context: dict(age, pregnant, infection, o2_device, spo2_scale)
    """
    v = validate_vitals(vitals.get("hr"), vitals.get("rr"), vitals.get("sbp"),
                        vitals.get("temp"), vitals.get("spo2"))
    avpu = vitals.get("avpu","A")
    reasons = []

    # Red flags (fail-safe)
    if v["sbp"] is not None and v["sbp"] < 90: reasons.append("SBP<90")
    if v["spo2"] is not None and v["spo2"] < 90: reasons.append("SpO₂<90%")
    if str(avpu).upper() != "A": reasons.append("Altered mentation (AVPU)")
    if flags.get("seizure"): reasons.append("Seizure")
    if flags.get("pph"): reasons.append("Post-partum haemorrhage")

    # Scores (gated)
    news2_score, news2_hits, news2_review, news2_urgent = calc_NEWS2(
        v["rr"], v["spo2"], v["sbp"], v["hr"], v["temp"], avpu,
        context.get("o2_device","Air"), context.get("spo2_scale",1)
    )
    q_score, q_hits, q_high = calc_qSOFA(v["rr"], v["sbp"], avpu) if context.get("infection") else (0,[],False)
    meows = calc_MEOWS(v["hr"], v["rr"], v["sbp"], v["temp"], v["spo2"]) if context.get("pregnant") else dict(red=[],yellow=[])
    pews_sc, pews_meta, pews_high, pews_watch = calc_PEWS(context.get("age"), v["rr"], v["hr"], context.get("behavior","Normal"), v["spo2"]) if (context.get("age") is not None and context.get("age")<18) else (0,{},False,False)

    # Colour logic
    colour = "GREEN"
    if reasons or news2_urgent or q_high or (len(meows["red"])>0) or pews_high:
        colour = "RED"
    elif news2_review or (len(meows["yellow"])>0) or pews_watch:
        colour = "YELLOW"

    details = {
        "NEWS2": dict(score=news2_score, hits=news2_hits, review=news2_review, urgent=news2_urgent),
        "qSOFA": dict(score=q_score, hits=q_hits, high=q_high),
        "MEOWS": meows,
        "PEWS": dict(score=pews_sc, high=pews_high, watch=pews_watch),
        "reasons": reasons
    }
    return colour, details

# ---------------------- UI HELPERS ----------------------
def triage_pill(color:str):
    c = (color or "").upper()
    cls = "red" if c=="RED" else "yellow" if c=="YELLOW" else "green"
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

# ---- TRIAGE BANNER HELPER (uses rule engine, hardened) ----
def render_triage_banner(hr, rr, sbp, temp, spo2, avpu,
                         rf_sbp, rf_spo2, rf_avpu, rf_seizure, rf_pph, complaint):
    # Build inputs (coerce to safe types)
    vitals = dict(
        hr=_num(hr),
        rr=_num(rr),
        sbp=_num(sbp),
        temp=_num(temp),
        spo2=_num(spo2),
        avpu=(str(avpu).strip().upper() if avpu is not None else "A")
    )
    flags = dict(seizure=bool(rf_seizure), pph=bool(rf_pph))

    # Pull context safely from session (with defaults)
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

    # Decide + explain
    colour, details = triage_decision(vitals, flags, context)

    st.markdown("### Triage decision")
    triage_pill(colour)

    why = []
    if details["reasons"]:
        why += details["reasons"]
    if details["NEWS2"]["urgent"]:
        why.append(f"NEWS2 {details['NEWS2']['score']} (≥7)")
    elif details["NEWS2"]["review"]:
        why.append(f"NEWS2 {details['NEWS2']['score']} (≥5)")
    if details["qSOFA"]["high"]:
        why.append(f"qSOFA {details['qSOFA']['score']} (≥2)")
    if context.get("pregnant") and details["MEOWS"]["red"]:
        why.append("MEOWS red band")
    if (age is not None and age < 18) and details["PEWS"]["high"]:
        why.append(f"PEWS {details['PEWS']['score']} (≥6)")

    st.caption("Why: " + (", ".join(why) if why else "thresholds not met"))

    with st.expander("Score details"):
        st.write(details)


def facility_card(row):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"#### 🏥 {row['name']}  ", unsafe_allow_html=True)
        sub = f"ETA ~ {row['eta_min']} min • {row['km']} km • ICU open: {row['ICU_open']} • Acceptance: {row['accept']}%"
        st.markdown(f'<div class="small">{sub}</div>', unsafe_allow_html=True)
        st.markdown("**Specialties**")
        cap_badges(row.get("specialties",""))
        st.markdown("**High-end equipment**")
        cap_badges(row.get("highend",""))
        st.markdown('<hr class="soft" />', unsafe_allow_html=True)
        cta1, cta2 = st.columns(2)
        pick = cta1.button("Select as destination", key=f"pick_{row['name']}")
        alt  = cta2.button("Add as alternate", key=f"alt_{row['name']}")
        st.markdown('</div>', unsafe_allow_html=True)
        return pick, alt

# ---------------------- GEOMETRY & UTILITIES ----------------------
def interpolate_route(lat1, lon1, lat2, lon2, n=20):
    # Simple straight-line polyline (demo). In pilots, swap with OSRM/Mapbox.
    return [[lat1 + (lat2-lat1)*i/(n-1), lon1 + (lon2-lon1)*i/(n-1)] for i in range(n)]

def traffic_factor_for_hour(hr):
    # crude “rush-hour” model
    if 8 <= hr <= 10 or 17 <= hr <= 20: return 1.5   # heavy
    if 7 <= hr < 8 or 10 < hr < 12 or 15 <= hr < 17: return 1.2   # moderate
    return 1.0  # free

def dist_km(lat1, lon1, lat2, lon2):
    R=6371
    dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def now_ts(): return time.time()
def minutes(a,b):
    if not a or not b: return None
    return int((b-a)/60)

# ---------------------- SCORES (standards-aligned) ----------------------
def news2_scale1_spo2(s): return 3 if s<=91 else 2 if s<=93 else 1 if s<=95 else 0
def news2_scale2_spo2(s): return 3 if s<=83 else 2 if s<=85 else 1 if s<=86 else 0 if s<=91 else 1
def calc_NEWS2(rr, spo2, sbp, hr, temp, avpu, o2_device, spo2_scale):
    sc=0
    sc += 3 if rr<=8 else 1 if rr<=11 else 0 if rr<=20 else 2 if rr<=24 else 3
    sc += news2_scale2_spo2(spo2) if spo2_scale==2 else news2_scale1_spo2(spo2)
    if o2_device=="O2": sc+=2
    sc += 3 if sbp<=90 else 2 if sbp<=100 else 1 if sbp<=110 else 3 if sbp>219 else 0
    sc += 3 if hr<=40 else 1 if hr<=50 else 0 if hr<=90 else 1 if hr<=110 else 2 if hr<=130 else 3
    sc += 3 if temp<=35 else 1 if temp<=36 else 0 if temp<=38 else 1 if temp<=39 else 2
    if avpu!="A": sc+=3
    return sc, (sc>=5), (sc>=7)

def calc_qSOFA(rr, sbp, avpu):
    s = (1 if rr>=22 else 0) + (1 if sbp<=100 else 0) + (1 if avpu!="A" else 0)
    return s, (s>=2)

def calc_MEOWS(rr, spo2, hr, sbp, temp, avpu):
    def band(v, rules):
        for t,c in rules:
            if t(v): return c
        return "green"
    m=dict(
        rr   = band(rr,  [(lambda v:v<10,"red"),(lambda v:v<=20,"green"),(lambda v:v<=30,"yellow"),(lambda v:v>30,"red")]),
        spo2 = band(spo2,[(lambda v:v<92,"red"),(lambda v:v<95,"yellow"),(lambda v:v>=95,"green")]),
        hr   = band(hr,  [(lambda v:v<50,"red"),(lambda v:v<=100,"green"),(lambda v:v<=120,"yellow"),(lambda v:v>120,"red")]),
        sbp  = band(sbp, [(lambda v:v<90,"red"),(lambda v:v<=140,"green"),(lambda v:v<=160,"yellow"),(lambda v:v>160,"red")]),
        temp = band(temp,[(lambda v:v<35,"red"),(lambda v:v<=38,"green"),(lambda v:v<39,"yellow"),(lambda v:v>=39,"red")]),
        avpu = "green" if avpu=="A" else "red"
    )
    reds=sum(1 for v in m.values() if v=="red"); yell=sum(1 for v in m.values() if v=="yellow")
    trig=(reds>=1 or yell>=2)
    return ("Red" if reds else ("Yellow" if trig else "Green")), trig

def calc_PEWS(age, age_band, rr, hr, spo2, o2_device, behavior, cap_refill=2):
    if age>=18: return False, None, False
    beh = 1 if behavior=="Irritable" else 2 if behavior=="Lethargic" else 0
    crt = 2 if cap_refill>2 else 0
    hiRR = dict(Infant=50,Toddler=40,Child=30,Adolescent=25).get(age_band,30)
    hiHR = dict(Infant=180,Toddler=160,Child=140,Adolescent=120).get(age_band,120)
    s=beh+crt
    if rr>hiRR: s+=2
    s += 2 if spo2<92 else 1 if spo2<95 else 0
    if o2_device=="O2": s+=2
    if hr>hiHR: s+=2
    return True, s, (s>=8)

def tri_color(v):
    red = v["rf_sbp"] or v["rf_spo2"] or v["rf_avpu"] or v["rf_seizure"] or v["rf_pph"]
    red = red or (v["sbp"]<90 or v["spo2"]<90 or v["rr"]>30 or v["rr"]<8 or v["hr"]>130 or v["temp"]>39.5 or v["temp"]<35)
    if not red:
        yellow = (v["hr"]>110 or v["rr"]>22 or v["spo2"]<93 or v["sbp"]<100 or v["temp"]>38.5) or (v["complaint"]=="Cardiac")
        return "YELLOW" if yellow else "GREEN"
    if v["complaint"]=="Maternal" and v["rf_pph"]: return "RED"
    return "RED"

# ---------------------- DEMO FACILITIES (East Khasi Hills) ----------------------
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

# ---- Schema safety helpers ----
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

# ---------- Synthetic data seeding ----------
RESUS = ["Airway positioning","Oxygen","IV fluids","Uterotonics","TXA","Bleeding control","Antibiotics","Nebulization","Immobilization","AED/CPR"]

def seed_referrals(n=300, rng_seed=42):
    st.session_state.referrals.clear()
    rng = random.Random(rng_seed)
    conds = ["Maternal","Trauma","Stroke","Cardiac","Sepsis","Other"]
    facs  = st.session_state.facilities
    base  = time.time() - 7*24*3600  # last 7 days

    for i in range(n):
        cond = rng.choices(conds, weights=[0.22,0.23,0.18,0.18,0.14,0.05])[0]
        # vitals (rough distributions)
        hr   = rng.randint(80, 145)
        sbp  = rng.randint(85, 140)
        rr   = rng.randint(14, 32)
        spo2 = rng.randint(88, 98)
        temp = round(36 + rng.random()*3, 1)
        avpu = "A"
        rf   = dict(
            rf_sbp   = (sbp < 90 and rng.random() < 0.6),
            rf_spo2  = (spo2 < 90 and rng.random() < 0.6),
            rf_avpu  = False, rf_seizure=False,
            rf_pph   = (cond=="Maternal" and rng.random()<0.35)
        )
        vit = dict(hr=hr, rr=rr, sbp=sbp, temp=temp, spo2=spo2, avpu=avpu, complaint=cond, **rf)
        color = tri_color(vit)                        # RED/YELLOW/GREEN
        severity = {"RED":"Critical", "YELLOW":"Moderate", "GREEN":"Non-critical"}[color]

        # geo + facility
        lat, lon = rand_geo(rng)
        dest = rng.choice(facs)
        dkm  = dist_km(lat, lon, dest["lat"], dest["lon"])
        # simple ETA from speed (rural avg ~36 km/h) with traffic multiplier
        ts_first = base + rng.randint(0, 7*24*3600)
        hr_of_day = datetime.fromtimestamp(ts_first).hour
        traffic_mult = traffic_factor_for_hour(hr_of_day)          # 1.0/1.2/1.5
        speed_kmh = rng.choice([30, 36, 45])                       # simulate road mix
        eta_min = max(5, int(dkm / speed_kmh * 60 * traffic_mult))

        # route polyline (for pydeck PathLayer)
        route = interpolate_route(lat, lon, dest["lat"], dest["lon"], n=24)

        # ambulance lifecycle
        amb_avail = (rng.random() > 0.25)
        t_dec = ts_first + rng.randint(60, 6*60)
        t_disp = t_dec + (rng.randint(2*60, 10*60) if amb_avail else rng.randint(15*60, 45*60))
        t_arr  = t_disp + (eta_min*60)
        t_hov  = t_arr + rng.randint(5*60, 20*60)

        st.session_state.referrals.append(dict(
            id=f"S{i:04d}",
            patient=dict(name=f"Pt{i:04d}", age=rng.randint(1,85), sex=("Female" if rng.random()<0.5 else "Male"),
                         id="", location=dict(lat=lat,lon=lon)),
            referrer=dict(name=rng.choice(["Dr. Rai","Dr. Khonglah","ANM Pynsuk"]), facility=rng.choice(
                ["PHC Mawlai","CHC Smit","CHC Pynursla","District Hospital Shillong","Tertiary Shillong Hub"]
            )),
            provisionalDx=("PPH" if cond=="Maternal" else rng.choice(["Sepsis","Head injury","STEMI","Stroke?","—"])),
            resuscitation=rng.sample(RESUS, rng.randint(0,3)),
            triage=dict(complaint=cond, decision=dict(color=color), hr=hr, sbp=sbp, rr=rr, temp=temp, spo2=spo2, avpu=avpu),
            clinical=dict(summary="Auto-seeded"),
            severity=severity,      # Critical / Moderate / Non-critical
            reasons=dict(severity=True, bedOrICUUnavailable=(rng.random()<0.2), specialTest=(rng.random()<0.3), requiredCapabilities=[]),
            dest=dest["name"],
            transport=dict(eta_min=eta_min, traffic=traffic_mult, speed_kmh=speed_kmh, ambulance=rng.choice(["BLS","ALS","ALS + Vent"])),
            route=route,            # list of [lat, lon] points
            times=dict(first_contact_ts=ts_first, decision_ts=t_dec, dispatch_ts=t_disp, arrive_dest_ts=t_arr, handover_ts=t_hov),
            status=rng.choice(["HANDOVER","ARRIVE_DEST","DEPART_SCENE"]),   # mix
            ambulance_available=amb_avail
        
        ))

# ---------------------- SESSION ----------------------
if "facilities" not in st.session_state:
    st.session_state.facilities = default_facilities(count=15)
if "referrals" not in st.session_state:
    st.session_state.referrals = []
if "active_fac" not in st.session_state:
    st.session_state.active_fac = st.session_state.facilities[0]["name"]

# normalize schema
st.session_state.facilities = [normalize_facility(x) for x in st.session_state.facilities]

# auto-seed on first run (ensures ≥100)
if len(st.session_state.referrals) < 100:
    seed_referrals(n=300)

# ---------------------- UI TABS ----------------------
st.title("AHECN – Streamlit MVP v1.8 (East Khasi Hills)")

tabs = st.tabs(["Referrer","Ambulance / EMT","Receiving Hospital","Government","Data / Admin","Facility Admin"])

# ======== Referrer ========
with tabs[0]:
    st.subheader("Patient & Referrer")
    c1,c2,c3 = st.columns(3)
    with c1:
        p_name = st.text_input("Patient name","Sita Devi")
        p_age  = st.number_input("Age",0,120,28)
        p_sex  = st.selectbox("Sex",["Female","Male","Other"])
    with c2:
        p_id   = st.text_input("Patient ID (ABDM/Local)","")
        r_name = st.text_input("Referrer name","Dr. Rao / ASHA Poonam")
        r_fac  = st.text_input("Referrer facility","PHC Mawlai")
    with c3:
        p_lat  = st.number_input("Lat",value=25.58,format="%.4f")
        p_lon  = st.number_input("Lon",value=91.89,format="%.4f")
        p_dx   = st.text_input("Provisional diagnosis","PPH; suspected retained placenta")
    ocr = st.text_area("Notes / OCR (paste)",height=100)

    st.subheader("Vitals + Scores")
    v1,v2,v3 = st.columns(3)
    with v1:
        hr   = st.number_input("HR", 0,250,118)
        sbp  = st.number_input("SBP",0,300,92)
        rr   = st.number_input("RR", 0,80,26)
        temp = st.number_input("Temp °C",30.0,43.0,38.4,step=0.1)
    with v2:
        spo2 = st.number_input("SpO₂ %",50,100,92)
        avpu = st.selectbox("AVPU",["A","V","P","U"],index=0)
        complaint = st.selectbox("Chief complaint",["Maternal","Trauma","Stroke","Cardiac","FeverConfusion","Sepsis","Other"],index=0)
        rf_sbp = st.checkbox("Red flag: SBP <90",False)
    with v3:
        rf_spo2 = st.checkbox("Red flag: SpO₂ <90%",False)
        rf_avpu = st.checkbox("Red flag: AVPU ≠ A",False)
        rf_seizure = st.checkbox("Red flag: Seizure",False)
        rf_pph = st.checkbox("Red flag: PPH",value=("Maternal" in complaint))

    o2_device = st.selectbox("O₂ device",["Air","O2"])
    spo2_scale= st.selectbox("SpO₂ scale (NEWS2)",[1,2],index=0)
    pews_ageband = st.selectbox("PEWS age band",["Infant","Toddler","Child","Adolescent"],index=2)
    pews_beh     = st.selectbox("PEWS behavior",["Normal","Irritable","Lethargic"],index=0)

    # Scores
    n_score, n_review, n_emerg = calc_NEWS2(rr,spo2,sbp,hr,temp,avpu,o2_device,spo2_scale)
    st.write(f"NEWS2: **{n_score}** {'• EMERGENCY' if n_emerg else '• review' if n_review else ''}")
    q_score, q_high = calc_qSOFA(rr,sbp,avpu)
    st.write(f"qSOFA: **{q_score}** {'• ≥2 high risk' if q_high else ''}")
    m_band, m_trig = calc_MEOWS(rr,spo2,hr,sbp,temp,avpu)
    st.write(f"MEOWS: **{m_band}** {'• trigger' if m_trig else ''}")
    if p_age < 18:
        _, pews_score, pews_high = calc_PEWS(p_age,pews_ageband,rr,hr,spo2,o2_device,pews_beh)
        st.write(f"PEWS: **{pews_score}** {'• ≥8 high risk' if pews_high else ''}")
    else:
        st.caption("PEWS disabled for ≥18y")

    # Hero triage banner
    render_triage_banner(hr, rr, sbp, temp, spo2, avpu, rf_sbp, rf_spo2, rf_avpu, rf_seizure, rf_pph, complaint)
        
    # Resuscitation interventions
    st.subheader("Resuscitation / Stabilization done (tick all applied)")
    cols = st.columns(5)
    RESUS_LIST = ["Airway positioning","Oxygen","IV fluids","Uterotonics","TXA","Bleeding control","Antibiotics","Nebulization","Immobilization","AED/CPR"]
    resus_done = [r for i,r in enumerate(RESUS_LIST) if cols[i%5].checkbox(r, False)]

    st.subheader("Reason(s) for referral + capabilities needed")
    c1,c2 = st.columns(2)
    with c1:
        ref_beds  = st.checkbox("No ICU/bed available",False)
        ref_tests = st.checkbox("Special intervention/test required",True)
        ref_severity = True
    need_caps=[]
    if ref_tests:
        st.caption("Select required capabilities for this case")
        cap_cols = st.columns(5)
        CAP_LIST = ["ICU","Ventilator","BloodBank","OR","CT","Thrombolysis","OBGYN_OT","CathLab","Dialysis","Neurosurgery"]
        for i,cap in enumerate(CAP_LIST):
            pre = (cap in ["ICU","BloodBank","OBGYN_OT"]) if complaint=="Maternal" else False
            if cap_cols[i%5].checkbox(cap, value=pre, key=f"cap_{cap}"):
                need_caps.append(cap)

    # Facility matching (cards)
    st.markdown("### Facility matching")
    if st.button("Find matched facilities"):
        rows=[]
        for f in st.session_state.facilities:
            caps_ok = all(f["caps"].get(c,0)==1 for c in need_caps)
            if not caps_ok: continue
            km = dist_km(p_lat, p_lon, f["lat"], f["lon"])
            ready = (min(f["ICU_open"],3)/3)*0.5 + f["acceptanceRate"]*0.5
            proximity = max(0, 1-(km/60))
            score = 0.6*ready + 0.3*proximity + 0.1
            rows.append(dict(
                name=f["name"], km=round(km,1),
                eta_min=int(km/0.6) if km>0 else 5,
                ICU_open=f["ICU_open"],
                accept=int(round(f["acceptanceRate"]*100,0)),
                specialties=", ".join([s for s,v in f["specialties"].items() if v]) or "—",
                highend=", ".join([i for i,v in f["highend"].items() if v]) or "—",
                score=int(round(score*100,0))
            ))
        if not rows:
            st.warning("No capability-fit facilities. Try relaxing requirements.")
        else:
            ranked = pd.DataFrame(rows).sort_values(["score","km"], ascending=[False,True]).head(10)
            st.session_state["_matched_primary"]=None
            st.session_state["_matched_alts"]=set()
            st.markdown("### Suggested destinations")
            for _, r in ranked.iterrows():
                pick, alt = facility_card(r)
                if pick: st.session_state["_matched_primary"]=r["name"]
                if alt:  st.session_state["_matched_alts"].add(r["name"])
            if not st.session_state["_matched_primary"]:
                st.session_state["_matched_primary"] = ranked.iloc[0]["name"]
            st.info(f"Primary: {st.session_state['_matched_primary']} • Alternates: {', '.join(st.session_state['_matched_alts']) or '—'}")

    st.markdown("### Referral details")
    colA, colB, colC = st.columns(3)
    with colA:
        priority = st.selectbox("Transport priority", ["Routine","Urgent","STAT"], index=1)
    with colB:
        amb_type = st.selectbox("Ambulance type", ["BLS","ALS","ALS + Vent","Neonatal"], index=1)
    with colC:
        consent = st.checkbox("Patient/family consent obtained", value=True)

    primary = st.session_state.get("_matched_primary")
    alternates = sorted(list(st.session_state.get("_matched_alts", [])))

    def _save_referral(dispatch=False):
        if not primary:
            st.error("Select a primary destination from 'Find matched facilities' above.")
            return
        vit=dict(hr=hr, rr=rr, sbp=sbp, temp=temp, spo2=spo2, avpu=avpu,
                 rf_sbp=rf_sbp, rf_spo2=rf_spo2, rf_avpu=rf_avpu, rf_seizure=rf_seizure, rf_pph=rf_pph, complaint=complaint)
        ref = dict(
            id="R"+str(int(time.time()))[-6:],
            patient=dict(name=p_name, age=int(p_age), sex=p_sex, id=p_id, location=dict(lat=float(p_lat), lon=float(p_lon))),
            referrer=dict(name=r_name, facility=r_fac),
            provisionalDx=p_dx,
            resuscitation=resus_done,
            triage=dict(complaint=complaint, decision=dict(color=tri_color(vit)), hr=hr, sbp=sbp, rr=rr, temp=temp, spo2=spo2, avpu=avpu),
            clinical=dict(summary=" ".join(ocr.split()[:60])),
            reasons=dict(severity=True, bedOrICUUnavailable=ref_beds, specialTest=ref_tests, requiredCapabilities=need_caps),
            dest=primary,
            alternates=alternates,
            transport=dict(priority=priority, ambulance=amb_type, consent=bool(consent)),
            times=dict(first_contact_ts=now_ts(), decision_ts=now_ts()),
            status="PREALERT",
            ambulance_available=None
        )
        if dispatch:
            ref["times"]["dispatch_ts"]=now_ts()
            ref["status"]="DISPATCHED"
            ref["ambulance_available"]=True
        st.session_state.referrals.insert(0, ref)
        st.success(f"Referral {ref['id']} → {primary} created" + (" and DISPATCHED" if dispatch else ""))

    col1, col2 = st.columns(2)
    if col1.button("Create referral"):
        _save_referral(dispatch=False)
    if col2.button("Create & dispatch now"):
        _save_referral(dispatch=True)

# ======== Ambulance / EMT ========
with tabs[1]:
    st.subheader("Active jobs (availability • route • live ETA)")
    avail = st.radio("Ambulance availability", ["Available","Unavailable"], horizontal=True)

    active = [r for r in st.session_state.referrals if r["status"] in
              ["PREALERT","DISPATCHED","ARRIVE_SCENE","DEPART_SCENE","ARRIVE_DEST"]]
    if not active:
        st.info("No active jobs")
    else:
        # pick one to visualize
        ids = [f"{r['id']} • {r['patient']['name']} • {r['triage']['complaint']} • {r['triage']['decision']['color']}" for r in active]
        pick = st.selectbox("Select case", ids, index=0)
        r = active[ids.index(pick)]

        # stage buttons
        c1,c2,c3,c4,c5 = st.columns(5)
        if c1.button("Dispatch"):     r["times"]["dispatch_ts"]=now_ts(); r["status"]="DISPATCHED"; r["ambulance_available"]=(avail=="Available")
        if c2.button("Arrive scene"): r["times"]["arrive_scene_ts"]=now_ts(); r["status"]="ARRIVE_SCENE"
        if c3.button("Depart scene"): r["times"]["depart_scene_ts"]=now_ts(); r["status"]="DEPART_SCENE"
        if c4.button("Arrive dest"):  r["times"]["arrive_dest_ts"]=now_ts(); r["status"]="ARRIVE_DEST"
        if c5.button("Handover"):     r["times"]["handover_ts"]=now_ts();  r["status"]="HANDOVER"

        # Live traffic toggle (recompute ETA)
        st.markdown("### Route & live traffic")
        traffic_state = st.radio("Traffic", ["Free","Moderate","Heavy"],
                                 index=0 if r["transport"].get("traffic",1.0)==1.0 else 1 if r["transport"]["traffic"]<=1.2 else 2,
                                 horizontal=True)
        tf = {"Free":1.0,"Moderate":1.2,"Heavy":1.5}[traffic_state]
        r["transport"]["traffic"] = tf
        # recompute ETA from stored distance (approx from route endpoints)
        if r.get("route"):
            p1, p2 = r["route"][0], r["route"][-1]
            dkm = dist_km(p1[0],p1[1],p2[0],p2[1])
            speed = r["transport"].get("speed_kmh", 36)
            eta_min = max(5, int(dkm / speed * 60 * tf))
            r["transport"]["eta_min"] = eta_min

        # Show ETA + triage
        left, right = st.columns([1,3])
        with left:
            st.write(f"**ETA:** {r['transport'].get('eta_min','—')} min")
            st.write(f"**Ambulance:** {r['transport'].get('ambulance','—')}")
            st.write("**Triage:**"); triage_pill(r['triage']['decision']['color'])

        # Map with route (pydeck PathLayer)
        if r.get("route"):
            path = [dict(path=[[pt[1], pt[0]] for pt in r["route"]])]  # lon,lat order for pydeck
            layer = pdk.Layer(
                "PathLayer",
                data=path,
                get_path="path",
                get_color=[16,185,129,200],
                width_scale=5, width_min_pixels=3,
            )
            v = pdk.ViewState(latitude=r["route"][0][0], longitude=r["route"][0][1], zoom=10)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=v, map_style="mapbox://styles/mapbox/dark-v10"))
        else:
            st.caption("No route saved in this record.")

# ======== Receiving Hospital ========
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
            st.write(f"**{r['patient']['name']}** — {r['triage']['complaint']} ", unsafe_allow_html=True)
            triage_pill(r['triage']['decision']['color'])
            st.write(f"| Dx: **{r['provisionalDx']}**")

            open_key = f"open_{r['id']}"
            if st.button("Open case", key=open_key):
                isbar = f"""I: {r['patient']['name']}, {r['patient']['age']} {r['patient']['sex']}
Dx (provisional): {r['provisionalDx']}
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

# ======== Government (KPIs + Heat Map + Patterns) ========
with tabs[3]:
    st.subheader("Government – Master Dashboard (SLA • Severity • Flow • Supply)")

    # Filters
    tri_filter = st.selectbox("Triage",["All","RED","YELLOW","GREEN"],index=0)
    sev_filter = st.selectbox("Severity",["All","Critical","Moderate","Non-critical"],index=0)
    cond_filter= st.selectbox("Condition",["All","Maternal","Trauma","Stroke","Cardiac","Sepsis","Other"],index=0)

    data = st.session_state.referrals.copy()
    fac_by_name = {f["name"]:f for f in st.session_state.facilities}

    if tri_filter!="All": data=[r for r in data if r["triage"]["decision"]["color"]==tri_filter]
    if sev_filter!="All": data=[r for r in data if r.get("severity")==sev_filter]
    if cond_filter!="All": data=[r for r in data if r["triage"]["complaint"]==cond_filter]

    total = len(data) or 1
    reds  = [r for r in data if r["triage"]["decision"]["color"]=="RED"]
    with_disp=[r for r in data if r["times"].get("dispatch_ts")]
    pct_red_60 = int(100* len([r for r in reds if r["times"].get("arrive_dest_ts")
                               and minutes(r["times"]["first_contact_ts"], r["times"]["arrive_dest_ts"])<=60])/(len(reds) or 1))
    pct_disp_10 = int(100* len([r for r in with_disp if minutes(r["times"]["first_contact_ts"], r["times"]["dispatch_ts"])<=10])/(len(with_disp) or 1))
    accepted   = len([r for r in data if r["status"] in ["ARRIVE_DEST","HANDOVER"]])
    rejected   = len([r for r in data if r.get("reasons",{}).get("rejected")])

    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi_tile("% RED ≤60m", f"{pct_red_60}%")
    with k2: kpi_tile("% Dispatch ≤10m", f"{pct_disp_10}%")
    with k3: kpi_tile("Acceptance rate", f"{int(100*accepted/total)}%")
    with k4: kpi_tile("Rejection rate", f"{int(100*rejected/total)}%")

    st.markdown("### Severity mix")
    sev_series = pd.Series([r.get("severity","—") for r in data]).value_counts()
    st.bar_chart(sev_series, use_container_width=True)

    st.markdown("### Referral funnel")
    s1 = len(data)
    s2 = len([r for r in data if r["times"].get("dispatch_ts")])
    s3 = len([r for r in data if r["status"] in ["ARRIVE_DEST","HANDOVER"]])
    funnel_df = pd.DataFrame({"stage":["Referrals","Dispatched","Arrived/Handover"],"count":[s1,s2,s3]}).set_index("stage")
    st.bar_chart(funnel_df, use_container_width=True)

    st.markdown("### Acceptance by receiving facility")
    by_fac = pd.Series([r["dest"] for r in data if r["status"] in ["ARRIVE_DEST","HANDOVER"]]).value_counts().head(15)
    st.bar_chart(by_fac, use_container_width=True)

    st.markdown("### Rejection reasons")
    rej = pd.Series([r.get("reasons",{}).get("reject_reason","—") for r in data if r.get("reasons",{}).get("rejected")]).value_counts()
    if not rej.empty: st.bar_chart(rej, use_container_width=True)
    else: st.caption("No recorded rejections in current filter.")

    st.markdown("### Geo density (cases)")
    if data:
        mdf = pd.DataFrame([dict(lat=r["patient"]["location"]["lat"], lon=r["patient"]["location"]["lon"]) for r in data])
        st.map(mdf, use_container_width=True)


# ======== Data / Admin ========
with tabs[4]:
    st.subheader("Seed / Import / Export (JSON & CSV)")
    seed_n = st.slider("Seed referrals (synthetic)", 100, 1000, 300, step=50)
    if st.button("Seed synthetic data"):
        seed_referrals(n=seed_n)
        st.success(f"Seeded {seed_n} referrals")

    st.download_button("Export JSON", data=json.dumps(dict(referrals=st.session_state.referrals, facilities=st.session_state.facilities), indent=2),
                       file_name="ahecn_data.json", mime="application/json")
    if st.button("Export referrals (CSV)"):
        if st.session_state.referrals:
            out=pd.DataFrame(st.session_state.referrals)
            st.download_button("Download CSV", data=out.to_csv(index=False), file_name="ahecn_referrals.csv", mime="text/csv")

    upload = st.file_uploader("Import JSON", type=["json"])
    if upload:
        data = json.load(upload)
        st.session_state.referrals = data.get("referrals", [])
        imported_fac = data.get("facilities", st.session_state.facilities)
        st.session_state.facilities = [normalize_facility(x) for x in imported_fac]
        st.success("Imported (schema normalized)")

# ======== Facility Admin ========
with tabs[5]:
    st.subheader("Facility capabilities & readiness (edit live)")

    # generate more demo facilities
    st.markdown("**Generate more demo facilities**")
    new_n = st.slider("Number of facilities", 10, 30, len(st.session_state.facilities), step=1)
    if st.button("Regenerate facilities"):
        st.session_state.facilities = [normalize_facility(x) for x in default_facilities(count=new_n)]
        st.success(f"Generated {new_n} facilities")

    fac_df = facilities_df()
    st.dataframe(fac_df, use_container_width=True)

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
