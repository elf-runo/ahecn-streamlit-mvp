# --- AHECN Streamlit MVP v1.6 (responsive dashboards + synthetic data) ---

import math, json, time, random, statistics
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="AHECN – Streamlit MVP v1.6", layout="wide")

# ------------------------------ Helpers ------------------------------
def dist_km(lat1, lon1, lat2, lon2):
    R=6371
    dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def minutes(a,b):
    if not a or not b: return None
    return int((b - a)/60)

def now_ts(): return time.time()
def iso(ts): return datetime.fromtimestamp(ts).isoformat(timespec="seconds")

# NEWS2
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

def calc_qSOFA(rr, sbp, avpu_altered):
    s = (1 if rr>=22 else 0) + (1 if sbp<=100 else 0) + (1 if avpu_altered else 0)
    return s, (s>=2)

def calc_MEOWS(rr, spo2, hr, sbp, temp, avpu):
    def band(v, rules):
        for t,c in rules:
            if t(v): return c
        return "green"
    m = dict(
        rr   = band(rr,  [(lambda v:v<10,"red"),(lambda v:v<=20,"green"),(lambda v:v<=30,"yellow"),(lambda v:v>30,"red")]),
        spo2 = band(spo2,[(lambda v:v<92,"red"),(lambda v:v<95,"yellow"),(lambda v:v>=95,"green")]),
        hr   = band(hr,  [(lambda v:v<50,"red"),(lambda v:v<=100,"green"),(lambda v:v<=120,"yellow"),(lambda v:v>120,"red")]),
        sbp  = band(sbp, [(lambda v:v<90,"red"),(lambda v:v<=140,"green"),(lambda v:v<=160,"yellow"),(lambda v:v>160,"red")]),
        temp = band(temp,[(lambda v:v<35,"red"),(lambda v:v<=38,"green"),(lambda v:v<39,"yellow"),(lambda v:v>=39,"red")]),
        avpu = "green" if avpu=="A" else "red"
    )
    reds = sum(1 for v in m.values() if v=="red")
    yell = sum(1 for v in m.values() if v=="yellow")
    trig = (reds>=1 or yell>=2)
    return ("Red" if reds else ("Yellow" if trig else "Green")), trig

def calc_PEWS(age, age_band, rr, hr, spo2, o2_device, behavior, cap_refill=2):
    if age>=18: return False, None, False
    beh = 1 if behavior=="Irritable" else 2 if behavior=="Lethargic" else 0
    crt = 2 if cap_refill>2 else 0
    hiRR = dict(Infant=50,Toddler=40,Child=30,Adolescent=25).get(age_band,30)
    hiHR = dict(Infant=180,Toddler=160,Child=140,Adolescent=120).get(age_band,120)
    s = beh+crt
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

def composite_score(f, km, need_caps):
    cap = 1 if all(f["caps"].get(c,0)==1 for c in need_caps) else 0
    ready = (min(f["ICU_open"],3)/3)*0.5 + f["acceptanceRate"]*0.5
    far = max(0, 1-(km/50))
    return 0.6*ready + 0.25*far + 0.15*cap

# ------------------------------ Demo Data ------------------------------
DEFAULT_FAC = [
  dict(name="District Hospital X", lat=26.8467, lon=80.9462, ICU_open=2, acceptanceRate=0.86,
       caps=dict(ICU=1,Ventilator=1,BloodBank=1,OR=1,CT=1,Thrombolysis=1,OBGYN_OT=0,CathLab=0,Dialysis=1,Neurosurgery=0)),
  dict(name="Women & Child Hospital", lat=26.86, lon=80.98, ICU_open=1, acceptanceRate=0.78,
       caps=dict(ICU=1,Ventilator=1,BloodBank=1,OR=1,CT=0,Thrombolysis=0,OBGYN_OT=1,CathLab=0,Dialysis=0,Neurosurgery=0)),
  dict(name="Neuro Medical Center", lat=26.87, lon=80.93, ICU_open=0, acceptanceRate=0.72,
       caps=dict(ICU=1,Ventilator=1,BloodBank=0,OR=0,CT=1,Thrombolysis=1,OBGYN_OT=0,CathLab=0,Dialysis=0,Neurosurgery=1)),
  dict(name="City Cardiac & Cath Lab", lat=26.90, lon=80.95, ICU_open=3, acceptanceRate=0.90,
       caps=dict(ICU=1,Ventilator=1,BloodBank=0,OR=0,CT=0,Thrombolysis=0,OBGYN_OT=0,CathLab=1,Dialysis=0,Neurosurgery=0)),
  dict(name="Trauma & Ortho Center", lat=26.83, lon=80.91, ICU_open=1, acceptanceRate=0.81,
       caps=dict(ICU=1,Ventilator=1,BloodBank=1,OR=1,CT=1,Thrombolysis=0,OBGYN_OT=0,CathLab=0,Dialysis=1,Neurosurgery=0)),
]

if "facilities" not in st.session_state: st.session_state.facilities = DEFAULT_FAC.copy()
if "referrals"  not in st.session_state: st.session_state.referrals = []
if "active_fac" not in st.session_state: st.session_state.active_fac = st.session_state.facilities[0]["name"]

# ------------------------------ UI ------------------------------
st.title("AHECN – Streamlit MVP v1.6")
tabs = st.tabs(["Referrer", "Ambulance / EMT", "Receiving Hospital", "Government", "Data & Admin"])

# ===== Referrer =====
with tabs[0]:
    st.subheader("Patient & Referrer")
    c1,c2,c3 = st.columns(3)
    with c1:
        p_name = st.text_input("Patient name", "Sita Devi")
        p_age  = st.number_input("Age", 0, 120, 28)
        p_sex  = st.selectbox("Sex", ["Female","Male","Other"])
    with c2:
        p_id   = st.text_input("Patient ID (ABDM/Local)", "")
        r_name = st.text_input("Referrer name", "Dr. Rao / ASHA Poonam")
        r_fac  = st.text_input("Referrer facility", "PHC Dhanbad")
    with c3:
        p_lat  = st.number_input("Lat", value=26.85, format="%.4f")
        p_lon  = st.number_input("Lon", value=80.95, format="%.4f")
        p_dx   = st.text_input("Provisional diagnosis", "PPH; suspected retained placenta")
    ocr = st.text_area("Notes/OCR (paste text; used for summary)", height=100)

    st.subheader("Triage + Scores")
    v1,v2,v3 = st.columns(3)
    with v1:
        hr   = st.number_input("HR", 0, 250, 118)
        sbp  = st.number_input("SBP", 0, 300, 92)
        rr   = st.number_input("RR", 0, 80, 26)
        temp = st.number_input("Temp °C", 30.0, 43.0, 38.4, step=0.1)
    with v2:
        spo2 = st.number_input("SpO₂ %", 50, 100, 92)
        avpu = st.selectbox("AVPU", ["A","V","P","U"], index=0)
        complaint = st.selectbox("Chief complaint", ["Maternal","Trauma","Stroke","Cardiac","FeverConfusion","Sepsis","Other"], index=0)
        rf_sbp = st.checkbox("Flag: SBP <90", value=False)
    with v3:
        rf_spo2 = st.checkbox("Flag: SpO₂ <90%", value=False)
        rf_avpu = st.checkbox("Flag: AVPU ≠ A", value=False)
        rf_seizure = st.checkbox("Flag: Seizure", value=False)
        rf_pph = st.checkbox("Flag: PPH", value=("Maternal" in complaint))

    o2_device   = st.selectbox("O₂ device", ["Air","O2"])
    spo2_scale  = st.selectbox("SpO₂ scale (NEWS2)", [1,2], index=0)
    pews_ageband= st.selectbox("PEWS age band", ["Infant","Toddler","Child","Adolescent"], index=2)
    pews_beh    = st.selectbox("PEWS behavior", ["Normal","Irritable","Lethargic"], index=0)

    # Display scores
    n_score, n_review, n_emerg = calc_NEWS2(rr, spo2, sbp, hr, temp, avpu, o2_device, spo2_scale)
    st.write(f"NEWS2: **{n_score}** {'• EMERGENCY' if n_emerg else '• review' if n_review else ''}")
    q_score, q_high = calc_qSOFA(rr, sbp, avpu!="A")
    st.write(f"qSOFA: **{q_score}** {'• ≥2 high risk' if q_high else ''}")
    m_band, m_trig = calc_MEOWS(rr, spo2, hr, sbp, temp, avpu)
    st.write(f"MEOWS: **{m_band}** {'• trigger' if m_trig else ''}")
    if p_age < 18:
        _, pews_score, pews_high = calc_PEWS(p_age, pews_ageband, rr, hr, spo2, o2_device, pews_beh)
        st.write(f"PEWS: **{pews_score}** {'• ≥8 high risk' if pews_high else ''}")
    else:
        st.caption("PEWS disabled for ≥18y")

    # Referral reasons + capabilities
    st.subheader("Referral & Facility Matching")
    c1,c2 = st.columns(2)
    with c1:
        ref_beds = st.checkbox("Reason: No ICU/bed", value=False)
        ref_tests= st.checkbox("Reason: Special intervention/test", value=True)
    need_caps=[]
    if ref_tests:
        st.caption("Select required capabilities:")
        cols = st.columns(5)
        CAP_LIST = ["ICU","Ventilator","BloodBank","OR","CT","Thrombolysis","OBGYN_OT","CathLab","Dialysis","Neurosurgery"]
        for i,cap in enumerate(CAP_LIST):
            if cols[i%5].checkbox(cap, value=(cap in ["ICU","BloodBank","OBGYN_OT"] if complaint=="Maternal" else False), key=f"cap_{cap}"):
                need_caps.append(cap)

    if st.button("Find matched facilities"):
        rows=[]
        for f in st.session_state.facilities:
            if not all(f["caps"].get(c,0)==1 for c in need_caps): continue
            km = dist_km(p_lat, p_lon, f["lat"], f["lon"])
            score = composite_score(f, km, need_caps)
            rows.append(dict(name=f["name"], km=round(km,1), ICU_open=f["ICU_open"], accept=int(round(f["acceptanceRate"]*100,0)), score=int(round(score*100,0))))
        if rows:
            df=pd.DataFrame(rows).sort_values(["score","km"], ascending=[False,True])
            st.dataframe(df, use_container_width=True)
            chosen = st.selectbox("Select destination", [r["name"] for r in rows])
            if st.button("Create referral"):
                vit = dict(hr=hr, rr=rr, sbp=sbp, temp=temp, spo2=spo2, avpu=avpu,
                           rf_sbp=rf_sbp, rf_spo2=rf_spo2, rf_avpu=rf_avpu, rf_seizure=rf_seizure, rf_pph=rf_pph, complaint=complaint)
                ref = dict(
                    id="R"+str(int(time.time()))[-6:],
                    patient=dict(name=p_name, age=int(p_age), sex=p_sex, id=p_id, location=dict(lat=float(p_lat), lon=float(p_lon))),
                    referrer=dict(name=r_name, facility=r_fac),
                    provisionalDx=p_dx,
                    triage=dict(complaint=complaint, decision=dict(color=tri_color(vit)), hr=hr, sbp=sbp, rr=rr, temp=temp, spo2=spo2, avpu=avpu),
                    clinical=dict(summary=" ".join(ocr.split()[:40])),
                    reasons=dict(severity=True, bedOrICUUnavailable=ref_beds, specialTest=ref_tests, requiredCapabilities=need_caps),
                    dest=chosen,
                    times=dict(first_contact_ts=now_ts(), decision_ts=now_ts()),
                    status="PREALERT"
                )
                st.session_state.referrals.insert(0, ref)
                st.success(f"Referral {ref['id']} → {chosen} created")

# ===== Ambulance =====
with tabs[1]:
    st.subheader("Active Jobs")
    active = [r for r in st.session_state.referrals if r["status"] in ["PREALERT","DISPATCHED","ARRIVE_SCENE","DEPART_SCENE","ARRIVE_DEST"]]
    if not active: st.info("No active jobs")
    for r in active:
        st.markdown(f"**{r['patient']['name']}** • {r['triage']['complaint']} • **{r['triage']['decision']['color']}** → **{r['dest']}**")
        c1,c2,c3,c4,c5 = st.columns(5)
        if c1.button("Dispatch", key=f"d_{r['id']}"): r["times"]["dispatch_ts"]=now_ts(); r["status"]="DISPATCHED"
        if c2.button("Arrive scene", key=f"a_{r['id']}"): r["times"]["arrive_scene_ts"]=now_ts(); r["status"]="ARRIVE_SCENE"
        if c3.button("Depart scene", key=f"ds_{r['id']}"): r["times"]["depart_scene_ts"]=now_ts(); r["status"]="DEPART_SCENE"
        if c4.button("Arrive dest", key=f"ad_{r['id']}"): r["times"]["arrive_dest_ts"]=now_ts(); r["status"]="ARRIVE_DEST"
        if c5.button("Handover", key=f"h_{r['id']}"): r["times"]["handover_ts"]=now_ts(); r["status"]="HANDOVER"
        st.caption(f"Stage: {r['status']}")

# ===== Receiving Hospital =====
with tabs[2]:
    st.subheader("Incoming referrals")
    fac_names=[f["name"] for f in st.session_state.facilities]
    st.session_state.active_fac = st.selectbox("Facility", fac_names, index=fac_names.index(st.session_state.active_fac) if st.session_state.active_fac in fac_names else 0)
    incoming = [r for r in st.session_state.referrals if r["dest"]==st.session_state.active_fac and r["status"] in ["PREALERT","DISPATCHED","ARRIVE_SCENE","DEPART_SCENE","ARRIVE_DEST"]]
    if not incoming: st.info("No incoming referrals")
    for r in incoming:
        st.write(f"**{r['patient']['name']}** — {r['triage']['complaint']} • **{r['triage']['decision']['color']}** | Provisional Dx: **{r['provisionalDx']}**")
        if st.button("Open case", key=f"open_{r['id']}"):
            isbar = f"""I: {r['patient']['name']}, {r['patient']['age']} {r['patient']['sex']}
Dx (provisional): {r['provisionalDx']}
S: {r['triage']['complaint']}; triage {r['triage']['decision']['color']}
B: HR {r['triage']['hr']}, SBP {r['triage']['sbp']}, RR {r['triage']['rr']}, Temp {r['triage']['temp']}, SpO2 {r['triage']['spo2']}, AVPU {r['triage']['avpu']}
A: —
R: {"Bed/ICU unavailable; " if r['reasons'].get('bedOrICUUnavailable') else ""}{"Special test; " if r['reasons'].get('specialTest') else ""}Severity
Notes: {r['clinical'].get('summary',"—")}
"""
            st.code(isbar)
            c1,c2 = st.columns(2)
            if c1.button("Accept", key=f"acc_{r['id']}"):
                r["status"]="ARRIVE_DEST"; r["times"]["arrive_dest_ts"]=now_ts(); st.success("Accepted")
            if c2.button("Request divert", key=f"div_{r['id']}"):
                r["reasons"]["bedOrICUUnavailable"]=True; r["status"]="PREALERT"; st.warning("Divert requested")

# ===== Government =====
with tabs[3]:
    st.subheader("SLA & Condition KPIs (Filters)")
    tri_filter = st.selectbox("Triage", ["All","RED","YELLOW","GREEN"], index=0)
    cond_filter= st.selectbox("Condition", ["All","Maternal","Trauma","Stroke","Cardiac","FeverConfusion","Sepsis","Other"], index=0)

    data = st.session_state.referrals
    if tri_filter!="All": data=[r for r in data if r["triage"]["decision"]["color"]==tri_filter]
    if cond_filter!="All": data=[r for r in data if r["triage"]["complaint"]==cond_filter]

    # KPI tiles
    reds=[r for r in data if r["triage"]["decision"]["color"]=="RED"]
    pct_red_60 = int(100* len([r for r in reds if r["times"].get("arrive_dest_ts") and minutes(r["times"]["first_contact_ts"], r["times"]["arrive_dest_ts"])<=60])/len(reds)) if reds else 0
    with_disp=[r for r in data if r["times"].get("dispatch_ts")]
    pct_disp_10 = int(100* len([r for r in with_disp if minutes(r["times"]["first_contact_ts"], r["times"]["dispatch_ts"])<=10])/len(with_disp)) if with_disp else 0

    k1,k2 = st.columns(2)
    k1.metric("% RED ≤60 min", f"{pct_red_60}%")
    k2.metric("% Dispatch ≤10 min", f"{pct_disp_10}%")

    st.markdown("### Condition bundle tiles")
    bundle = {}
    for cond in ["Maternal","Trauma","Stroke","Cardiac","Sepsis"]:
        subset=[r for r in data if r["triage"]["complaint"]==cond]
        if not subset: bundle[cond]=(0,"—"); continue
        tmins=[]
        for r in subset:
            if r["times"].get("arrive_dest_ts"):
                tmins.append(minutes(r["times"]["first_contact_ts"], r["times"]["arrive_dest_ts"]))
        med = f"{statistics.median(tmins)}m" if tmins else "—"
        bundle[cond]=(len(subset), med)
    bcols=st.columns(5)
    for i,cond in enumerate(["Maternal","Trauma","Stroke","Cardiac","Sepsis"]):
        count, med = bundle[cond]
        bcols[i].metric(cond, count, help=f"Median transfer: {med}")

    st.markdown("### Bottlenecks")
    # ICU open = 0 and frequent divert
    fac_df = pd.DataFrame(st.session_state.facilities)
    if not fac_df.empty:
        zero_icu = fac_df[fac_df["ICU_open"]==0]["name"].tolist()
        st.write("Facilities with ICU_open = 0:", ", ".join(zero_icu) if zero_icu else "None")
    diverts=[r for r in st.session_state.referrals if r["reasons"].get("bedOrICUUnavailable")]
    if diverts:
        top_div = pd.Series([r["dest"] for r in diverts]).value_counts().head(5).reset_index()
        top_div.columns=["Facility","Divert flags"]
        st.dataframe(top_div, use_container_width=True)

    st.markdown("### Top receiving facilities (count)")
    if data:
        s = pd.Series([r["dest"] for r in data]).value_counts().reset_index()
        s.columns = ["Facility","Cases"]
        st.dataframe(s, use_container_width=True)
        st.download_button("Download KPIs (CSV)", data=s.to_csv(index=False), file_name="ahecn_kpis.csv", mime="text/csv")
    else:
        st.caption("No data for selected filters")

# ===== Data & Admin =====
with tabs[4]:
    st.subheader("Seed / Import / Export / Facility editor")
    seed_n = st.slider("Seed referrals (synthetic)", 50, 500, 200, step=50)
    if st.button("Seed synthetic data"):
        st.session_state.referrals.clear()
        rng = random.Random(42)
        conds=["Maternal","Trauma","Stroke","Cardiac","Sepsis","Other"]
        colors=["RED","YELLOW","GREEN"]
        fac_names=[f["name"] for f in st.session_state.facilities]
        base = time.time() - 7*24*3600  # last 7 days
        for i in range(seed_n):
            cond=rng.choice(conds)
            col = rng.choices(colors, weights=[0.3,0.5,0.2])[0]
            ts = base + rng.randint(0,7*24*3600)
            t_disp = ts + rng.randint(2*60, 30*60)  # 2–30m
            t_arrdest = ts + rng.randint(20*60, 180*60)  # 20–180m
            st.session_state.referrals.append(dict(
                id=f"S{i:04d}",
                patient=dict(name=f"Pt{i:04d}", age=rng.randint(1,85), sex=("Female" if rng.random()<0.5 else "Male"),
                             id="", location=dict(lat=26.84+rng.random()*0.06, lon=80.90+rng.random()*0.08)),
                referrer=dict(name="Dr. Demo", facility="PHC Demo"),
                provisionalDx=("PPH" if cond=="Maternal" else "—"),
                triage=dict(complaint=cond, decision=dict(color=col), hr=rng.randint(80,140), sbp=rng.randint(85,140),
                            rr=rng.randint(14,32), temp=round(36+rng.random()*3,1), spo2=rng.randint(88,98), avpu="A"),
                clinical=dict(summary="Auto-seeded"),
                reasons=dict(severity=True, bedOrICUUnavailable=(rng.random()<0.15), specialTest=(rng.random()<0.2), requiredCapabilities=[]),
                dest=rng.choice(fac_names),
                times=dict(first_contact_ts=ts, decision_ts=ts+60, dispatch_ts=t_disp, arrive_dest_ts=t_arrdest),
                status="HANDOVER" if rng.random()<0.8 else "ARRIVE_DEST"
            ))
        st.success(f"Seeded {seed_n} referrals")

    st.download_button("Export JSON", data=json.dumps(dict(referrals=st.session_state.referrals, facilities=st.session_state.facilities), indent=2),
                       file_name="ahecn_data.json", mime="application/json")

    upload = st.file_uploader("Import JSON", type=["json"])
    if upload:
        data=json.load(upload)
        st.session_state.referrals = data.get("referrals", [])
        st.session_state.facilities = data.get("facilities", st.session_state.facilities)
        st.success("Imported")

    st.markdown("#### Facility editor")
    fac_df = pd.DataFrame(st.session_state.facilities)
    st.dataframe(fac_df[["name","ICU_open","acceptanceRate"]], use_container_width=True)
    target = st.selectbox("Select facility", [f["name"] for f in st.session_state.facilities])
    new_icu = st.number_input("ICU_open (int)", 0, 20, value=[f for f in st.session_state.facilities if f["name"]==target][0]["ICU_open"])
    new_acc = st.slider("Acceptance rate", 0.0, 1.0, value=[f for f in st.session_state.facilities if f["name"]==target][0]["acceptanceRate"], step=0.01)
    if st.button("Update facility"):
        for f in st.session_state.facilities:
            if f["name"]==target:
                f["ICU_open"]=int(new_icu); f["acceptanceRate"]=float(new_acc)
        st.success("Facility updated")
