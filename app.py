# AHECN – Streamlit MVP v1.7 (Investor Demo)
# Roles: Referrer • Ambulance/EMT • Receiving Hospital • Government • Data/Admin • Facility Admin
# Region: East Khasi Hills, Meghalaya (synthetic geo + facilities)

import math, json, time, random, statistics
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="AHECN – Streamlit MVP v1.7", layout="wide")

# ---------------------- Geometry & Utilities ----------------------
def dist_km(lat1, lon1, lat2, lon2):
    R=6371
    dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def now_ts(): return time.time()
def minutes(a,b): 
    if not a or not b: return None
    return int((b-a)/60)

# ---------------------- Scores (standards-aligned) ----------------------
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

# ---------------------- Demo Facilities (East Khasi Hills) ----------------------
EH_BASE = dict(lat_min=25.45, lat_max=25.65, lon_min=91.80, lon_max=91.95)
def rand_geo(rng):
    return (EH_BASE["lat_min"]+rng.random()*(EH_BASE["lat_max"]-EH_BASE["lat_min"]),
            EH_BASE["lon_min"]+rng.random()*(EH_BASE["lon_max"]-EH_BASE["lon_min"]))

SPECIALTIES = ["Obstetrics","Paediatrics","Cardiology","Neurology","Orthopaedics","General Surgery","Anaesthesia","ICU"]
INTERVENTIONS = ["CathLab","OBGYN_OT","CT","MRI","Dialysis","Thrombolysis","Ventilator","BloodBank","OR","Neurosurgery"]

def default_facilities():
    # ---- Schema safety helpers ----
REQ_CAPS = ["ICU","Ventilator","BloodBank","OR","CT","Thrombolysis","OBGYN_OT","CathLab","Dialysis","Neurosurgery"]

def normalize_facility(f):
    f = dict(f)  # shallow copy
    f.setdefault("name", "Unknown Facility")
    f.setdefault("type", "PHC")
    f.setdefault("ICU_open", 0)
    f.setdefault("acceptanceRate", 0.75)
    f.setdefault("lat", 25.58)
    f.setdefault("lon", 91.89)

    # nested dicts
    caps = f.get("caps", {}) or {}
    f["caps"] = {k: int(bool(caps.get(k, 0))) for k in REQ_CAPS}

    specs = f.get("specialties", {}) or {}
    f["specialties"] = {s: int(bool(specs.get(s, 0))) for s in SPECIALTIES}

    hi = f.get("highend", {}) or {}
    f["highend"] = {i: int(bool(hi.get(i, 0))) for i in INTERVENTIONS}
    return f

def facilities_df():
    """Flat, guaranteed columns for the admin table."""
    fac = [normalize_facility(x) for x in st.session_state.facilities]
    rows = [{"name": x["name"], "type": x["type"], "ICU_open": x["ICU_open"], "acceptanceRate": x["acceptanceRate"]} for x in fac]
    return pd.DataFrame(rows)

    rng=random.Random(17)
    names=[
        "Civil Hospital Shillong","NEIGRIHMS","Nazareth Hospital","Ganesh Das Maternal & Child",
        "Shillong Polyclinic & Trauma Center","Smit Community Health Centre","Pynursla CHC",
        "Mawsynram PHC","Sohra Civil Hospital","Madansynram CHC","Jowai (ref) Hub","Mawlai CHC"
    ]
    fac=[]
    for n in names:
        lat,lon=rand_geo(rng)
        caps={c:int(rng.random()<0.7) for c in ["ICU","Ventilator","BloodBank","OR","CT","Thrombolysis","OBGYN_OT","CathLab","Dialysis","Neurosurgery"]}
        spec={s:int(rng.random()<0.6) for s in SPECIALTIES}
        hi={i:int(rng.random()<0.5) for i in INTERVENTIONS}
        fac.append(dict(
            name=n, lat=lat, lon=lon,
            ICU_open=rng.randint(0,4),
            acceptanceRate=round(0.7+rng.random()*0.25,2),
            caps=caps, specialties=spec, highend=hi,
            type=rng.choice(["PHC","CHC","District Hospital","Tertiary"])
        ))
    return fac

# ---------------------- Session ----------------------
if "facilities" not in st.session_state: st.session_state.facilities = default_facilities()
if "referrals"  not in st.session_state: st.session_state.referrals = []
if "active_fac" not in st.session_state: st.session_state.active_fac = st.session_state.facilities[0]["name"]
# Ensure facilities schema is consistent even if older JSONs were loaded
st.session_state.facilities = [normalize_facility(x) for x in st.session_state.facilities]

# ---------------------- UI ----------------------
st.title("AHECN – Streamlit MVP v1.7 (East Khasi Hills)")

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

    # Resuscitation interventions (performed before referral)
    st.subheader("Resuscitation / Stabilization done (tick all applied)")
    cols = st.columns(5)
    RESUS = ["Airway positioning","Oxygen","IV fluids","Uterotonics","TXA","Bleeding control","Antibiotics","Nebulization","Immobilization","AED/CPR"]
    resus_done = [r for i,r in enumerate(RESUS) if cols[i%5].checkbox(r, False)]

    st.subheader("Reason(s) for referral + capabilities needed")
    c1,c2 = st.columns(2)
    with c1:
        ref_beds  = st.checkbox("No ICU/bed available",False)
        ref_tests = st.checkbox("Special intervention/test required",True)
        ref_severity = True  # by design when triage RED/YELLOW
    need_caps=[]
    if ref_tests:
        st.caption("Select required capabilities for this case")
        cap_cols = st.columns(5)
        CAP_LIST = ["ICU","Ventilator","BloodBank","OR","CT","Thrombolysis","OBGYN_OT","CathLab","Dialysis","Neurosurgery"]
        for i,cap in enumerate(CAP_LIST):
            pre = (cap in ["ICU","BloodBank","OBGYN_OT"]) if complaint=="Maternal" else False
            if cap_cols[i%5].checkbox(cap, value=pre, key=f"cap_{cap}"):
                need_caps.append(cap)

    # Facility matching (adds preparedness badges: specialties + high-end)
    if st.button("Find matched facilities"):
        rows=[]
        for f in st.session_state.facilities:
            caps_ok = all(f["caps"].get(c,0)==1 for c in need_caps)
            if not caps_ok: continue
            km = dist_km(p_lat, p_lon, f["lat"], f["lon"])
            ready = (min(f["ICU_open"],3)/3)*0.5 + f["acceptanceRate"]*0.5
            far = max(0,1-(km/50))
            score = 0.6*ready + 0.25*far + 0.15*(1 if caps_ok else 0)
            rows.append(dict(
                name=f["name"], km=round(km,1), ICU_open=f["ICU_open"],
                accept=int(round(f["acceptanceRate"]*100,0)),
                specialties=", ".join([s for s,v in f["specialties"].items() if v]) or "—",
                highend=", ".join([i for i,v in f["highend"].items() if v]) or "—",
                score=int(round(score*100,0))
            ))
        if rows:
            df=pd.DataFrame(rows).sort_values(["score","km"],ascending=[False,True])
            st.dataframe(df, use_container_width=True)
            chosen = st.selectbox("Select destination", [r["name"] for r in rows])
            if st.button("Create referral"):
                vit=dict(hr=hr,rr=rr,sbp=sbp,temp=temp,spo2=spo2,avpu=avpu,rf_sbp=rf_sbp,rf_spo2=rf_spo2,
                         rf_avpu=rf_avpu,rf_seizure=rf_seizure,rf_pph=rf_pph,complaint=complaint)
                ref = dict(
                    id="R"+str(int(time.time()))[-6:],
                    patient=dict(name=p_name,age=int(p_age),sex=p_sex,id=p_id,location=dict(lat=float(p_lat),lon=float(p_lon))),
                    referrer=dict(name=r_name,facility=r_fac),
                    provisionalDx=p_dx,
                    resuscitation=resus_done,
                    triage=dict(complaint=complaint,decision=dict(color=tri_color(vit)),hr=hr,sbp=sbp,rr=rr,temp=temp,spo2=spo2,avpu=avpu),
                    clinical=dict(summary=" ".join(ocr.split()[:60])),
                    reasons=dict(severity=ref_severity,bedOrICUUnavailable=ref_beds,specialTest=ref_tests,requiredCapabilities=need_caps),
                    dest=chosen,
                    times=dict(first_contact_ts=now_ts(),decision_ts=now_ts()),
                    status="PREALERT",
                    ambulance_available=None
                )
                st.session_state.referrals.insert(0,ref)
                st.success(f"Referral {ref['id']} → {chosen} created")

# ======== Ambulance / EMT ========
with tabs[1]:
    st.subheader("Active jobs (availability + 5-stage lifecycle)")
    # EMT availability toggle
    avail = st.radio("Ambulance availability", ["Available","Unavailable"], horizontal=True)
    # list jobs
    active = [r for r in st.session_state.referrals if r["status"] in ["PREALERT","DISPATCHED","ARRIVE_SCENE","DEPART_SCENE","ARRIVE_DEST"]]
    if not active: st.info("No active jobs")
    for r in active:
        st.markdown(f"**{r['patient']['name']}** • {r['triage']['complaint']} • **{r['triage']['decision']['color']}** → **{r['dest']}**")
        c1,c2,c3,c4,c5 = st.columns(5)
        if c1.button("Dispatch",key=f"d_{r['id']}"): r["times"]["dispatch_ts"]=now_ts(); r["status"]="DISPATCHED"; r["ambulance_available"]=(avail=="Available")
        if c2.button("Arrive scene",key=f"a_{r['id']}"): r["times"]["arrive_scene_ts"]=now_ts(); r["status"]="ARRIVE_SCENE"
        if c3.button("Depart scene",key=f"ds_{r['id']}"): r["times"]["depart_scene_ts"]=now_ts(); r["status"]="DEPART_SCENE"
        if c4.button("Arrive dest",key=f"ad_{r['id']}"): r["times"]["arrive_dest_ts"]=now_ts(); r["status"]="ARRIVE_DEST"
        if c5.button("Handover",key=f"h_{r['id']}"): r["times"]["handover_ts"]=now_ts(); r["status"]="HANDOVER"
        st.caption(f"Stage: {r['status']} • Ambulance available at dispatch: {r.get('ambulance_available')}")

# ======== Receiving Hospital ========
with tabs[2]:
    st.subheader("Incoming referrals & case actions")
    fac_names=[f["name"] for f in st.session_state.facilities]
    st.session_state.active_fac = st.selectbox("Facility", fac_names, index=fac_names.index(st.session_state.active_fac))
    incoming=[r for r in st.session_state.referrals if r["dest"]==st.session_state.active_fac and r["status"] in
              ["PREALERT","DISPATCHED","ARRIVE_SCENE","DEPART_SCENE","ARRIVE_DEST"]]
    if not incoming: st.info("No incoming referrals")
    for r in incoming:
        st.write(f"**{r['patient']['name']}** — {r['triage']['complaint']} • **{r['triage']['decision']['color']}** | Dx: **{r['provisionalDx']}**")
        if st.button("Open case",key=f"open_{r['id']}"):
            isbar=f"""I: {r['patient']['name']}, {r['patient']['age']} {r['patient']['sex']}
Dx (provisional): {r['provisionalDx']}
S: {r['triage']['complaint']}; triage {r['triage']['decision']['color']}
B: HR {r['triage']['hr']}, SBP {r['triage']['sbp']}, RR {r['triage']['rr']}, Temp {r['triage']['temp']}, SpO2 {r['triage']['spo2']}, AVPU {r['triage']['avpu']}
A: Resus: {", ".join(r.get('resuscitation',[])) or "—"}
R: {"Bed/ICU unavailable; " if r['reasons'].get('bedOrICUUnavailable') else ""}{"Special test; " if r['reasons'].get('specialTest') else ""}Severity
Notes: {r['clinical'].get('summary',"—")}
"""
            st.code(isbar)
            c1,c2,c3 = st.columns(3)
            if c1.button("Accept",key=f"acc_{r['id']}"):
                r["status"]="ARRIVE_DEST"; r["times"]["arrive_dest_ts"]=now_ts(); st.success("Accepted")
            reject_reason = c2.selectbox("Reject reason",["—","No ICU bed","No specialist","Equipment down","Over capacity","Outside scope"], key=f"rejrs_{r['id']}")
            if c3.button("Reject",key=f"rej_{r['id']}") and reject_reason!="—":
                r["status"]="PREALERT"; 
                r["reasons"]["rejected"]=True; r["reasons"]["reject_reason"]=reject_reason
                st.warning(f"Requested divert / rejected: {reject_reason}")

# ======== Government (KPIs + Heat Map + Patterns) ========
with tabs[3]:
    st.subheader("SLA & Condition KPIs (Real-time)")
    tri_filter = st.selectbox("Triage",["All","RED","YELLOW","GREEN"],index=0)
    cond_filter= st.selectbox("Condition",["All","Maternal","Trauma","Stroke","Cardiac","FeverConfusion","Sepsis","Other"],index=0)
    origin_type= st.selectbox("Referred from (type)",["All","PHC","CHC","District Hospital","Tertiary"],index=0)
    dest_type  = st.selectbox("Receiving type",["All","PHC","CHC","District Hospital","Tertiary"],index=0)

    data = st.session_state.referrals.copy()
    # enrich with types
    fac_by_name = {f["name"]:f for f in st.session_state.facilities}
    for r in data:
        r["dest_type"] = fac_by_name.get(r["dest"],{}).get("type","—")
        r["origin_type"]= r["referrer"]["facility"].split()[0] if r["referrer"]["facility"] else "PHC"

    if tri_filter!="All": data=[r for r in data if r["triage"]["decision"]["color"]==tri_filter]
    if cond_filter!="All": data=[r for r in data if r["triage"]["complaint"]==cond_filter]
    if origin_type!="All": data=[r for r in data if r.get("origin_type")==origin_type]
    if dest_type!="All": data=[r for r in data if r.get("dest_type")==dest_type]

    reds=[r for r in data if r["triage"]["decision"]["color"]=="RED"]
    pct_red_60 = int(100* len([r for r in reds if r["times"].get("arrive_dest_ts") and minutes(r["times"]["first_contact_ts"], r["times"]["arrive_dest_ts"])<=60])/len(reds)) if reds else 0
    with_disp=[r for r in data if r["times"].get("dispatch_ts")]
    pct_disp_10 = int(100* len([r for r in with_disp if minutes(r["times"]["first_contact_ts"], r["times"]["dispatch_ts"])<=10])/len(with_disp)) if with_disp else 0
    accepted   = len([r for r in data if r["status"] in ["ARRIVE_DEST","HANDOVER"]])
    rejected   = len([r for r in data if r["reasons"].get("rejected")])
    total      = len(data) or 1

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("% RED ≤60m", f"{pct_red_60}%")
    k2.metric("% Dispatch ≤10m", f"{pct_disp_10}%")
    k3.metric("Acceptance rate", f"{int(100*accepted/total)}%")
    k4.metric("Rejection rate", f"{int(100*rejected/total)}%")

    st.markdown("### Case heat map (locations)")
    if data:
        mdf = pd.DataFrame([dict(lat=r["patient"]["location"]["lat"], lon=r["patient"]["location"]["lon"]) for r in data])
        st.map(mdf, use_container_width=True)
    else:
        st.caption("No data")

    st.markdown("### Patterns & Workload")
    c1,c2 = st.columns(2)
    # referral reasons
    reasons = pd.Series([("Special test" if r["reasons"].get("specialTest") else "No special test") for r in data]).value_counts()
    c1.bar_chart(reasons, use_container_width=True)
    # categories
    cats = pd.Series([r["triage"]["complaint"] for r in data]).value_counts()
    c2.bar_chart(cats, use_container_width=True)

    st.markdown("### Time performance")
    durations=[]
    for r in data:
        if r["times"].get("handover_ts") and r["times"].get("dispatch_ts"):
            durations.append(minutes(r["times"]["dispatch_ts"], r["times"]["handover_ts"]))
        elif r["times"].get("arrive_dest_ts") and r["times"].get("dispatch_ts"):
            durations.append(minutes(r["times"]["dispatch_ts"], r["times"]["arrive_dest_ts"]))
    avg_time = f"{int(sum(durations)/len(durations))} min" if durations else "—"
    st.write("Average time (Dispatch → Handover/Arrive dest): **", avg_time, "**")

    st.markdown("### Ambulance availability at dispatch")
    av_series = pd.Series(["Available" if r.get("ambulance_available") else "Unavailable" for r in data if r.get("ambulance_available") is not None]).value_counts()
    if not av_series.empty: st.bar_chart(av_series, use_container_width=True)
    else: st.caption("No dispatch records yet")

    st.markdown("### Download KPIs (CSV)")
    if data:
        out = pd.DataFrame([dict(id=r["id"], complaint=r["triage"]["complaint"], triage=r["triage"]["decision"]["color"],
                                 dest=r["dest"], dest_type=r.get("dest_type"), accepted=(r["status"] in ["ARRIVE_DEST","HANDOVER"]),
                                 rejected=r["reasons"].get("rejected",False), reject_reason=r["reasons"].get("reject_reason",""),
                                 ambulance_available=r.get("ambulance_available")) for r in data])
        st.download_button("Download", data=out.to_csv(index=False), file_name="ahecn_kpis.csv", mime="text/csv")

# ======== Data / Admin ========
with tabs[4]:
    st.subheader("Seed / Import / Export (JSON & CSV)")
    seed_n = st.slider("Seed referrals (synthetic)", 150, 300, 200, step=25)
    if st.button("Seed synthetic data"):
        st.session_state.referrals.clear()
        rng=random.Random(42)
        conds=["Maternal","Trauma","Stroke","Cardiac","Sepsis","Other"]
        colors=["RED","YELLOW","GREEN"]
        fac_names=[f["name"] for f in st.session_state.facilities]
        base = time.time() - 5*24*3600  # last 5 days
        for i in range(seed_n):
            cond=rng.choices(conds, weights=[0.2,0.25,0.2,0.2,0.1,0.05])[0]
            col = rng.choices(colors, weights=[0.3,0.5,0.2])[0]
            ts = base + rng.randint(0,5*24*3600)
            # EMT availability & timings
            amb_avail = (rng.random() > 0.2)
            t_disp = ts + rng.randint(2*60, 20*60) if amb_avail else ts + rng.randint(21*60, 60*60)
            t_arrdest = t_disp + rng.randint(25*60, 120*60)
            lat,lon = rand_geo(rng)
            st.session_state.referrals.append(dict(
                id=f"S{i:04d}",
                patient=dict(name=f"Pt{i:04d}", age=rng.randint(1,85), sex=("Female" if rng.random()<0.5 else "Male"),
                             id="", location=dict(lat=lat,lon=lon)),
                referrer=dict(name="Dr. Demo", facility=rng.choice(["PHC Mawlai","CHC Smit","District Hospital Shillong","Tertiary Shillong Hub"])),
                provisionalDx=("PPH" if cond=="Maternal" else rng.choice(["—","Sepsis","Head injury","STEMI","Stroke?"])),
                resuscitation=rng.sample(RESUS, rng.randint(0,3)),
                triage=dict(complaint=cond, decision=dict(color=col), hr=rng.randint(80,140), sbp=rng.randint(85,140),
                            rr=rng.randint(14,32), temp=round(36+rng.random()*3,1), spo2=rng.randint(88,98), avpu="A"),
                clinical=dict(summary="Auto-seeded"),
                reasons=dict(severity=True, bedOrICUUnavailable=(rng.random()<0.2), specialTest=(rng.random()<0.3), requiredCapabilities=[]),
                dest=rng.choice(fac_names),
                times=dict(first_contact_ts=ts, decision_ts=ts+60, dispatch_ts=t_disp, arrive_dest_ts=t_arrdest,
                           handover_ts=(t_arrdest + rng.randint(5*60,25*60) if rng.random()<0.7 else None)),
                status="HANDOVER",
                ambulance_available=amb_avail
            ))
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
fac_df = facilities_df()
st.dataframe(fac_df, use_container_width=True)

target = st.selectbox("Select facility", [f["name"] for f in st.session_state.facilities])
# pull normalized record for editing
F = next(f for f in st.session_state.facilities if f["name"] == target)

    c1,c2 = st.columns(2)
    with c1:
        new_icu = st.number_input("ICU_open", 0, 30, value=int(F["ICU_open"]))
        new_acc = st.slider("Acceptance rate", 0.0, 1.0, value=float(F["acceptanceRate"]), step=0.01)
    with c2:
        st.caption("Toggle key specialties")
        for s in SPECIALTIES:
            F["specialties"][s] = st.checkbox(s, value=bool(F["specialties"].get(s,0)), key=f"spec_{s}")
    st.caption("High-end interventional equipment")
    hi_cols = st.columns(5)
    for i,cap in enumerate(INTERVENTIONS):
        F["highend"][cap] = hi_cols[i%5].checkbox(cap, value=bool(F["highend"].get(cap,0)), key=f"hi_{cap}")
    if st.button("Update facility"):
        F["ICU_open"]=int(new_icu); F["acceptanceRate"]=float(new_acc)
        st.success("Facility updated")

