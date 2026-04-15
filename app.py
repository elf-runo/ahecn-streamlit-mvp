import streamlit as st
import pandas as pd
import altair as alt
import time
import random
import os
import hashlib
from datetime import datetime, timedelta

# --- Page Configuration (MUST BE FIRST) ---
st.set_page_config(
    page_title="MCECN Command Center", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ==========================================
# 🎨 MCECN BRANDING & UI/UX CSS INJECTION
# ==========================================
st.markdown("""
<style>
    /* Global App Background & Typography */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* --- THE MCECN GLOBAL BRAND HEADER --- */
    .mcecn-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        padding: 1.8rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -5px rgba(30, 58, 138, 0.4);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .mcecn-title-wrapper {
        display: flex;
        flex-direction: column;
    }
    .mcecn-title {
        font-size: 2.8rem;
        font-weight: 900;
        margin: 0;
        line-height: 1.1;
        letter-spacing: 1.5px;
        background: linear-gradient(to right, #48BBEA, #FFFFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .mcecn-subtitle {
        font-size: 1.05rem;
        font-weight: 500;
        color: #94A3B8;
        margin: 5px 0 0 0;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .mcecn-status {
        background: rgba(6, 182, 212, 0.15);
        border: 1px solid rgba(6, 182, 212, 0.4);
        color: #22D3EE;
        padding: 0.6rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: 1px;
    }
    
    .pulse-dot {
        height: 10px;
        width: 10px;
        background-color: #22D3EE;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 10px #22D3EE;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 211, 238, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(34, 211, 238, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 211, 238, 0); }
    }
    
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 14px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0px 4px 20px rgba(15, 23, 42, 0.04) !important;
        padding: 1.8rem !important;
        transition: box-shadow 0.3s ease-in-out, transform 0.3s ease;
    }
    
    /* Buttons & Typography */
    button[kind="primary"] {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25) !important;
    }
    h1, h2, h3, h4 { color: #0F172A !important; font-weight: 800 !important; }
    [data-testid="stMetricValue"] { color: #0F172A !important; font-size: 2.4rem !important; font-weight: 900 !important; }
    
    /* Timeline styling */
    .timeline-container {
        display: flex; justify-content: space-between; align-items: center; 
        background: #F1F5F9; padding: 15px 25px; border-radius: 10px; margin-bottom: 20px;
    }
    .timeline-step {
        display: flex; flex-direction: column; align-items: center; text-align: center;
        flex: 1; position: relative;
    }
    .timeline-step:not(:last-child)::after {
        content: ''; position: absolute; top: 15px; right: -50%; width: 100%; height: 3px; background: #CBD5E1; z-index: 1;
    }
    .timeline-step.active:not(:last-child)::after { background: #3B82F6; }
    .timeline-icon {
        width: 30px; height: 30px; border-radius: 50%; background: #CBD5E1; color: white; 
        display: flex; align-items: center; justify-content: center; z-index: 2; font-weight: bold; margin-bottom: 8px;
    }
    .timeline-step.active .timeline-icon { background: #3B82F6; box-shadow: 0 0 10px rgba(59, 130, 246, 0.5); }
    .timeline-label { font-size: 0.8rem; font-weight: 600; color: #64748B; }
    .timeline-step.active .timeline-label { color: #0F172A; }
</style>
""", unsafe_allow_html=True)

# --- Architecture Imports ---
from clinical_engine import validated_triage_decision
from scoring_engine import calculate_facility_score
from routing_engine import get_eta, haversine_km
from analytics_engine import mortality_risk

# --- State Management & Fleet Rosters ---
if 'active_case' not in st.session_state: st.session_state.active_case = None
if 'transfer_initiated' not in st.session_state: st.session_state.transfer_initiated = False
if 'patient_accepted' not in st.session_state: st.session_state.patient_accepted = False
if 'match_results' not in st.session_state: st.session_state.match_results = None
if 'civic_override_active' not in st.session_state: st.session_state.civic_override_active = False
if 'bed_locked' not in st.session_state: st.session_state.bed_locked = False

PILOTS = ["Khraw", "Mewan", "Donbok", "Bantei", "Pynskhem", "Lamphrang"]
EMTS = ["Dr. Sarah", "Paramedic Grace", "Paramedic John", "Nurse Riba"]

# --- DATA LOADER ---
def load_datasets_v3():
    fac_file, icd_file = 'meghalaya_facilities.csv', 'icd_catalogue.csv'
    f_path, i_path = None, None
    for root, _, files in os.walk("."):
        if fac_file in files and not f_path: f_path = os.path.join(root, fac_file)
        if icd_file in files and not i_path: i_path = os.path.join(root, icd_file)
    if not f_path or not i_path:
        st.error(f"CRITICAL: Missing '{fac_file}' or '{icd_file}' in repository.")
        st.stop()
    try:
        fac_df = pd.read_csv(f_path, encoding='utf-8-sig', on_bad_lines='skip')
        icd_df = pd.read_csv(i_path, encoding='utf-8-sig', on_bad_lines='skip')
    except Exception as e:
        st.error(f"CRITICAL: Failed to read CSV. Error: {e}")
        st.stop()
    icd_df.columns = [str(c).replace('ï»¿', '').strip().lower() for c in icd_df.columns]
    rename_map = {'icd-10': 'icd10', 'code': 'icd10'}
    icd_df = icd_df.rename(columns=rename_map)
    for c in ["lat","lon"]:
        if c in fac_df.columns: fac_df[c] = pd.to_numeric(fac_df[c], errors='coerce')
    return fac_df, icd_df

facilities_df, icd_df = load_datasets_v3()
    
# --- Sidebar Navigation & RBAC Security ---
with st.sidebar:
    st.markdown("""
        <div style='padding-bottom: 2rem;'>
            <h2 style='color: #1E3A8A !important; margin-bottom: 0;'>MCECN OS</h2>
            <p style='color: #64748B; font-size: 0.85rem; font-weight: 600; margin-top: 0; letter-spacing: 1px;'>ENTERPRISE v7.0</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🔐 Access Control")
    simulated_role = st.selectbox("Simulate Active User Login:", [
        "State Health Command (Macro View)", "Authorized Community Node (ASHA/CFR)", 
        "Director: NEIGRIHMS", "Director: Woodland Hospital", "108 ERC Dispatcher"
    ], label_visibility="collapsed")
    st.session_state.user_role = simulated_role 
    
    st.markdown("---")
    st.subheader("🧭 System Navigation")
    nav_selection = st.radio(
        "Select Module:",
        ["108 ERC INTAKE CONSOLE", "REFERRAL INITIATION", "ACTIVE TRANSIT TELEMETRY", "RECEIVING HOSPITAL BAY", "STATE COMMAND & AI"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("🟢 Secure Connection Established")

st.markdown("""
<div class="mcecn-header">
    <div class="mcecn-title-wrapper">
        <h1 class="mcecn-title">MCECN</h1>
        <p class="mcecn-subtitle">Meghalaya Comprehensive Emergency Care Network</p>
    </div>
    <div class="mcecn-status"><span class="pulse-dot"></span> LIVE NETWORK</div>
</div>
""", unsafe_allow_html=True)

# Helper for Timeline
def render_timeline(step_idx):
    steps = ["108 Intake", "AI Triage", "Fleet Dispatched", "Active Transit", "ED Handover"]
    html = "<div class='timeline-container'>"
    for i, s in enumerate(steps):
        active_class = "active" if i <= step_idx else ""
        icon = "✓" if i < step_idx else (str(i+1) if i == step_idx else "○")
        html += f"<div class='timeline-step {active_class}'><div class='timeline-icon'>{icon}</div><div class='timeline-label'>{s}</div></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ==========================================
# VIEW 0: 108 ERC INTAKE CONSOLE
# ==========================================
if nav_selection == "108 ERC INTAKE CONSOLE":
    st.header("📞 108 Emergency Response Center (Intake)")
    st.caption("Layer 1: Unified Call Capture & Ecosystem Routing")
    
    render_timeline(0)
    
    with st.container(border=True):
        st.subheader("Statewide Call Funnel (Live Feb 2026 Audit Baseline)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Ringing Calls", "8,789", "Monthly Load")
        c2.metric("Calls Answered", "5,762 (65.6%)", "-34% Access Friction", delta_color="inverse")
        c3.metric("Abandoned in Queue", "1,437 (16.3%)", "High Risk", delta_color="inverse")
        c4.metric("Missed Calls", "1,475 (16.8%)", "High Risk", delta_color="inverse")
        
        st.markdown("---")
        st.markdown("### 🚦 MCECN Demand Lane Splitter")
        st.caption("Preventing routine transfers from consuming critical emergency readiness.")
        l1, l2, l3 = st.columns(3)
        with l1:
            st.error("**🚨 Emergency Queue (48%)**\n\n**647 Active Cases**\n\n*Reserved for ALS/BLS Fleet*")
        with l2:
            st.warning("**🔄 Inter-Facility Transfer (IFT) (50%)**\n\n**669 Active Cases**\n\n*Routed to Dedicated IFT Fleet*")
        with l3:
            st.info("**📅 Scheduled (ISFT) (2%)**\n\n**29 Active Cases**\n\n*Routed to Empanelled Cabs*")

    with st.container(border=True):
        st.subheader("🎧 Live Operator Queue")
        
        st.markdown("**Incoming Call: 108-MEGH-0942 (Ringing...)**")
        st.warning("⏱️ **SLA Timer:** 00:42 (Target: Dispatch within 2 minutes)")
        
        c_q1, c_q2 = st.columns(2)
        with c_q1:
            st.text_input("Caller Location", "Mawphlang, East Khasi Hills")
            st.text_input("Raw Chief Complaint", "Severe breathing issue, turning blue")
        with c_q2:
            st.selectbox("MCECN Lane Assignment", ["🚨 Emergency Response", "🔄 Hospital Transfer (IFT)"])
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡ Escalate to MCECN Triage Intelligence", type="primary", use_container_width=True):
                st.success("Call Captured & Logged. Proceed to **REFERRAL INITIATION** tab to execute AI Clinical Triage.")

# ==========================================
# VIEW 1: REFERRAL INITIATION
# ==========================================
elif nav_selection == "REFERRAL INITIATION":
    st.header("Clinical Triage & Referral")
    st.caption("Layer 2: Secure, dual-vector triage and topography-aware facility matching.")
    render_timeline(1)
    
    with st.container(border=True):
        st.subheader("1. Patient Physiology & Context")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            patient_name = st.text_input("Patient Name (E2EE)", "John Doe")
            age = st.number_input("Age (years)", 0, 120, 35)
            pregnant = st.checkbox("Pregnant", value=False)
        with c2:
            rr = st.number_input("RR", 0, 80, 22)
            spo2 = st.number_input("SpO₂", 50, 100, 92)
            avpu = st.selectbox("AVPU", ["A","V","P","U"], index=0)
        with c3:
            hr = st.number_input("HR", 0, 250, 118)
            sbp = st.number_input("SBP", 0, 300, 92)
            temp = st.number_input("Temp °C", 30.0, 43.0, 38.4, step=0.1)
        with c4:
            src_lat = st.number_input("Origin Latitude", value=25.586936, format="%.6f") 
            src_lon = st.number_input("Origin Longitude", value=91.809418, format="%.6f")

    with st.container(border=True):
        st.subheader("2. Provisional Diagnosis")
        col_b, col_d = st.columns(2)
        with col_b:
            bundle = st.selectbox("Case Bundle", sorted(icd_df["bundle"].unique().tolist()))
        with col_d:
            dfb = icd_df[icd_df["bundle"] == bundle].copy()
            dx = st.selectbox("Select Diagnosis", dfb["label"].tolist())
            icd_row = dfb[dfb["label"] == dx].iloc[0].to_dict()
        required_caps = [x.strip() for x in (icd_row.get("default_caps","") or "").split(";") if x.strip()]

    with st.container(border=True):
        st.subheader("3. Clinical Rationale & Pre-Transfer Stabilization")
        auto_rationales = [f"General escalation of care required for {dx}."]
        caps_lower = [c.lower() for c in required_caps]
        
        if "neurosurgeon" in caps_lower or "neurosurgery" in caps_lower: auto_rationales.insert(0, "Emergent Neurosurgical evaluation required.")
        if "cathlab" in caps_lower or "cardiologist" in caps_lower: auto_rationales.insert(0, "Requires emergent Cath Lab activation.")
        if "icu" in caps_lower or "ventilator" in caps_lower: auto_rationales.insert(0, "Requires immediate ICU admission.")
        if "bloodbank" in caps_lower or "surgeon" in caps_lower: auto_rationales.insert(0, "Requires emergent surgical control.")
        if bundle == "Maternal": auto_rationales.insert(0, "High-risk obstetric emergency requiring tertiary OBGYN OT.")
        
        selected_rationale = st.selectbox("Standardized Reason for Transfer", auto_rationales)
        
        st.markdown("---")
        initiator_role = st.radio("Initiating Provider Status:", ["Attending Medical Officer (Doctor)", "Staff Nurse / ANM (No Doctor on Duty)"], horizontal=True)
        
        interventions_str = icd_row.get("default_interventions", "")
        available_interventions = [x.strip() for x in str(interventions_str).split(";") if x.strip()]
        
        completed_interventions = []
        surgical_override = False
        
        if available_interventions:
            completed_interventions = st.multiselect("Select life-saving interventions already administered:", available_interventions)
            
        if initiator_role == "Attending Medical Officer (Doctor)":
            surgical_override = st.checkbox("⚠️ **SURGICAL EMERGENCY OVERRIDE:** Patient requires immediate surgical control unavailable here. Hemodynamic resuscitation is ongoing in transit.")

    vitals = {"hr": hr, "rr": rr, "sbp": sbp, "temp": temp, "spo2": spo2, "avpu": avpu}
    context = {"age": age, "pregnant": pregnant, "o2_device": "Air", "spo2_scale": 1, "behavior": "Normal"}
    
    try:
        triage_color, meta = validated_triage_decision(vitals=vitals, icd_row=icd_row, context=context)
    except Exception as e:
        st.error(f"🚨 Clinical Engine Failure: {e}")
        st.stop()

    with st.container(border=True):
        st.subheader("4. Dual-Vector Triage Result")
        pill = {"RED":"CRITICAL (RED)", "YELLOW":"URGENT (YELLOW)", "GREEN":"STABLE (GREEN)"}[triage_color]
        if triage_color == "RED": st.error(pill)
        elif triage_color == "YELLOW": st.warning(pill)
        else: st.success(pill)
        st.markdown(f"**Primary Driver:** {meta['primary_driver']} | **Reason:** {meta['reason']} | **Severity Index:** {meta['severity_index']:.2f}")

    with st.container(border=True):
        st.subheader("5. Facility Matching (Gated Clinical Safety)")
        
        if st.button("🔍 Run AI Facility Matcher", type="primary", use_container_width=True):
            with st.spinner("Analyzing topography, capabilities, and capacity..."):
                origin = (float(src_lat), float(src_lon))
                results = []
                sev = float(meta.get("severity_index", 0.0))

                for _, row in facilities_df.iterrows():
                    f_dict = row.to_dict()
                    dest = (float(f_dict.get("lat", 0.0)), float(f_dict.get("lon", 0.0)))
                    f_dict["ownership"] = str(f_dict.get("ownership", "Private") or "Private")
                    try: route_eta = get_eta(origin, dest, speed_kmh=40.0, is_hilly_terrain=True)
                    except: route_eta = 999.0

                    try:
                        score, details = calculate_facility_score(
                            facility=f_dict, required_caps=required_caps, eta=route_eta,
                            triage_color=triage_color, severity_index=sev, case_type=bundle
                        )
                    except: continue

                    if score > 0 or details.get("gate_capacity") == "WARNING_ED_STABILIZATION_ONLY":
                        results.append({
                            "facility": f_dict["name"], "score": score, "eta": round(route_eta, 1),
                            "ownership": f_dict["ownership"], "mortality_risk": mortality_risk(sev, route_eta, pathology=bundle), "scoring_details": details
                        })

                st.session_state.match_results = sorted(results, key=lambda x: (-x["score"], x["eta"]))

        if st.session_state.get('match_results') is not None:
            results = st.session_state.match_results
            
            if not results:
                st.error("CRITICAL ALERT: ZERO STATEWIDE CAPACITY. Initiate on-site ED stabilization.")
            else:
                st.success(f"✅ AI routing complete. Found {len(results)} viable destinations.")
                radio_options = [f"{r['facility']} (Score: {r['score']})" for r in results[:5]]
                selected_option = st.radio("Select Destination Facility:", radio_options)
                selected_fac_name = selected_option.split(" (Score:")[0]
                selected_fac = next(r for r in results if r["facility"] == selected_fac_name)
                
                with st.expander(f"📊 Explainable AI Logic for {selected_fac['facility']}"):
                    details = selected_fac["scoring_details"]
                    tier = details.get('clinical_tier', 'Tier 1: Definitive Care')
                    st.markdown(f"**1. Clinical Capability Escalon ({tier})**")
                    if details.get('gate_capacity') == "WARNING_ED_STABILIZATION_ONLY":
                        st.error("⚠️ **Capacity Gate Override:** ICU Full. Facility selected for immediate ED Resuscitation ONLY.")

                dispatch_disabled = False
                dispatch_warning = ""
                
                if triage_color == "RED":
                    if initiator_role == "Attending Medical Officer (Doctor)":
                        if not completed_interventions and not surgical_override:
                            dispatch_disabled = True
                            dispatch_warning = "🚨 **IFT PROTOCOL LOCK:** Patient must be hemodynamically stabilized prior to transport. Select interventions administered or authorize Surgical Override."
                    else:
                        dispatch_warning = "⚠️ **NURSE-LED INITIATION:** No doctor on site. Patient un-resuscitated. Receiving ED will be pre-alerted."
                
                if dispatch_warning:
                    if dispatch_disabled: st.error(dispatch_warning)
                    else: st.warning(dispatch_warning)

                if st.button("🚀 Initiate E2EE Transfer & Dispatch Transport", type="primary", disabled=dispatch_disabled):
                    ideal_fleet = "ALS" if triage_color == "RED" or meta['severity_index'] >= 0.6 else "BLS"
                    allocated_fleet, fleet_status = ideal_fleet, "OPTIMAL"
                    model, plate, driver, emt_name = f"{ideal_fleet}-{random.randint(101,250)}", f"ML-05-{random.randint(1000,9999)}", random.choice(PILOTS), random.choice(EMTS)
                    
                    st.session_state.active_case = {
                        "patient_name": patient_name, "age": age, "vitals": vitals, "diagnosis": dx,
                        "bundle": bundle, "triage_color": triage_color, "severity_index": meta['severity_index'],
                        "destination": selected_fac, "dispatch_time": datetime.now().strftime("%H:%M:%S"),
                        "rationale": selected_rationale, "interventions": completed_interventions, 
                        "rejection_log": [], "medical_orders": [],
                        "fleet": {"ideal": ideal_fleet, "allocated": allocated_fleet, "status": fleet_status, "driver": driver, "plate": plate, "vehicle": model, "emt": emt_name},
                        "nurse_led": (initiator_role == "Staff Nurse / ANM (No Doctor on Duty)"),
                        "surgical_override": surgical_override
                    }
                    st.session_state.transfer_initiated = True
                    st.session_state.patient_accepted = False
                    st.session_state.match_results = None 
                    st.rerun()

# ==========================================
# VIEW 2: ACTIVE TRANSIT TELEMETRY
# ==========================================
elif nav_selection == "ACTIVE TRANSIT TELEMETRY":
    st.header("Active Transit & Telemetry Dashboard")
    render_timeline(3)

    with st.container(border=True):
        if not st.session_state.active_case:
            st.info("No active dispatch. Initiate a transfer from the Referral tab.")
        else:
            case = st.session_state.active_case
            dest = case["destination"]
            if "transit_log" not in case: case["transit_log"] = []

            st.error(f"🚑 PRIORITY {case['triage_color']} EN ROUTE TO {dest['facility'].upper()} [{case['fleet']['allocated']} UNIT]")
            st.markdown(f"**Unit Details:** {case['fleet']['vehicle']} ({case['fleet']['plate']}) | **Pilot:** {case['fleet']['driver']} | **Lead EMT:** {case['fleet']['emt']}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Topography-Adjusted ETA", f"{dest['eta']} min")
            c2.metric("Modeled Mortality Risk", f"{dest.get('mortality_risk', 0.0)}%")
            c3.metric("Severity Index", f"{case.get('severity_index', 0.0):.2f}")
            c4.metric("Dispatch Time", case["dispatch_time"])

            st.markdown("---")

            if case.get("medical_orders"):
                st.warning("🎙️ **INCOMING MEDICAL COMMAND ORDER**")
                st.markdown(f"### 🩺 {case['medical_orders'][-1]['order']}")
                st.markdown("---")
            
            col_actions, col_vitals, col_traffic = st.columns([1.5, 1.2, 1.2])
            with col_actions:
                st.subheader("⚡ Med-Legal Ledger")
                a1, a2, a3 = st.columns(3)
                if a1.button("💉 Push ACLS", use_container_width=True): case["transit_log"].append({"action": "ACLS Administered"})
                if a2.button("🫁 Airway", use_container_width=True): case["transit_log"].append({"action": "Airway Secured"})
                if a3.button("🩸 Tourniquet", use_container_width=True): case["transit_log"].append({"action": "Hemorrhage Control"})
                    
            with col_vitals:
                st.subheader("📡 Live Bluetooth Telemetry")
                st.caption("Streamed from Unit Defibrillator")
                
                live_hr = case['vitals']['hr'] + random.randint(-3, 3)
                live_spo2 = case['vitals']['spo2'] + random.randint(-1, 1)
                
                hr_color = "inverse" if live_hr > 130 or live_hr < 50 else "normal"
                spo2_color = "inverse" if live_spo2 < 90 else "normal"
                
                v1, v2 = st.columns(2)
                v1.metric("Live HR", f"{live_hr} bpm", delta=f"{live_hr - case['vitals']['hr']}", delta_color=hr_color)
                v2.metric("Live SpO2", f"{live_spo2} %", delta=f"{live_spo2 - case['vitals']['spo2']}", delta_color=spo2_color)
                st.markdown("**Vehicle Oxygen Reserves:** 82% 🟢")

            with col_traffic:
                st.subheader("🚧 ETA Drift Telemetry")
                traffic_delay = st.slider("Live Traffic/Blockade Delay (Mins)", 0, 60, 0)
                live_eta = dest['eta'] + traffic_delay
                st.metric("Live GPS ETA", f"{live_eta} mins", delta=f"+{traffic_delay} mins" if traffic_delay > 0 else "On Time", delta_color="inverse")
                
                if traffic_delay > 15 and case['triage_color'] == 'RED':
                    if st.button("🌐 INITIATE CIVIC OVERRIDE & ALS INTERCEPT", type="primary", use_container_width=True):
                        st.session_state.civic_override_active = True
                        st.rerun()
                        
                if st.session_state.get('civic_override_active', False):
                    st.markdown("---")
                    st.error("🚨 **STATE EMERGENCY PROTOCOL: GREEN CORRIDOR ENGAGED**")
                    with st.status("📡 Establishing handshake with Meghalaya Police CAD...", expanded=True) as status:
                        st.write("Auth Token: MCECN-AUTH-994A...")
                        time.sleep(0.8)
                        st.write("Transmitting coordinates to Highway Patrol Sub-Station...")
                        time.sleep(0.8)
                        st.write("Unit MEGH-P-42 (Interceptor) confirming intercept vector...")
                        time.sleep(1.2)
                        status.update(label="✅ POLICE ESCORT SECURED & DISPATCHED", state="complete", expanded=False)
                    c_gc1, c_gc2 = st.columns(2)
                    c_gc1.info("🚔 **Interceptor ETA to Ambulance:** 04 Mins")
                    c_gc2.warning("🚦 **Traffic Command:** Next 4 intersections locked to Green.")

# ==========================================
# VIEW 3: RECEIVING HOSPITAL
# ==========================================
elif nav_selection == "RECEIVING HOSPITAL BAY":
    st.header("Emergency Department Receiving Board")
    render_timeline(4)

    with st.container(border=True):
        case = st.session_state.get('active_case')
        if not case:
            st.info("ED Bay Clear. No incoming critical transfers.")
        else:
            dest = case["destination"]
            dest_name = dest['facility']
            st.success(f"🏥 **SECURE TERMINAL ACTIVE:** YOU ARE VIEWING AS **{dest_name.upper()}**")
            st.markdown("---")
            
            st.error(f"🚑 **INCOMING PRIORITY {case['triage_color']} AMBULANCE: ETA {dest['eta']} mins**")
            
            if case.get("nurse_led"): st.error("⚠️ **NURSE-LED PHC INITIATION: NO DOCTOR ON SITE. PREPARE IMMEDIATE ED STABILIZATION.**")
            if case.get("surgical_override"): st.warning("⚠️ **SURGICAL OVERRIDE: Patient requires immediate surgical control. Resuscitation ongoing.**")

            st.markdown("---")
            st.markdown("### 📄 Pre-Hospital Clinical Handover (ISBAR)")
            
            nurse_alert = "\n[⚠️ NURSE-LED PHC INITIATION]" if case.get("nurse_led") else ""
            surg_alert = "\n[⚠️ SURGICAL EMERGENCY OVERRIDE]" if case.get("surgical_override") else ""

            isbar_text = f"""[ISBAR CLINICAL HANDOVER]{nurse_alert}{surg_alert}
Status: PRIORITY {case['triage_color']} (Severity: {case.get('severity_index', 0.0):.2f})
I - IDENTIFICATION: Patient: {case['patient_name']}, {case['age']} Y/O
S - SITUATION: Emergency dispatch to {dest['facility']}. Provisional DX: {case['diagnosis']}
B - BACKGROUND: Bundle: {case['bundle']}. Rationale: {case.get('rationale', 'N/A')}
A - ASSESSMENT: HR: {case['vitals']['hr']} | SBP: {case['vitals']['sbp']} | RR: {case['vitals']['rr']} | SpO2: {case['vitals']['spo2']}% | Temp: {case['vitals']['temp']}°C | AVPU: {case['vitals']['avpu']}
"""
            st.code(isbar_text, language="markdown")
            
            st.markdown("---")
            st.markdown("### 🛏️ Transactional Capacity Management")
            if 'bed_locked' not in st.session_state: st.session_state.bed_locked = False
                
            if not st.session_state.bed_locked:
                if dest['scoring_details'].get('icu_beds', 0) > 0:
                    st.info(f"Available Critical Care Beds: {dest['scoring_details']['icu_beds']}")
                    if st.button("🔒 Lock 1x ICU Bed on State Registry", type="primary"):
                        with st.spinner("Writing cryptographic reservation to state ledger..."):
                            time.sleep(1)
                            st.session_state.bed_locked = True
                            st.rerun()
                else:
                    st.error("⚠️ Zero ICU Beds Available. Admitting to ED Resuscitation Bay.")
            else:
                st.success(f"🔒 **STATE GRID UPDATED:** 1x ICU Bed reserved for Unit {case['fleet']['plate']}.")

            st.markdown("---")
            st.markdown("### 🎙️ Medical Command Uplink")
            dynamic_orders = ["Select Order...", "Direct to Cath Lab (Bypass ED)", "Initiate Massive Transfusion Protocol", "Push IV Labetalol"]
            col_order, col_send = st.columns([3, 1])
            with col_order:
                selected_order = st.selectbox("Standardized Counter-Orders", dynamic_orders, label_visibility="collapsed")
            with col_send:
                if st.button("📡 Transmit Order", type="primary", use_container_width=True, disabled=(selected_order == "Select Order...")):
                    if "medical_orders" not in case: case["medical_orders"] = []
                    case["medical_orders"].append({"time": datetime.now().strftime("%H:%M:%S"), "order": selected_order})
                    st.rerun()
                    
            if not st.session_state.patient_accepted:
                st.markdown("### Decision Matrix")
                if st.button("✅ Acknowledge & Accept Patient", type="primary", use_container_width=True):
                    st.session_state.patient_accepted = True
                    st.rerun()
            else:
                st.success("✅ Patient accepted. Telemetry linked to ED monitors.")

# ==========================================
# VIEW 4: STATE COMMAND & AI
# ==========================================
elif nav_selection == "STATE COMMAND & AI":
    st.header("State Command & Control")

    if st.session_state.user_role != "State Health Command (Macro View)":
        st.error("🛑 **CLEARANCE VIOLATION**")
        st.markdown("View compartmentalized for State Health Officials only.")
    else:
        st.caption("Zero-PHI Analytics & Autonomous Logistics Engine (Data synced to Feb 2026 Audit)")

        tab_fleet, tab_geo, tab_perf, tab_econ, tab_roi = st.tabs([
            "🚨 108 System Optimization", 
            "🌍 Case-Type Intelligence", 
            "🏥 Hospital Acceptance Heatboard",
            "💰 MHIS Fiscal Governance",
            "📈 Investment ROI Simulator"
        ])

        with tab_fleet:
            st.subheader("Statewide Workforce & Deployment Stress")
            st.caption("Identifying over-burdened crews and failing bases to optimize deployment.")
            
            col_f1, col_f2 = st.columns([1.2, 1])
            with col_f1:
                st.markdown("**District Base Performance (P90 Target Misses)**")
                base_data = pd.DataFrame({
                    "Base": ["JARAIN (WJH)", "JOWAI (WJH)", "SOHIONG (EKH)", "TURA CIVIL (WGH)", "NONGPOH (RB)"],
                    "Target Met %": [27.3, 27.8, 38.1, 38.1, 62.4]
                })
                st.altair_chart(alt.Chart(base_data).mark_bar(cornerRadiusEnd=4).encode(
                    y=alt.Y('Base:N', sort='x'),
                    x=alt.X('Target Met %:Q'),
                    color=alt.condition(alt.datum['Target Met %'] < 40, alt.value('#E11D48'), alt.value('#3B82F6'))
                ).properties(height=250), use_container_width=True)
                
            with col_f2:
                st.error("⚠️ **CRITICAL WORKFORCE ALERT**")
                st.markdown("Feb 2026 Audit identifies **Manpower (81 cases)** as the absolute leading cause of fleet deassignments.")
                st.markdown("""
                **AI Repositioning Recommendations:**
                * 🔄 **Shift 2 BLS Relievers** to Jarain (Severe Fatigue Risk).
                * 🚑 **Upgrade Jowai Corridor** to ALS coverage.
                * 🌙 **Night Transport Risk** identified in West Garo Hills.
                """)

        with tab_geo:
            st.subheader("Meghalaya Case-Type Intelligence (Feb 2026)")
            st.caption("Real-time breakdown of critical emergency categories.")
            
            c1, c2 = st.columns(2)
            with c1:
                path_data = pd.DataFrame({
                    "Pathology": ["Obstetric Emergency", "RTA (Trauma)", "Abdominal Pain", "Breathing Difficulty", "Unconscious"],
                    "Volume": [218, 63, 56, 44, 41]
                })
                st.altair_chart(alt.Chart(path_data).mark_arc(innerRadius=40).encode(
                    theta=alt.Theta(field="Volume", type="quantitative"),
                    color=alt.Color(field="Pathology", type="nominal"),
                    tooltip=["Pathology", "Volume"]
                ).properties(height=300), use_container_width=True)
                
            with c2:
                st.info("💡 **Clinical Routing Insight:**")
                st.markdown("""
                **Obstetric & Neonatal (218 cases)** is the single largest high-volume cluster. 
                * **Current Failure:** Outcomes are highly pathway-dependent, but routing is currently blind to NICU/OBGYN availability.
                * **MCECN Solution:** Dual-Vector triage now strictly forces MEOWS physiological scoring and routes exclusively to facilities with proven Maternal OT capability.
                """)

        with tab_perf:
            st.subheader("Hospital Acceptance & Diversion Heatboard")
            st.caption("Live statewide command summary showing blind referral reductions.")
            
            heat_data = pd.DataFrame({
                "Hospital": ["NEIGRIHMS", "Shillong Civil", "Woodland Hosp", "Bethany Hosp", "Tura Civil"],
                "Acceptance": ["94%", "42%", "88%", "85%", "71%"],
                "Status": ["Green", "Red", "Green", "Amber", "Amber"],
                "Primary Diversion Reason": ["None", "Zero ICU Beds (Surge)", "None", "CT Scanner Down", "Specialist Scrubbed In"]
            })
            
            def color_status(val):
                if val == 'Green': return 'color: #10B981; font-weight: bold;'
                elif val == 'Red': return 'color: #E11D48; font-weight: bold;'
                return 'color: #F59E0B; font-weight: bold;'
            
            st.dataframe(heat_data.style.map(color_status, subset=['Status']), use_container_width=True, hide_index=True)

        with tab_econ:
            st.subheader("MHIS Health Economics & Fiscal Governance")
            st.markdown("**(Deterministic Projections Based on Audit Baselines)**")
            
            k1, k2, k3 = st.columns(3)
            k1.metric("State Total Referrals", "1,345")
            k2.metric("MHIS Fiscal Leakage", "₹ 2.4 Cr", delta="- Drain to Private Sector", delta_color="inverse")
            k3.metric("ALOS Uncompensated Burden", "₹ 18.5 L", delta="Bed-Blocking Penalties", delta_color="inverse")
            st.caption("Leakage is driven entirely by zero-capacity diversions at Government hubs.")

        with tab_roi:
            st.subheader("📈 Policy Investment ROI Simulator")
            st.caption("Dynamically forecast the impact of capital investments on the 108 network.")
            
            st.markdown("### Step 1: Adjust Capital Allocation")
            col_roi1, col_roi2 = st.columns(2)
            with col_roi1:
                als_added = st.slider("➕ Add New ALS Ambulances to Fleet", 0, 50, 0)
                bls_added = st.slider("➕ Add New BLS Reliever Crews", 0, 30, 0)
            with col_roi2:
                ift_split = st.checkbox("🔄 Activate Dedicated IFT Fleet (Split Queue)")
                hosp_upgrade = st.checkbox("🏥 Fund 20-Bed Step-Down Ward at Shillong Civil")
                
            st.markdown("### Step 2: Projected Statewide Outcomes")
            
            # ROI Math Engine
            base_p90 = 51.0
            base_compliance = 60.3
            
            new_p90 = base_p90 - (als_added * 0.3) - (bls_added * 0.1) - (6.0 if ift_split else 0.0) - (2.0 if hosp_upgrade else 0.0)
            new_comp = base_compliance + (als_added * 0.6) + (bls_added * 0.2) + (12.0 if ift_split else 0.0) + (4.0 if hosp_upgrade else 0.0)
            
            new_p90 = max(20.0, new_p90)
            new_comp = min(99.0, new_comp)
            lives_saved = int((new_comp - base_compliance) * 4.2)
            
            r1, r2, r3 = st.columns(3)
            r1.metric("Rural P90 Response Time", f"{new_p90:.1f} mins", f"{new_p90 - base_p90:.1f} mins vs Baseline", delta_color="inverse")
            r2.metric("NHM Target Compliance (≤30m)", f"{new_comp:.1f} %", f"+{new_comp - base_compliance:.1f} % vs Baseline")
            r3.metric("Estimated Critical Lives Saved", f"{lives_saved} / month", "+ Preventable Mortalities Avoided")
            
            if new_comp > 85.0:
                st.success("🌟 **OPTIMAL SYSTEM ARCHITECTURE ACHIEVED:** The combination of dedicated IFT routing and targeted fleet expansion successfully bends the mortality curve into NHM compliance.")
