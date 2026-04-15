import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
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
# 🎨 BRANDING & MATERIAL ICONS CSS INJECTION
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');
    
    .stApp { background-color: #F8FAFC; font-family: 'Inter', 'Segoe UI', sans-serif; }
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
    
    /* Material Icon Classes */
    .icon-blue { color: #1E3A8A; vertical-align: bottom; padding-right: 8px; font-size: 1.8rem; }
    .icon-red { color: #E11D48; vertical-align: bottom; padding-right: 8px; font-size: 1.8rem; }
    .icon-slate { color: #64748B; vertical-align: bottom; padding-right: 8px; font-size: 1.5rem; }
    .icon-green { color: #10B981; vertical-align: bottom; padding-right: 8px; font-size: 1.5rem; }
    
    /* The MCECN Global Brand Header */
    .mcecn-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        padding: 1.8rem 2.5rem; border-radius: 16px; color: white;
        margin-bottom: 2rem; box-shadow: 0 10px 30px -5px rgba(30, 58, 138, 0.4);
        display: flex; justify-content: space-between; align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .mcecn-title { font-size: 2.8rem; font-weight: 900; margin: 0; line-height: 1.1; letter-spacing: 1.5px; background: linear-gradient(to right, #48BBEA, #FFFFFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .mcecn-subtitle { font-size: 1.05rem; font-weight: 500; color: #94A3B8; margin: 5px 0 0 0; letter-spacing: 2px; text-transform: uppercase; }
    .mcecn-status { background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.4); color: #22D3EE; padding: 0.6rem 1.2rem; border-radius: 50px; font-weight: 700; font-size: 0.9rem; display: flex; align-items: center; gap: 10px; letter-spacing: 1px; }
    .pulse-dot { height: 10px; width: 10px; background-color: #22D3EE; border-radius: 50%; display: inline-block; box-shadow: 0 0 10px #22D3EE; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 211, 238, 0.7); } 70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(34, 211, 238, 0); } 100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 211, 238, 0); } }
    
    /* UI Cards & Buttons */
    [data-testid="stVerticalBlockBorderWrapper"] { background-color: #FFFFFF !important; border-radius: 14px !important; border: 1px solid #E2E8F0 !important; box-shadow: 0px 4px 20px rgba(15, 23, 42, 0.04) !important; padding: 1.8rem !important; }
    button[kind="primary"] { background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; padding: 0.6rem 1.2rem !important; }
    h1, h2, h3, h4 { color: #0F172A !important; font-weight: 800 !important; }
    [data-testid="stMetricValue"] { color: #0F172A !important; font-size: 2.4rem !important; font-weight: 900 !important; }
    
    /* Input formatting */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input { border-radius: 8px !important; border: 1px solid #CBD5E1 !important; background-color: #F8FAFC !important; color: #0F172A !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

# --- Architecture Imports ---
from clinical_engine import validated_triage_decision
from scoring_engine import calculate_facility_score
from routing_engine import get_eta, haversine_km
from analytics_engine import mortality_risk
from synthetic_cases import seed_synthetic_referrals_v2

# --- State Management & Fleet Rosters ---
if 'active_case' not in st.session_state: st.session_state.active_case = None
if 'synthetic_data' not in st.session_state: st.session_state.synthetic_data = None
if 'transfer_initiated' not in st.session_state: st.session_state.transfer_initiated = False
if 'patient_accepted' not in st.session_state: st.session_state.patient_accepted = False
if 'match_results' not in st.session_state: st.session_state.match_results = None
if 'civic_override_active' not in st.session_state: st.session_state.civic_override_active = False
if 'bed_locked' not in st.session_state: st.session_state.bed_locked = False

# --- LIVE QUEUE & HANDOFF STATE (PATCH 1) ---
if 'draft_case' not in st.session_state: st.session_state.draft_case = None
if 'main_nav' not in st.session_state: st.session_state.main_nav = "108 ERC INTAKE CONSOLE"
if 'call_queue' not in st.session_state:
    st.session_state.call_queue = [
        {"id": "108-MEGH-0942", "loc": "Mawphlang, EKH", "lat": 25.449, "lon": 91.758, "complaint": "Severe breathing issue, turning blue"},
        {"id": "108-MEGH-0943", "loc": "Tikrikilla, WGH", "lat": 25.926, "lon": 90.134, "complaint": "Pregnant woman, heavy bleeding"},
        {"id": "108-MEGH-0944", "loc": "Nongpoh NH-40", "lat": 25.900, "lon": 91.879, "complaint": "Road traffic accident, severe head injury"},
        {"id": "108-MEGH-0945", "loc": "Cherrapunji, EKH", "lat": 25.270, "lon": 91.732, "complaint": "Elderly male, sudden weakness on right side"},
        {"id": "108-MEGH-0946", "loc": "Tura Civil, WGH", "lat": 25.514, "lon": 90.203, "complaint": "Massive chest pain, sweating profusely"},
        {"id": "108-MEGH-0947", "loc": "Jowai, WJH", "lat": 25.448, "lon": 92.199, "complaint": "High fever, seizures in a toddler"},
        {"id": "108-MEGH-0948", "loc": "Baghmara, SGH", "lat": 25.201, "lon": 90.622, "complaint": "Snake bite, leg swelling rapidly"},
        {"id": "108-MEGH-0949", "loc": "Khliehriat, EJH", "lat": 25.361, "lon": 92.365, "complaint": "Severe abdominal pain, vomiting blood"},
        {"id": "108-MEGH-0950", "loc": "Dawki, WJH", "lat": 25.185, "lon": 92.015, "complaint": "Fell from height, suspected spinal injury"},
        {"id": "108-MEGH-0951", "loc": "Williamnagar, EGH", "lat": 25.586, "lon": 90.612, "complaint": "Child drank kerosene"},
        {"id": "108-MEGH-0952", "loc": "Mairang, EWKH", "lat": 25.564, "lon": 91.638, "complaint": "Asthma attack, inhaler not working"},
        {"id": "108-MEGH-0953", "loc": "Shillong PB, EKH", "lat": 25.578, "lon": 91.883, "complaint": "Unconscious female, found on street"}
    ]

PILOTS = ["Khraw", "Mewan", "Donbok", "Bantei", "Pynskhem", "Lamphrang"]
EMTS = ["Dr. Sarah", "Paramedic Grace", "Paramedic John", "Nurse Riba"]

# --- DATA LOADER ---
@st.cache_data
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

# --- Plotly Gantt Helper ---
def render_plotly_timeline(current_step):
    steps = ["108 Call Intake", "AI Clinical Triage", "Fleet Dispatch", "Active Transit", "ED Handover"]
    now = datetime.now()
    
    data = []
    for i, step in enumerate(steps):
        status = "Completed" if i < current_step else ("Active" if i == current_step else "Pending")
        start_time = now - timedelta(minutes=(5 - i)*2)
        end_time = start_time + timedelta(minutes=2)
        data.append({"Stage": step, "Start": start_time, "Finish": end_time, "Status": status})
        
    df = pd.DataFrame(data)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Stage", color="Status",
                      color_discrete_map={"Completed": "#94A3B8", "Active": "#3B82F6", "Pending": "#E2E8F0"})
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(showgrid=True, gridcolor='#F1F5F9', linecolor='#CBD5E1', showticklabels=False, title=""),
                      yaxis=dict(showgrid=False, title="", tickfont=dict(size=13, weight='bold', color='#0F172A')),
                      showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=180)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- Altair Tufte Styler Helper ---
def apply_tufte_theme(chart):
    return chart.configure_view(strokeWidth=0).configure_axis(
        grid=False, domainColor="#CBD5E1", tickColor="#CBD5E1", labelColor="#64748B", titleColor="#0F172A",
        labelFont='Inter, sans-serif', titleFont='Inter, sans-serif'
    )
    
# --- Sidebar Navigation & RBAC Security ---
with st.sidebar:
    st.markdown("""
        <div style='padding-bottom: 2rem;'>
            <h2 style='color: #1E3A8A !important; margin-bottom: 0;'>MCECN OS</h2>
            <p style='color: #64748B; font-size: 0.85rem; font-weight: 600; margin-top: 0; letter-spacing: 1px;'>ENTERPRISE v7.0</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h4 style='color: #0F172A; font-size: 1rem;'><span class='material-symbols-outlined icon-slate' style='font-size:1.2rem;'>admin_panel_settings</span> Access Control</h4>", unsafe_allow_html=True)
    simulated_role = st.selectbox("Simulate Active User Login:", [
        "State Health Command (Macro View)", "Authorized Community Node (ASHA/CFR)", "Citizen Self-Booking App",
        "Director: NEIGRIHMS", "Director: Woodland Hospital", "108 ERC Dispatcher"
    ], label_visibility="collapsed")
    st.session_state.user_role = simulated_role 
    
    st.markdown("---")
    st.markdown("<h4 style='color: #0F172A; font-size: 1rem;'><span class='material-symbols-outlined icon-slate' style='font-size:1.2rem;'>explore</span> System Navigation</h4>", unsafe_allow_html=True)
    
    # --- PROGRAMMABLE NAVIGATION (PATCH 2) ---
    nav_selection = st.radio(
        "Select Module:",
        ["108 ERC INTAKE CONSOLE", "REFERRAL INITIATION", "ACTIVE TRANSIT TELEMETRY", "RECEIVING HOSPITAL BAY", "STATE COMMAND & AI"],
        key="main_nav",
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Secure Connection Established")

st.markdown("""
<div class="mcecn-header">
    <div class="mcecn-title-wrapper">
        <h1 class="mcecn-title">MCECN</h1>
        <p class="mcecn-subtitle">Meghalaya Comprehensive Emergency Care Network</p>
    </div>
    <div class="mcecn-status"><span class="pulse-dot"></span> LIVE NETWORK</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# VIEW 0: 108 ERC INTAKE CONSOLE
# ==========================================
if st.session_state.main_nav == "108 ERC INTAKE CONSOLE":
    st.markdown("<h2><span class='material-symbols-outlined icon-blue'>headset_mic</span> 108 Emergency Response Center</h2>", unsafe_allow_html=True)
    st.caption("Layer 1: Unified Call Capture & Ecosystem Routing")
    
    render_plotly_timeline(0)
    
    with st.container(border=True):
        st.markdown("<h3><span class='material-symbols-outlined icon-slate'>monitoring</span> Statewide Call Funnel (Live Feb 2026 Audit Baseline)</h3>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Ringing Calls", "8,789", "Monthly Load")
        c2.metric("Calls Answered", "5,762 (65.6%)", "-34% Access Friction", delta_color="inverse")
        c3.metric("Abandoned in Queue", "1,437 (16.3%)", "High Risk", delta_color="inverse")
        c4.metric("Missed Calls", "1,475 (16.8%)", "High Risk", delta_color="inverse")
        
        st.markdown("---")
        st.markdown("<h3><span class='material-symbols-outlined icon-slate'>call_split</span> MCECN Demand Lane Splitter</h3>", unsafe_allow_html=True)
        st.caption("Preventing routine transfers from consuming critical emergency readiness.")
        l1, l2, l3 = st.columns(3)
        with l1:
            st.error("**Emergency Queue (48%)**\n\n**647 Active Cases**\n\n*Reserved for ALS/BLS Fleet*")
        with l2:
            st.warning("**Inter-Facility Transfer (IFT) (50%)**\n\n**669 Active Cases**\n\n*Routed to Dedicated IFT Fleet*")
        with l3:
            st.info("**Scheduled (ISFT) (2%)**\n\n**29 Active Cases**\n\n*Routed to Empanelled Cabs*")

    # --- DYNAMIC QUEUE & ESCALATION (PATCH 3) ---
    with st.container(border=True):
        st.markdown("<h3><span class='material-symbols-outlined icon-slate'>support_agent</span> Live Operator Queue (12 Active Cases)</h3>", unsafe_allow_html=True)
        st.warning("SLA Timer: 00:42 (Target: Dispatch within 2 minutes)")
        
        c_q1, c_q2 = st.columns([1.5, 1])
        with c_q1:
            call_options = [f"{c['id']} - {c['loc']}" for c in st.session_state.call_queue]
            selected_call_str = st.selectbox("Select Active Call from Priority Queue:", call_options)
            active_call = next(c for c in st.session_state.call_queue if f"{c['id']} - {c['loc']}" == selected_call_str)
            
            st.text_input("Caller Location (Triangulated)", active_call['loc'], disabled=True)
            st.text_input("Raw Chief Complaint (Transcribed)", active_call['complaint'], disabled=True)
            
        with c_q2:
            st.selectbox("MCECN Lane Assignment", ["Emergency Response", "Hospital Transfer (IFT)"])
            st.markdown("<br>", unsafe_allow_html=True)
            
            # The Fix: Using a Streamlit callback to safely alter widget state
            def trigger_escalation(call_payload):
                st.session_state.draft_case = call_payload
                st.session_state.main_nav = "REFERRAL INITIATION"

            st.button(
                "Escalate to MCECN Triage Intelligence", 
                type="primary", 
                use_container_width=True, 
                on_click=trigger_escalation, 
                args=(active_call,)
            )

# ==========================================
# VIEW 1: REFERRAL INITIATION
# ==========================================
elif st.session_state.main_nav == "REFERRAL INITIATION":
    st.markdown("<h2><span class='material-symbols-outlined icon-blue'>medical_services</span> Clinical Triage & Referral</h2>", unsafe_allow_html=True)
    st.caption("Layer 2: Secure, dual-vector triage and topography-aware facility matching.")
    render_plotly_timeline(1)
    
    # Optional gateway rendering if ASHA/Citizen
    if st.session_state.user_role in ["Authorized Community Node (ASHA/CFR)", "Citizen Self-Booking App"]:
        is_asha = (st.session_state.user_role == "Authorized Community Node (ASHA/CFR)")
        if is_asha:
            st.caption("Authorized Node: Initiating heavily subsidized Tier-3 Health Ride for vulnerable populations.")
        else:
            st.caption("Direct-to-Citizen App: Book a standard-rate Health Cab to your nearest clinic.")
            
        with st.container(border=True):
            st.subheader("Patient Request Details")
            c1, c2, c3 = st.columns(3)
            patient_name = c1.text_input("Patient Name", "Local Resident")
            age = c2.number_input("Age", 0, 120, 45)
            symptom_cat = c3.selectbox("Primary Complaint", ["Orthopedic (Fracture/Sprain) [GREEN]", "Fever/Infection [GREEN]", "Maternal (Routine Checkup) [GREEN]", "Moderate Abdominal Pain [YELLOW]", "Severe Chest Pain [RED]", "Unconscious / Bleeding [RED]"])
            c_lat, c_lon = st.columns(2)
            src_lat = c_lat.number_input("Pickup Latitude", value=25.586936, format="%.6f") 
            src_lon = c_lon.number_input("Pickup Longitude", value=91.809418, format="%.6f")
            
            is_critical = "[RED]" in symptom_cat
            is_yellow = "[YELLOW]" in symptom_cat
            
            if st.button("Find Healthcare Facilities & Check Fleet", type="primary"):
                with st.spinner("Calculating Topography..."):
                    time.sleep(1.5)
                    origin = (float(src_lat), float(src_lon))
                    results = []
                    for _, row in facilities_df.iterrows():
                        f_dict = row.to_dict()
                        dest = (float(f_dict.get("lat", 0.0)), float(f_dict.get("lon", 0.0)))
                        try: route_eta = get_eta(origin, dest, speed_kmh=40.0, is_hilly_terrain=True)
                        except: route_eta = 999.0
                        prox_score = max(0, 100 - (route_eta * 1.5))
                        results.append({"facility": f_dict["name"], "eta": round(route_eta, 1), "score": prox_score, "details": f_dict})
                    st.session_state.asha_match_results = sorted(results, key=lambda x: x["eta"])
                    st.session_state.demo_als_avail = random.choices([True, False], weights=[0.2, 0.8])[0]
                    st.session_state.demo_bls_avail = random.choices([True, False], weights=[0.2, 0.8])[0]
            
            if st.session_state.get('asha_match_results') is not None:
                res = st.session_state.asha_match_results
                radio_opts = [f"{r['facility']} (ETA: {r['eta']} mins)" for r in res[:5]]
                sel_opt = st.radio("Select Destination:", radio_opts)
                
                if is_critical:
                    als_avail, bls_avail = st.session_state.demo_als_avail, st.session_state.demo_bls_avail
                    if als_avail or bls_avail:
                        st.success("State Ambulance Available.")
                        button_text, allocated_fleet, fleet_status = "Dispatch Emergency Ambulance", ("ALS" if als_avail else "BLS"), ("OPTIMAL" if als_avail else "DOWNGRADE_RISK")
                    else:
                        st.error("STATE FLEET EXHAUSTED.")
                        st.warning("CONTINUITY OF CARE FAILSAFE: Authorizing Scoop & Run via empanelled Cab.")
                        button_text, allocated_fleet, fleet_status = "AUTHORIZE EMERGENCY CAB OVERRIDE", "CAB", "CRITICAL_CAB_FALLBACK"
                else:
                    button_text, allocated_fleet, fleet_status = "Dispatch Standard Health Cab", "CAB", "OPTIMAL"

                if st.button(button_text, type="primary"):
                    sel_fac_name = sel_opt.split(" (ETA:")[0]
                    sel_fac_details = next(r for r in res if r["facility"] == sel_fac_name)["details"]
                    sel_fac_details["scoring_details"] = {"gate_capacity": "PASSED", "clinical_tier": "Basic"}
                    
                    if allocated_fleet == "ALS": model, plate, driver, emt_name = f"ALS-{random.randint(101,130)}", f"ML-05-A-{random.randint(1000,9999)}", random.choice(PILOTS), random.choice(EMTS)
                    elif allocated_fleet == "BLS": model, plate, driver, emt_name = f"BLS-{random.randint(201,250)}", f"ML-05-B-{random.randint(1000,9999)}", random.choice(PILOTS), random.choice(EMTS)
                    else: driver, plate, model, emt_name = random.choice(PILOTS), random.choice(["ML-05-C-8842", "ML-05-T-1123"]), random.choice(["Maruti Swift", "Tata Sumo"]), None
                        
                    triage_col = "RED" if is_critical else ("YELLOW" if is_yellow else "GREEN")
                    sev_idx = 0.85 if is_critical else (0.45 if is_yellow else 0.1)
                    
                    st.session_state.active_case = {
                        "patient_name": patient_name, "age": age, "vitals": {"hr": 110 if is_critical else 80, "rr": 24 if is_critical else 16, "sbp": 90 if is_critical else 120, "temp": 37.0, "spo2": 88 if is_critical else 98, "avpu": "V" if is_critical else "A"}, 
                        "diagnosis": symptom_cat.split(" [")[0], "bundle": "Trauma/Cardiac" if is_critical else "Standard", "triage_color": triage_col, "severity_index": sev_idx,
                        "destination": {"facility": sel_fac_name, "eta": next(r for r in res if r["facility"] == sel_fac_name)["eta"], "scoring_details": {"gate_capacity": "PASSED"}}, 
                        "dispatch_time": datetime.now().strftime("%H:%M:%S"), "rationale": "Direct Booking", "interventions": [], "rejection_log": [], "medical_orders": [],
                        "fleet": {"ideal": "ALS" if is_critical else "CAB", "allocated": allocated_fleet, "status": fleet_status, "driver": driver, "plate": plate, "vehicle": model, "emt": emt_name},
                        "nurse_led": False, "surgical_override": False
                    }
                    st.session_state.transfer_initiated = True
                    st.session_state.patient_accepted = False
                    if fleet_status == "CRITICAL_CAB_FALLBACK": st.session_state.civic_override_active = True
                    st.rerun()
        st.stop()

    # --- STANDARD HOSPITAL-TO-HOSPITAL VIEW ---
    with st.container(border=True):
        st.markdown("<h3><span class='material-symbols-outlined icon-slate'>vital_signs</span> 1. Patient Physiology & Context</h3>", unsafe_allow_html=True)
        
        # --- REVISED HANDOFF CATCHER (EXPLICIT 108 ATTRIBUTION) ---
        draft = st.session_state.get('draft_case')
        if draft:
            # Custom HTML for an aggressive, enterprise-grade CAD Alert
            st.markdown(f"""
            <div style='background-color: #FEF2F2; border: 1px solid #EF4444; border-left: 6px solid #EF4444; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
                <h4 style='color: #B91C1C; margin-top: 0; margin-bottom: 10px;'><span class='material-symbols-outlined' style='vertical-align: bottom;'>emergency</span> 108 FIELD EMERGENCY: ACTIVE CAD HANDOFF</h4>
                <div style='color: #7F1D1D; font-size: 0.95rem; line-height: 1.5;'>
                    <b>Data Origin:</b> 108 Central ERC Dispatch ➔ MCECN Telephony Sidecar<br>
                    <b>Call ID:</b> {draft['id']} | <b>Triangulated Location:</b> {draft['loc']}<br>
                    <b>NLP Transcribed Complaint:</b> <i>"{draft['complaint']}"</i>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 **SYSTEM SHIFT:** Origin detected as non-clinical field location. Standard IFT protocol locks are bypassed. Proceed with Scoop & Run logic.")
            
            default_lat = float(draft['lat'])
            default_lon = float(draft['lon'])
            default_name = "Unknown (108 Field Intake)"
        else:
            # Standard IFT UI (No 108 Draft)
            default_lat = 25.586936
            default_lon = 91.809418
            default_name = "John Doe"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            patient_name = st.text_input("Patient Name (E2EE)", default_name)
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
            src_lat = st.number_input("Origin Latitude", value=default_lat, format="%.6f") 
            src_lon = st.number_input("Origin Longitude", value=default_lon, format="%.6f")

    with st.container(border=True):
        st.markdown("<h3><span class='material-symbols-outlined icon-slate'>stethoscope</span> 2. Provisional Diagnosis</h3>", unsafe_allow_html=True)
        col_b, col_d = st.columns(2)
        with col_b:
            bundle = st.selectbox("Case Bundle", sorted(icd_df["bundle"].unique().tolist()))
        with col_d:
            dfb = icd_df[icd_df["bundle"] == bundle].copy()
            dx = st.selectbox("Select Diagnosis", dfb["label"].tolist())
            icd_row = dfb[dfb["label"] == dx].iloc[0].to_dict()
        required_caps = [x.strip() for x in (icd_row.get("default_caps","") or "").split(";") if x.strip()]

    with st.container(border=True):
        st.markdown("<h3><span class='material-symbols-outlined icon-slate'>assignment</span> 3. Clinical Rationale & Pre-Transfer Stabilization</h3>", unsafe_allow_html=True)
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
            surgical_override = st.checkbox("SURGICAL EMERGENCY OVERRIDE: Patient requires immediate surgical control unavailable here. Hemodynamic resuscitation is ongoing in transit.")

    vitals = {"hr": hr, "rr": rr, "sbp": sbp, "temp": temp, "spo2": spo2, "avpu": avpu}
    context = {"age": age, "pregnant": pregnant, "o2_device": "Air", "spo2_scale": 1, "behavior": "Normal"}
    
    try:
        triage_color, meta = validated_triage_decision(vitals=vitals, icd_row=icd_row, context=context)
    except Exception as e:
        st.error(f"Clinical Engine Failure: {e}")
        st.stop()

    with st.container(border=True):
        st.markdown("<h3><span class='material-symbols-outlined icon-slate'>emergency_share</span> 4. Dual-Vector Triage Result</h3>", unsafe_allow_html=True)
        pill = {"RED":"CRITICAL (RED)", "YELLOW":"URGENT (YELLOW)", "GREEN":"STABLE (GREEN)"}[triage_color]
        if triage_color == "RED": st.error(pill)
        elif triage_color == "YELLOW": st.warning(pill)
        else: st.success(pill)
        st.markdown(f"**Primary Driver:** {meta['primary_driver']} | **Reason:** {meta['reason']} | **Severity Index:** {meta['severity_index']:.2f}")

    with st.container(border=True):
        st.markdown("<h3><span class='material-symbols-outlined icon-slate'>share_location</span> 5. Facility Matching (Gated Clinical Safety)</h3>", unsafe_allow_html=True)
        
        if st.button("Run AI Facility Matcher", type="primary", use_container_width=True):
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
                        score, details = calculate_facility_score(facility=f_dict, required_caps=required_caps, eta=route_eta, triage_color=triage_color, severity_index=sev, case_type=bundle)
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
                st.success(f"AI routing complete. Found {len(results)} viable destinations.")
                radio_options = [f"{r['facility']} (Score: {r['score']})" for r in results[:5]]
                selected_option = st.radio("Select Destination Facility:", radio_options)
                selected_fac_name = selected_option.split(" (Score:")[0]
                selected_fac = next(r for r in results if r["facility"] == selected_fac_name)
                
                with st.expander(f"Explainable AI Logic for {selected_fac['facility']}"):
                    details = selected_fac["scoring_details"]
                    tier = details.get('clinical_tier', 'Tier 1: Definitive Care')
                    st.markdown(f"**1. Clinical Capability Escalon ({tier})**")
                    if details.get('gate_capacity') == "WARNING_ED_STABILIZATION_ONLY":
                        st.error("Capacity Gate Override: ICU Full. Facility selected for immediate ED Resuscitation ONLY.")

                dispatch_disabled = False
                dispatch_warning = ""
                
                if triage_color == "RED":
                    if initiator_role == "Attending Medical Officer (Doctor)":
                        if not completed_interventions and not surgical_override:
                            dispatch_disabled = True
                            dispatch_warning = "IFT PROTOCOL LOCK: Patient must be hemodynamically stabilized prior to transport. Select interventions administered or authorize Surgical Override."
                    else:
                        dispatch_warning = "NURSE-LED INITIATION: No doctor on site. Patient un-resuscitated. Receiving ED will be pre-alerted."
                
                if dispatch_warning:
                    if dispatch_disabled: st.error(dispatch_warning)
                    else: st.warning(dispatch_warning)

                if st.button("Initiate E2EE Transfer & Dispatch Transport", type="primary", disabled=dispatch_disabled):
                    ideal_fleet = "ALS" if triage_color == "RED" or meta['severity_index'] >= 0.6 else "BLS"
                    allocated_fleet, fleet_status = ideal_fleet, "OPTIMAL"
                    model, plate, driver, emt_name = f"{ideal_fleet}-{random.randint(101,250)}", f"ML-05-{random.randint(1000,9999)}", random.choice(PILOTS), random.choice(EMTS)
                    
                    st.session_state.active_case = {
                        "patient_name": patient_name, "age": age, "vitals": vitals, "diagnosis": dx,
                        "bundle": bundle, "triage_color": triage_color, "severity_index": meta['severity_index'],
                        "destination": selected_fac, "dispatch_time": datetime.now().strftime("%H:%M:%S"),
                        "rationale": selected_rationale, "interventions": completed_interventions, 
                        "viable_destinations": results, "current_dest_index": next(i for i, r in enumerate(results) if r["facility"] == selected_fac["facility"]),      
                        "rejection_log": [], "medical_orders": [],
                        "fleet": {"ideal": ideal_fleet, "allocated": allocated_fleet, "status": fleet_status, "driver": driver, "plate": plate, "vehicle": model, "emt": emt_name},
                        "nurse_led": (initiator_role == "Staff Nurse / ANM (No Doctor on Duty)"),
                        "surgical_override": surgical_override
                    }
                    st.session_state.transfer_initiated = True
                    st.session_state.patient_accepted = False
                    st.session_state.match_results = None 
                    st.session_state.draft_case = None # Clear draft on successful dispatch
                    st.rerun()

# ==========================================
# VIEW 2: ACTIVE TRANSIT TELEMETRY
# ==========================================
elif st.session_state.main_nav == "ACTIVE TRANSIT TELEMETRY":
    st.markdown("<h2><span class='material-symbols-outlined icon-blue'>ambulance</span> Active Transit & Telemetry Dashboard</h2>", unsafe_allow_html=True)
    render_plotly_timeline(3)

    with st.container(border=True):
        if not st.session_state.active_case:
            st.info("No active dispatch. Initiate a transfer from the Referral tab.")
        else:
            case = st.session_state.active_case
            dest = case["destination"]
            is_cab = (case['fleet']['allocated'] == 'CAB')
            if "transit_log" not in case: case["transit_log"] = []

            if is_cab:
                if case['triage_color'] == 'RED':
                    st.error(f"CRITICAL CAB FALLBACK TO {dest['facility'].upper()}")
                else:
                    st.info(f"TIER-3 CAB DISPATCHED TO {dest['facility'].upper()}")
            else:
                st.error(f"PRIORITY {case['triage_color']} EN ROUTE TO {dest['facility'].upper()} [{case['fleet']['allocated']} UNIT]")
                
            st.markdown(f"**Unit Details:** {case['fleet']['vehicle']} ({case['fleet']['plate']}) | **Pilot:** {case['fleet']['driver']} | **Lead EMT:** {case['fleet']['emt']}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Topography-Adjusted ETA", f"{dest['eta']} min")
            c2.metric("Modeled Mortality Risk", f"{dest.get('mortality_risk', 0.0)}%")
            c3.metric("Severity Index", f"{case.get('severity_index', 0.0):.2f}")
            c4.metric("Dispatch Time", case["dispatch_time"])

            st.markdown("---")

            if case.get("medical_orders"):
                st.warning("INCOMING MEDICAL COMMAND ORDER")
                st.markdown(f"### {case['medical_orders'][-1]['order']}")
                st.markdown("---")
            
            col_actions, col_vitals, col_traffic = st.columns([1.5, 1.2, 1.2])
            with col_actions:
                st.markdown("<h3><span class='material-symbols-outlined icon-slate'>receipt_long</span> Med-Legal Ledger</h3>", unsafe_allow_html=True)
                a1, a2, a3 = st.columns(3)
                if a1.button("Push ACLS", use_container_width=True): case["transit_log"].append({"action": "ACLS Administered"})
                if a2.button("Airway", use_container_width=True): case["transit_log"].append({"action": "Airway Secured"})
                if a3.button("Tourniquet", use_container_width=True): case["transit_log"].append({"action": "Hemorrhage Control"})
                    
            with col_vitals:
                st.markdown("<h3><span class='material-symbols-outlined icon-slate'>bluetooth</span> Live Bluetooth Telemetry</h3>", unsafe_allow_html=True)
                st.caption("Streamed from Unit Defibrillator")
                
                live_hr = case['vitals']['hr'] + random.randint(-3, 3)
                live_spo2 = case['vitals']['spo2'] + random.randint(-1, 1)
                
                hr_color = "inverse" if live_hr > 130 or live_hr < 50 else "normal"
                spo2_color = "inverse" if live_spo2 < 90 else "normal"
                
                v1, v2 = st.columns(2)
                v1.metric("Live HR", f"{live_hr} bpm", delta=f"{live_hr - case['vitals']['hr']}", delta_color=hr_color)
                v2.metric("Live SpO2", f"{live_spo2} %", delta=f"{live_spo2 - case['vitals']['spo2']}", delta_color=spo2_color)
                st.markdown("**Vehicle Oxygen Reserves:** 82% (Nominal)")

            with col_traffic:
                st.markdown("<h3><span class='material-symbols-outlined icon-slate'>traffic</span> ETA Drift Telemetry</h3>", unsafe_allow_html=True)
                traffic_delay = st.slider("Live Traffic/Blockade Delay (Mins)", 0, 60, 0)
                live_eta = dest['eta'] + traffic_delay
                st.metric("Live GPS ETA", f"{live_eta} mins", delta=f"+{traffic_delay} mins" if traffic_delay > 0 else "On Time", delta_color="inverse")
                
                if traffic_delay > 15 and case['triage_color'] == 'RED':
                    if st.button("INITIATE CIVIC OVERRIDE & ESCORT", type="primary", use_container_width=True):
                        st.session_state.civic_override_active = True
                        st.rerun()
                        
                if st.session_state.get('civic_override_active', False):
                    st.markdown("---")
                    st.error("STATE EMERGENCY PROTOCOL: GREEN CORRIDOR ENGAGED")
                    with st.status("Establishing handshake with Multi-Agency CAD...", expanded=True) as status:
                        st.write("Auth Token: MCECN-AUTH-994A...")
                        time.sleep(0.6)
                        st.write("Police Command: Unit MEGH-P-42 dispatched for intercept...")
                        time.sleep(0.6)
                        st.write("Traffic Command: Overriding signals on Shillong-Guwahati corridor...")
                        time.sleep(0.6)
                        st.write("Community Action: Broadcasting RED ALERT to Radio Partners (Red FM 93.5)...")
                        time.sleep(0.8)
                        status.update(label="ESCORT SECURED & CORRIDOR CLEARED", state="complete", expanded=False)
                    c_gc1, c_gc2 = st.columns(2)
                    c_gc1.info("Interceptor ETA: 04 Mins")
                    c_gc2.warning("Traffic: Next 4 nodes locked to Green.")

# ==========================================
# VIEW 3: RECEIVING HOSPITAL
# ==========================================
elif st.session_state.main_nav == "RECEIVING HOSPITAL BAY":
    st.markdown("<h2><span class='material-symbols-outlined icon-blue'>local_hospital</span> Emergency Department Receiving Board</h2>", unsafe_allow_html=True)
    render_plotly_timeline(4)
    
    tab_active, tab_analytics, tab_outcomes = st.tabs(["Active Receiving Board", "ED Operations Analytics", "Clinical Milestones"])

    with tab_active:
        with st.container(border=True):
            case = st.session_state.get('active_case')
            if not case:
                st.info("ED Bay Clear. No incoming critical transfers.")
            else:
                dest = case["destination"]
                dest_name = dest['facility']
                st.success(f"SECURE TERMINAL ACTIVE: YOU ARE VIEWING AS **{dest_name.upper()}**")
                st.markdown("---")
                
                st.error(f"INCOMING PRIORITY {case['triage_color']} AMBULANCE: ETA {dest['eta']} mins")
                
                if case.get("nurse_led"): st.error("NURSE-LED PHC INITIATION: NO DOCTOR ON SITE. PREPARE IMMEDIATE ED STABILIZATION.")
                if case.get("surgical_override"): st.warning("SURGICAL OVERRIDE: Patient requires immediate surgical control. Resuscitation ongoing.")

                st.markdown("---")
                st.markdown("<h3><span class='material-symbols-outlined icon-slate'>description</span> Pre-Hospital Clinical Handover (ISBAR)</h3>", unsafe_allow_html=True)
                
                nurse_alert = "\n[NURSE-LED PHC INITIATION]" if case.get("nurse_led") else ""
                surg_alert = "\n[SURGICAL EMERGENCY OVERRIDE]" if case.get("surgical_override") else ""
                
                diversion_text = "\n[DIVERSION AUDIT]: " + " -> ".join([f"Bypassed {r['facility']} ({r['reason']})" for r in case.get("rejection_log", [])]) if case.get("rejection_log") else ""

                isbar_text = f"""[ISBAR CLINICAL HANDOVER]{nurse_alert}{surg_alert}
Status: PRIORITY {case['triage_color']} (Severity: {case.get('severity_index', 0.0):.2f})
I - IDENTIFICATION: Patient: {case['patient_name']}, {case['age']} Y/O
S - SITUATION: Emergency dispatch to {dest['facility']}. Provisional DX: {case['diagnosis']}
B - BACKGROUND: Bundle: {case['bundle']}. Rationale: {case.get('rationale', 'N/A')}
Pre-Transfer Interventions: {', '.join(case.get('interventions', [])) if case.get('interventions') else 'None / Not Logged'}{diversion_text}
A - ASSESSMENT: HR: {case['vitals']['hr']} | SBP: {case['vitals']['sbp']} | RR: {case['vitals']['rr']} | SpO2: {case['vitals']['spo2']}% | Temp: {case['vitals']['temp']}°C | AVPU: {case['vitals']['avpu']}
"""
                st.code(isbar_text, language="markdown")
                
                st.markdown("---")
                st.markdown("<h3><span class='material-symbols-outlined icon-slate'>bed</span> Transactional Capacity Management</h3>", unsafe_allow_html=True)
                if 'bed_locked' not in st.session_state: st.session_state.bed_locked = False
                    
                if not st.session_state.bed_locked:
                    if dest['scoring_details'].get('icu_beds', 0) > 0:
                        st.info(f"Available Critical Care Beds: {dest['scoring_details']['icu_beds']}")
                        if st.button("Lock 1x ICU Bed on State Registry", type="primary"):
                            with st.spinner("Writing cryptographic reservation to state ledger..."):
                                time.sleep(1)
                                st.session_state.bed_locked = True
                                st.rerun()
                    else:
                        st.error("Zero ICU Beds Available. Admitting to ED Resuscitation Bay.")
                else:
                    st.success(f"STATE GRID UPDATED: 1x ICU Bed reserved for Unit {case['fleet']['plate']}.")

                st.markdown("---")
                st.markdown("<h3><span class='material-symbols-outlined icon-slate'>settings_remote</span> Medical Command Uplink</h3>", unsafe_allow_html=True)
                dynamic_orders = ["Select Order...", "Direct to Cath Lab (Bypass ED)", "Initiate Massive Transfusion Protocol", "Push IV Labetalol"]
                col_order, col_send = st.columns([3, 1])
                with col_order:
                    selected_order = st.selectbox("Standardized Counter-Orders", dynamic_orders, label_visibility="collapsed")
                with col_send:
                    if st.button("Transmit Order", type="primary", use_container_width=True, disabled=(selected_order == "Select Order...")):
                        if "medical_orders" not in case: case["medical_orders"] = []
                        case["medical_orders"].append({"time": datetime.now().strftime("%H:%M:%S"), "order": selected_order})
                        st.rerun()
                        
                if not st.session_state.patient_accepted:
                    st.markdown("### Decision Matrix")
                    col_acc, col_rej = st.columns(2)
                    with col_acc:
                        if st.button("Acknowledge & Accept Patient", type="primary", use_container_width=True):
                            st.session_state.patient_accepted = True
                            st.rerun()
                    with col_rej:
                        with st.expander("Reject & Trigger AI Reroute"):
                            current_idx = case.get("current_dest_index", 0)
                            viable_list = case.get("viable_destinations", [])
                            if current_idx + 1 < len(viable_list):
                                next_dest = viable_list[current_idx + 1]
                                st.info(f"**AI Fallback Target:** If you divert this case, the AI will immediately reroute to **{next_dest['facility']}** (ETA: {next_dest['eta']} mins).")
                            else:
                                st.error("**AI Fallback Target:** ZERO viable facilities remaining. Rejecting forces on-site stabilization.")

                            reject_reason = st.selectbox("Standardized Reason for Diversion", ["Select Reason...", "Critical Care Beds Suddenly Full (Surge)", "Required Specialist Scrubbed In", "CT Scanner Down"])
                            if st.button("Confirm Diversion", type="primary", disabled=reject_reason == "Select Reason..."):
                                case["rejection_log"].append({"facility": dest_name, "reason": reject_reason, "time": datetime.now().strftime("%H:%M:%S")})
                                if current_idx + 1 < len(viable_list):
                                    next_dest = viable_list[current_idx + 1]
                                    case["destination"] = next_dest
                                    case["current_dest_index"] = current_idx + 1
                                    st.session_state.bed_locked = False
                                    st.success(f"Diverting. AI locking onto {next_dest['facility']}...")
                                    time.sleep(1.5)
                                    st.rerun()
                                else:
                                    st.error("CRITICAL: Zero fallback facilities remaining.")
                else:
                    st.success("Patient accepted. Telemetry linked to ED monitors.")

    with tab_analytics:
        st.markdown("<h3><span class='material-symbols-outlined icon-slate'>monitoring</span> Institutional Operations Analytics</h3>", unsafe_allow_html=True)
        if st.session_state.user_role in ["State Health Command (Macro View)", "Authorized Community Node (ASHA/CFR)", "Citizen Self-Booking App", "108 ERC Dispatcher"]:
            st.warning("ACCESS DENIED: View restricted to Facility Directors.")
        else:
            active_hospital = st.session_state.user_role.replace("Director: ", "")
            h_seed = int(hashlib.md5(active_hospital.encode()).hexdigest(), 16) % 10000
            rng = random.Random(h_seed)
            base_vol = rng.randint(110, 450)
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(f"Accepted at {active_hospital}", f"{base_vol}", f"+{rng.randint(2, 18)}% MoM")
            kpi2.metric("Local Diversions", f"{rng.randint(5, int(base_vol * 0.15))}", f"{rng.choice(['+', '-'])}{rng.randint(1, 5)}% MoM", delta_color="inverse")
            kpi3.metric("Fleet Arrival Delay (SLA)", f"+{round(rng.uniform(1.2, 8.5), 1)} mins", f"+0.{rng.randint(1, 9)} mins", delta_color="inverse")
            kpi4.metric("Avg Triage-to-Bed Time", f"{round(rng.uniform(4.5, 14.0), 1)} mins", f"-1.{rng.randint(1, 5)} mins")
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown(f"**Local Diversion Autopsy ({active_hospital})**")
                rejection_data = pd.DataFrame({"Reason": ["Zero ICU Beds", "Neurosurgeon Unavailable", "CT Scanner Down", "Internal Code Black"], "Count": [rng.randint(1, 10) for _ in range(4)]})
                chart1 = alt.Chart(rejection_data).mark_arc(innerRadius=50).encode(theta=alt.Theta(field="Count", type="quantitative"), color=alt.Color(field="Reason", type="nominal"))
                st.altair_chart(apply_tufte_theme(chart1), use_container_width=True)
            with col_chart2:
                st.markdown("**Inbound Acuity Distribution**")
                path_data = pd.DataFrame({"Pathology": ["Cardiac", "Trauma", "Stroke", "Maternal"], "Volume": [rng.randint(10, int(base_vol*0.4)) for _ in range(4)]})
                chart2 = alt.Chart(path_data).mark_bar().encode(x=alt.X('Pathology:N', sort='-y'), y='Volume:Q', color=alt.value('#1E3A8A'))
                st.altair_chart(apply_tufte_theme(chart2), use_container_width=True)

    with tab_outcomes:
        st.markdown("<h3><span class='material-symbols-outlined icon-slate'>fact_check</span> Clinical Milestone & Outcome Reconciliation</h3>", unsafe_allow_html=True)
        if st.session_state.user_role in ["State Health Command (Macro View)", "Authorized Community Node (ASHA/CFR)", "Citizen Self-Booking App", "108 ERC Dispatcher"]:
            st.warning("ACCESS DENIED: View restricted to Facility Directors.")
        else:
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                st.markdown("#### 24-Hour Stabilization Check")
                for i in range(2):
                    with st.container(border=True):
                        st.markdown(f"**ID: MCECN-24H-{random.randint(100,999)}** | {random.choice(['Severe Trauma', 'STEMI', 'Stroke'])}")
                        b1, b2, b3 = st.columns(3)
                        if b1.button("Stabilized", key=f"m1_a_{i}"): st.success("Logged.")
                        if b2.button("Deceased", key=f"m1_d_{i}"): st.error("Logged.")
                        if b3.button("Re-Referred", key=f"m1_r_{i}"): st.warning("Logged.")
            with col_q2:
                st.markdown("#### ALOS Disposition Check")
                for i in range(2):
                    with st.container(border=True):
                        st.markdown(f"**ID: MCECN-ALOS-{random.randint(100,999)}** | {random.choice(['Maternal Hemorrhage', 'Pediatric Resp'])}")
                        b1, b2, b3 = st.columns(3)
                        if b1.button("Discharged", key=f"m2_d_{i}"): st.success("Logged.")
                        if b2.button("Admitted", key=f"m2_s_{i}"): st.info("Logged.")
                        if b3.button("Deceased in ICU", key=f"m2_x_{i}"): st.error("Logged.")

# ==========================================
# VIEW 4: STATE COMMAND & AI
# ==========================================
elif st.session_state.main_nav == "STATE COMMAND & AI":
    st.markdown("<h2><span class='material-symbols-outlined icon-blue'>dashboard_customize</span> State Command & Control</h2>", unsafe_allow_html=True)

    if st.session_state.user_role != "State Health Command (Macro View)":
        st.error("CLEARANCE VIOLATION: View compartmentalized for State Health Officials only.")
    else:
        st.caption("Zero-PHI Analytics & Autonomous Logistics Engine")

        def _now_ts(): return time.time()
        def _rand_geo(rng): return (25.5 + rng.random()*0.2, 91.8 + rng.random()*0.2)
        def _dist_km(lat1, lon1, lat2, lon2): return haversine_km((lat1, lon1), (lat2, lon2))
        def _interp_route(lat1, lon1, lat2, lon2, n): return []
        def _traffic(hr): return 1.2 if 8 <= hr <= 20 else 1.0

        col_b1, col_b2 = st.columns([1, 4])
        with col_b1:
            if st.button("Inject 1,000 Synthetic Cases", type="primary"):
                with st.spinner("Generating clinical, financial, & fleet telemetry..."):
                    try:
                        raw_data = seed_synthetic_referrals_v2(
                            n=1000, facilities=facilities_df.to_dict('records'), icd_df=icd_df,
                            validated_triage_decision_fn=validated_triage_decision, now_ts_fn=_now_ts,
                            rand_geo_fn=_rand_geo, dist_km_fn=_dist_km, interpolate_route_fn=_interp_route,
                            traffic_factor_fn=_traffic, rng_seed=42
                        )
                        safe_records = []
                        for r in raw_data:
                            sev_idx = r["triage"]["decision"]["score_details"].get("severity_index", 0.0)
                            bundle_name = r["provisionalDx"]["case_type"]
                            ideal = "ALS" if r["triage"]["decision"]["color"] == "RED" or sev_idx >= 0.6 else "BLS"
                            unit_id = f"ALS-{random.randint(101, 115)}" if (ideal=="ALS") else (f"BLS-{random.randint(201, 230)}" if random.random()>0.2 else "CAB-CIVILIAN")
                            mhis_base = {"Cardiac": 120000, "Trauma": 180000, "Maternal": 45000, "Stroke": 85000, "Pediatric": 25000}
                            mhis_val = mhis_base.get(bundle_name, 50000) * random.uniform(0.9, 1.1)
                            fc_gov = random.random() < 0.65
                            route = "Govt -> Private (Leakage)" if (fc_gov and random.random() < 0.3) else ("Govt -> Govt" if fc_gov else "Private -> Private")
                            
                            safe_records.append({
                                "bundle": bundle_name, "triage_color": r["triage"]["decision"]["color"],
                                "actual_trip_time": r["transport"]["eta_min"] + random.uniform(1.0, 15.0),
                                "unit_id": unit_id, "mhis_value": mhis_val, "routing_path": route,
                                "origin_district": random.choice(["East Khasi Hills", "West Garo Hills", "Jaintia Hills"]),
                                "uncompensated_burden": max(0, random.randint(-1, 6)) * 5000 if "Govt" in route else 0,
                                "pre_hospital_intervention": "Yes" if random.choice([True, False]) else "No",
                                "mortality_risk": min(99.9, mortality_risk(sev_idx, r["transport"]["eta_min"], bundle_name)),
                                "status": random.choices(["Accepted", "Diverted"], weights=[0.78, 0.22])[0],
                                "mission_type": f"Optimal {ideal}" if unit_id.startswith(ideal) else "Fallback",
                                "dest_facility": r["dest"],
                                "facility_ownership": r["facility_ownership"]
                            })
                        st.session_state.synthetic_data = pd.DataFrame(safe_records)
                        st.success("Data Injected.")
                    except Exception as e:
                        st.error(f"Generation Failed: {e}")
        with col_b2:
            if st.session_state.synthetic_data is not None:
                st.success("🟢 LIVE: Dashboard driven by 1,000-case synthetic Machine Learning engine.")
            else:
                st.info("🟡 BASELINE: Dashboard showing hardcoded Meghalaya Feb 2026 Audit data. Click 'Inject 1,000 Cases' to activate dynamic ML graphs.")

        st.markdown("---")

        tab_fleet, tab_geo, tab_perf, tab_econ, tab_roi = st.tabs([
            "108 System Optimization", 
            "Case-Type Intelligence", 
            "Hospital Heatboard",
            "MHIS Fiscal Governance",
            "Investment ROI Simulator"
        ])

        df_synth = st.session_state.synthetic_data

        with tab_fleet:
            st.markdown("<h3><span class='material-symbols-outlined icon-slate'>work_history</span> Statewide Workforce & Deployment Stress</h3>", unsafe_allow_html=True)
            col_f1, col_f2 = st.columns([1.5, 1])
            
            with col_f1:
                if df_synth is None:
                    st.markdown("**District Base Performance (P90 Target Misses - Feb 2026 Audit)**")
                    base_data = pd.DataFrame({"Base": ["JARAIN", "JOWAI", "SOHIONG", "TURA", "NONGPOH"], "Target Met %": [27.3, 27.8, 38.1, 38.1, 62.4]})
                    base_chart = alt.Chart(base_data).mark_bar(cornerRadiusEnd=4, size=20).encode(y=alt.Y('Base:N', sort='x', title=None), x=alt.X('Target Met %:Q', title="Target Met (%)"), color=alt.condition(alt.datum['Target Met %'] < 40, alt.value('#E11D48'), alt.value('#94A3B8'))).properties(height=250)
                    st.altair_chart(apply_tufte_theme(base_chart), use_container_width=True)
                else:
                    st.markdown("**State Fleet Workload Distribution (Dynamic Synthetic)**")
                    fleet_stats = df_synth[df_synth['unit_id'] != 'CAB-CIVILIAN'].groupby('unit_id').size().reset_index(name='Trips')
                    fleet_chart = alt.Chart(fleet_stats).mark_bar(size=15).encode(x=alt.X('unit_id:N', sort='-y', title=None), y=alt.Y('Trips:Q', title=None), color=alt.condition(alt.datum.Trips > fleet_stats['Trips'].quantile(0.8), alt.value('#E11D48'), alt.value('#94A3B8'))).properties(height=250)
                    st.altair_chart(apply_tufte_theme(fleet_chart), use_container_width=True)
                    
            with col_f2:
                st.error("CRITICAL WORKFORCE ALERT")
                st.markdown("Audit identifies **Manpower (81 cases)** as the leading cause of fleet deassignments.")
                st.markdown("**AI Repositioning Recommendations:**\n* **Shift 2 BLS Relievers** to Jarain (Severe Fatigue Risk).\n* **Upgrade Jowai Corridor** to ALS coverage.\n* **Night Transport Risk** identified in West Garo Hills.")

        with tab_geo:
            st.markdown("<h3><span class='material-symbols-outlined icon-slate'>public</span> Meghalaya Case-Type Intelligence</h3>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if df_synth is None:
                    path_data = pd.DataFrame({"Pathology": ["Obstetric Emergency", "RTA (Trauma)", "Abdominal Pain", "Breathing Difficulty", "Unconscious"], "Volume": [218, 63, 56, 44, 41]})
                else:
                    path_data = df_synth['bundle'].value_counts().reset_index()
                    path_data.columns = ['Pathology', 'Volume']
                
                path_chart = alt.Chart(path_data).mark_arc(innerRadius=40).encode(theta=alt.Theta(field="Volume", type="quantitative"), color=alt.Color(field="Pathology", type="nominal", scale=alt.Scale(scheme='blues')), tooltip=["Pathology", "Volume"]).properties(height=300)
                st.altair_chart(apply_tufte_theme(path_chart), use_container_width=True)
                
            with c2:
                if df_synth is None:
                    st.info("Clinical Routing Insight:")
                    st.markdown("**Obstetric & Neonatal (218 cases)** is the single largest high-volume cluster. \n* **Current Failure:** Outcomes are highly pathway-dependent, but routing is blind to NICU/OBGYN availability.\n* **MCECN Solution:** Dual-Vector triage forces MEOWS scoring and routes exclusively to proven Maternal OT facilities.")
                else:
                    st.error("EPIDEMIC ANOMALY DETECTOR (Sub-District Alerts)")
                    anomaly_data = pd.DataFrame({"Origin": ["Mawphlang", "Tikrikilla", "Khliehriat"], "Flag": ["Pediatric Resp", "Maternal Hemorrhage", "Acute Trauma"], "Deviation": ["+420% Surge", "+185% Surge", "+210% Surge"]})
                    st.dataframe(anomaly_data, use_container_width=True, hide_index=True)

        with tab_perf:
            st.markdown("<h3><span class='material-symbols-outlined icon-slate'>domain_verification</span> Institutional Performance & Heatboard</h3>", unsafe_allow_html=True)
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                st.markdown("**Hospital Acceptance Heatboard**")
                heat_data = pd.DataFrame({"Hospital": ["NEIGRIHMS", "Shillong Civil", "Woodland Hosp", "Bethany Hosp", "Tura Civil"], "Acceptance": ["94%", "42%", "88%", "85%", "71%"], "Status": ["Green", "Red", "Green", "Amber", "Amber"], "Primary Diversion Reason": ["None", "Zero ICU Beds (Surge)", "None", "CT Scanner Down", "Specialist Scrubbed In"]})
                def color_status(val):
                    if val == 'Green': return 'color: #10B981; font-weight: bold;'
                    elif val == 'Red': return 'color: #E11D48; font-weight: bold;'
                    return 'color: #F59E0B; font-weight: bold;'
                st.dataframe(heat_data.style.map(color_status, subset=['Status']), use_container_width=True, hide_index=True)
            with c_p2:
                if df_synth is not None:
                    st.markdown("**Mortality Shift Analysis (Efficacy)**")
                    mortality_shift = pd.DataFrame({"Phase": ["< 24h (Pre-Hospital/ED Failure)", "> 24h (ICU/Ward Complication)"], "Pre-MCECN OS": [65, 35], "Post-MCECN OS": [22, 78]}).melt(id_vars="Phase", var_name="Era", value_name="Percentage of Total Mortalities")
                    shift_chart = alt.Chart(mortality_shift).mark_bar().encode(x=alt.X('Percentage of Total Mortalities:Q', stack='normalize', axis=alt.Axis(format='%')), y=alt.Y('Era:N', sort=['Pre-MCECN OS', 'Post-MCECN OS']), color=alt.Color('Phase:N', scale=alt.Scale(domain=['< 24h (Pre-Hospital/ED Failure)', '> 24h (ICU/Ward Complication)'], range=['#E11D48', '#94A3B8']))).properties(height=200)
                    st.altair_chart(apply_tufte_theme(shift_chart), use_container_width=True)
                else:
                    st.info("Inject Synthetic Data to view Mortality Shift Analysis.")

        with tab_econ:
            st.markdown("<h3><span class='material-symbols-outlined icon-slate'>account_balance</span> MHIS Health Economics & Fiscal Governance</h3>", unsafe_allow_html=True)
            if df_synth is None:
                k1, k2, k3 = st.columns(3)
                k1.metric("State Total Referrals", "1,345")
                k2.metric("MHIS Fiscal Leakage", "₹ 2.4 Cr", delta="- Drain to Private Sector", delta_color="inverse")
                k3.metric("ALOS Uncompensated Burden", "₹ 18.5 L", delta="Bed-Blocking Penalties", delta_color="inverse")
                st.caption("Baseline metrics from Feb 2026 Audit.")
            else:
                retained_rev = df_synth[df_synth['routing_path'] == 'Govt -> Govt']['mhis_value'].sum()
                leakage_val = df_synth[df_synth['routing_path'] == 'Govt -> Private (Leakage)']['mhis_value'].sum()
                total_burden = df_synth['uncompensated_burden'].sum()
                
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("State Total Referrals", f"{len(df_synth)}")
                k2.metric("Public Revenue Retained", f"₹ {retained_rev/10000000:.2f} Cr")
                k3.metric("MHIS Fiscal Leakage", f"₹ {leakage_val/10000000:.2f} Cr", delta="- Drain to Private Sector", delta_color="inverse")
                k4.metric("ALOS Uncompensated Burden", f"₹ {total_burden/100000:.2f} L", delta="Bed-Blocking Penalties", delta_color="inverse")
                
                st.markdown("---")
                c_e1, c_e2 = st.columns(2)
                with c_e1:
                    st.markdown("**District Capital Flight Radar**")
                    flight_data = df_synth[df_synth['routing_path'] == 'Govt -> Private (Leakage)']
                    flight_chart = alt.Chart(flight_data).mark_bar(size=20).encode(x=alt.X('sum(mhis_value):Q', title="Leakage (₹)"), y=alt.Y('origin_district:N', sort='-x', title=None), color=alt.value('#E11D48')).properties(height=200)
                    st.altair_chart(apply_tufte_theme(flight_chart), use_container_width=True)
                with c_e2:
                    st.markdown("**Uncompensated ALOS Burden by Pathology**")
                    burden_bar = alt.Chart(df_synth[df_synth['uncompensated_burden'] > 0]).mark_bar().encode(x=alt.X('bundle:N', sort='-y', title=None), y=alt.Y('sum(uncompensated_burden):Q', title="Total Penalty (₹)"), color=alt.value('#F59E0B')).properties(height=200)
                    st.altair_chart(apply_tufte_theme(burden_bar), use_container_width=True)

        with tab_roi:
            st.markdown("<h3><span class='material-symbols-outlined icon-slate'>insights</span> Policy Investment ROI Simulator</h3>", unsafe_allow_html=True)
            
            st.markdown("#### Step 1: Adjust Capital Allocation")
            col_roi1, col_roi2 = st.columns(2)
            with col_roi1:
                als_added = st.slider("Add New ALS Ambulances", 0, 50, 0)
                bls_added = st.slider("Add New BLS Relievers", 0, 30, 0)
            with col_roi2:
                ift_split = st.checkbox("Activate Dedicated IFT Fleet (Split Queue)")
                hosp_upgrade = st.checkbox("Fund 20-Bed Step-Down Ward at Shillong Civil")
                
            st.markdown("#### Step 2: Projected Statewide Outcomes")
            base_p90 = 51.0
            base_compliance = 60.3
            
            new_p90 = max(20.0, base_p90 - (als_added * 0.3) - (bls_added * 0.1) - (6.0 if ift_split else 0.0) - (2.0 if hosp_upgrade else 0.0))
            new_comp = min(99.0, base_compliance + (als_added * 0.6) + (bls_added * 0.2) + (12.0 if ift_split else 0.0) + (4.0 if hosp_upgrade else 0.0))
            lives_saved = int((new_comp - base_compliance) * 4.2)
            
            r1, r2, r3 = st.columns(3)
            r1.metric("Rural P90 Response", f"{new_p90:.1f} mins", f"{new_p90 - base_p90:.1f} mins vs Baseline", delta_color="inverse")
            r2.metric("NHM Target Compliance (≤30m)", f"{new_comp:.1f} %", f"+{new_comp - base_compliance:.1f} % vs Baseline")
            r3.metric("Estimated Critical Lives Saved", f"{lives_saved} / month", "+ Preventable Mortalities Avoided")
            
            if new_comp > 85.0:
                st.success("OPTIMAL SYSTEM ARCHITECTURE ACHIEVED: Dedicated IFT routing and fleet expansion bends mortality into NHM compliance.")
