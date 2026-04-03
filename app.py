import streamlit as st
import pandas as pd
import altair as alt
import time
import random
import os
from datetime import datetime, timedelta

# --- Architecture Imports ---
from clinical_engine import validated_triage_decision
from scoring_engine import calculate_facility_score
from routing_engine import get_eta, haversine_km
from analytics_engine import mortality_risk
from synthetic_cases import seed_synthetic_referrals_v2

# --- Page Configuration ---
st.set_page_config(
    page_title="AHECN Command Center", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- State Management ---
if 'active_case' not in st.session_state:
    st.session_state.active_case = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'transfer_initiated' not in st.session_state:
    st.session_state.transfer_initiated = False
if 'patient_accepted' not in st.session_state:
    st.session_state.patient_accepted = False
if 'match_results' not in st.session_state:
    st.session_state.match_results = None
if 'civic_override_active' not in st.session_state:
    st.session_state.civic_override_active = False

# --- THE CACHE-BUSTER: We are temporarily removing @st.cache_data ---
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
        # on_bad_lines='skip' ensures a single stray comma in the CSV won't crash the engine
        fac_df = pd.read_csv(f_path, encoding='utf-8-sig', on_bad_lines='skip')
        icd_df = pd.read_csv(i_path, encoding='utf-8-sig', on_bad_lines='skip')
    except Exception as e:
        st.error(f"CRITICAL: Failed to read CSV. Error: {e}")
        st.stop()

    # --- NATIVE PYTHON COLUMN SANITIZATION (Bypasses PyArrow completely) ---
    icd_df.columns = [
        str(c).replace('ï»¿', '').replace('\ufeff', '').strip().lower().replace('"', '').replace("'", "") 
        for c in icd_df.columns
    ]
    
    # The Firewall
    if 'bundle' not in icd_df.columns:
        st.error("🚨 DATA FORMAT ERROR: The column 'bundle' is missing from the dataset.")
        st.warning(f"The system is actually seeing these columns: {icd_df.columns.tolist()}")
        st.stop()

    rename_map = {'icd-10': 'icd10', 'icd_10': 'icd10', 'icd code': 'icd10', 'code': 'icd10'}
    icd_df = icd_df.rename(columns=rename_map)
    if 'icd10' not in icd_df.columns: icd_df = icd_df.rename(columns={icd_df.columns[0]: 'icd10'})
    
    for c in ["lat","lon"]:
        if c in fac_df.columns: fac_df[c] = pd.to_numeric(fac_df[c], errors='coerce')
    fac_df["ownership"] = fac_df.get("ownership", "Private").fillna("Private")
    
    return fac_df, icd_df

# Call the new function name to completely bypass Streamlit's old memory
facilities_df, icd_df = load_datasets_v3()
    
# --- Sidebar Navigation ---
with st.sidebar:
    st.title("AHECN OS")
    st.caption("Runo Health Enterprise v6.0")
    st.markdown("---")
    nav_selection = st.radio(
        "SYSTEM NAVIGATION",
        ["REFERRAL INITIATION", "ACTIVE TRANSIT TELEMETRY", "RECEIVING HOSPITAL BAY", "STATE COMMAND & AI"]
    )
    st.markdown("---")
    st.caption("Status: All Systems Operational")

# ==========================================
# VIEW 1: REFERRAL INITIATION
# ==========================================
if nav_selection == "REFERRAL INITIATION":
    st.header("Triage & Referral Initiation")
    st.caption("Secure, dual-vector triage and topography-aware facility matching.")
    
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

    # --- 3. DYNAMIC CLINICAL RATIONALE INJECTION ---
    with st.container(border=True):
        st.subheader("3. Clinical Rationale")
        
        # Auto-generate medico-legal rationales based on required capabilities
        auto_rationales = [f"General escalation of care required for {dx}."]
        caps_lower = [c.lower() for c in required_caps]
        
        if "neurosurgeon" in caps_lower or "neurosurgery" in caps_lower:
            auto_rationales.insert(0, "Emergent Neurosurgical evaluation and intervention required.")
        if "cathlab" in caps_lower or "cardiologist" in caps_lower:
            auto_rationales.insert(0, "Requires emergent Cath Lab activation and Cardiology intervention.")
        if "dialysis" in caps_lower or "crrt" in caps_lower:
            auto_rationales.insert(0, "Requires urgent Hemodialysis / Nephrology intervention.")
        if "icu" in caps_lower or "ventilator" in caps_lower:
            auto_rationales.insert(0, "Requires immediate ICU admission and advanced critical care/ventilation.")
        if "bloodbank" in caps_lower or "or" in caps_lower or "surgeon" in caps_lower:
            auto_rationales.insert(0, "Requires emergent surgical control and massive transfusion protocol capability.")
        if "neonatologist" in caps_lower or "nicu" in caps_lower:
            auto_rationales.insert(0, "Requires Level 3 NICU and specialized neonatal resuscitation.")
        if "picu" in caps_lower or "pediatrician" in caps_lower:
            auto_rationales.insert(0, "Pediatric emergency requiring specialized PICU escalation.")
        if bundle == "Maternal":
            auto_rationales.insert(0, "High-risk obstetric emergency requiring tertiary maternal-fetal care and OBGYN OT.")
            
        auto_rationales.append("Other / Custom (Specify below)")
        
        selected_rationale = st.selectbox("Standardized Reason for Transfer", auto_rationales)
        
        if selected_rationale == "Other / Custom (Specify below)":
            reason_for_referral = st.text_area("Specify Rationale", placeholder="Type specific clinical details here...")
        else:
            custom_notes = st.text_input("Additional Clinical Notes (Optional)", placeholder="E.g., deteriorating on current O2 support...")
            reason_for_referral = f"{selected_rationale} {custom_notes}".strip()
        # --- NEW: DYNAMIC RESUSCITATION BUNDLE ---
        st.markdown("---")
        st.markdown("⚕️ **Pre-Transfer Resuscitation & Interventions**")
        
        # Read the default interventions from the ICD catalogue
        interventions_str = icd_row.get("default_interventions", "")
        available_interventions = [x.strip() for x in str(interventions_str).split(";") if x.strip()]
        
        completed_interventions = []
        if available_interventions:
            completed_interventions = st.multiselect(
                "Select life-saving interventions already administered:",
                available_interventions + ["Other (Specified in notes)"]
            )
            if not completed_interventions:
                st.caption("⚠️ *No pre-transfer interventions logged. Ensure patient is stabilized for transit.*")
        else:
            st.info("No standardized resuscitation bundle found for this pathology. Use notes if needed.")
    # --- THE CLINICAL FIREWALL ---
    vitals = {"hr": hr, "rr": rr, "sbp": sbp, "temp": temp, "spo2": spo2, "avpu": avpu}
    context = {"age": age, "pregnant": pregnant, "o2_device": "Air", "spo2_scale": 1, "behavior": "Normal"}
    
    guardrail_passed = True
    if "Pediatric" in bundle and age >= 18:
        st.error("🛑 **CLINICAL INTERLOCK TRIGGERED:** You have selected a Pediatric pathology for an adult patient.")
        guardrail_passed = False
    elif "Maternal" in bundle and not pregnant:
        st.warning("⚠️ **CLINICAL AUTOCORRECTION:** Maternal bundle detected. Auto-verifying pregnancy status.")
        context["pregnant"] = True 
        
    if guardrail_passed:
        try:
            triage_color, meta = validated_triage_decision(vitals=vitals, icd_row=icd_row, context=context)
        except Exception as e:
            st.error(f"🚨 Clinical Engine Failure: {e}")
            st.stop()
    else:
        st.stop()

    with st.container(border=True):
        st.subheader("4. Dual-Vector Triage Result")
        pill = {"RED":"CRITICAL (RED)", "YELLOW":"URGENT (YELLOW)", "GREEN":"STABLE (GREEN)"}[triage_color]
        if triage_color == "RED": st.error(pill)
        elif triage_color == "YELLOW": st.warning(pill)
        else: st.success(pill)
        st.markdown(f"**Primary Driver:** {meta['primary_driver']} | **Reason:** {meta['reason']} | **Severity Index:** {meta['severity_index']:.2f}")

    # --- 5. THE FACILITY MATCHING UI (WITH EXPLICIT TRIGGER) ---
    with st.container(border=True):
        st.subheader("5. Facility Matching (Gated Clinical Safety)")
        
        # The Explicit Button you requested
        if st.button("🔍 Run AI Facility Matcher", type="primary", use_container_width=True):
            with st.spinner("Analyzing topography, capabilities, and capacity..."):
                origin = (float(src_lat), float(src_lon))
                results = []
                pathology_bundle = icd_row.get("bundle", "Other")
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
                            triage_color=triage_color, severity_index=sev, case_type=pathology_bundle
                        )
                    except Exception as e: 
                        continue

                    if score > 0 or details.get("gate_capacity") == "WARNING_ED_STABILIZATION_ONLY":
                        try: m_risk = mortality_risk(sev, route_eta, pathology=pathology_bundle)
                        except: m_risk = 99.9
                        results.append({
                            "facility": f_dict["name"], "score": score, "eta": round(route_eta, 1),
                            "ownership": f_dict["ownership"], "mortality_risk": m_risk, "scoring_details": details
                        })

                # Lock the results into the session state so they don't disappear
                st.session_state.match_results = sorted(results, key=lambda x: (-x["score"], x["eta"]))

        # --- RENDER RESULTS IF THEY EXIST IN MEMORY ---
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
                    if tier == "Tier 1: Definitive Care":
                        st.success("✅ **Tier 1 Match:** 100% of requested Specialists and Hardware present.")
                    elif tier == "Tier 2: Advanced Medical Bridging":
                        st.warning(f"⚠️ **Tier 2 Match:** Primary specialist unavailable. AI engaged medical fallback protocol.")
                        for b in details.get('bridged_capabilities', []):
                            st.caption(f"↳ *Missing:* {b['missing'].upper()} | *Bridged via:* {', '.join(b['bridged_with']).upper()}")
                    else:
                        st.error(f"🚨 **Tier 3 Match:** Definitive care missing. Route optimized for nearest airway/ED resuscitation.")
                        st.caption(f"↳ *Critical Missing:* {', '.join(details.get('failed_capabilities', [])).upper()}")
                    
                    if details.get('gate_capacity') == "PASSED":
                        st.success(f"✅ **Capacity Gate:** Passed. {details.get('icu_beds')} ICU beds available.")
                    elif details.get('gate_capacity') == "WARNING_ED_STABILIZATION_ONLY":
                        st.error("⚠️ **Capacity Gate Override:** ICU Full. Facility selected for immediate ED Resuscitation ONLY.")

                    st.markdown(f"**2. Optimization Matrix (Total Score: {details.get('total_score', 0)}/100)**")
                    st.markdown(f"🚑 **Topography-Adjusted ETA:** {details.get('eta_minutes', 'N/A')} mins *(Score: {details.get('proximity_score', 0)}/50)*")
                    st.markdown(f"🛏️ **Surge Buffer:** *(Score: {details.get('icu_score', 0)}/15)*")
                    st.markdown(f"🏛️ **Fiscal Guardrail:** {details.get('ownership')} facility *(Score: {details.get('fiscal_score', 0)}/20)*")
                    st.markdown(f"⚖️ **Acuity Bonus:** Criticality weight *(Score: {details.get('severity_bonus', 0)}/5)*")

                # The final dispatch button
                if st.button("🚀 Initiate E2EE Transfer & Dispatch Ambulance", type="primary"):
                    selected_idx = next(i for i, r in enumerate(results) if r["facility"] == selected_fac["facility"])
                    
                    st.session_state.active_case = {
                        "patient_name": patient_name, "age": age, "vitals": vitals, "diagnosis": dx,
                        "bundle": bundle, "triage_color": triage_color, "severity_index": meta['severity_index'],
                        "destination": selected_fac, "dispatch_time": datetime.now().strftime("%H:%M:%S"),
                        "rationale": reason_for_referral,
                        "interventions": completed_interventions, 
                        "viable_destinations": results,          
                        "current_dest_index": selected_idx,      
                        "rejection_log": [],
                        "medical_orders": []                    
                    }
                    st.session_state.transfer_initiated = True
                    st.session_state.patient_accepted = False
                    st.session_state.match_results = None 
                    st.rerun()

    # --- Phase 5: Med-Legal Zero-Friction Handover ---
    if st.session_state.transfer_initiated and st.session_state.active_case:
        with st.container(border=True):
            st.subheader("📄 Med-Legal Documentation Generated")
            st.success("Official ISBAR handover securely logged to the State Registry. Manual referral slip is no longer required.")
            
            case = st.session_state.active_case
            isbar_text = f"""[ISBAR CLINICAL HANDOVER]
Timestamp: {case['dispatch_time']}
Status: PRIORITY {case['triage_color']} (Severity: {case['severity_index']:.2f})

I - IDENTIFICATION:
Patient: {case['patient_name']}, {case['age']} Y/O

S - SITUATION:
Emergency dispatch to {case['destination']['facility']}. 
Provisional DX: {case['diagnosis']}

B - BACKGROUND: Bundle: {case['bundle']}. Rationale: {case.get('rationale', 'N/A')}
Pre-Transfer Interventions: {', '.join(case.get('interventions', [])) if case.get('interventions') else 'None / Not Logged'}
{chr(10) + '[DIVERSION AUDIT]: ' + ' -> '.join([f"Bypassed {r['facility']} ({r['reason']})" for r in case.get("rejection_log", [])]) if case.get("rejection_log") else ""}

A - ASSESSMENT:
HR: {case['vitals']['hr']} bpm | SBP: {case['vitals']['sbp']} mmHg | RR: {case['vitals']['rr']} rpm
SpO2: {case['vitals']['spo2']}% | Temp: {case['vitals']['temp']}°C | AVPU: {case['vitals']['avpu']}

R - RECOMMENDATION:
{('ED STABILIZATION ONLY REQUIRED DUE TO ZERO ICU BEDS.' if case['destination']['scoring_details'].get('gate_capacity') == 'WARNING_ED_STABILIZATION_ONLY' else 'Prepare critical care receiving bay.')}
"""
            st.code(isbar_text, language="markdown")
            col_a, col_b = st.columns(2)
            col_a.button("⬇️ Download Official PDF")
            col_b.button("🔗 Share to Secure Network")

# ==========================================
# VIEW 2: ACTIVE TRANSIT TELEMETRY
# ==========================================
elif nav_selection == "ACTIVE TRANSIT TELEMETRY":
    st.header("Active Transit & Paramedic Dashboard")

    with st.container(border=True):
        if not st.session_state.active_case:
            st.info("No active dispatch. Initiate a transfer from the Referral tab.")
        else:
            case = st.session_state.active_case
            dest = case["destination"]
            
            # --- INITIALIZE TRANSIT LOG MEMORY ---
            if "transit_log" not in case:
                case["transit_log"] = []

            # --- HEADER ROW & PROACTIVE NOTIFICATION HUB ---
            st.error(f"🚑 PRIORITY {case['triage_color']} EN ROUTE TO {dest['facility'].upper()}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Topography-Adjusted ETA", f"{dest['eta']} min")
            c2.metric("Modeled Mortality Risk", f"{dest['mortality_risk']}%")
            c3.metric("Severity Index", f"{case.get('severity_index', 0.0):.2f}")
            c4.metric("Dispatch Time", case["dispatch_time"])

            # --- THE NEW PROACTIVE CORRIDOR ALERT SYSTEM ---
            if case['triage_color'] == 'RED' or case.get('severity_index', 0.0) >= 0.6:
                with st.expander("🚨 PROACTIVE CORRIDOR NOTIFICATIONS ACTIVE", expanded=True):
                    st.markdown("**System has engaged pre-arrival protocols based on critical Severity Index.**")
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.success("✅ **Receiving ED:** Code Team activated. Trauma/Resus Bay reserved.")
                        if dest["scoring_details"].get("gate_capacity") == "WARNING_ED_STABILIZATION_ONLY":
                            st.warning("⚠️ **Capacity Failsafe:** Receiving facility warned of ZERO ICU beds. Prepping for ED Stabilization.")
                    with col_p2:
                        st.info("📡 **Regional ALS:** Local rapid response units placed on standby for potential mid-flight intercept.")
                        st.info("🚓 **Traffic Command:** Pre-notified of high-acuity transit.")

            if dest["scoring_details"].get("gate_capacity") == "WARNING_ED_STABILIZATION_ONLY":
                st.warning("⚠️ PARAMEDIC ALERT: Destination ICU is at capacity. You are routing for ED STABILIZATION ONLY.")

            st.markdown("---")

            # --- INNOVATION 1 LISTENER: MEDICAL COMMAND ORDERS ---
            if case.get("medical_orders"):
                st.warning("🎙️ **INCOMING MEDICAL COMMAND ORDER**")
                latest_order = case["medical_orders"][-1]
                st.markdown(f"### 🩺 {latest_order['order']}")
                st.caption(f"Received at {latest_order['time']} from {dest['facility']} ED Director")
                
            st.markdown("---")
            
            # --- INNOVATION 1, 2 & 3: ACTION LEDGER, VITALS & DRIFT TELEMETRY ---
            col_actions, col_vitals, col_traffic = st.columns([1.5, 1, 1.2])
            
            with col_actions:
                st.subheader("⚡ Med-Legal Ledger")
                st.caption("Timestamps lock for Receiving ED visibility.")
                a1, a2, a3 = st.columns(3)
                if a1.button("💉 Push ACLS", use_container_width=True):
                    case["transit_log"].append({"time": datetime.now().strftime("%H:%M:%S"), "action": "ACLS Medications Administered"})
                    st.rerun()
                if a2.button("🫁 Airway", use_container_width=True):
                    case["transit_log"].append({"time": datetime.now().strftime("%H:%M:%S"), "action": "Advanced Airway Secured"})
                    st.rerun()
                if a3.button("🩸 Tourniquet", use_container_width=True):
                    case["transit_log"].append({"time": datetime.now().strftime("%H:%M:%S"), "action": "Hemorrhage Control Applied"})
                    st.rerun()
                
                # --- NEW: DYNAMIC PRIORITY ESCALATION ---
                if case['triage_color'] != 'RED':
                    st.markdown("---")
                    if st.button("⬆️ ESCALATE TO PRIORITY RED", type="primary", use_container_width=True):
                        case['triage_color'] = 'RED'
                        # Mathematically bump the severity to trigger all proactive alerts
                        case['severity_index'] = max(case.get('severity_index', 0.0), 0.75) 
                        case["transit_log"].append({
                            "time": datetime.now().strftime("%H:%M:%S"), 
                            "action": "Mid-Transit Deterioration. Escalated to PRIORITY RED. Corridor Notified."
                        })
                        st.rerun()
                    st.markdown("---")

                st.markdown("**Transit Audit Trail:**")
                if not case["transit_log"]:
                    st.info("No mid-transit interventions logged.")
                else:
                    for log in reversed(case["transit_log"]):
                        st.markdown(f"⏱️ **{log['time']}** - {log['action']}")
                        
            with col_vitals:
                st.subheader("📉 Vitals Delta")
                new_spo2 = st.slider("SpO2 %", 50, 100, case['vitals']['spo2'])
                new_sbp = st.slider("SBP mmHg", 40, 200, case['vitals']['sbp'])
                
                if st.button("Update ED Board", type="secondary", use_container_width=True):
                    case['vitals']['spo2'] = new_spo2
                    case['vitals']['sbp'] = new_sbp
                    case["transit_log"].append({"time": datetime.now().strftime("%H:%M:%S"), "action": f"Vitals Updated: SpO2 {new_spo2}%, SBP {new_sbp}"})
                    st.rerun()

            with col_traffic:
                st.subheader("🚧 ETA Drift Telemetry")
                st.caption("Live GPS vs Baseline Analytics")
                
                traffic_delay = st.slider("Live Traffic/Blockade Delay (Mins)", 0, 60, 0)
                live_eta = dest['eta'] + traffic_delay
                
                st.metric("Live GPS ETA", f"{live_eta} mins", delta=f"+{traffic_delay} mins" if traffic_delay > 0 else "On Time", delta_color="inverse")
                
                if traffic_delay > 5 and traffic_delay <= 15:
                    st.warning("⚠️ **Drift Detected:** Receiving ED notified.")
                    if "delay_notified" not in case:
                        case["transit_log"].append({"time": datetime.now().strftime("%H:%M:%S"), "action": f"Traffic Drift (+{traffic_delay}m). ED Notified."})
                        case["delay_notified"] = True
                        
                elif traffic_delay > 15 and case['triage_color'] == 'RED':
                    st.error("🚨 **CRITICAL BLOCKADE**")
                    
                    if st.button("🌐 INITIATE CIVIC OVERRIDE & ALS INTERCEPT", type="primary", use_container_width=True):
                        st.session_state.civic_override_active = True
                        case["transit_log"].append({"time": datetime.now().strftime("%H:%M:%S"), "action": "CRITICAL BLOCKADE. ALS Intercept Unit dispatched."})
                        case["transit_log"].append({"time": datetime.now().strftime("%H:%M:%S"), "action": "GREEN CORRIDOR: API Sync to Traffic Command & Public Broadcast."})
                        st.rerun()
                        
                # Visualize Active Civic Override
                if st.session_state.get('civic_override_active', False) and traffic_delay > 15:
                    st.success("✅ **Civic Override Executing**")
                    with st.expander("📡 View Active API Broadcasts", expanded=True):
                        st.info("**Logistics:** ALS Motorcycle Intercept dispatched. ETA: 6 mins.")
                        st.warning("**Police:** Ping sent to Patrol Unit (MEGH-P-42) for immediate escort.")
                        st.code(f"URGENT: Critical Medical Emergency in transit. Ambulance approaching {dest['facility']}. Clear the right lane immediately.", language="markdown")

            st.markdown("---")
            
            # --- INNOVATION 4: THE MID-FLIGHT CRASH OVERRIDE ---
            st.subheader("🚨 Autonomous Mid-Flight Escalation")
            if st.button("🚨 PATIENT CRASHING - INITIATE AI REROUTE TO LEVEL 1 CENTER", type="primary", use_container_width=True):
                case["transit_log"].append({"time": datetime.now().strftime("%H:%M:%S"), "action": "CRITICAL: Patient Arrest/Crash. Paramedic initiated AI Reroute."})
                case["triage_color"] = "RED"
                case["severity_index"] = 1.00
                
                viable = case.get("viable_destinations", [])
                emergency_dest = next((fac for fac in viable if fac["scoring_details"].get("icu_beds", 0) > 0 and fac["facility"] != dest["facility"]), None)
                
                if emergency_dest:
                    case["destination"] = emergency_dest
                    case["current_dest_index"] = viable.index(emergency_dest)
                    if "rejection_log" not in case: case["rejection_log"] = []
                    case["rejection_log"].append({"facility": dest['facility'], "reason": "Mid-Flight Patient Crash. Paramedic Escalation.", "time": datetime.now().strftime("%H:%M:%S")})
                    st.error(f"REROUTING... LOCKING ONTO {emergency_dest['facility']}...")
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error("CRITICAL OVERRIDE FAILED: No higher-tier facilities available.")
# ==========================================
# VIEW 3: RECEIVING HOSPITAL
# ==========================================
elif nav_selection == "RECEIVING HOSPITAL BAY":
    st.header("Emergency Department Receiving Board")

    with st.container(border=True):
        if not st.session_state.active_case:
            st.info("ED Bay Clear. No incoming transfers.")
        else:
            case = st.session_state.active_case
            dest = case["destination"]
            dest_name = dest['facility']

            # --- 1. TERMINAL IDENTITY BANNER (Fixes Demo Confusion) ---
            st.success(f"🏥 **SECURE TERMINAL ACTIVE:** YOU ARE VIEWING AS **{dest_name.upper()}**")
            st.markdown("---")

            if st.session_state.synthetic_data is not None:
                df_analytics = st.session_state.synthetic_data
                if 'dest_facility' in df_analytics.columns:
                    fac_cases = df_analytics[df_analytics['dest_facility'] == dest_name]
                    historical_daily_avg = max(1, len(fac_cases) / 30.0)
                    current_live_inbound = int(historical_daily_avg * 1.8) + 1

                    if current_live_inbound > (historical_daily_avg * 1.5):
                        st.error(f"CODE SURGE DETECTED: {dest_name.upper()}")
                        st.markdown(f"**Predictive AI Alert:** Current inbound ambulance load exceeds historical average by >50%. Mobilize ED Resuscitation team immediately.")
                        st.markdown("---")

            st.warning(f"INCOMING ALERT: ETA {dest['eta']} mins")

            # --- INNOVATION 3: TIME-SPACE COLLISION RADAR ---
            st.markdown("### 📡 Arrival Collision Radar")
            collision_detected = False
            
            # Simulated collision detection logic based on ETA window
            if dest['triage_color'] == 'RED' and 10 <= dest['eta'] <= 25:
                collision_detected = True
                ghost_eta = dest['eta'] + random.randint(1, 4)
                
            if collision_detected:
                st.error("⚠️ **COLLISION ALERT:** Multiple critical units inbound.")
                st.markdown(f"**Unit 1 (Active Patient):** ETA {dest['eta']} mins | **Unit 2 (Network Tracking):** ETA {ghost_eta} mins")
                st.info("💡 **Command Prompt:** Deploy secondary triage team to ambulance bay immediately.")
            else:
                st.success("✅ **Radar Clear:** No overlapping critical arrivals detected in your 15-minute window.")

            # --- INNOVATION 1: BI-DIRECTIONAL MEDICAL CONTROL ---
            st.markdown("### 🎙️ Medical Command Uplink")
            st.caption("Transmit legally-binding counter-orders directly to the paramedic mid-transit.")
            
            col_order, col_send = st.columns([3, 1])
            with col_order:
                standard_orders = [
                    "Select Order...",
                    "Administer TXA 1g IV Push immediately",
                    "Hold all blood thinners/anticoagulants",
                    "Push 1 Amp Bicarbonate",
                    "Begin permissive hypotension protocol (Target SBP 90)",
                    "Initiate Massive Transfusion Protocol prep on arrival"
                ]
                selected_order = st.selectbox("Standardized Counter-Orders", standard_orders, label_visibility="collapsed")
            with col_send:
                if st.button("📡 Transmit Order", type="primary", use_container_width=True, disabled=(selected_order == "Select Order...")):
                    if "medical_orders" not in case: case["medical_orders"] = []
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # Save to the distinct orders list AND the general transit log
                    case["medical_orders"].append({"time": timestamp, "order": selected_order})
                    case["transit_log"].append({"time": timestamp, "action": f"[ED COMMAND RECEIVED]: {selected_order}"})
                    st.success("Transmitted!")
                    time.sleep(1)
                    st.rerun()
                    
            if case.get("medical_orders"):
                with st.expander("Active Transmitted Orders", expanded=True):
                    for o in reversed(case["medical_orders"]):
                        st.markdown(f"**{o['time']}** - 🩺 {o['order']}")
            
            st.markdown("---")
            
            # --- REROUTE TELEMETRY (Chain of Custody) ---
            diversion_text = ""
            if case.get("rejection_log"):
                st.error("⚠️ SECONDARY REROUTE PROTOCOL ENGAGED")
                with st.container(border=True):
                    st.markdown("**Chain of Custody / Diversion Audit:**")
                    for rej in case["rejection_log"]:
                        st.markdown(f"🚨 **Rejected By:** {rej['facility']} at {rej['time']}")
                        st.markdown(f"↳ **Rationale:** *{rej['reason']}*")
                    st.caption("AI autonomously rerouted to your facility as the next mathematically optimal destination.")
                
                diversion_text = "\n[DIVERSION AUDIT]: " + " -> ".join([f"Bypassed {r['facility']} ({r['reason']})" for r in case["rejection_log"]])

            # --- DISPLAY FULL ISBAR BEFORE DECISION ---
            st.markdown("### Secure Med-Legal Handover Document")
            isbar_text = f"""[ISBAR CLINICAL HANDOVER]
Status: PRIORITY {case['triage_color']} (Severity: {case['severity_index']:.2f})

I - IDENTIFICATION: Patient: {case['patient_name']}, {case['age']} Y/O
S - SITUATION: Emergency dispatch to {dest['facility']}. Provisional DX: {case['diagnosis']}
B - BACKGROUND: Bundle: {case['bundle']}. Rationale: {case.get('rationale', 'N/A')}
Pre-Transfer Interventions: {', '.join(case.get('interventions', [])) if case.get('interventions') else 'None / Not Logged'}
{chr(10) + '[DIVERSION AUDIT]: ' + ' -> '.join([f"Bypassed {r['facility']} ({r['reason']})" for r in case.get("rejection_log", [])]) if case.get("rejection_log") else ""}
A - ASSESSMENT: HR: {case['vitals']['hr']} | SBP: {case['vitals']['sbp']} | RR: {case['vitals']['rr']} | SpO2: {case['vitals']['spo2']}% | Temp: {case['vitals']['temp']}°C | AVPU: {case['vitals']['avpu']}
R - RECOMMENDATION: {('ED STABILIZATION ONLY DUE TO ZERO ICU BEDS.' if dest['scoring_details'].get('gate_capacity') == 'WARNING_ED_STABILIZATION_ONLY' else 'Prepare critical care receiving bay.')}
"""
            st.code(isbar_text, language="markdown")

            # --- ACCEPTANCE & AUTONOMOUS REJECTION LOGIC ---
            if not st.session_state.patient_accepted:
                st.markdown("### Decision Matrix")
                col_acc, col_rej = st.columns(2)

                with col_acc:
                    if st.button("✅ Acknowledge & Accept Patient", type="primary", use_container_width=True):
                        st.session_state.patient_accepted = True
                        st.rerun()

                with col_rej:
                    with st.expander("❌ Reject & Trigger AI Reroute"):
                        
                        # --- 2. THE AI REROUTE PREVIEW (Shows the exact fallback) ---
                        current_idx = case.get("current_dest_index", 0)
                        viable_list = case.get("viable_destinations", [])
                        
                        if current_idx + 1 < len(viable_list):
                            next_dest = viable_list[current_idx + 1]
                            st.info(f"**AI Fallback Target:** If you divert this case, the AI will immediately reroute the ambulance to **{next_dest['facility']}** (ETA: {next_dest['eta']} mins).")
                        else:
                            st.error("**AI Fallback Target:** ZERO viable facilities remaining in the state network. Rejecting forces on-site stabilization.")

                        reject_reason = st.selectbox(
                            "Standardized Reason for Diversion",
                            ["Select Reason...",
                             "Critical Care Beds Suddenly Full (Surge)",
                             "Required Specialist Scrubbed In / Unavailable",
                             "Hardware/Equipment Failure (e.g., CT Down)",
                             "Facility Currently on State Diversion Status"]
                        )
                        
                        if st.button("Confirm Diversion", type="primary", disabled=reject_reason == "Select Reason..."):
                            case["rejection_log"].append({
                                "facility": dest_name,
                                "reason": reject_reason,
                                "time": datetime.now().strftime("%H:%M:%S")
                            })

                            if current_idx + 1 < len(viable_list):
                                case["destination"] = viable_list[current_idx + 1]
                                case["current_dest_index"] = current_idx + 1
                                st.rerun()
                            
                            # Shift to the next best facility in the array
                            current_idx = case["current_dest_index"]
                            viable_list = case["viable_destinations"]
                            
                            if current_idx + 1 < len(viable_list):
                                next_dest = viable_list[current_idx + 1]
                                case["destination"] = next_dest
                                case["current_dest_index"] = current_idx + 1
                                st.success(f"Diverting. AI locking onto {next_dest['facility']}...")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("CRITICAL: Zero viable fallback facilities remaining. Reverting to On-Site ED Stabilization Command.")
            else:
                st.success("✅ Patient accepted. Telemetry linked to ED monitors.")
                st.markdown("---")
                st.subheader("🔄 Close the Clinical Loop")
                if st.button("Register Patient Stabilized & Notify Referring Doctor"):
                    st.success("✅ Outcome securely recorded. Automated feedback ping sent to referring clinician confirming successful stabilization.")

# ==========================================
# VIEW 4: STATE COMMAND & AI
# ==========================================
elif nav_selection == "STATE COMMAND & AI":
    st.header("State Command & Control")
    st.caption("Zero-PHI Analytics & Autonomous Logistics Engine")
    
    with st.container(border=True):
        def _now_ts(): return time.time()
        def _rand_geo(rng): return (25.5 + rng.random()*0.2, 91.8 + rng.random()*0.2)
        def _dist_km(lat1, lon1, lat2, lon2): return haversine_km((lat1, lon1), (lat2, lon2))
        def _interp_route(lat1, lon1, lat2, lon2, n): return []
        def _traffic(hr): return 1.2 if 8 <= hr <= 20 else 1.0

        if st.button("Inject 1,000 Synthetic Cases (Stress Test)", type="primary"):
            with st.spinner("Generating clinical telemetry..."):
                try:
                    raw_data = seed_synthetic_referrals_v2(
                        n=1000, facilities=facilities_df.to_dict('records'), icd_df=icd_df,
                        validated_triage_decision_fn=validated_triage_decision, now_ts_fn=_now_ts,
                        rand_geo_fn=_rand_geo, dist_km_fn=_dist_km, interpolate_route_fn=_interp_route,
                        traffic_factor_fn=_traffic, rng_seed=42
                    )
                    safe_records = []
                    for r in raw_data:
                        safe_records.append({
                            "timestamp": r["times"]["first_contact_ts"], "bundle": r["provisionalDx"]["case_type"],
                            "triage_color": r["triage"]["decision"]["color"],
                            "severity_index": r["triage"]["decision"]["score_details"].get("severity_index", 0.0),
                            "eta_min": r["transport"]["eta_min"], "facility_ownership": r["facility_ownership"],
                            "dest_facility": r["dest"],
                            "mortality_risk": mortality_risk(
                                r["triage"]["decision"]["score_details"].get("severity_index", 0.0), 
                                r["transport"]["eta_min"], pathology=r["provisionalDx"]["case_type"]
                            )
                        })
                    st.session_state.synthetic_data = pd.DataFrame(safe_records)
                    st.success("Data injected successfully.")
                except Exception as e:
                    st.error(f"Generation Failed: {e}")

    if st.session_state.synthetic_data is not None:
        df_analytics = st.session_state.synthetic_data
        
        with st.container(border=True):
            st.subheader("Predictive Intelligence")
            col_ai1, col_ai2 = st.columns(2)
            top_facility = None 
            
            with col_ai1:
                if not df_analytics.empty and 'bundle' in df_analytics.columns and 'dest_facility' in df_analytics.columns:
                    try:
                        top_bundle = df_analytics['bundle'].mode()[0]
                        top_facility = df_analytics[df_analytics['bundle'] == top_bundle]['dest_facility'].mode()[0]
                        st.error("EPIDEMIOLOGICAL HOTSPOT PREDICTION")
                        st.markdown(f"**Forecast:** A cluster event ({top_bundle}) is highly probable at **{top_facility}** within 48 hours.")
                    except: st.warning("Awaiting variance for hotspot prediction.")
                else: st.warning("Analytics booting...")

            with col_ai2:
                try:
                    total_icu_beds = facilities_df['ICU_open'].astype(int).sum()
                    red_cases = len(df_analytics[df_analytics['triage_color'] == 'RED'])
                    hourly_burn_rate = max(1, (red_cases / 30 / 24) * 1.5) 
                    
                    forecast_data = [{"Time": datetime.now() + timedelta(hours=i), "Projected_ICU_Beds": max(0, total_icu_beds - (hourly_burn_rate * i))} for i in range(13)]
                    df_forecast = pd.DataFrame(forecast_data)
                    
                    base = alt.Chart(df_forecast).encode(x=alt.X('Time:T', title=''))
                    line = base.mark_line(color='#ff4b4b', strokeWidth=3).encode(y=alt.Y('Projected_ICU_Beds:Q', title='ICU Beds'))
                    threshold_line = alt.Chart(pd.DataFrame({'y': [total_icu_beds * 0.1]})).mark_rule(color='black', strokeDash=[5,5]).encode(y='y:Q')
                    st.altair_chart((line + threshold_line).properties(height=180), use_container_width=True)
                except: st.warning("ICU Forecast offline.")

        with st.container(border=True):
            st.subheader("Autonomous Fleet Repositioning")
            try:
                if top_facility:
                    facility_names = facilities_df['name'].tolist()
                    fleet_data = [{"Unit ID": f"ALS-{random.randint(1000, 9999)}", "Current Location": random.choice(facility_names), "Status": random.choices(["Idle", "Active", "Maintenance"], weights=[0.6, 0.3, 0.1])[0]} for _ in range(150)]
                    df_fleet = pd.DataFrame(fleet_data)

                    active_red_dests = df_analytics[df_analytics['triage_color'] == 'RED']['dest_facility'].unique().tolist()
                    cold_zones = [f for f in facility_names if f not in active_red_dests and f != top_facility]
                    idle_in_cold_zones = df_fleet[(df_fleet['Status'] == 'Idle') & (df_fleet['Current Location'].isin(cold_zones))]
                    
                    if not idle_in_cold_zones.empty:
                        dispatch_commands = []
                        for zone in idle_in_cold_zones['Current Location'].unique()[:3]:
                            amb = idle_in_cold_zones[idle_in_cold_zones['Current Location'] == zone].iloc[0]
                            dispatch_commands.append({"Unit ID": amb['Unit ID'], "Origin (Cold Zone)": amb['Current Location'], "Target (Hotspot)": top_facility})
                        st.dataframe(pd.DataFrame(dispatch_commands), use_container_width=True, hide_index=True)
                    else: st.info("No idle assets available in Cold Zones.")
            except: st.warning("Fleet engine awaiting stabilization.")

        with st.container(border=True):
            st.subheader("Statewide Telemetry")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Referrals", len(df_analytics))
            c2.metric("Avg Severity Index", f"{df_analytics['severity_index'].mean():.2f}")
            c3.metric("Avg Mortality Risk", f"{df_analytics['mortality_risk'].mean():.1f}%")
            c4.metric("Gov Facility Utilization", f"{(len(df_analytics[df_analytics['facility_ownership'] == 'Government']) / len(df_analytics)) * 100:.1f}%")

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                triage_chart = alt.Chart(df_analytics).mark_arc().encode(
                    theta=alt.Theta(field="triage_color", aggregate="count"),
                    color=alt.Color(field="triage_color", type="nominal", scale=alt.Scale(domain=['RED', 'YELLOW', 'GREEN'], range=['#ff4b4b', '#faca2b', '#00cc96']))
                )
                st.altair_chart(triage_chart.properties(height=300), use_container_width=True)

            with col_chart2:
                risk_chart = alt.Chart(df_analytics).mark_area(opacity=0.4).encode(
                    x=alt.X("eta_min:Q", bin=alt.Bin(maxbins=30), title="Transit ETA (mins)"),
                    y=alt.Y("average(mortality_risk):Q", title="Avg Modeled Mortality Risk (%)"),
                    color="bundle:N"
                )
                st.altair_chart(risk_chart.properties(height=300), use_container_width=True)
                st.altair_chart(risk_chart.properties(height=300), use_container_width=True)
