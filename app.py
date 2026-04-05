import streamlit as st
import pandas as pd
import altair as alt
import time
import random
import os
import hashlib
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

# --- State Management & Fleet Rosters ---
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

# Mock State Fleet Roster
PILOTS = ["Khraw", "Mewan", "Donbok", "Bantei", "Pynskhem", "Lamphrang", "Kynsai", "Banshanlang"]
EMTS = ["Dr. Sarah", "Dr. Aibor", "Paramedic Grace", "Paramedic John", "Nurse Riba", "Paramedic Samuel"]

# --- THE CACHE-BUSTER ---
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

    icd_df.columns = [
        str(c).replace('ï»¿', '').replace('\ufeff', '').strip().lower().replace('"', '').replace("'", "") 
        for c in icd_df.columns
    ]
    
    if 'bundle' not in icd_df.columns:
        st.error("🚨 DATA FORMAT ERROR: The column 'bundle' is missing from the dataset.")
        st.stop()

    rename_map = {'icd-10': 'icd10', 'icd_10': 'icd10', 'icd code': 'icd10', 'code': 'icd10'}
    icd_df = icd_df.rename(columns=rename_map)
    if 'icd10' not in icd_df.columns: icd_df = icd_df.rename(columns={icd_df.columns[0]: 'icd10'})
    
    for c in ["lat","lon"]:
        if c in fac_df.columns: fac_df[c] = pd.to_numeric(fac_df[c], errors='coerce')
    fac_df["ownership"] = fac_df.get("ownership", "Private").fillna("Private")
    
    return fac_df, icd_df

facilities_df, icd_df = load_datasets_v3()
    
# --- Sidebar Navigation & RBAC Security ---
with st.sidebar:
    st.title("AHECN OS")
    st.caption("Runo Health Enterprise v6.0")
    
    st.markdown("---")
    st.subheader("🔐 Security & Access Control")
    simulated_role = st.selectbox(
        "Simulate Active User Login:",
        [
            "State Health Command (Macro View)", 
            "Authorized Community Node (ASHA/CFR)", 
            "Citizen Self-Booking App", 
            "Director: NEIGRIHMS", 
            "Director: Woodland Hospital", 
            "Director: Bethany Hospital"
        ]
    )
    st.session_state.user_role = simulated_role
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
    
    # ---------------------------------------------------------
    # THE UNIVERSAL COMMUNITY & CITIZEN GATEWAY
    # ---------------------------------------------------------
    if st.session_state.user_role in ["Authorized Community Node (ASHA/CFR)", "Citizen Self-Booking App"]:
        is_asha = (st.session_state.user_role == "Authorized Community Node (ASHA/CFR)")
        
        st.header("Health Transport Booking")
        if is_asha:
            st.caption("🔐 Authorized Node: Initiating heavily subsidized Tier-3 Health Ride for vulnerable populations.")
        else:
            st.caption("📱 Direct-to-Citizen App: Book a standard-rate Health Cab to your nearest clinic.")
            
        with st.container(border=True):
            st.subheader("Patient Request Details")
            c1, c2, c3 = st.columns(3)
            patient_name = c1.text_input("Patient Name", "Local Resident")
            age = c2.number_input("Age", 0, 120, 45)
            
            symptom_cat = c3.selectbox("Primary Complaint", [
                "Orthopedic (Fracture/Sprain) [GREEN]", 
                "Fever/Infection [GREEN]", 
                "Maternal (Routine Checkup) [GREEN]", 
                "Moderate Abdominal Pain / Laceration [YELLOW]",
                "Severe Chest Pain [RED]", 
                "Unconscious / Bleeding [RED]"
            ])
            
            c_lat, c_lon = st.columns(2)
            src_lat = c_lat.number_input("Pickup Latitude", value=25.586936, format="%.6f") 
            src_lon = c_lon.number_input("Pickup Longitude", value=91.809418, format="%.6f")
            
            is_critical = "[RED]" in symptom_cat
            is_yellow = "[YELLOW]" in symptom_cat
            
            if is_critical:
                st.error("🚨 **AI ESCALATION TRIGGERED:** Volatile pathology detected. Highest Priority Transport Required.")
            elif is_yellow:
                st.warning("⚠️ **AI CLEARANCE:** Condition urgent but stable (YELLOW). Authorized for standard Tier-3 Cab Transport.")
            else:
                st.success("✅ **AI CLEARANCE:** Condition stable (GREEN). Authorized for standard Tier-3 Cab Transport.")
                
            st.markdown("---")
            
            if st.button("🔍 Find Healthcare Facilities & Check Fleet", type="primary"):
                with st.spinner("Calculating Topography & Pinging State Fleet..."):
                    time.sleep(1.5) # Dramatic pause for the Dynamic Failsafe effect
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
                    
                    # Pre-roll fleet availability heavily weighted to FAIL for the Red-Override Demo
                    st.session_state.demo_als_avail = random.choices([True, False], weights=[0.2, 0.8])[0]
                    st.session_state.demo_bls_avail = random.choices([True, False], weights=[0.2, 0.8])[0]
            
            if st.session_state.get('asha_match_results') is not None:
                res = st.session_state.asha_match_results
                st.markdown("**Available Institutions (Ranked by Proximity):**")
                
                radio_opts = [f"{r['facility']} (ETA: {r['eta']} mins)" for r in res[:5]]
                sel_opt = st.radio("Select Destination (Nearest or Patient's Choice):", radio_opts)
                
                # --- THE DYNAMIC FAILSAFE & TRANSPORT UI ---
                if is_critical:
                    st.markdown("### 🚑 State Fleet Dispatch Radar")
                    als_avail = st.session_state.demo_als_avail
                    bls_avail = st.session_state.demo_bls_avail
                    
                    if als_avail or bls_avail:
                        st.success("✅ **State Ambulance Available:** Unit located within safe ETA.")
                        button_text = "🚑 Dispatch Emergency Ambulance"
                        allocated_fleet = "ALS" if als_avail else "BLS"
                        fleet_status = "OPTIMAL" if als_avail else "DOWNGRADE_RISK"
                    else:
                        st.error("❌ **STATE FLEET EXHAUSTED:** Zero ALS/BLS units available within safe ETA.")
                        st.warning("💡 **CONTINUITY OF CARE FAILSAFE:** Patient life at immediate risk. You are authorized to bypass standard protocol and summon an empanelled civilian Tier-3 Cab for a 'Scoop & Run'.")
                        button_text = "🚨 AUTHORIZE EMERGENCY CAB OVERRIDE (SCOOP & RUN)"
                        allocated_fleet = "CAB"
                        fleet_status = "CRITICAL_CAB_FALLBACK"
                else:
                    if is_asha:
                        st.info("💳 **Payment Method:** 100% State Subsidized / MHIS Covered (Authorized by Node)")
                    else:
                        st.warning("💳 **Payment Method:** Citizen Self-Pay (₹25/km Standard Rate). UPI mandate required upon boarding.")
                    button_text = "🚖 Dispatch Standard Health Cab"
                    allocated_fleet = "CAB"
                    fleet_status = "OPTIMAL"

                # --- DISPATCH EXECUTION LOGIC ---
                if st.button(button_text, type="primary"):
                    sel_fac_name = sel_opt.split(" (ETA:")[0]
                    sel_fac_details = next(r for r in res if r["facility"] == sel_fac_name)["details"]
                    sel_fac_details["scoring_details"] = {"gate_capacity": "PASSED", "clinical_tier": "Basic"}
                    
                    driver, plate, model, emt_name = None, None, None, None
                    
                    if allocated_fleet == "ALS":
                        model, plate, driver, emt_name = f"ALS-{random.randint(101,130)}", f"ML-05-A-{random.randint(1000,9999)}", random.choice(PILOTS), random.choice(EMTS)
                    elif allocated_fleet == "BLS":
                        model, plate, driver, emt_name = f"BLS-{random.randint(201,250)}", f"ML-05-B-{random.randint(1000,9999)}", random.choice(PILOTS), random.choice(EMTS)
                    else:
                        driver, plate, model = random.choice(PILOTS), random.choice(["ML-05-C-8842", "ML-05-T-1123"]), random.choice(["Maruti Swift", "Tata Sumo"])
                        
                    triage_col = "RED" if is_critical else ("YELLOW" if is_yellow else "GREEN")
                    sev_idx = 0.85 if is_critical else (0.45 if is_yellow else 0.1)
                    rationale = "Authorized Subsidized Community Transport" if is_asha else "Citizen Direct App Booking"
                    clean_diagnosis = symptom_cat.split(" [")[0] # Removes the [GREEN]/[RED] tag for the handover doc
                    
                    st.session_state.active_case = {
                        "patient_name": patient_name, "age": age, 
                        "vitals": {"hr": 110 if is_critical else (95 if is_yellow else 80), "rr": 24 if is_critical else (20 if is_yellow else 16), "sbp": 90 if is_critical else (110 if is_yellow else 120), "temp": 37.0, "spo2": 88 if is_critical else (94 if is_yellow else 98), "avpu": "V" if is_critical else "A"}, 
                        "diagnosis": clean_diagnosis, "bundle": "Trauma/Cardiac" if is_critical else ("Urgent Care" if is_yellow else "Green / Standard"), 
                        "triage_color": triage_col, "severity_index": sev_idx,
                        "destination": {"facility": sel_fac_name, "eta": next(r for r in res if r["facility"] == sel_fac_name)["eta"], "scoring_details": {"gate_capacity": "PASSED"}}, 
                        "dispatch_time": datetime.now().strftime("%H:%M:%S"),
                        "rationale": rationale,
                        "interventions": [], "viable_destinations": [], "current_dest_index": 0, "rejection_log": [], "medical_orders": [],
                        "fleet": {"ideal": "ALS" if is_critical else "CAB", "allocated": allocated_fleet, "status": fleet_status, "driver": driver, "plate": plate, "vehicle": model, "emt": emt_name}
                    }
                    st.session_state.transfer_initiated = True
                    st.session_state.patient_accepted = False
                    st.session_state.asha_match_results = None 
                    
                    if fleet_status == "CRITICAL_CAB_FALLBACK":
                        st.session_state.civic_override_active = True
                    st.rerun()
        st.stop()

    # ---------------------------------------------------------
    # STANDARD HOSPITAL-TO-HOSPITAL VIEW
    # ---------------------------------------------------------
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

    with st.container(border=True):
        st.subheader("3. Clinical Rationale")
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

        st.markdown("---")
        st.markdown("⚕️ **Pre-Transfer Resuscitation & Interventions**")
        
        interventions_str = icd_row.get("default_interventions", "")
        available_interventions = [x.strip() for x in str(interventions_str).split(";") if x.strip()]
        
        completed_interventions = []
        if available_interventions:
            completed_interventions = st.multiselect(
                "Select life-saving interventions already administered:",
                available_interventions + ["Other (Specified in notes)"]
            )
        else:
            st.info("No standardized resuscitation bundle found for this pathology. Use notes if needed.")

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

    with st.container(border=True):
        st.subheader("5. Facility Matching (Gated Clinical Safety)")
        
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
                    if tier == "Tier 1: Definitive Care":
                        st.success("✅ **Tier 1 Match:** 100% of requested Specialists and Hardware present.")
                    elif tier == "Tier 2: Advanced Medical Bridging":
                        st.warning(f"⚠️ **Tier 2 Match:** Primary specialist unavailable. AI engaged medical fallback protocol.")
                    else:
                        st.error(f"🚨 **Tier 3 Match:** Definitive care missing. Route optimized for nearest airway/ED resuscitation.")
                    
                    if details.get('gate_capacity') == "PASSED":
                        st.success(f"✅ **Capacity Gate:** Passed. {details.get('icu_beds')} ICU beds available.")
                    elif details.get('gate_capacity') == "WARNING_ED_STABILIZATION_ONLY":
                        st.error("⚠️ **Capacity Gate Override:** ICU Full. Facility selected for immediate ED Resuscitation ONLY.")

                if st.button("🚀 Initiate E2EE Transfer & Dispatch Transport", type="primary"):
                    selected_idx = next(i for i, r in enumerate(results) if r["facility"] == selected_fac["facility"])
                    
                    ideal_fleet = "ALS" if triage_color == "RED" or meta['severity_index'] >= 0.6 else ("BLS" if triage_color == "YELLOW" else "CAB")
                    als_available = random.choices([True, False], weights=[0.4, 0.6])[0] 
                    bls_available = random.choices([True, False], weights=[0.5, 0.5])[0] 
                    
                    driver, plate, model, emt_name = None, None, None, None
                    
                    if ideal_fleet == "ALS":
                        if als_available: 
                            allocated_fleet, fleet_status = "ALS", "OPTIMAL"
                            model, plate, driver, emt_name = f"ALS-{random.randint(101,130)}", f"ML-05-A-{random.randint(1000,9999)}", random.choice(PILOTS), random.choice(EMTS)
                        elif bls_available: 
                            allocated_fleet, fleet_status = "BLS", "DOWNGRADE_RISK"
                            model, plate, driver, emt_name = f"BLS-{random.randint(201,250)}", f"ML-05-B-{random.randint(1000,9999)}", random.choice(PILOTS), random.choice(EMTS)
                        else: 
                            allocated_fleet, fleet_status = "CAB", "CRITICAL_CAB_FALLBACK"
                            driver, plate, model = random.choice(PILOTS), random.choice(["ML-05-C-8842", "ML-05-T-1123"]), random.choice(["Maruti Swift", "Tata Sumo"])
                    elif ideal_fleet == "BLS":
                        if bls_available: 
                            allocated_fleet, fleet_status = "BLS", "OPTIMAL"
                            model, plate, driver, emt_name = f"BLS-{random.randint(201,250)}", f"ML-05-B-{random.randint(1000,9999)}", random.choice(PILOTS), random.choice(EMTS)
                        elif als_available: 
                            allocated_fleet, fleet_status = "ALS", "RESOURCE_WASTE"
                            model, plate, driver, emt_name = f"ALS-{random.randint(101,130)}", f"ML-05-A-{random.randint(1000,9999)}", random.choice(PILOTS), random.choice(EMTS)
                        else: 
                            allocated_fleet, fleet_status = "CAB", "CRITICAL_CAB_FALLBACK"
                            driver, plate, model = random.choice(PILOTS), random.choice(["ML-05-C-8842", "ML-05-T-1123"]), random.choice(["Maruti Swift", "Tata Sumo"])
                    else:
                        allocated_fleet, fleet_status = "CAB", "OPTIMAL"
                        driver, plate, model = random.choice(PILOTS), random.choice(["ML-05-C-8842", "ML-05-T-1123"]), random.choice(["Maruti Swift", "Tata Sumo"])
                    
                    st.session_state.active_case = {
                        "patient_name": patient_name, "age": age, "vitals": vitals, "diagnosis": dx,
                        "bundle": bundle, "triage_color": triage_color, "severity_index": meta['severity_index'],
                        "destination": selected_fac, "dispatch_time": datetime.now().strftime("%H:%M:%S"),
                        "rationale": reason_for_referral,
                        "interventions": completed_interventions, 
                        "viable_destinations": results,          
                        "current_dest_index": selected_idx,      
                        "rejection_log": [],
                        "medical_orders": [],
                        "fleet": {"ideal": ideal_fleet, "allocated": allocated_fleet, "status": fleet_status, "driver": driver, "plate": plate, "vehicle": model, "emt": emt_name}
                    }
                    st.session_state.transfer_initiated = True
                    st.session_state.patient_accepted = False
                    st.session_state.match_results = None 
                    
                    if fleet_status == "CRITICAL_CAB_FALLBACK":
                        st.session_state.civic_override_active = True
                    st.rerun()

    if st.session_state.transfer_initiated and st.session_state.active_case:
        with st.container(border=True):
            st.subheader("📄 Med-Legal Documentation Generated")
            st.success("Official ISBAR handover securely logged to the State Registry.")
            case = st.session_state.active_case
            
            transit_note = f"via [ {case['fleet']['allocated']} AMBULANCE ({case['fleet']['vehicle']} - Plate: {case['fleet']['plate']}) | EMT: {case['fleet']['emt']} ]"
            if case['fleet']['allocated'] == 'CAB':
                transit_note = f"via [ TIER-3 CAB ({case['fleet']['plate']}) - NO EMT ON BOARD | Driver: {case['fleet']['driver']} ]"
                
            isbar_text = f"""[ISBAR CLINICAL HANDOVER]
Status: PRIORITY {case['triage_color']} (Severity: {case['severity_index']:.2f})
I - IDENTIFICATION: Patient: {case['patient_name']}, {case['age']} Y/O
S - SITUATION: Emergency dispatch to {case['destination']['facility']} {transit_note}. Provisional DX: {case['diagnosis']}
B - BACKGROUND: Bundle: {case['bundle']}. Rationale: {case.get('rationale', 'N/A')}
Pre-Transfer Interventions: {', '.join(case.get('interventions', [])) if case.get('interventions') else 'None / Not Logged'}
{chr(10) + '[DIVERSION AUDIT]: ' + ' -> '.join([f"Bypassed {r['facility']} ({r['reason']})" for r in case.get("rejection_log", [])]) if case.get("rejection_log") else ""}
A - ASSESSMENT: HR: {case['vitals']['hr']} | SBP: {case['vitals']['sbp']} | RR: {case['vitals']['rr']} | SpO2: {case['vitals']['spo2']}% | Temp: {case['vitals']['temp']}°C | AVPU: {case['vitals']['avpu']}
"""
            st.code(isbar_text, language="markdown")

# ==========================================
# VIEW 2: ACTIVE TRANSIT TELEMETRY
# ==========================================
elif nav_selection == "ACTIVE TRANSIT TELEMETRY":
    st.header("Active Transit & Telemetry Dashboard")

    with st.container(border=True):
        if not st.session_state.active_case:
            st.info("No active dispatch. Initiate a transfer from the Referral tab.")
        else:
            case = st.session_state.active_case
            dest = case["destination"]
            is_cab = (case['fleet']['allocated'] == 'CAB')
            if "transit_log" not in case: case["transit_log"] = []

            fleet_type = case['fleet']['allocated']
            if is_cab:
                if case['triage_color'] == 'RED':
                    st.error(f"🚨 **CRITICAL CAB FALLBACK TO {dest['facility'].upper()}**")
                    st.warning("⚠️ **CONTROL ROOM OVERRIDE:** This is a RED Category case in a civilian vehicle. Telemetry and hospital handover are now locked and actively managed by State Dispatch.")
                    st.markdown(f"**Vehicle:** {case['fleet']['vehicle']} ({case['fleet']['plate']})")
                else:
                    st.info(f"🚖 **TIER-3 CAB DISPATCHED TO {dest['facility'].upper()}**")
                    st.markdown(f"**Vehicle:** {case['fleet']['vehicle']} ({case['fleet']['plate']}) | **Driver:** {case['fleet']['driver']}")
                    st.caption("Standard transport. No ED pre-alert required.")
            else:
                st.error(f"🚑 PRIORITY {case['triage_color']} EN ROUTE TO {dest['facility'].upper()} [{fleet_type} UNIT]")
                st.markdown(f"**Unit Details:** {case['fleet']['vehicle']} ({case['fleet']['plate']}) | **Pilot:** {case['fleet']['driver']} | **Lead EMT:** {case['fleet']['emt']}")
            
            if case['fleet']['status'] == "DOWNGRADE_RISK":
                st.warning("⚠️ **FLEET WARNING:** Patient acuity requires ALS. Due to state shortage, BLS allocated. Proceed with maximum caution.")
            elif case['fleet']['status'] == "CRITICAL_CAB_FALLBACK":
                st.error("🚨 **CRITICAL CAB FALLBACK:** State ambulance fleet exhausted. Un-equipped Tier-3 Cab dispatched to save life. Proactive Police Escort Engaged.")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Topography-Adjusted ETA", f"{dest['eta']} min")
            if not is_cab or case['fleet']['status'] == "CRITICAL_CAB_FALLBACK":
                c2.metric("Modeled Mortality Risk", f"{dest.get('mortality_risk', 0.0)}%")
                c3.metric("Severity Index", f"{case.get('severity_index', 0.0):.2f}")
            c4.metric("Dispatch Time", case["dispatch_time"])

            st.markdown("---")

            if case.get("medical_orders") and not is_cab:
                st.warning("🎙️ **INCOMING MEDICAL COMMAND ORDER**")
                st.markdown(f"### 🩺 {case['medical_orders'][-1]['order']}")
                st.markdown("---")
            
            if is_cab:
                col_traffic, = st.columns(1)
            else:
                col_actions, col_vitals, col_traffic = st.columns([1.5, 1, 1.2])
                with col_actions:
                    st.subheader("⚡ Med-Legal Ledger")
                    a1, a2, a3 = st.columns(3)
                    if a1.button("💉 Push ACLS", use_container_width=True):
                        case["transit_log"].append({"time": datetime.now().strftime("%H:%M:%S"), "action": "ACLS Administered"})
                        st.rerun()
                    if a2.button("🫁 Airway", use_container_width=True):
                        case["transit_log"].append({"time": datetime.now().strftime("%H:%M:%S"), "action": "Airway Secured"})
                        st.rerun()
                    if a3.button("🩸 Tourniquet", use_container_width=True):
                        case["transit_log"].append({"time": datetime.now().strftime("%H:%M:%S"), "action": "Hemorrhage Control"})
                        st.rerun()
                        
                with col_vitals:
                    st.subheader("📉 Vitals Delta")
                    new_spo2 = st.slider("SpO2 %", 50, 100, case['vitals']['spo2'])
                    new_sbp = st.slider("SBP mmHg", 40, 200, case['vitals']['sbp'])
                    if st.button("Update ED Board"):
                        case['vitals']['spo2'], case['vitals']['sbp'] = new_spo2, new_sbp
                        st.rerun()

            with col_traffic:
                st.subheader("🚧 ETA Drift Telemetry")
                traffic_delay = st.slider("Live Traffic/Blockade Delay (Mins)", 0, 60, 0)
                live_eta = dest['eta'] + traffic_delay
                st.metric("Live GPS ETA", f"{live_eta} mins", delta=f"+{traffic_delay} mins" if traffic_delay > 0 else "On Time", delta_color="inverse")
                
                if traffic_delay > 15 and (case['triage_color'] == 'RED' or case['fleet']['status'] == "CRITICAL_CAB_FALLBACK"):
                    if st.button("🌐 INITIATE CIVIC OVERRIDE & ALS INTERCEPT", type="primary", use_container_width=True):
                        st.session_state.civic_override_active = True
                        st.rerun()
                        
                if st.session_state.get('civic_override_active', False):
                    st.success("✅ **Civic Override Executing**")
                    with st.expander("📡 View Active API Broadcasts", expanded=True):
                        st.warning("**Police:** Ping sent to Patrol Unit (MEGH-P-42) for immediate escort to Receiving ED.")

# ==========================================
# VIEW 3: RECEIVING HOSPITAL
# ==========================================
elif nav_selection == "RECEIVING HOSPITAL BAY":
    st.header("Emergency Department Receiving Board")

    tab_active, tab_analytics, tab_outcomes = st.tabs(["🚨 Active Receiving Board", "📊 ED Operations Analytics", "🗃️ Clinical Milestones"])

    with tab_active:
        with st.container(border=True):
            case = st.session_state.get('active_case')
            
            # THE NEW VISIBILITY GATE: Hide cabs unless they are RED
            if not case or (case['fleet']['allocated'] == 'CAB' and case['triage_color'] != 'RED'):
                st.info("ED Bay Clear. No incoming critical transfers.")
            else:
                dest = case["destination"]
                dest_name = dest['facility']

                st.success(f"🏥 **SECURE TERMINAL ACTIVE:** YOU ARE VIEWING AS **{dest_name.upper()}**")
                st.markdown("---")
                
                is_cab = (case['fleet']['allocated'] == 'CAB')
                
                if is_cab:
                    st.error(f"🚨 **CRITICAL SCOOP & RUN ALERT: ETA {dest['eta']} mins**")
                    st.markdown(f"**Vehicle:** {case['fleet']['vehicle']} ({case['fleet']['plate']}) | **Status:** Un-resuscitated transit.")
                    st.warning("📡 **COMMUNICATION ROUTING:** Driver comms disabled. Handover and updates are being actively managed by State Control Room.")
                else:
                    st.error(f"🚑 **INCOMING PRIORITY {case['triage_color']} AMBULANCE: ETA {dest['eta']} mins**")
                    st.markdown(f"**Unit:** {case['fleet']['vehicle']} ({case['fleet']['plate']}) | **Pilot:** {case['fleet']['driver']} | **Lead EMT:** {case['fleet']['emt']}")
                
                st.markdown("---")
                st.markdown("### 🎙️ Medical Command Uplink")
                st.caption("Transmit legally-binding counter-orders directly to the paramedic mid-transit. *(Disabled if Cab Transport)*")
                
                bundle = case.get('bundle', 'Other')
                dynamic_orders = ["Select Order...", "Direct to Cath Lab (Bypass ED)", "Initiate Massive Transfusion Protocol", "Push IV Labetalol", "Other / Custom Order (Type below)"]

                col_order, col_send = st.columns([3, 1])
                with col_order:
                    selected_order = st.selectbox("Standardized Counter-Orders", dynamic_orders, label_visibility="collapsed", disabled=is_cab)
                    custom_order = st.text_input("Type Custom Medical Order:", placeholder="E.g., Push 1mg Atropine IV...", disabled=is_cab) if selected_order == "Other / Custom Order (Type below)" else ""
                with col_send:
                    if selected_order == "Other / Custom Order (Type below)": st.markdown("<br>", unsafe_allow_html=True) 
                    final_order = custom_order if selected_order == "Other / Custom Order (Type below)" else selected_order
                    is_disabled = is_cab or selected_order == "Select Order..." or (selected_order == "Other / Custom Order (Type below)" and not custom_order.strip())
                    if st.button("📡 Transmit Order", type="primary", use_container_width=True, disabled=is_disabled):
                        if "medical_orders" not in case: case["medical_orders"] = []
                        case["medical_orders"].append({"time": datetime.now().strftime("%H:%M:%S"), "order": final_order})
                        st.rerun()
                        
                if not st.session_state.patient_accepted:
                    st.markdown("### Decision Matrix")
                    col_acc, col_rej = st.columns(2)

                    with col_acc:
                        if st.button("✅ Acknowledge & Accept Patient", type="primary", use_container_width=True):
                            st.session_state.patient_accepted = True
                            st.rerun()

                    with col_rej:
                        with st.expander("❌ Reject & Trigger AI Reroute"):
                            current_idx = case.get("current_dest_index", 0)
                            viable_list = case.get("viable_destinations", [])
                            
                            # Show the doctor exactly where the AI will send the patient next
                            if current_idx + 1 < len(viable_list):
                                next_dest = viable_list[current_idx + 1]
                                st.info(f"**AI Fallback Target:** If you divert this case, the AI will immediately reroute to **{next_dest['facility']}** (ETA: {next_dest['eta']} mins).")
                            else:
                                st.error("**AI Fallback Target:** ZERO viable facilities remaining. Rejecting forces on-site stabilization.")

                            # The Restored Reason Dropdown
                            reject_reason = st.selectbox(
                                "Standardized Reason for Diversion",
                                ["Select Reason...", "Critical Care Beds Suddenly Full (Surge)", "Required Specialist Scrubbed In / Unavailable", "Hardware/Equipment Failure (e.g., CT Down)", "Facility Currently on State Diversion Status"]
                            )
                            
                            if st.button("Confirm Diversion", type="primary", disabled=reject_reason == "Select Reason..."):
                                case["rejection_log"].append({"facility": dest_name, "reason": reject_reason, "time": datetime.now().strftime("%H:%M:%S")})
                                
                                if current_idx + 1 < len(viable_list):
                                    next_dest = viable_list[current_idx + 1]
                                    case["destination"] = next_dest
                                    case["current_dest_index"] = current_idx + 1
                                    st.success(f"Diverting. AI locking onto {next_dest['facility']}...")
                                    time.sleep(1.5)
                                    st.rerun()
                                else:
                                    st.error("CRITICAL: Zero viable fallback facilities remaining. Reverting to On-Site ED Stabilization Command.")
                else:
                    st.success("✅ Patient accepted. Telemetry linked to ED monitors.")

    with tab_analytics:
        st.subheader("🏥 Institutional Operations Analytics")
        if st.session_state.user_role in ["State Health Command (Macro View)", "Authorized Community Node (ASHA/CFR)", "Citizen Self-Booking App"]:
            st.warning("⚠️ **ACCESS DENIED:** View restricted to Facility Directors. \n\n*Demo Tip: Change your role in the sidebar to a Hospital Director.*")
        else:
            active_hospital = st.session_state.user_role.replace("Director: ", "")
            st.success(f"🔐 **DATA SILO ACTIVE:** Displaying isolated operational telemetry for {active_hospital} only.")
            
            h_seed = int(hashlib.md5(active_hospital.encode()).hexdigest(), 16) % 10000
            rng = random.Random(h_seed)
            
            base_vol = rng.randint(110, 450)
            div_vol = rng.randint(5, int(base_vol * 0.15))
            delay_mins = round(rng.uniform(1.2, 8.5), 1)
            t2b_mins = round(rng.uniform(4.5, 14.0), 1)
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(f"Accepted at {active_hospital}", f"{base_vol}", f"+{rng.randint(2, 18)}% MoM")
            kpi2.metric("Local Diversions", f"{div_vol}", f"{rng.choice(['+', '-'])}{rng.randint(1, 5)}% MoM", delta_color="inverse")
            kpi3.metric("Fleet Arrival Delay (SLA)", f"+{delay_mins} mins", f"+0.{rng.randint(1, 9)} mins", delta_color="inverse")
            kpi4.metric("Avg Triage-to-Bed Time", f"{t2b_mins} mins", f"-1.{rng.randint(1, 5)} mins")
            
            st.markdown("---")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown(f"**Local Diversion Autopsy ({active_hospital})**")
                st.caption("Primary reasons this facility rejected critical transfers.")
                
                reasons = ["Zero ICU Beds", "Neurosurgeon Unavailable", "CT Scanner Down", "Internal Code Black", "Blood Bank Depleted"]
                counts = [rng.randint(1, 10) for _ in range(5)]
                rejection_data = pd.DataFrame({"Reason": reasons, "Count": counts})
                rejection_data = rejection_data[rejection_data["Count"] > 0]
                
                st.altair_chart(alt.Chart(rejection_data).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="Count", type="quantitative"),
                    color=alt.Color(field="Reason", type="nominal"),
                    tooltip=["Reason", "Count"]
                ).properties(height=300), use_container_width=True)
                
            with col_chart2:
                st.markdown("**Inbound Acuity Distribution**")
                st.caption("Pathology breakdown of cases received this month.")
                
                paths = ["Cardiac", "Trauma", "Stroke", "Maternal", "Pediatric", "Sepsis"]
                vols = [rng.randint(10, int(base_vol*0.4)) for _ in range(6)]
                path_data = pd.DataFrame({"Pathology": paths, "Volume": vols})
                
                st.altair_chart(alt.Chart(path_data).mark_bar().encode(
                    x=alt.X('Pathology:N', sort='-y'),
                    y='Volume:Q',
                    color=alt.value('#1f77b4'),
                    tooltip=["Pathology", "Volume"]
                ).properties(height=300), use_container_width=True)

    with tab_outcomes:
        st.subheader("🗃️ Clinical Milestone & Outcome Reconciliation")
        if st.session_state.user_role in ["State Health Command (Macro View)", "Authorized Community Node (ASHA/CFR)", "Citizen Self-Booking App"]:
            st.warning("⚠️ **ACCESS DENIED:** View restricted to Facility Directors. \n\n*Demo Tip: Change your role in the sidebar to a Hospital Director.*")
        else:
            st.markdown("Clear pending milestones below to close the state mortality audit loop. *(Note: Duty officers receive 1-click WhatsApp 'Magic Links' to log these asynchronously).*")
            
            active_hospital = st.session_state.user_role.replace("Director: ", "")
            h_seed = int(hashlib.md5(active_hospital.encode()).hexdigest(), 16) % 10000
            rng = random.Random(h_seed + 3) 
            
            col_q1, col_q2 = st.columns(2)
            
            with col_q1:
                st.markdown("### ⏱️ 24-Hour Stabilization Check")
                st.caption("Assessing Pre-Hospital & ED Resuscitation Efficacy")
                for i in range(2):
                    with st.container(border=True):
                        st.markdown(f"**ID: AHECN-24H-{rng.randint(100,999)}** | {random.choice(['Severe Trauma', 'STEMI', 'Stroke'])}")
                        st.caption("Arrived: ~24 Hours Ago")
                        b1, b2, b3 = st.columns(3)
                        if b1.button("🟢 Stabilized / Admitted", key=f"m1_a_{i}", use_container_width=True): st.success("Logged.")
                        if b2.button("🔴 Deceased in ED", key=f"m1_d_{i}", type="primary", use_container_width=True): st.error("Logged.")
                        if b3.button("↪️ Re-Referred", key=f"m1_r_{i}", use_container_width=True): st.warning("Logged.")

            with col_q2:
                st.markdown("### 🛏️ ALOS Disposition Check")
                st.caption("Assessing Resource Utilization & Final Outcome")
                for i in range(2):
                    with st.container(border=True):
                        st.markdown(f"**ID: AHECN-ALOS-{rng.randint(100,999)}** | {random.choice(['Maternal Hemorrhage', 'Pediatric Resp'])}")
                        st.caption(f"Arrived: {rng.randint(3, 8)} Days Ago (ALOS Exceeded)")
                        b1, b2, b3 = st.columns(3)
                        if b1.button("🟢 Discharged Home", key=f"m2_d_{i}", use_container_width=True): st.success("Logged.")
                        if b2.button("🟡 Still Admitted", key=f"m2_s_{i}", use_container_width=True): st.info("Logged.")
                        if b3.button("🔴 Deceased in ICU", key=f"m2_x_{i}", type="primary", use_container_width=True): st.error("Logged.")

# ==========================================
# VIEW 4: STATE COMMAND & AI
# ==========================================
elif nav_selection == "STATE COMMAND & AI":
    st.header("State Command & Control")

    if st.session_state.user_role != "State Health Command (Macro View)":
        st.error("🛑 **CLEARANCE VIOLATION**")
        st.markdown("View compartmentalized for State Health Officials only.")
    else:
        st.caption("Zero-PHI Analytics & Autonomous Logistics Engine")

        with st.container(border=True):
            def _now_ts(): return time.time()
            def _rand_geo(rng): return (25.5 + rng.random()*0.2, 91.8 + rng.random()*0.2)
            def _dist_km(lat1, lon1, lat2, lon2): return haversine_km((lat1, lon1), (lat2, lon2))
            def _interp_route(lat1, lon1, lat2, lon2, n): return []
            def _traffic(hr): return 1.2 if 8 <= hr <= 20 else 1.0

            if st.button("Inject 1,000 Synthetic Cases (Stress Test)", type="primary"):
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
                            had_intervention = random.choice([True, False])
                            sev_idx = r["triage"]["decision"]["score_details"].get("severity_index", 0.0)
                            bundle_name = r["provisionalDx"]["case_type"]
                            base_mortality = mortality_risk(sev_idx, r["transport"]["eta_min"], pathology=bundle_name)
                            final_mortality = base_mortality * random.uniform(0.75, 0.85) if had_intervention else base_mortality

                            # Fleet & Unit ID Allocation Logic
                            ideal = "ALS" if r["triage"]["decision"]["color"] == "RED" or sev_idx >= 0.6 else "BLS"
                            als_avail = random.choices([True, False], weights=[0.75, 0.25])[0]
                            bls_avail = random.choices([True, False], weights=[0.8, 0.2])[0]
                            
                            if ideal == "ALS":
                                if als_avail: 
                                    mission_type = "Optimal: ALS for Critical"
                                    allocated_fleet = "ALS"
                                    unit_id = f"ALS-{random.randint(101, 115)}"
                                elif bls_avail: 
                                    mission_type = "Risk: BLS deployed for Critical Case"
                                    allocated_fleet = "BLS"
                                    unit_id = f"BLS-{random.randint(201, 230)}"
                                else: 
                                    mission_type = "Desperation: Cab used for Critical"
                                    allocated_fleet = "CAB"
                                    unit_id = "CAB-CIVILIAN"
                            elif ideal == "BLS":
                                if bls_avail: 
                                    mission_type = "Optimal: BLS for Stable"
                                    allocated_fleet = "BLS"
                                    unit_id = f"BLS-{random.randint(201, 230)}"
                                elif als_avail: 
                                    mission_type = "Waste: ALS deployed for Stable Case"
                                    allocated_fleet = "ALS"
                                    unit_id = f"ALS-{random.randint(101, 115)}"
                                else: 
                                    mission_type = "Desperation: Cab used for Critical"
                                    allocated_fleet = "CAB"
                                    unit_id = "CAB-CIVILIAN"
                            else:
                                mission_type = "Optimal: Cab for Green"
                                allocated_fleet = "CAB"
                                unit_id = "CAB-CIVILIAN"

                            trip_delay = random.uniform(1.0, 15.0)
                            actual_trip_time = r["transport"]["eta_min"] + trip_delay

                            # MHIS ECONOMICS ENGINE
                            mhis_base = {"Cardiac": 120000, "Trauma": 180000, "Maternal": 45000, "Stroke": 85000, "Pediatric": 25000, "Neonatal": 40000}
                            mhis_val = mhis_base.get(bundle_name, 50000) * random.uniform(0.9, 1.1)
                            
                            first_choice_gov = random.choices([True, False], weights=[0.65, 0.35])[0]
                            if first_choice_gov:
                                if random.random() < 0.3: 
                                    route = "Govt -> Private (Leakage)"
                                    final_sec = "Private"
                                else:
                                    route = "Govt -> Govt (Retained)"
                                    final_sec = "Government"
                            else:
                                route = "Private -> Private (Standard)"
                                final_sec = "Private"
                                
                            alos_var = random.randint(-1, 6)
                            burden = max(0, alos_var) * 5000 if final_sec == "Government" else 0

                            safe_records.append({
                                "timestamp": r["times"]["first_contact_ts"], "bundle": bundle_name,
                                "triage_color": r["triage"]["decision"]["color"],
                                "severity_index": sev_idx,
                                "eta_min": r["transport"]["eta_min"], 
                                "actual_trip_time": actual_trip_time,
                                "facility_ownership": r["facility_ownership"],
                                "dest_facility": r["dest"],
                                "pre_hospital_intervention": "Yes" if had_intervention else "No",
                                "mortality_risk": min(99.9, final_mortality),
                                "origin_district": random.choice(["East Khasi Hills", "West Garo Hills", "Jaintia Hills", "Ri-Bhoi", "South Garo Hills"]),
                                "status": random.choices(["Accepted", "Diverted"], weights=[0.78, 0.22])[0],
                                "mission_type": mission_type,
                                "allocated_fleet": allocated_fleet,
                                "unit_id": unit_id,
                                "mhis_value": mhis_val,
                                "routing_path": route,
                                "uncompensated_burden": burden
                            })
                        st.session_state.synthetic_data = pd.DataFrame(safe_records)
                        st.success("Data injected successfully.")
                    except Exception as e:
                        st.error(f"Generation Failed: {e}")

        if st.session_state.synthetic_data is not None:
            df_analytics = st.session_state.synthetic_data

            tab_fleet, tab_geo, tab_perf, tab_econ = st.tabs([
                "🚁 Fleet Logistics & Workload", 
                "🌍 Epidemiology & Anomalies", 
                "🏆 Institutional Performance",
                "💰 MHIS Fiscal Governance"
            ])

            with tab_fleet:
                st.subheader("Ambulance Workload & Unit Performance Auditor")
                st.caption("Identifying over-burdened crews and tracking under-utilized physical assets to optimize state dispatch.")
                
                if 'unit_id' in df_analytics.columns:
                    # Filter out civilian cabs to only analyze state ambulances
                    fleet_stats = df_analytics[df_analytics['unit_id'] != 'CAB-CIVILIAN'].groupby('unit_id').agg(
                        Total_Trips=('unit_id', 'count'),
                        Avg_Trip_Time=('actual_trip_time', 'mean'),
                        Critical_Cases=('triage_color', lambda x: (x == 'RED').sum())
                    ).reset_index()
                    
                    fleet_stats['Avg_Trip_Time'] = fleet_stats['Avg_Trip_Time'].round(1)
                    fleet_stats = fleet_stats.sort_values(by='Total_Trips', ascending=False)

                    col_f1, col_f2 = st.columns([1.5, 1])
                    
                    with col_f1:
                        st.markdown("**State Fleet Workload Distribution**")
                        workload_chart = alt.Chart(fleet_stats).mark_bar(cornerRadiusEnd=4).encode(
                            x=alt.X('unit_id:N', sort='-y', title="Ambulance Unit ID"),
                            y=alt.Y('Total_Trips:Q', title="Total Missions Executed"),
                            color=alt.condition(
                                alt.datum.Total_Trips > fleet_stats['Total_Trips'].quantile(0.8),
                                alt.value('#ff4b4b'), 
                                alt.value('#1f77b4')  
                            ),
                            tooltip=['unit_id', 'Total_Trips', 'Critical_Cases']
                        ).properties(height=300)
                        st.altair_chart(workload_chart, use_container_width=True)

                    with col_f2:
                        st.markdown("**Top Performing Units (High Volume)**")
                        st.dataframe(fleet_stats.head(6), hide_index=True, use_container_width=True)
                        
                        st.markdown("**Under-Utilized Units (Idle Risk)**")
                        st.dataframe(fleet_stats.tail(3), hide_index=True, use_container_width=True)

                    st.info("💡 **Logistics Insight:** The chart clearly highlights Fleet Misallocation. Units highlighted in red are executing 2x the state average and are at severe risk of mechanical failure and crew burnout. Conversely, units at the far right are severely under-utilized. The State Dispatch Command should immediately reposition the under-utilized units to the districts generating the heaviest mission volume.")

            with tab_geo:
                st.subheader("Statewide Epidemiology & Anomaly Detection")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Case Type Referrals (Statewide Pathology Load)**")
                    st.caption("Real-time breakdown of critical emergency categories.")
                    if 'bundle' in df_analytics.columns:
                        bundle_counts = df_analytics['bundle'].value_counts().reset_index()
                        bundle_counts.columns = ['Pathology', 'Volume']
                        st.altair_chart(alt.Chart(bundle_counts).mark_arc(innerRadius=40).encode(
                            theta=alt.Theta(field="Volume", type="quantitative"),
                            color=alt.Color(field="Pathology", type="nominal"),
                            tooltip=["Pathology", "Volume"]
                        ).properties(height=250), use_container_width=True)

                with c2:
                    st.markdown("**Temporal Surge Radar (Time of Day)**")
                    st.caption("Case volumes mapped against 24-hour cycles to guide shift staffing.")
                    hours = list(range(24))
                    surge_vols = [random.randint(10, 30) if h < 6 or h > 20 else random.randint(40, 100) for h in hours]
                    st.altair_chart(alt.Chart(pd.DataFrame({"Hour": hours, "Referrals": surge_vols})).mark_area(color="#faca2b", opacity=0.6).encode(
                        x=alt.X("Hour:O", title="Hour of Day (24H)"), y=alt.Y("Referrals:Q", title="Volume")
                    ).properties(height=250), use_container_width=True)
                    
                st.markdown("---")
                st.error("🦠 **EPIDEMIC ANOMALY DETECTOR (Sub-District Alerts)**")
                st.caption("AI-flagged statistical deviations indicating potential localized outbreaks or clinical failures.")
                
                anomaly_data = pd.DataFrame({
                    "Origin Block/District": ["Mawphlang (East Khasi)", "Tikrikilla (West Garo)", "Khliehriat (Jaintia)"],
                    "Institution Type": ["Govt CHC", "Private PHC", "Govt Civil Hosp"],
                    "Pathology Flag": ["Pediatric Respiratory", "Maternal Hemorrhage", "Acute Trauma"],
                    "Deviation": ["+420% Surge", "+185% Surge", "+210% Surge"],
                    "AI Recommendation": ["Deploy Mobile Ped Unit. Investigate Viral Pneumonia.", "Audit referring physician protocols.", "Check highway conditions for major MVA."]
                })
                st.dataframe(anomaly_data, use_container_width=True, hide_index=True)

            with tab_perf:
                st.subheader("Platform Efficacy & Institutional Quality Assurance")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("24-Hour Resuscitation Survival", "89.4%", "+4.2% Post-AHECN Launch")
                c2.metric("Statewide ALOS Compliance", "72%", "28% Bed-Blocking Rate", delta_color="inverse")
                c3.metric("Secondary Re-Referral (Bounce) Rate", "6.8%", "114 Unnecessary Transfers", delta_color="inverse")

                st.markdown("---")
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    st.markdown("**Statewide Institutional Leaderboard**")
                    st.caption("Ranked by Acceptance, 24H Survival, and Low Bounce Rates.")
                    perf_data = pd.DataFrame({
                        "Institution": ["NEIGRIHMS", "Woodland Hosp", "Bethany Hosp", "Civil Hosp Shillong", "Tura Civil Hosp"],
                        "Sector": ["Govt", "Private", "Private", "Govt", "Govt"],
                        "Acceptance %": ["94%", "88%", "85%", "78%", "71%"],
                        "24H Survival": ["92%", "89%", "87%", "81%", "76%"],
                        "Bounce Rate (Re-Referred)": ["1.2%", "3.4%", "4.1%", "12.5%", "18.2%"]
                    })
                    st.dataframe(perf_data, use_container_width=True, hide_index=True)
                
                with col_p2:
                    st.markdown("**Mortality Shift Analysis (AHECN Efficacy)**")
                    st.caption("Tracking when mortalities occur to validate Pre-Hospital platform success.")
                    
                    mortality_shift = pd.DataFrame({
                        "Phase": ["< 24h (Pre-Hospital/ED Failure)", "> 24h (ICU/Ward Complication)"],
                        "Pre-AHECN OS": [65, 35],
                        "Post-AHECN OS": [22, 78]
                    }).melt(id_vars="Phase", var_name="Era", value_name="Percentage of Total Mortalities")
                    
                    shift_chart = alt.Chart(mortality_shift).mark_bar().encode(
                        x=alt.X('Percentage of Total Mortalities:Q', stack='normalize', axis=alt.Axis(format='%')),
                        y=alt.Y('Era:N', sort=['Pre-AHECN OS', 'Post-AHECN OS']),
                        color=alt.Color('Phase:N', scale=alt.Scale(domain=['< 24h (Pre-Hospital/ED Failure)', '> 24h (ICU/Ward Complication)'], range=['#ff4b4b', '#5c5c5c']))
                    ).properties(height=200)
                    st.altair_chart(shift_chart, use_container_width=True)
                    
                st.info("💡 **Policy Governance Insight:** The Platform has successfully shifted mortality risk. Prior to this software, 65% of deaths happened in the first 24 hours (ambulances getting lost, wrong hospitals). Now, 24-hour survival is up to 89.4%. However, **Tura Civil Hospital** shows an 18.2% 'Bounce Rate', indicating they are actively accepting critical patients they lack the infrastructure to treat, requiring targeted capacity audits rather than punitive action.")

            with tab_econ:
                st.subheader("MHIS Health Economics & Fiscal Governance")
                
                df_econ = df_analytics.copy()
                
                retained_rev = df_econ[df_econ['routing_path'] == 'Govt -> Govt (Retained)']['mhis_value'].sum()
                leakage_val = df_econ[df_econ['routing_path'] == 'Govt -> Private (Leakage)']['mhis_value'].sum()
                total_burden = df_econ['uncompensated_burden'].sum()
                
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("State Total Referrals", f"{len(df_econ)}")
                k2.metric("Public Revenue Retained", f"₹ {retained_rev/10000000:.2f} Cr")
                k3.metric("MHIS Fiscal Leakage", f"₹ {leakage_val/10000000:.2f} Cr", delta="- Drain to Private Sector", delta_color="inverse")
                k4.metric("ALOS Uncompensated Burden", f"₹ {total_burden/100000:.2f} L", delta="Bed-Blocking Penalties", delta_color="inverse")

                st.markdown("---")
                
                c_e1, c_e2 = st.columns(2)
                
                with c_e1:
                    st.markdown("**1. The MHIS Twin Ledger (Pathology-Weighted)**")
                    st.caption("Comparing Government claims retained vs. Insurance capital lost to forced private diversions.")
                    
                    ledger_data = df_econ[df_econ['routing_path'].isin(['Govt -> Govt (Retained)', 'Govt -> Private (Leakage)'])]
                    ledger_chart = alt.Chart(ledger_data).mark_bar().encode(
                        x=alt.X('sum(mhis_value):Q', title="Total MHIS Value (₹)"),
                        y=alt.Y('routing_path:N', title=""),
                        color=alt.Color('routing_path:N', scale=alt.Scale(domain=['Govt -> Govt (Retained)', 'Govt -> Private (Leakage)'], range=['#00cc96', '#ff4b4b']), legend=None)
                    ).properties(height=200)
                    st.altair_chart(ledger_chart, use_container_width=True)

                with c_e2:
                    st.markdown("**2. District Capital Flight Radar**")
                    st.caption("Identifying which districts bleed the most MHIS funds due to zero-capacity diversions.")
                    
                    flight_data = df_econ[df_econ['routing_path'] == 'Govt -> Private (Leakage)']
                    flight_chart = alt.Chart(flight_data).mark_bar().encode(
                        x=alt.X('sum(mhis_value):Q', title="Leakage (₹)"),
                        y=alt.Y('origin_district:N', sort='-x', title="Origin District"),
                        color=alt.value('#ff4b4b')
                    ).properties(height=200)
                    st.altair_chart(flight_chart, use_container_width=True)
                    
                st.markdown("---")
                st.markdown("**3. Uncompensated ALOS Burden (Bed-Blocking Economics)**")
                st.caption("Tracking fixed-package financial penalties incurred by Government hospitals when patients overstay benchmark ALOS (e.g., awaiting step-down beds).")
                
                burden_bar = alt.Chart(df_econ[df_econ['uncompensated_burden'] > 0]).mark_bar().encode(
                    x=alt.X('bundle:N', sort='-y', title="Pathology Category"),
                    y=alt.Y('sum(uncompensated_burden):Q', title="Total Penalty (₹)"),
                    color=alt.value('#faca2b')
                ).properties(height=250)
                st.altair_chart(burden_bar, use_container_width=True)
                
                st.info(f"💡 **MHIS Policy Formulation:** Over ₹{leakage_val/10000000:.2f} Crores was lost to the private sector this month exclusively because Government facilities lacked capacity. Furthermore, delayed discharges caused an uncompensated ALOS burden of ₹{total_burden/100000:.2f} Lakhs. By investing in step-down wards and a 20-bed ICU annex in the worst-performing districts, the state can instantly recapture this capital within the MHIS framework.")
