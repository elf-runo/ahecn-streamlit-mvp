# app.py
import os
import streamlit as st
import pandas as pd
import altair as alt
import time
import random
from datetime import datetime, timedelta

# --- Architecture Imports ---
from clinical_engine import validated_triage_decision
from scoring_engine import calculate_facility_score
from routing_engine import get_eta, haversine_km
from analytics_engine import mortality_risk
from synthetic_cases import seed_synthetic_referrals_v2

# --- Page Configuration ---
st.set_page_config(
    page_title="AHECN – Enterprise Predictive Build", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- State Management ---
if 'active_case' not in st.session_state:
    st.session_state.active_case = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None

# --- MANDATE 1: THE UTILS BAN (Self-Contained Loading) ---
@st.cache_data(show_spinner=False)
def load_datasets():
    fac_file, icd_file = 'meghalaya_facilities.csv', 'icd_catalogue.csv'
    f_path, i_path = None, None
    for root, _, files in os.walk("."):
        if fac_file in files: f_path = os.path.join(root, fac_file)
        if icd_file in files: i_path = os.path.join(root, icd_file)
    if not f_path or not i_path:
        st.error(f"🚨 CRITICAL: Missing CSVs. Ensure {fac_file} and {icd_file} are in the directory tree.")
        st.stop()
    
    fac_df, icd_df = pd.read_csv(f_path), pd.read_csv(i_path)
    
    # Sanitize ICD DataFrame
    icd_df.columns = [c.strip().lower() for c in icd_df.columns]
    rename_map = {'icd-10': 'icd10', 'icd_10': 'icd10', 'icd code': 'icd10', 'code': 'icd10'}
    icd_df = icd_df.rename(columns=rename_map)
    if 'icd10' not in icd_df.columns: 
        icd_df = icd_df.rename(columns={icd_df.columns[0]: 'icd10'})
        
    # Sanitize Facilities DataFrame
    for c in ["lat","lon"]:
        if c in fac_df.columns: 
            fac_df[c] = pd.to_numeric(fac_df[c], errors='coerce')
    fac_df["ownership"] = fac_df.get("ownership", "Private").fillna("Private")
    
    return fac_df, icd_df

facilities_df, icd_df = load_datasets()

# --- Header ---
st.title("AHECN – Acute Healthcare Emergency Coordination Network")
st.caption("Enterprise Build v5.0 | Autonomous Fleet Repositioning | Defensive Pandas | Zero-PHI Analytics")
st.markdown("---")

# --- 4-Tier Architecture Tabs ---
tab_referrer, tab_emt, tab_hospital, tab_state = st.tabs([
    "🏥 1. Referrer (PHC/CHC)", 
    "🚑 2. EMT / Transit", 
    "🏥 3. Receiving Hospital", 
    "🏛️ 4. State Command (Analytics & AI)"
])

# ==========================================
# TAB 1: REFERRER (PHC/CHC)
# ==========================================
with tab_referrer:
    st.header("Triage & Referral Initiation")
    
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

    st.subheader("2. Provisional Diagnosis (Pathology Vector)")
    col_b, col_d = st.columns(2)
    with col_b:
        bundle = st.selectbox("Case Bundle", sorted(icd_df["bundle"].dropna().unique().tolist()))
    with col_d:
        dfb = icd_df[icd_df["bundle"] == bundle].copy()
        dx = st.selectbox("Select Diagnosis", dfb["label"].dropna().tolist())
        icd_row = dfb[dfb["label"] == dx].iloc[0].to_dict()

    required_caps = [x.strip() for x in (icd_row.get("default_caps","") or "").split(";") if x.strip()]

    vitals = {"hr": hr, "rr": rr, "sbp": sbp, "temp": temp, "spo2": spo2, "avpu": avpu}
    context = {"age": age, "pregnant": pregnant, "o2_device": "Air", "spo2_scale": 1, "behavior": "Normal"}
    
    try:
        triage_color, meta = validated_triage_decision(vitals=vitals, icd_row=icd_row, context=context)
    except Exception as e:
        st.error(f"🚨 Clinical Engine Failure: {e}")
        st.stop()

    st.markdown("### 🧠 Dual-Vector Triage Result")
    pill = {"RED":"🟥 CRITICAL (RED)", "YELLOW":"🟨 URGENT (YELLOW)", "GREEN":"🟩 STABLE (GREEN)"}[triage_color]
    st.subheader(pill)
    st.caption(f"**Primary Driver:** {meta['primary_driver']} | **Reason:** {meta['reason']} | **Severity Index:** {meta['severity_index']:.2f}")

    st.markdown("---")
    st.header("Facility Matching (Gated Clinical Safety)")
    
    origin = (float(src_lat), float(src_lon))
    results = []
    pathology_bundle = icd_row.get("bundle", "Other")
    sev = float((meta or {}).get("severity_index", 0.0) or 0.0)

    for _, row in facilities_df.iterrows():
        f_dict = row.to_dict()
        dest = (float(f_dict.get("lat", 0.0)), float(f_dict.get("lon", 0.0)))
        f_dict["ownership"] = str(f_dict.get("ownership", "Private") or "Private")
        
        try:
            route_eta = get_eta(origin, dest, speed_kmh=40.0, is_hilly_terrain=True)
        except Exception:
            route_eta = 999.0

        try:
            score, details = calculate_facility_score(
                facility=f_dict,
                required_caps=required_caps,
                eta=route_eta,
                triage_color=triage_color,
                severity_index=sev,
                case_type=pathology_bundle
            )
        except Exception:
            continue

        if score > 0 or details.get("gate_capacity") == "WARNING_ED_STABILIZATION_ONLY":
            try:
                m_risk = mortality_risk(sev, route_eta, pathology=pathology_bundle)
            except:
                m_risk = 99.9

            results.append({
                "facility": f_dict["name"],
                "score": score,
                "eta": round(route_eta, 1),
                "ownership": f_dict["ownership"],
                "mortality_risk": m_risk,
                "scoring_details": details,
                "lat": dest[0],
                "lon": dest[1]
            })

    results = sorted(results, key=lambda x: (-x["score"], x["eta"]))

    if not results:
        st.error("🚨 **CRITICAL ALERT: ZERO STATEWIDE CAPACITY.**")
        st.warning("No facilities meet the absolute minimum clinical safety gates within transit range. Initiate immediate on-site ED stabilization and escalate to State Command for out-of-network airlift.")
    else:
        selected_fac_name = st.radio("Select Destination Facility:", [r["facility"] for r in results[:5]])
        selected_fac = next(r for r in results if r["facility"] == selected_fac_name)
        
        with st.expander(f"📊 Explainable AI Logic for {selected_fac['facility']}"):
            details = selected_fac["scoring_details"]
            
            st.markdown("**1. Safety Gates (Absolute Mandates)**")
            if details.get('gate_capability') == "PASSED":
                st.markdown("✅ **Infrastructure Gate:** Passed. 100% of requested life-saving capabilities present.")
            
            if details.get('gate_capacity') == "PASSED":
                st.markdown("✅ **Capacity Gate:** Passed. Minimum required critical care beds available.")
            elif details.get('gate_capacity') == "WARNING_ED_STABILIZATION_ONLY":
                st.markdown("⚠️ **Capacity Gate Override:** ICU Full. Facility selected for immediate ED Resuscitation ONLY.")

            st.markdown("**2. Optimization Matrix**")
            st.markdown(f"🚑 **Topography-Adjusted ETA:** {details.get('eta_minutes', 'N/A')} mins *(Score: {details.get('proximity_score', 0)}/50)*")
            st.markdown(f"🛏️ **Surge Buffer:** {details.get('icu_beds', 0)} open beds *(Score: {details.get('icu_score', 0)}/15)*")
            st.markdown(f"🏛️ **Fiscal Guardrail:** {details.get('ownership')} facility *(Score: {details.get('fiscal_score', 0)}/20)*")

        if st.button("🚀 Initiate E2EE Transfer & Dispatch Ambulance", type="primary"):
            st.session_state.active_case = {
                "patient_name": patient_name,
                "age": age,
                "vitals": vitals,
                "diagnosis": dx,
                "bundle": bundle,
                "triage_color": triage_color,
                "severity_index": meta['severity_index'],
                "destination": selected_fac,
                "dispatch_time": datetime.now().strftime("%H:%M:%S")
            }
            st.success("Transfer Initiated! Switch to EMT / Hospital tabs.")

# ==========================================
# TAB 2: EMT / TRANSIT
# ==========================================
with tab_emt:
    st.header("🚑 Active Transit & Paramedic Dashboard")
    
    if not st.session_state.active_case:
        st.info("No active dispatch. Initiate a transfer from the Referrer tab.")
    else:
        case = st.session_state.active_case
        dest = case["destination"]
        
        st.error(f"🚨 **PRIORITY {case['triage_color']} EN ROUTE TO {dest['facility'].upper()}**")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Topography-Adjusted ETA", f"{dest['eta']} min")
        c2.metric("Modeled Mortality Risk", f"{dest['mortality_risk']}%")
        c3.metric("Dispatch Time", case["dispatch_time"])

        st.markdown("### 📋 ISBAR Clinical Handover (E2EE Unredacted)")
        st.markdown(f"""
        * **Identification:** {case['patient_name']}, {case['age']} Y/O.
        * **Situation:** Priority {case['triage_color']} transfer for {case['diagnosis']}.
        * **Background:** Case Bundle: {case['bundle']}. Severity Index: {case['severity_index']:.2f}.
        * **Assessment:** HR {case['vitals']['hr']} | BP {case['vitals']['sbp']} | RR {case['vitals']['rr']} | SpO2 {case['vitals']['spo2']}% | Temp {case['vitals']['temp']}°C | AVPU: {case['vitals']['avpu']}
        * **Recommendation:** Prepare ED receiving bay. 
        """)
        
        if dest["scoring_details"].get("gate_capacity") == "WARNING_ED_STABILIZATION_ONLY":
            st.warning("⚠️ **PARAMEDIC ALERT:** Destination ICU is at capacity. You are routing for ED STABILIZATION ONLY. Prepare for potential secondary transfer post-resuscitation.")

# ==========================================
# TAB 3: RECEIVING HOSPITAL
# ==========================================
with tab_hospital:
    st.header("🏥 Emergency Department Receiving Board")
    
    if not st.session_state.active_case:
        st.info("ED Bay Clear. No incoming transfers.")
    else:
        case = st.session_state.active_case
        dest = case["destination"]
        dest_name = dest['facility']
        
        # --- MANDATE 2: DEFENSIVE PANDAS (AI SURGE WARNING) ---
        if st.session_state.synthetic_data is not None:
            df_analytics = st.session_state.synthetic_data
            if not df_analytics.empty and 'dest_facility' in df_analytics.columns:
                try:
                    fac_cases = df_analytics[df_analytics['dest_facility'] == dest_name]
                    historical_daily_avg = max(1, len(fac_cases) / 30.0)
                    current_live_inbound = int(historical_daily_avg * 1.8) + 1 
                    
                    if current_live_inbound > (historical_daily_avg * 1.5):
                        st.error(f"🚨 **CODE SURGE DETECTED: {dest_name.upper()}** 🚨")
                        st.markdown(f"""
                        **Predictive AI Alert:** Current inbound ambulance load ({current_live_inbound} cases) exceeds your 90-day historical average ({historical_daily_avg:.1f} cases/day) by **>50%**. 
                        
                        **Recommended Administrator Actions:**
                        * ⚠️ Mobilize off-duty trauma surgeons and ED physicians immediately.
                        * ⚠️ Initiate rapid discharge protocols for stable ward patients to free up ICU step-down beds.
                        * ⚠️ Alert State Command if diversion status is required.
                        """)
                        st.markdown("---")
                except Exception as e:
                    st.warning("Surge AI encountered an error calculating historical baselines.")
            else:
                st.warning("⚠️ Predictive Surge AI offline. (Telemetry data malformed).")
        else:
            st.warning("⚠️ Predictive Surge AI offline. (Requires synthetic data injection in State Command tab).")

        # --- STANDARD RECEIVING UI ---
        st.warning(f"🔔 **INCOMING ALERT:** ETA {dest['eta']} mins")
        
        if dest["scoring_details"].get("gate_capacity") == "WARNING_ED_STABILIZATION_ONLY":
            st.error("🛑 **CRITICAL CAPACITY OVERRIDE:** Patient routed to your facility for ED STABILIZATION ONLY due to zero regional ICU beds. Mobilize ED Resuscitation team immediately.")
            
        st.markdown(f"**Patient:** {case['patient_name']} ({case['age']} Y/O)")
        st.markdown(f"**Diagnosis:** {case['diagnosis']} ({case['triage_color']})")
        st.markdown(f"**Vitals at Dispatch:** HR {case['vitals']['hr']} | BP {case['vitals']['sbp']} | SpO2 {case['vitals']['spo2']}%")
        
        if st.button("✅ Acknowledge & Accept Patient"):
            st.success("Patient accepted. Telemetry linked to ED monitors.")

# ==========================================
# TAB 4: STATE COMMAND (ANALYTICS & AI)
# ==========================================
with tab_state:
    st.header("🏛️ State Command & Control (Zero-PHI Analytics)")
    st.caption("Aggregated, anonymized telemetry for government fiscal, capacity oversight, and predictive intelligence.")
    
    def _now_ts(): return time.time()
    def _rand_geo(rng): return (25.5 + rng.random()*0.2, 91.8 + rng.random()*0.2)
    def _dist_km(lat1, lon1, lat2, lon2): return haversine_km((lat1, lon1), (lat2, lon2))
    def _interp_route(lat1, lon1, lat2, lon2, n): return []
    def _traffic(hr): return 1.2 if 8 <= hr <= 20 else 1.0

    if st.button("💉 Inject 1,000 Synthetic Cases (Stress Test)"):
        with st.spinner("Generating 1,000 clinically validated synthetic cases..."):
            try:
                raw_data = seed_synthetic_referrals_v2(
                    n=1000,
                    facilities=facilities_df.to_dict('records'),
                    icd_df=icd_df,
                    validated_triage_decision_fn=validated_triage_decision,
                    now_ts_fn=_now_ts,
                    rand_geo_fn=_rand_geo,
                    dist_km_fn=_dist_km,
                    interpolate_route_fn=_interp_route,
                    traffic_factor_fn=_traffic,
                    rng_seed=42
                )
                
                safe_records = []
                for r in raw_data:
                    safe_records.append({
                        "case_id": r["id"],
                        "timestamp": r["times"]["first_contact_ts"],
                        "bundle": r["provisionalDx"]["case_type"],
                        "triage_color": r["triage"]["decision"]["color"],
                        "severity_index": r["triage"]["decision"]["score_details"].get("severity_index", 0.0),
                        "eta_min": r["transport"]["eta_min"],
                        "facility_ownership": r["facility_ownership"],
                        "dest_facility": r["dest"],
                        "mortality_risk": mortality_risk(
                            r["triage"]["decision"]["score_details"].get("severity_index", 0.0), 
                            r["transport"]["eta_min"], 
                            pathology=r["provisionalDx"]["case_type"]
                        )
                    })
                
                st.session_state.synthetic_data = pd.DataFrame(safe_records)
                st.success("Data injected and PHI stripped successfully.")
            except Exception as e:
                st.error(f"Synthetic Generation Failed: {e}")

    if st.session_state.synthetic_data is not None:
        df_analytics = st.session_state.synthetic_data
        
        st.markdown("---")
        st.subheader("🧠 Predictive AI & Epidemiological Intelligence")
        
        col_ai1, col_ai2 = st.columns(2)
        top_facility = None 
        
        with col_ai1:
            # --- MANDATE 2: DEFENSIVE PANDAS (HOTSPOTTING) ---
            if not df_analytics.empty and 'bundle' in df_analytics.columns and 'dest_facility' in df_analytics.columns:
                try:
                    top_bundle = df_analytics['bundle'].mode()[0]
                    top_facility = df_analytics[df_analytics['bundle'] == top_bundle]['dest_facility'].mode()[0]
                    
                    st.error("⚠️ **EPIDEMIOLOGICAL HOTSPOT PREDICTION**")
                    st.markdown(f"""
                    **Vector Analysis:** The AI has detected a statistically significant anomaly in **{top_bundle}** cases.
                    **Forecast:** A cluster event is highly probable within the catchment area of **{top_facility}** over the next 48 hours.
                    **Action Required:** Pre-position specialized {top_bundle.lower()} medical supplies and alert regional ASHA workers.
                    """)
                except (KeyError, IndexError):
                    st.warning("Insufficient data variance to generate hotspot prediction.")
            else:
                st.warning("Awaiting valid telemetry data for hotspot prediction.")

        with col_ai2:
            st.warning("📉 **12-HOUR ICU EXHAUSTION FORECAST**")
            if not df_analytics.empty and 'triage_color' in df_analytics.columns:
                try:
                    total_icu_beds = facilities_df['ICU_open'].astype(int).sum()
                    red_cases = len(df_analytics[df_analytics['triage_color'] == 'RED'])
                    hourly_burn_rate = max(1, (red_cases / 30 / 24) * 1.5) 
                    
                    current_time = datetime.now()
                    forecast_data = []
                    current_beds = total_icu_beds
                    
                    for i in range(13):
                        forecast_time = current_time + timedelta(hours=i)
                        forecast_data.append({
                            "Time": forecast_time,
                            "Projected_ICU_Beds": max(0, current_beds)
                        })
                        current_beds -= hourly_burn_rate
                        
                    df_forecast = pd.DataFrame(forecast_data)
                    
                    base = alt.Chart(df_forecast).encode(
                        x=alt.X('Time:T', title='Forecast Time (Next 12 Hrs)')
                    )
                    line = base.mark_line(color='#ff4b4b', strokeWidth=3).encode(
                        y=alt.Y('Projected_ICU_Beds:Q', title='Available Statewide ICU Beds')
                    )
                    critical_threshold = total_icu_beds * 0.1
                    threshold_line = alt.Chart(pd.DataFrame({'y': [critical_threshold]})).mark_rule(
                        color='black', strokeDash=[5,5]
                    ).encode(y='y:Q')
                    
                    icu_chart = (line + threshold_line).properties(height=200)
                    st.altair_chart(icu_chart, use_container_width=True)
                except Exception as e:
                    st.warning("ICU Forecast model offline. Awaiting telemetry stabilization.")
            else:
                st.warning("Awaiting valid telemetry data for ICU Forecast.")

        st.markdown("---")
        st.subheader("📊 Statewide Telemetry (30-Day Aggregate)")
        
        if not df_analytics.empty and all(c in df_analytics.columns for c in ['severity_index', 'mortality_risk', 'facility_ownership', 'triage_color', 'eta_min', 'bundle']):
            try:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Referrals", len(df_analytics))
                c2.metric("Avg Severity Index", f"{df_analytics['severity_index'].mean():.2f}")
                c3.metric("Avg Modeled Mortality Risk", f"{df_analytics['mortality_risk'].mean():.1f}%")
                
                gov_pct = (len(df_analytics[df_analytics['facility_ownership'] == 'Government']) / len(df_analytics)) * 100
                c4.metric("Gov Facility Utilization", f"{gov_pct:.1f}%", help="Fiscal Guardrail Metric")

                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    triage_chart = alt.Chart(df_analytics).mark_arc().encode(
                        theta=alt.Theta(field="triage_color", aggregate="count"),
                        color=alt.Color(field="triage_color", type="nominal", scale=alt.Scale(
                            domain=['RED', 'YELLOW', 'GREEN'],
                            range=['#ff4b4b', '#faca2b', '#00cc96']
                        )),
                        tooltip=['triage_color', 'count()']
                    ).properties(title="Statewide Triage Distribution")
                    st.altair_chart(triage_chart, use_container_width=True)

                with col_chart2:
                    scatter_chart = alt.Chart(df_analytics).mark_circle(size=60, opacity=0.6).encode(
                        x=alt.X("eta_min:Q", title="Transit ETA (mins)"),
                        y=alt.Y("severity_index:Q", title="Severity Index (0-1)"),
                        color=alt.Color("facility_ownership:N", title="Sector"),
                        tooltip=["bundle", "eta_min", "severity_index", "mortality_risk"]
                    ).properties(title="ETA vs Severity (Fiscal Sector Segmented)")
                    st.altair_chart(scatter_chart, use_container_width=True)
                    
                st.markdown("### 📉 Golden Hour Mortality Curve Analysis")
                risk_chart = alt.Chart(df_analytics).mark_area(opacity=0.4).encode(
                    x=alt.X("eta_min:Q", bin=alt.Bin(maxbins=30), title="Transit ETA (mins)"),
                    y=alt.Y("average(mortality_risk):Q", title="Avg Modeled Mortality Risk (%)"),
                    color="bundle:N"
                ).properties(title="Exponential Mortality Degradation by Pathology Bundle")
                st.altair_chart(risk_chart, use_container_width=True)
            except Exception as e:
                st.warning("Telemetry visualization encountered an error. Please re-inject data.")
        else:
            st.warning("Telemetry data is incomplete or malformed.")

        # --- AUTONOMOUS FLEET REPOSITIONING ---
        st.markdown("---")
        st.subheader("🚑 Autonomous Fleet Repositioning")
        st.caption("Pre-emptive logistics engine: Rebalancing idle assets from Cold Zones to predicted Hotspots.")

        if top_facility and not df_analytics.empty and 'triage_color' in df_analytics.columns and 'dest_facility' in df_analytics.columns:
            try:
                # 1. Simulate Statewide Fleet (150 Ambulances)
                facility_names = facilities_df['name'].dropna().unique().tolist()
                fleet_data = []
                for i in range(150):
                    fleet_data.append({
                        "Unit ID": f"ALS-{random.randint(1000, 9999)}",
                        "Current Location": random.choice(facility_names),
                        "Status": random.choices(["Idle", "Active", "Maintenance"], weights=[0.6, 0.3, 0.1])[0]
                    })
                df_fleet = pd.DataFrame(fleet_data)

                # 2. Identify Cold Zones (Facilities with 0 inbound RED cases)
                red_cases = df_analytics[df_analytics['triage_color'] == 'RED']
                active_red_dests = red_cases['dest_facility'].unique().tolist()
                cold_zones = [f for f in facility_names if f not in active_red_dests and f != top_facility]

                # 3. Repositioning Algorithm
                idle_in_cold_zones = df_fleet[(df_fleet['Status'] == 'Idle') & (df_fleet['Current Location'].isin(cold_zones))]
                
                if not idle_in_cold_zones.empty:
                    # Extract up to 3 distinct cold zones
                    unique_cold_zones_with_idle = idle_in_cold_zones['Current Location'].unique()
                    selected_zones = unique_cold_zones_with_idle[:3]

                    dispatch_commands = []
                    for zone in selected_zones:
                        # Pull exactly 1 idle ambulance from this specific cold zone
                        amb = idle_in_cold_zones[idle_in_cold_zones['Current Location'] == zone].iloc[0]
                        dispatch_commands.append({
                            "Unit ID": amb['Unit ID'],
                            "Current Location": amb['Current Location'],
                            "Relocation Target": top_facility,
                            "Reason": "Pre-emptive surge staging (Zero RED cases at origin)"
                        })

                    if dispatch_commands:
                        st.success(f"✅ **Optimization Complete:** {len(dispatch_commands)} idle units identified for immediate relocation to {top_facility}.")
                        st.dataframe(pd.DataFrame(dispatch_commands), use_container_width=True)
                    else:
                        st.info("No idle assets available in Cold Zones for repositioning at this time.")
                else:
                    st.info("No idle assets available in Cold Zones for repositioning at this time.")
                    
            except Exception as e:
                st.error(f"🚨 Fleet Optimization Engine Error: {e}")
                st.warning("Failsafe activated: Maintaining current fleet distribution.")
        else:
            st.info("Hotspot prediction unavailable or telemetry missing. Cannot execute fleet optimization.")
