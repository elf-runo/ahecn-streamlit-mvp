# AHECN – Streamlit MVP v1.8 (Investor Demo + UI/UX)
# Roles: Referrer • Ambulance/EMT • Receiving Hospital • Government • Data/Admin • Facility Admin
# Region: East Khasi Hills, Meghalaya (synthetic geo + facilities)

import math, json, time, random
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import os

# === LOAD ICD CATALOG FROM CSV ===
def load_icd_catalogue():
    """Load ICD catalog from CSV file with error handling."""
    try:
        # Try to read from CSV file
        df = pd.read_csv('icd_catalogue.csv')
        
        # Convert to the format expected by the app
        icd_lut = []
        for _, row in df.iterrows():
            icd_lut.append({
                "icd_code": row['icd10'],
                "label": row['label'],
                "case_type": row['bundle'],
                "age_min": row['age_min'],
                "age_max": row['age_max'],
                "default_interventions": "",  # You can add this column to CSV if needed
                "default_caps": row['default_caps'].split(';') if pd.notna(row['default_caps']) else []
            })
        return icd_lut
    except Exception as e:
        st.error(f"Error loading ICD catalog: {str(e)}")
        # Fallback to minimal catalog if CSV fails
        return [
            {"icd_code":"S06.0", "label":"Concussion", "case_type":"Trauma", "age_min":0, "age_max":120,
             "default_interventions":"Airway positioning;Cervical immobilization;IV fluids", "default_caps":["CT"]},
            {"icd_code":"O72.1", "label":"Postpartum haemorrhage", "case_type":"Maternal", "age_min":12, "age_max":55,
             "default_interventions":"Uterotonics;TXA;IV fluids;Bleeding control", "default_caps":["ICU","BloodBank","OBGYN_OT"]},
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

# ---------------------- THEME & GLOBAL CSS ----------------------
st.set_page_config(page_title="AHECN MVP v1.8", layout="wide", initial_sidebar_state="collapsed")
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
.recommended { background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 12px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# [REST OF YOUR EXISTING CODE REMAINS THE SAME - only showing modified sections]
# ... (keep all your existing helper functions, scoring engines, etc.)

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

    # Vitals Section
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
        rf_sbp = st.checkbox("Red flag: SBP <90", False)
    with v3:
        rf_spo2 = st.checkbox("Red flag: SpO₂ <90%", False)
        rf_avpu = st.checkbox("Red flag: AVPU ≠ A", False)
        rf_seizure = st.checkbox("Red flag: Seizure", False)
        rf_pph = st.checkbox("Red flag: PPH", value=(complaint == "Maternal"))

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
    st.write(f"qSOFA: **{q_score}** {'• ≥2 high risk' if q_high else ''}")

    meows = calc_MEOWS(hr, rr, sbp, temp, spo2)
    m_band = "Red" if meows["red"] else ("Yellow" if meows["yellow"] else "Green")
    m_trig = bool(meows["red"] or meows["yellow"])
    if complaint == "Maternal":
        st.write(f"MEOWS: **{m_band}** {'• trigger' if m_trig else ''}")
    else:
        st.caption("MEOWS applies to maternal cases only.")

    if p_age < 18:
        pews_sc, pews_meta, pews_high, pews_watch = calc_PEWS(p_age, rr, hr, pews_beh, spo2)
        st.write(f"PEWS: **{pews_sc}** {'• ≥6 high risk' if pews_high else ('• watch' if pews_watch else '')}")
    else:
        st.caption("PEWS disabled for ≥18y")

    # Triage decision banner
    render_triage_banner(hr, rr, sbp, temp, spo2, avpu, rf_sbp, rf_spo2, rf_avpu, rf_seizure, rf_pph, complaint)

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
            
            # NEW: Display recommended capabilities from CSV
            default_caps = row.get("default_caps", [])
            if default_caps:
                st.markdown(f'<div class="recommended"><strong>💡 Recommended capabilities:</strong> {", ".join(default_caps)}</div>', 
                           unsafe_allow_html=True)
            
            # Display ICD details
            st.info(f"**Selected:** {row['label']} ({row['icd_code']}) • Age range: {row['age_min']}-{row['age_max']} years")
        else:
            st.warning("No ICD codes match your search/filters. Try different criteria or check 'Show all diagnoses'.")
            chosen_icd = None
            row = None
            default_iv = []
            default_caps = []

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
                    default_caps = row.get("default_caps", [])
                    
                    # Display recommended capabilities
                    if default_caps:
                        st.markdown(f'<div class="recommended"><strong>💡 Recommended capabilities:</strong> {", ".join(default_caps)}</div>', 
                                   unsafe_allow_html=True)
                    
                    # Optional interventions
                    st.markdown("**Suggested interventions**")
                    for i, item in enumerate(default_iv):
                        if st.checkbox(item, value=False, key=f"iv_{i}"):
                            iv_selected.append(item) if 'iv_selected' in locals() else iv_selected.extend([item])
                else:
                    row = None
                    default_iv = []
                    default_caps = []
                    iv_selected = []
            else:
                st.info("No ICD suggestions for this age & case type")
                chosen_icd = None
                row = None
                default_iv = []
                default_caps = []
                iv_selected = []
        
        # Additional notes
        dx_free = st.text_input("Additional notes (optional)", "")
        
        # Diagnosis payload for non-doctors
        dx_payload = dict(code=row["icd_code"] if chosen_icd else "", 
                         label=referral_reason or (row["label"] if chosen_icd else ""), 
                         case_type=str(complaint))
        
        if not referral_reason and not chosen_icd:
            st.error("Please provide a reason for referral or select an ICD diagnosis")

    # NEW: Auto-apply recommended capabilities when ICD is selected
    if 'default_caps' in locals() and default_caps and referrer_role == "Doctor/Physician":
        if st.button("🔄 Apply Recommended Capabilities", key="apply_caps"):
            # Store the default capabilities to pre-fill the checkboxes
            st.session_state.recommended_caps = default_caps
            st.success(f"Applied {len(default_caps)} recommended capabilities")
            st.rerun()

    # Referral reasons and capabilities
    st.subheader("Reason(s) for referral + capabilities needed")
    c1, c2 = st.columns(2)
    with c1:
        ref_beds = st.checkbox("No ICU/bed available", False)
        ref_tests = st.checkbox("Special intervention/test required", True)
        ref_severity = True
    
    # Initialize need_caps with recommended ones if available
    if 'recommended_caps' in st.session_state and st.session_state.recommended_caps:
        pre_selected_caps = st.session_state.recommended_caps
    else:
        pre_selected_caps = []
    
    need_caps = []
    if ref_tests:
        st.caption("Select required capabilities for this case")
        cap_cols = st.columns(5)
        CAP_LIST = ["ICU", "Ventilator", "BloodBank", "OR", "CT", "Thrombolysis", "OBGYN_OT", "CathLab", "Dialysis", "Neurosurgery"]
        
        # Apply pre-selected capabilities based on ICD
        for i, cap in enumerate(CAP_LIST):
            # Set default value based on ICD recommendation or maternal condition
            default_val = cap in pre_selected_caps or (complaint == "Maternal" and cap in ["ICU", "BloodBank", "OBGYN_OT"])
            
            if cap_cols[i % 5].checkbox(cap, value=default_val, key=f"cap_{cap}"):
                need_caps.append(cap)

    # [REST OF YOUR REFERRER TAB CODE REMAINS THE SAME]
    # ... (facility matching, transport details, etc.)

# [REST OF YOUR TABS REMAIN UNCHANGED]
