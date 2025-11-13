# AHECN – Streamlit MVP v1.9 (Enhanced Analytics & Demo Visualizations)
import math
import json
import time
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import altair as alt
import os
import requests
import urllib.parse
import asyncio
import threading
from collections import deque

# === PAGE CONFIG MUST BE FIRST STREAMLIT COMMAND ===
st.set_page_config(
    page_title="AHECN MVP v1.9",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === FREE ROUTING CONFIGURATION ===
ROUTING_PROVIDER = 'osrm'
OSRM_BASE_URL = "http://router.project-osrm.org"
ORS_BASE_URL = "https://api.openrouteservice.org"

# Cache configuration
DISTANCE_CACHE = {}
CACHE_DURATION = timedelta(hours=24)

# === REAL-TIME EVENT SYSTEM ===
class RealTimeEventSystem:
    def __init__(self):
        self.events = []
        self.subscribers = []
    
    def publish_event(self, event_type, data, user=None, facility=None):
        """Publish system-wide event"""
        event = {
            "id": f"evt_{int(time.time())}_{len(self.events)}",
            "timestamp": time.time(),
            "type": event_type,
            "data": data,
            "user": user or "System",
            "facility": facility,
            "read_by": set()
        }
        
        self.events.append(event)
        st.session_state.system_activities.insert(0, event)
        
        # Keep only last 100 events
        if len(st.session_state.system_activities) > 100:
            st.session_state.system_activities = st.session_state.system_activities[:100]
        
        # Create notification for relevant users
        self.create_notification(event)
        
        return event
    
    def create_notification(self, event):
        """Create targeted notifications based on event type"""
        notification = {
            "id": f"notif_{event['id']}",
            "event_id": event["id"],
            "timestamp": event["timestamp"],
            "type": event["type"],
            "title": self.get_notification_title(event),
            "message": self.get_notification_message(event),
            "priority": self.get_notification_priority(event),
            "target_users": self.get_target_users(event),
            "facility": event.get("facility"),
            "read": False,
            "action_required": self.is_action_required(event)
        }
        
        st.session_state.notifications.append(notification)
        
        # Keep only last 50 notifications
        if len(st.session_state.notifications) > 50:
            st.session_state.notifications = st.session_state.notifications[:50]
    
    def get_notification_title(self, event):
        """Get appropriate title for event type"""
        titles = {
            "CASE_CREATED": "🆕 New Referral Created",
            "CASE_DISPATCHED": "🚑 Ambulance Dispatched", 
            "CASE_ACCEPTED": "✅ Case Accepted",
            "CASE_REJECTED": "❌ Case Rejected",
            "TRIAGE_OVERRIDE": "⚡ Triage Override",
            "AMBULANCE_ARRIVED": "🏥 Ambulance Arrived",
            "HANDOVER_COMPLETE": "🤝 Handover Complete",
            "CRITICAL_ALERT": "🚨 Critical Alert",
            "RESOURCE_LOW": "⚠️ Resource Alert",
            "SYSTEM_ALERT": "🔧 System Alert"
        }
        return titles.get(event["type"], "System Notification")
    
    def get_notification_message(self, event):
        """Generate appropriate message for event"""
        data = event["data"]
        
        if event["type"] == "CASE_CREATED":
            return f"New {data.get('complaint', 'case')} referral for {data.get('patient_name', 'patient')} - {data.get('triage_color', 'Unknown')} triage"
        
        elif event["type"] == "CASE_DISPATCHED":
            return f"Ambulance {data.get('ambulance_type', '')} dispatched to {data.get('facility', 'facility')}"
            
        elif event["type"] == "TRIAGE_OVERRIDE":
            return f"Dr. {event['user']} overrode triage from {data.get('from_color')} to {data.get('to_color')}"
            
        elif event["type"] == "CRITICAL_ALERT":
            return data.get('message', 'Critical situation requiring attention')
            
        elif event["type"] == "RESOURCE_LOW":
            return f"{data.get('resource')} running low: {data.get('level')}"
        
        return f"Event: {event['type']}"
    
    def get_notification_priority(self, event):
        """Determine notification priority"""
        high_priority = ["CRITICAL_ALERT", "TRIAGE_OVERRIDE", "RESOURCE_LOW"]
        medium_priority = ["CASE_REJECTED", "CASE_DISPATCHED", "AMBULANCE_ARRIVED"]
        
        if event["type"] in high_priority:
            return "HIGH"
        elif event["type"] in medium_priority:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_target_users(self, event):
        """Determine which users should receive this notification"""
        if event["type"] in ["CASE_ACCEPTED", "CASE_REJECTED", "AMBULANCE_ARRIVED"]:
            return ["receiving_hospital", "ambulance_crew"]
        elif event["type"] in ["CASE_CREATED", "TRIAGE_OVERRIDE"]:
            return ["referrer", "coordinator"]
        elif event["type"] in ["CRITICAL_ALERT", "RESOURCE_LOW"]:
            return ["all"]
        else:
            return ["coordinator"]
    
    def is_action_required(self, event):
        """Check if this notification requires action"""
        action_events = ["CASE_REJECTED", "CRITICAL_ALERT", "RESOURCE_LOW"]
        return event["type"] in action_events

# Initialize event system
event_system = RealTimeEventSystem()

# === REAL-TIME COMPONENT FUNCTIONS ===
def mark_notification_read(notification_id):
    """Mark notification as read"""
    for notification in st.session_state.notifications:
        if notification['id'] == notification_id:
            notification['read'] = True
            break

def mark_all_notifications_read():
    """Mark all notifications as read"""
    for notification in st.session_state.notifications:
        notification['read'] = True

def acknowledge_event(event_id):
    """Acknowledge an event"""
    for activity in st.session_state.system_activities:
        if activity['id'] == event_id:
            activity['acknowledged'] = True
            break

def show_event_details(event):
    """Show detailed event view"""
    st.sidebar.markdown("### 🔍 Event Details")
    st.sidebar.write(f"**Type:** {event['type']}")
    st.sidebar.write(f"**Time:** {datetime.fromtimestamp(event['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.write(f"**User:** {event.get('user', 'System')}")
    st.sidebar.write(f"**Facility:** {event.get('facility', 'N/A')}")
    st.sidebar.write("**Data:**")
    st.sidebar.json(event['data'])

def show_case_details(case):
    """Show detailed case view"""
    st.sidebar.markdown("### 📋 Case Details")
    st.sidebar.write(f"**Patient:** {case.get('patient_name')}")
    st.sidebar.write(f"**Complaint:** {case.get('complaint')}")
    st.sidebar.write(f"**Triage:** {case.get('triage_color')}")
    st.sidebar.write(f"**Status:** {case.get('status')}")
    st.sidebar.write(f"**Facility:** {case.get('facility')}")

def update_case_eta(case_id):
    """Update case ETA"""
    if case_id in st.session_state.live_cases:
        st.session_state.live_cases[case_id]['eta'] = random.randint(5, 25)

def generate_demo_activities():
    """Generate demo activities for testing"""
    demo_events = [
        {
            "type": "CASE_CREATED",
            "data": {"patient_name": "Rani Devi", "complaint": "Maternal", "triage_color": "RED", "message": "New maternal hemorrhage case"},
            "user": "Dr. Sharma",
            "facility": "PHC Mawlai"
        },
        {
            "type": "CASE_DISPATCHED", 
            "data": {"ambulance_type": "ALS", "destination": "Civil Hospital", "message": "ALS ambulance dispatched"},
            "user": "Dispatcher",
            "facility": "Civil Hospital"
        },
        {
            "type": "TRIAGE_OVERRIDE",
            "data": {"from_color": "YELLOW", "to_color": "RED", "reason": "Clinical deterioration", "message": "Triage escalated to RED"},
            "user": "Dr. Kumar", 
            "facility": "NEIGRIHMS"
        },
        {
            "type": "RESOURCE_LOW",
            "data": {"resource": "O- Blood", "level": "3 units", "message": "O- blood supply critical"},
            "user": "System",
            "facility": "Blood Bank"
        }
    ]
    
    for event in demo_events:
        event_system.publish_event(**event)

def initialize_demo_cases():
    """Initialize demo live cases"""
    st.session_state.live_cases = {
        "CASE_001": {
            "patient_name": "Rani Devi",
            "complaint": "Maternal Hemorrhage", 
            "triage_color": "RED",
            "age": 28,
            "gender": "F",
            "status": "ENROUTE_SCENE",
            "facility": "Civil Hospital",
            "eta": 12,
            "latest_vitals": {"hr": 118, "sbp": 92, "rr": 26, "spo2": 88, "temp": 37.8}
        },
        "CASE_002": {
            "patient_name": "Amit Kumar",
            "complaint": "STEMI",
            "triage_color": "RED", 
            "age": 45,
            "gender": "M",
            "status": "ARRIVE_SCENE",
            "facility": "NEIGRIHMS",
            "eta": 8,
            "latest_vitals": {"hr": 105, "sbp": 110, "rr": 22, "spo2": 94, "temp": 36.8}
        },
        "CASE_003": {
            "patient_name": "Priya Singh",
            "complaint": "Stroke",
            "triage_color": "YELLOW",
            "age": 62, 
            "gender": "F",
            "status": "DISPATCHED",
            "facility": "District Hospital",
            "eta": 15,
            "latest_vitals": {"hr": 88, "sbp": 150, "rr": 18, "spo2": 96, "temp": 36.5}
        }
    }

def render_activity_item(activity):
    """Render individual activity item"""
    timestamp = datetime.fromtimestamp(activity["timestamp"]).strftime("%H:%M:%S")
    
    icons = {
        "CASE_CREATED": "🆕",
        "CASE_DISPATCHED": "🚑", 
        "CASE_ACCEPTED": "✅",
        "CASE_REJECTED": "❌",
        "TRIAGE_OVERRIDE": "⚡",
        "AMBULANCE_ARRIVED": "🏥",
        "HANDOVER_COMPLETE": "🤝",
        "CRITICAL_ALERT": "🚨",
        "RESOURCE_LOW": "⚠️",
        "SYSTEM_ALERT": "🔧"
    }
    
    colors = {
        "CASE_CREATED": "#3B82F6",
        "CASE_DISPATCHED": "#F59E0B", 
        "CASE_ACCEPTED": "#10B981",
        "CASE_REJECTED": "#EF4444",
        "TRIAGE_OVERRIDE": "#8B5CF6",
        "AMBULANCE_ARRIVED": "#06B6D4",
        "HANDOVER_COMPLETE": "#84CC16",
        "CRITICAL_ALERT": "#DC2626",
        "RESOURCE_LOW": "#D97706",
        "SYSTEM_ALERT": "#6B7280"
    }
    
    icon = icons.get(activity["type"], "🔔")
    color = colors.get(activity["type"], "#6B7280")
    
    with st.container():
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.markdown(f"<h3 style='color: {color}; margin: 0;'>{icon}</h3>", unsafe_allow_html=True)
            st.caption(timestamp)
        
        with col2:
            st.markdown(f"**{activity.get('user', 'System')}**")
            st.write(f"{activity['data'].get('message', activity['type'])}")
            
            if activity.get('facility'):
                st.caption(f"📍 {activity['facility']}")
            
            if activity["type"] in ["CASE_REJECTED", "CRITICAL_ALERT"]:
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Acknowledge", key=f"ack_{activity['id']}"):
                        acknowledge_event(activity['id'])
                        st.rerun()
                with col_b:
                    if st.button("View Details", key=f"view_{activity['id']}"):
                        show_event_details(activity)
        
        st.markdown("---")

def show_realtime_activity_feed():
    """Main real-time activity feed component"""
    st.markdown("### 🔴 Live System Activity")
    
    # Auto-refresh toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh every 10s", value=True)
    with col2:
        if st.button("Clear Read"):
            mark_all_notifications_read()
            st.rerun()
    with col3:
        if st.button("Simulate Activity"):
            generate_demo_activities()
            st.rerun()
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(10)
        st.rerun()
    
    # Show unread notifications count
    unread_count = len([n for n in st.session_state.notifications if not n.get('read', False)])
    if unread_count > 0:
        st.error(f"🚨 {unread_count} unread notifications requiring attention")
    
    # Activity feed
    activities = st.session_state.system_activities[:15]
    
    if not activities:
        st.info("No recent activity. System monitoring for new events...")
        return
    
    for activity in activities:
        render_activity_item(activity)

def render_notification_item(notification):
    """Render individual notification item"""
    timestamp = datetime.fromtimestamp(notification["timestamp"]).strftime("%H:%M:%S")
    
    priority_styles = {
        "HIGH": "🔴",
        "MEDIUM": "🟡", 
        "LOW": "🟢"
    }
    
    priority_icon = priority_styles.get(notification.get('priority', 'LOW'), '⚪')
    
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            read_status = "✓ " if notification.get('read') else ""
            st.markdown(f"**{priority_icon} {read_status}{notification['title']}**")
        
        with col2:
            st.caption(timestamp)
        
        with col3:
            if notification.get('action_required') and not notification.get('read'):
                st.error("Action Required")
            else:
                st.info("Acknowledged")
        
        st.write(notification['message'])
        
        if notification.get('facility'):
            st.caption(f"Facility: {notification['facility']}")
        
        if not notification.get('read'):
            col_a, col_b, col_c = st.columns([1, 1, 2])
            with col_a:
                if st.button("Mark Read", key=f"read_{notification['id']}"):
                    mark_notification_read(notification['id'])
                    st.rerun()
            with col_b:
                if st.button("View Event", key=f"event_{notification['id']}"):
                    event = next((e for e in st.session_state.system_activities 
                                if e['id'] == notification['event_id']), None)
                    if event:
                        show_event_details(event)
        
        st.markdown("---")

def show_notification_center():
    """Dedicated notification center"""
    st.markdown("### 📋 Notification Center")
    
    notifications = st.session_state.notifications
    
    if not notifications:
        st.info("No notifications at this time")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        show_all = st.checkbox("Show all", value=True)
    with col2:
        show_unread = st.checkbox("Unread only", value=False)
    with col3:
        priority_filter = st.selectbox("Priority", ["All", "HIGH", "MEDIUM", "LOW"])
    
    filtered_notifications = notifications
    
    if show_unread:
        filtered_notifications = [n for n in filtered_notifications if not n.get('read', False)]
    
    if priority_filter != "All":
        filtered_notifications = [n for n in filtered_notifications if n.get('priority') == priority_filter]
    
    for notification in filtered_notifications[:20]:
        render_notification_item(notification)

def render_live_case(case_id, case):
    """Render individual live case"""
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown(f"#### {case.get('patient_name', 'Unknown Patient')}")
            st.write(f"**{case.get('complaint', 'Unknown')}** • {case.get('age', 'N/A')} {case.get('gender', '')}")
            
            triage_color = case.get('triage_color', 'GREEN')
            badge_color = {
                'RED': '🔴', 
                'YELLOW': '🟡',
                'GREEN': '🟢'
            }.get(triage_color, '⚪')
            
            st.markdown(f"{badge_color} **{triage_color}** triage")
        
        with col2:
            status = case.get('status', 'UNKNOWN')
            progress_data = {
                "DISPATCHED": 25,
                "ENROUTE_SCENE": 50, 
                "ARRIVE_SCENE": 60,
                "DEPART_SCENE": 75,
                "ARRIVE_DEST": 90,
                "HANDOVER": 100
            }
            
            progress = progress_data.get(status, 0)
            st.progress(progress/100)
            st.caption(f"Status: {status.replace('_', ' ').title()}")
            
            if case.get('eta'):
                st.write(f"**ETA:** {case['eta']} min")
        
        with col3:
            if status in ["DISPATCHED", "ENROUTE_SCENE"]:
                if st.button("Update ETA", key=f"eta_{case_id}"):
                    update_case_eta(case_id)
                    st.rerun()
            
            if st.button("View Details", key=f"details_{case_id}"):
                show_case_details(case)
        
        if case.get('latest_vitals'):
            vitals = case['latest_vitals']
            vitals_cols = st.columns(5)
            vitals_cols[0].metric("HR", vitals.get('hr', '--'))
            vitals_cols[1].metric("SBP", vitals.get('sbp', '--'))
            vitals_cols[2].metric("SpO2", f"{vitals.get('spo2', '--')}%")
            vitals_cols[3].metric("RR", vitals.get('rr', '--'))
            vitals_cols[4].metric("Temp", f"{vitals.get('temp', '--')}°C")
        
        st.markdown("---")

def show_live_case_tracker():
    """Live case tracking dashboard"""
    st.markdown("### 🚨 Live Case Tracker")
    
    if not st.session_state.live_cases:
        initialize_demo_cases()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Case Status", ["All", "DISPATCHED", "ENROUTE", "ARRIVED", "HANDOVER"])
    with col2:
        triage_filter = st.selectbox("Triage Level", ["All", "RED", "YELLOW", "GREEN"])
    with col3:
        facility_filter = st.selectbox("Facility", ["All"] + list(set(
            case['facility'] for case in st.session_state.live_cases.values()
            if case.get('facility')
        )))
    
    filtered_cases = st.session_state.live_cases.copy()
    
    if status_filter != "All":
        filtered_cases = {k: v for k, v in filtered_cases.items() 
                         if v.get('status') == status_filter}
    
    if triage_filter != "All":
        filtered_cases = {k: v for k, v in filtered_cases.items() 
                         if v.get('triage_color') == triage_filter}
    
    if facility_filter != "All":
        filtered_cases = {k: v for k, v in filtered_cases.items() 
                         if v.get('facility') == facility_filter}
    
    for case_id, case in list(filtered_cases.items())[:10]:
        render_live_case(case_id, case)

def integrate_realtime_events():
    """Hook real-time events into existing workflow"""
    if "referrals" in st.session_state and st.session_state.referrals:
        latest_referral = st.session_state.referrals[0]
        
        referral_id = latest_referral.get('id')
        if referral_id not in st.session_state.get('published_referrals', set()):
            
            event_system.publish_event(
                event_type="CASE_CREATED",
                data={
                    "patient_name": latest_referral['patient']['name'],
                    "complaint": latest_referral['triage']['complaint'],
                    "triage_color": latest_referral['triage']['decision']['color'],
                    "referrer": latest_referral['referrer']['name'],
                    "facility": latest_referral.get('dest')
                },
                user=latest_referral['referrer']['name'],
                facility=latest_referral['referrer']['facility']
            )
            
            if 'published_referrals' not in st.session_state:
                st.session_state.published_referrals = set()
            st.session_state.published_referrals.add(referral_id)
            
            st.session_state.live_cases[referral_id] = {
                "patient_name": latest_referral['patient']['name'],
                "complaint": latest_referral['triage']['complaint'],
                "triage_color": latest_referral['triage']['decision']['color'],
                "age": latest_referral['patient']['age'],
                "gender": latest_referral['patient']['sex'],
                "status": latest_referral.get('status', 'PREALERT'),
                "facility": latest_referral.get('dest'),
                "latest_vitals": {
                    "hr": latest_referral['triage']['hr'],
                    "sbp": latest_referral['triage']['sbp'],
                    "rr": latest_referral['triage']['rr'],
                    "spo2": latest_referral['triage']['spo2'],
                    "temp": latest_referral['triage']['temp']
                }
            }

def publish_triage_override_event(original_color, override_color, reason, user):
    """Publish triage override event"""
    event_system.publish_event(
        event_type="TRIAGE_OVERRIDE",
        data={
            "from_color": original_color,
            "to_color": override_color,
            "reason": reason,
            "message": f"Triage overridden from {original_color} to {override_color}"
        },
        user=user,
        facility=st.session_state.get('active_facility')
    )

def publish_ambulance_dispatch_event(case_id, ambulance_type, destination):
    """Publish ambulance dispatch event"""
    event_system.publish_event(
        event_type="CASE_DISPATCHED",
        data={
            "case_id": case_id,
            "ambulance_type": ambulance_type,
            "destination": destination,
            "message": f"{ambulance_type} ambulance dispatched to {destination}"
        },
        user="Dispatcher",
        facility=destination
    )

# === FREE ROUTING CONFIGURATION UI ===
def show_free_routing_configuration():
    """
    Show free routing provider configuration in the UI
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🆓 Free Routing Configuration")
    
    # Provider selection
    provider = st.sidebar.selectbox(
        "Routing Provider",
        ["osrm", "graphhopper", "openrouteservice"],
        index=0,
        format_func=lambda x: {
            "osrm": "OSRM (Recommended - Free)",
            "graphhopper": "GraphHopper (Free Tier)", 
            "openrouteservice": "OpenRouteService (Free)"
        }[x]
    )
    
    # Traffic simulation settings
    st.sidebar.markdown("**Traffic Simulation**")
    enable_traffic = st.sidebar.checkbox("Simulate traffic patterns", value=True)
    
    if enable_traffic:
        st.sidebar.info("Traffic simulation considers:\n- Peak hours (7-10 AM, 5-8 PM)\n- Weekends vs weekdays")
    
    # Cache management
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Clear Cache"):
            DISTANCE_CACHE.clear()
            st.success("Route cache cleared!")
    with col2:
        st.metric("Cached Routes", len(DISTANCE_CACHE))
    
    # Provider info
    if provider == "osrm":
        st.sidebar.success("**OSRM**: Open Source • No API Key • Global Coverage")
    elif provider == "graphhopper":
        st.sidebar.info("**GraphHopper**: Free Tier • Good Accuracy")
    else:
        st.sidebar.info("**OpenRouteService**: Free with Registration")
    
    return provider, enable_traffic

# === ENHANCED FACILITY CARD WITH ROUTING INFO ===
def cap_badges(list_or_csv):
    """Display capability badges"""
    if isinstance(list_or_csv, str):
        items = [x.strip() for x in list_or_csv.split(",") if x.strip() and x.strip()!="—"]
    else:
        items = list_or_csv or []
    if not items:
        st.markdown('<span class="badge">—</span>', unsafe_allow_html=True)
        return
    cols = st.columns(min(4, max(1, len(items))))
    for i,cap in enumerate(items[:12]):
        cols[i%len(cols)].markdown(f'<span class="badge">{cap}</span>', unsafe_allow_html=True)

def enhanced_facility_card(row, rank, is_primary=False, is_alternate=False):
    """
    Enhanced facility card with routing information
    """
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Header with routing status indicator
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            if is_primary:
                st.markdown(f"#### 🏥 {row['name']} 🥇 <span class='priority-badge'>PRIMARY</span>", unsafe_allow_html=True)
            elif is_alternate:
                st.markdown(f"#### 🏥 {row['name']} 🥈 <span class='alternate-badge'>ALTERNATE</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"#### 🏥 {row['name']} #{rank}", unsafe_allow_html=True)
            
            # Routing status indicator
            if row.get('routing_success'):
                if row.get('estimated'):
                    st.markdown('<span class="badge warn">⚠ Estimated ETA</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge ok">✓ Live Routing</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge warn">⚠ Estimated ETA</span>', unsafe_allow_html=True)
        
        with header_col2:
            st.markdown(f"**Match Score: {row['score']}**", unsafe_allow_html=True)
        
        # Enhanced info with real routing data
        if isinstance(row['eta_min'], (int, float)):
            traffic_info = f" (Traffic: {row['traffic_factor']}x)" if row.get('traffic_factor', 1.0) > 1.0 else ""
            provider_info = f" • {row.get('routing_provider', '')}"
            sub = f"ETA ~ {row['eta_min']} min{traffic_info}{provider_info} • {row['km']} km • ICU beds: {row['ICU_open']} • Acceptance: {row['accept']}%"
        else:
            sub = f"Distance: {row['km']} km • ICU beds: {row['ICU_open']} • Acceptance: {row['accept']}%"
        
        st.markdown(f'<div class="small">{sub}</div>', unsafe_allow_html=True)
        
        # Enhanced scoring breakdown
        with st.expander("Enhanced Score Details"):
            details = row.get("scoring_details", {})
            st.write(f"**Capability Match:** {details.get('capability_score', 0)}")
            st.write(f"**Proximity (ETA-based):** {details.get('proximity_score', 0)}")
            if 'eta_minutes' in details and isinstance(details['eta_minutes'], (int, float)):
                st.write(f"**Driving Time:** {details['eta_minutes']} min")
            if 'traffic_factor' in details:
                st.write(f"**Traffic Impact:** {details['traffic_factor']}x")
            st.write(f"**ICU Availability:** {details.get('icu_score', 0)}")
            st.write(f"**Acceptance Rate:** {details.get('acceptance_score', 0)}")
            st.write(f"**Specialization Bonus:** {details.get('specialization_bonus', 0)}")
        
        st.markdown("**Specialties**")
        cap_badges(row.get("specialties",""))
        
        st.markdown("**High-end equipment**")
        cap_badges(row.get("highend",""))
        
        st.markdown('<hr class="soft" />', unsafe_allow_html=True)
        
        # Action buttons
        cta1, cta2 = st.columns(2)
        pick_label = "Select as primary" if not is_primary else "✓ Primary selected"
        alt_label = "Add as alternate" if not is_alternate else "✓ Alternate"
        
        pick = cta1.button(pick_label, key=f"pick_{row['name']}", disabled=is_primary)
        alt = cta2.button(alt_label, key=f"alt_{row['name']}", disabled=is_alternate)
        
        st.markdown('</div>', unsafe_allow_html=True)
        return pick, alt

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
            
            # Enhanced: Load default interventions properly
            default_interventions = []
            if 'default_interventions' in df.columns and pd.notna(row.get('default_interventions')):
                default_interventions = [iv.strip() for iv in str(row['default_interventions']).split(';') if iv.strip()]
            
            icd_lut.append({
                "icd_code": row['icd10'],
                "label": row['label'],
                "case_type": row['bundle'],
                "age_min": int(row['age_min']),
                "age_max": int(row['age_max']),
                "default_interventions": default_interventions,  # Now a list
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
         "default_interventions": ["IV fluids", "Uterotonics", "TXA"], "default_caps": ["ICU", "BloodBank", "OBGYN_OT", "OR", "Ventilator"]},
        {"icd_code": "O72.1", "label": "Immediate postpartum haemorrhage", "case_type": "Maternal", "age_min": 12, "age_max": 55,
         "default_interventions": ["IV fluids", "Uterotonics", "TXA"], "default_caps": ["ICU", "BloodBank", "OBGYN_OT", "OR", "Ventilator"]},
        {"icd_code": "O14.1", "label": "Severe pre-eclampsia", "case_type": "Maternal", "age_min": 12, "age_max": 55,
         "default_interventions": ["Magnesium sulfate", "BP control"], "default_caps": ["ICU", "OBGYN_OT"]},
        
        # Trauma
        {"icd_code": "S06.0", "label": "Concussion", "case_type": "Trauma", "age_min": 0, "age_max": 120,
         "default_interventions": ["Neuro checks", "Immobilization"], "default_caps": ["CT"]},
        {"icd_code": "S06.5", "label": "Traumatic subdural haemorrhage", "case_type": "Trauma", "age_min": 0, "age_max": 120,
         "default_interventions": ["Airway protection", "IV access"], "default_caps": ["CT", "Neurosurgery", "ICU", "OR"]},
        
        # Stroke
        {"icd_code": "I63.9", "label": "Cerebral infarction unspecified", "case_type": "Stroke", "age_min": 18, "age_max": 120,
         "default_interventions": ["BP control", "Glucose check"], "default_caps": ["CT", "Thrombolysis", "ICU"]},
        
        # Cardiac
        {"icd_code": "I21.9", "label": "Acute myocardial infarction unspecified", "case_type": "Cardiac", "age_min": 18, "age_max": 120,
         "default_interventions": ["Aspirin", "Oxygen", "IV access"], "default_caps": ["CathLab", "ICU"]},
        
        # Sepsis
        {"icd_code": "A41.9", "label": "Sepsis unspecified organism", "case_type": "Sepsis", "age_min": 0, "age_max": 120,
         "default_interventions": ["Antibiotics", "IV fluids", "Oxygen"], "default_caps": ["ICU"]},
        
        # Other
        {"icd_code": "J96.0", "label": "Acute respiratory failure", "case_type": "Other", "age_min": 0, "age_max": 120,
         "default_interventions": ["Oxygen", "Nebulization"], "default_caps": ["Ventilator", "ICU"]},
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
.priority-badge { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.35); }
.alternate-badge { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.35); }
.match-score { font-size: 0.75rem; color: #9ca3af; margin-top: 4px; }
.analytics-card { background: #1e293b; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid #3b82f6; }
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

def now_ts():
    """Get current timestamp."""
    return time.time()

# === ANALYTICS & VISUALIZATION FUNCTIONS ===
def create_time_series_analysis(referrals):
    """Create time series analysis of referrals by hour/day"""
    if not referrals:
        return pd.DataFrame()
    
    df_data = []
    for ref in referrals:
        if 'times' in ref:
            ts = ref['times'].get('first_contact_ts')
            if ts:
                try:
                    dt = datetime.fromtimestamp(ts)
                    df_data.append({
                        'datetime': dt,
                        'date': dt.date(),
                        'hour': dt.hour,
                        'day_of_week': dt.strftime('%A'),
                        'referral': 1,
                        'dispatched': 1 if ref['times'].get('dispatch_ts') else 0,
                        'arrived': 1 if ref['times'].get('arrive_dest_ts') else 0,
                        'triage_color': ref['triage']['decision']['color'],
                        'case_type': ref['triage']['complaint'],
                        'facility': ref.get('dest', 'Unknown')
                    })
                except (ValueError, TypeError):
                    continue
    
    if not df_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(df_data)
    return df

def create_funnel_analysis(referrals):
    """Create funnel analysis from referral to handover"""
    if not referrals:
        return {}
    
    stages = {
        'Referrals': len(referrals),
        'Dispatched': len([r for r in referrals if r['times'].get('dispatch_ts')]),
        'Arrived': len([r for r in referrals if r['times'].get('arrive_dest_ts')]),
        'Handover': len([r for r in referrals if r['times'].get('handover_ts')])
    }
    return stages

def create_sla_analysis(referrals):
    """Calculate SLA distributions for key time intervals"""
    sla_data = []
    
    for ref in referrals:
        try:
            times = ref.get('times', {})
            
            # Decision to Dispatch
            if times.get('decision_ts') and times.get('dispatch_ts'):
                decision_dispatch = (times['dispatch_ts'] - times['decision_ts']) / 60  # minutes
            else:
                decision_dispatch = None
                
            # Dispatch to Arrival
            if times.get('dispatch_ts') and times.get('arrive_dest_ts'):
                dispatch_arrival = (times['arrive_dest_ts'] - times['dispatch_ts']) / 60
            else:
                dispatch_arrival = None
                
            # Arrival to Handover
            if times.get('arrive_dest_ts') and times.get('handover_ts'):
                arrival_handover = (times['handover_ts'] - times['arrive_dest_ts']) / 60
            else:
                arrival_handover = None
                
            if any([decision_dispatch, dispatch_arrival, arrival_handover]):
                sla_data.append({
                    'case_id': ref['id'],
                    'triage_color': ref['triage']['decision']['color'],
                    'decision_dispatch': decision_dispatch,
                    'dispatch_arrival': dispatch_arrival,
                    'arrival_handover': arrival_handover
                })
        except (KeyError, TypeError, ValueError):
            continue
    
    return pd.DataFrame(sla_data) if sla_data else pd.DataFrame()

def create_triage_mix_analysis(referrals):
    """Analyze triage mix by facility and case type"""
    mix_data = []
    
    for ref in referrals:
        try:
            mix_data.append({
                'facility': ref.get('dest', 'Unknown'),
                'case_type': ref['triage']['complaint'],
                'triage_color': ref['triage']['decision']['color'],
                'used_ambulance': ref.get('transport', {}).get('ambulance') in ['BLS', 'ALS', 'ALS + Vent']
            })
        except (KeyError, TypeError):
            continue
    
    return pd.DataFrame(mix_data) if mix_data else pd.DataFrame()

def create_geo_analysis(referrals):
    """Create geographic analysis for heatmaps"""
    geo_data = []
    
    for ref in referrals:
        try:
            patient = ref.get('patient', {})
            location = patient.get('location', {})
            if location.get('lat') and location.get('lon'):
                geo_data.append({
                    'lat': float(location['lat']),
                    'lon': float(location['lon']),
                    'triage_color': ref['triage']['decision']['color'],
                    'case_type': ref['triage']['complaint'],
                    'facility': ref.get('dest', 'Unknown'),
                    'timestamp': ref['times'].get('first_contact_ts')
                })
        except (KeyError, TypeError, ValueError):
            continue
    
    return pd.DataFrame(geo_data) if geo_data else pd.DataFrame()

def create_ambulance_usage_analysis(referrals):
    """Analyze avoidable ambulance usage"""
    usage_data = []
    
    for ref in referrals:
        try:
            transport = ref.get('transport', {})
            used_ambulance = transport.get('ambulance') in ['BLS', 'ALS', 'ALS + Vent']
            triage_color = ref['triage']['decision']['color']
            
            usage_data.append({
                'triage_color': triage_color,
                'used_ambulance': used_ambulance,
                'case_type': ref['triage']['complaint'],
                'avoidable': used_ambulance and triage_color in ['GREEN', 'YELLOW']
            })
        except (KeyError, TypeError):
            continue
    
    return pd.DataFrame(usage_data) if usage_data else pd.DataFrame()

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
    hr, rr, sbp, temp, spo2 = (_num(hr), _num(rr), _num(sbp), _num(temp), _num(spo2))
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
    Score-only triage decision without ad-hoc flags
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
        
        # === ADD REAL-TIME EVENT: Triage Override ===
        publish_triage_override_event(
            base_colour,
            final_colour,
            override_reason,
            st.session_state.get('referrer_name', 'Unknown')
        )

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

# === ENHANCED FACILITY MATCHING SYSTEM ===
def calculate_facility_score(facility, required_caps, distance_km, case_type, triage_color):
    """
    Calculate comprehensive facility score based on:
    - Capability match (40%)
    - Proximity (30%) 
    - ICU availability (20%)
    - Acceptance rate (10%)
    - Case type specialization bonus
    """
    score = 0
    scoring_details = {}
    
    # 1. Capability Match (40% weight)
    if required_caps:
        capability_match = sum(1 for cap in required_caps if facility["caps"].get(cap, 0)) / len(required_caps)
    else:
        capability_match = 1.0  # Full score if no specific capabilities required
    
    score += capability_match * 40
    scoring_details["capability_score"] = round(capability_match * 40, 1)
    
    # 2. Proximity Score (30% weight) - closer is better
    proximity_score = max(0, 1 - (distance_km / 100))  # Linear decay up to 100km
    score += proximity_score * 30
    scoring_details["proximity_score"] = round(proximity_score * 30, 1)
    
    # 3. ICU Availability (20% weight)
    icu_beds = facility.get("ICU_open", 0)
    icu_score = min(1.0, icu_beds / 5.0)  # Max score at 5+ ICU beds
    score += icu_score * 20
    scoring_details["icu_score"] = round(icu_score * 20, 1)
    
    # 4. Acceptance Rate (10% weight)
    acceptance_rate = facility.get("acceptanceRate", 0.75)
    score += acceptance_rate * 10
    scoring_details["acceptance_score"] = round(acceptance_rate * 10, 1)
    
    # 5. Specialization Bonuses
    specialization_bonus = 0
    
    # Case type specialization matching
    if case_type in facility.get("specialties", {}):
        if facility["specialties"][case_type]:
            specialization_bonus += 5
    
    # High-end equipment bonus for critical cases
    if triage_color == "RED":
        high_end_count = sum(1 for eq in facility.get("highend", {}).values() if eq)
        specialization_bonus += min(5, high_end_count)  # Max 5 bonus points
    
    score += specialization_bonus
    scoring_details["specialization_bonus"] = specialization_bonus
    
    # Ensure score is within bounds
    final_score = min(100, max(0, score))
    scoring_details["total_score"] = round(final_score, 1)
    
    return final_score, scoring_details

def rank_facilities_for_case(origin_coords, required_caps, case_type, triage_color, top_k=8):
    """
    Enhanced facility ranking with comprehensive scoring
    """
    ranked_facilities = []
    
    # Validate inputs
    if not origin_coords or len(origin_coords) != 2:
        st.error("Invalid origin coordinates")
        return []
    
    if not hasattr(st.session_state, 'facilities') or not st.session_state.facilities:
        st.error("No facilities data available")
        return []
    
    for facility in st.session_state.facilities:
        try:
            # Validate facility data
            if not facility or 'lat' not in facility or 'lon' not in facility:
                continue
                
            # Calculate distance
            distance_km = dist_km(
                origin_coords[0], origin_coords[1],
                float(facility["lat"]), float(facility["lon"])
            )
            
            # Calculate comprehensive score
            score, scoring_details = calculate_facility_score(
                facility, required_caps, distance_km, case_type, triage_color
            )
            
            # Calculate ETA
            traffic_mult = traffic_factor_for_hour(datetime.now().hour)
            eta_min = eta_minutes_for(distance_km, traffic_mult)
            
            # Generate route
            route = interpolate_route(
                origin_coords[0], origin_coords[1],
                float(facility["lat"]), float(facility["lon"]), n=20
            )
            
            ranked_facilities.append({
                "name": facility.get("name", "Unknown Facility"),
                "type": facility.get("type", "Unknown"),
                "score": score,
                "scoring_details": scoring_details,
                "km": round(distance_km, 1),
                "eta_min": eta_min,
                "ICU_open": facility.get("ICU_open", 0),
                "accept": int(facility.get("acceptanceRate", 0.75) * 100),
                "specialties": ", ".join([s for s, v in facility.get("specialties", {}).items() if v]) or "—",
                "highend": ", ".join([e for e, v in facility.get("highend", {}).items() if v]) or "—",
                "route": route,
                "lat": float(facility["lat"]),
                "lon": float(facility["lon"])
            })
        except Exception as e:
            st.error(f"Error processing facility {facility.get('name', 'Unknown')}: {str(e)}")
            continue
    
    # Sort by score (descending) and distance (ascending)
    ranked_facilities.sort(key=lambda x: (-x["score"], x["km"]))
    
    return ranked_facilities[:top_k]

def facility_card(row, rank, is_primary=False, is_alternate=False):
    """
    Enhanced facility card with scoring details and priority indicators
    """
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Header with priority indicators
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            if is_primary:
                st.markdown(f"#### 🏥 {row['name']} 🥇 <span class='priority-badge'>PRIMARY</span>", unsafe_allow_html=True)
            elif is_alternate:
                st.markdown(f"#### 🏥 {row['name']} 🥈 <span class='alternate-badge'>ALTERNATE</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"#### 🏥 {row['name']} #{rank}", unsafe_allow_html=True)
        
        with header_col2:
            st.markdown(f"**Match Score: {row['score']}**", unsafe_allow_html=True)
        
        # Basic info
        sub = f"ETA ~ {row['eta_min']} min • {row['km']} km • ICU beds: {row['ICU_open']} • Acceptance: {row['accept']}%"
        st.markdown(f'<div class="small">{sub}</div>', unsafe_allow_html=True)
        
        # Scoring breakdown (expandable)
        with st.expander("Score Details"):
            details = row.get("scoring_details", {})
            st.write(f"**Capability Match:** {details.get('capability_score', 0)}")
            st.write(f"**Proximity:** {details.get('proximity_score', 0)}")
            st.write(f"**ICU Availability:** {details.get('icu_score', 0)}")
            st.write(f"**Acceptance Rate:** {details.get('acceptance_score', 0)}")
            st.write(f"**Specialization Bonus:** {details.get('specialization_bonus', 0)}")
        
        st.markdown("**Specialties**")
        cap_badges(row.get("specialties",""))
        
        st.markdown("**High-end equipment**")
        cap_badges(row.get("highend",""))
        
        st.markdown('<hr class="soft" />', unsafe_allow_html=True)
        
        # Action buttons
        cta1, cta2 = st.columns(2)
        pick_label = "Select as primary" if not is_primary else "✓ Primary selected"
        alt_label = "Add as alternate" if not is_alternate else "✓ Alternate"
        
        pick = cta1.button(pick_label, key=f"pick_{row['name']}", disabled=is_primary)
        alt = cta2.button(alt_label, key=f"alt_{row['name']}", disabled=is_alternate)
        
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
    
# === FREE ROUTING PROVIDER INTEGRATIONS ===
def get_route_osrm_free(origin_lat, origin_lon, dest_lat, dest_lon, profile='driving'):
    """
    Get route information from public OSRM instance (completely free)
    """
    try:
        # Format coordinates
        coordinates = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        
        # Build URL - using public OSRM instance
        url = f"{OSRM_BASE_URL}/route/v1/{profile}/{coordinates}"
        params = {
            'overview': 'false',
            'steps': 'false',
            'annotations': 'true'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['code'] == 'Ok' and data['routes']:
                route = data['routes'][0]
                
                return {
                    'distance_km': route['distance'] / 1000,  # Convert to km
                    'duration_min': route['duration'] / 60,   # Convert to minutes
                    'success': True,
                    'provider': 'OSRM (Free)'
                }
        
        return {'success': False, 'error': 'No route found', 'provider': 'OSRM'}
            
    except Exception as e:
        return {'success': False, 'error': str(e), 'provider': 'OSRM'}

def get_route_graphhopper_free(origin_lat, origin_lon, dest_lat, dest_lon, profile='car'):
    """
    Get route information from GraphHopper (free tier available)
    """
    try:
        # GraphHopper offers free tier with API key, but we'll use their demo key
        # For production, get a free API key from https://www.graphhopper.com/
        api_key = "demo_key"  # Replace with your free API key if needed
        
        url = "https://graphhopper.com/api/1/route"
        params = {
            'point': [f"{origin_lat},{origin_lon}", f"{dest_lat},{dest_lon}"],
            'vehicle': profile,
            'key': api_key,
            'type': 'json',
            'instructions': 'false',
            'elevation': 'false'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'paths' in data and data['paths']:
                path = data['paths'][0]
                
                return {
                    'distance_km': path['distance'] / 1000,
                    'duration_min': path['time'] / 60000,  # Convert ms to minutes
                    'success': True,
                    'provider': 'GraphHopper (Free)'
                }
        
        return {'success': False, 'error': 'No route found', 'provider': 'GraphHopper'}
            
    except Exception as e:
        return {'success': False, 'error': str(e), 'provider': 'GraphHopper'}

def get_route_openrouteservice_free(origin_lat, origin_lon, dest_lat, dest_lon, profile='driving-car'):
    """
    Get route information from OpenRouteService (free with API key)
    """
    try:
        # OpenRouteService offers free tier with registration
        # Get free API key from https://openrouteservice.org/
        api_key = "your_free_api_key_here"  # Optional for MVP
        
        url = f"{ORS_BASE_URL}/v2/directions/{profile}"
        headers = {
            'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        }
        if api_key and api_key != "your_free_api_key_here":
            headers['Authorization'] = api_key
            
        body = {
            "coordinates": [[origin_lon, origin_lat], [dest_lon, dest_lat]],
            "instructions": "false",
            "preference": "recommended"
        }
        
        response = requests.post(url, json=body, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'routes' in data and data['routes']:
                route = data['routes'][0]
                summary = route['summary']
                
                return {
                    'distance_km': summary['distance'] / 1000,
                    'duration_min': summary['duration'] / 60,
                    'success': True,
                    'provider': 'OpenRouteService'
                }
        
        return {'success': False, 'error': 'No route found', 'provider': 'OpenRouteService'}
            
    except Exception as e:
        return {'success': False, 'error': str(e), 'provider': 'OpenRouteService'}

def estimate_traffic_for_route(route_data, hour_of_day=None):
    """
    Estimate traffic based on time of day and route characteristics
    Uses heuristic rules since we don't have real traffic data
    """
    if hour_of_day is None:
        hour_of_day = datetime.now().hour
    
    base_duration = route_data.get('duration_min', 0)
    
    # Simple traffic estimation based on time of day
    traffic_multiplier = 1.0
    
    # Peak hours (morning and evening rush)
    if (7 <= hour_of_day <= 10) or (17 <= hour_of_day <= 20):
        traffic_multiplier = 1.3  # 30% longer during peak hours
    elif (11 <= hour_of_day <= 13) or (15 <= hour_of_day <= 16):
        traffic_multiplier = 1.1  # 10% longer during moderate hours
    
    # Weekend adjustment (lighter traffic)
    if datetime.now().weekday() >= 5:  # Saturday or Sunday
        traffic_multiplier *= 0.9  # 10% shorter on weekends
    
    estimated_duration = base_duration * traffic_multiplier
    
    return {
        'estimated_duration_min': estimated_duration,
        'traffic_multiplier': traffic_multiplier,
        'base_duration_min': base_duration,
        'peak_hour': (7 <= hour_of_day <= 10) or (17 <= hour_of_day <= 20)
    }

def get_route_info_free(origin_lat, origin_lon, dest_lat, dest_lon, provider=None):
    """
    Unified function to get route information from free providers
    """
    if provider is None:
        provider = ROUTING_PROVIDER
    
    # Create cache key
    cache_key = f"{origin_lat}_{origin_lon}_{dest_lat}_{dest_lon}_{provider}"
    
    # Check cache
    if cache_key in DISTANCE_CACHE:
        cached_data = DISTANCE_CACHE[cache_key]
        if datetime.now() - cached_data['timestamp'] < CACHE_DURATION:
            return cached_data['data']
    
    # Get route based on provider
    if provider == 'osrm':
        result = get_route_osrm_free(origin_lat, origin_lon, dest_lat, dest_lon)
    elif provider == 'graphhopper':
        result = get_route_graphhopper_free(origin_lat, origin_lon, dest_lat, dest_lon)
    elif provider == 'openrouteservice':
        result = get_route_openrouteservice_free(origin_lat, origin_lon, dest_lat, dest_lon)
    else:
        result = {'success': False, 'error': 'Unknown routing provider'}
    
    # If routing failed, fall back to straight-line distance calculation
    if not result.get('success'):
        straight_line_km = dist_km(origin_lat, origin_lon, dest_lat, dest_lon)
        # Estimate driving time based on straight-line distance (assuming average speed)
        estimated_driving_min = (straight_line_km / 40) * 60  # 40 km/h average
        
        result = {
            'success': True,
            'distance_km': straight_line_km,
            'duration_min': estimated_driving_min,
            'estimated': True,
            'provider': 'Straight-line Estimation'
        }
    
    # Add traffic estimation
    if result.get('success'):
        traffic_data = estimate_traffic_for_route(result)
        result.update(traffic_data)
    
    # Cache the result
    DISTANCE_CACHE[cache_key] = {
        'timestamp': datetime.now(),
        'data': result
    }
    
    return result

# === ENHANCED ANALYTICS FUNCTIONS ===
def create_enhanced_time_series(referrals):
    """Enhanced time series analysis with trends"""
    if not referrals:
        return pd.DataFrame()
    
    df_data = []
    for ref in referrals:
        try:
            ts = ref['times'].get('first_contact_ts', now_ts())
            dt = datetime.fromtimestamp(ts)
            
            df_data.append({
                'datetime': dt,
                'date': dt.date(),
                'hour': dt.hour,
                'day_of_week': dt.strftime('%A'),
                'week_number': dt.isocalendar()[1],
                'month': dt.month,
                'referral': 1,
                'triage_color': ref['triage']['decision']['color'],
                'case_type': ref['triage']['complaint'],
                'facility': ref.get('dest', 'Unknown'),
                'dispatched': 1 if ref['times'].get('dispatch_ts') else 0,
                'arrived': 1 if ref['times'].get('arrive_dest_ts') else 0,
                'handover': 1 if ref['times'].get('handover_ts') else 0
            })
        except (ValueError, TypeError, KeyError):
            continue
    
    return pd.DataFrame(df_data) if df_data else pd.DataFrame()

def calculate_rejection_rates(referrals):
    """Calculate rejection rates per facility"""
    facility_rejections = {}
    
    for ref in referrals:
        try:
            facility = ref.get('dest', 'Unknown')
            audit_log = ref.get('audit_log', [])
            
            # Check for rejection in audit log
            rejected = any(log.get('action') == 'CASE_REJECTED' for log in audit_log)
            
            if facility not in facility_rejections:
                facility_rejections[facility] = {'total': 0, 'rejected': 0}
            
            facility_rejections[facility]['total'] += 1
            if rejected:
                facility_rejections[facility]['rejected'] += 1
                
        except (KeyError, TypeError):
            continue
    
    # Calculate rejection rates
    rejection_rates = []
    for facility, stats in facility_rejections.items():
        rate = (stats['rejected'] / stats['total']) * 100 if stats['total'] > 0 else 0
        rejection_rates.append({
            'facility': facility,
            'total_referrals': stats['total'],
            'rejected': stats['rejected'],
            'rejection_rate': round(rate, 1)
        })
    
    return pd.DataFrame(rejection_rates)

def analyze_referral_reasons(referrals):
    """Analyze reasons for referral"""
    reasons_data = {
        'severity': 0,
        'bed_icu_unavailable': 0,
        'special_test': 0,
        'capabilities': {}
    }
    
    for ref in referrals:
        try:
            ref_reasons = ref.get('reasons', {})
            
            if ref_reasons.get('severity'):
                reasons_data['severity'] += 1
            if ref_reasons.get('bedOrICUUnavailable'):
                reasons_data['bed_icu_unavailable'] += 1
            if ref_reasons.get('specialTest'):
                reasons_data['special_test'] += 1
            
            # Count capabilities requested
            capabilities = ref_reasons.get('requiredCapabilities', [])
            for cap in capabilities:
                reasons_data['capabilities'][cap] = reasons_data['capabilities'].get(cap, 0) + 1
                
        except (KeyError, TypeError):
            continue
    
    return reasons_data

def analyze_medical_specialties(referrals):
    """Analyze medical specialty requests"""
    specialty_data = {}
    case_type_breakdown = {}
    
    for ref in referrals:
        try:
            case_type = ref['triage']['complaint']
            capabilities = ref.get('reasons', {}).get('requiredCapabilities', [])
            
            # Count case types
            case_type_breakdown[case_type] = case_type_breakdown.get(case_type, 0) + 1
            
            # Associate capabilities with case types
            for cap in capabilities:
                if cap not in specialty_data:
                    specialty_data[cap] = {'total': 0, 'by_case_type': {}}
                
                specialty_data[cap]['total'] += 1
                specialty_data[cap]['by_case_type'][case_type] = specialty_data[cap]['by_case_type'].get(case_type, 0) + 1
                
        except (KeyError, TypeError):
            continue
    
    return specialty_data, case_type_breakdown

def analyze_ambulance_utilization(referrals):
    """Analyze ambulance usage by triage category"""
    utilization_data = {'RED': {}, 'YELLOW': {}, 'GREEN': {}}
    
    for ref in referrals:
        try:
            triage_color = ref['triage']['decision']['color']
            transport = ref.get('transport', {})
            ambulance_type = transport.get('ambulance', 'None')
            used_ambulance = ambulance_type in ['BLS', 'ALS', 'ALS + Vent', 'Neonatal']
            
            if triage_color not in utilization_data:
                utilization_data[triage_color] = {}
            
            # Count by ambulance type
            utilization_data[triage_color][ambulance_type] = utilization_data[triage_color].get(ambulance_type, 0) + 1
            
        except (KeyError, TypeError):
            continue
    
    return utilization_data    

# === ENHANCED FACILITY MATCHING WITH FREE ROUTING ===
def calculate_enhanced_facility_score_free(facility, required_caps, route_data, case_type, triage_color):
    """
    Enhanced facility scoring with free routing data
    """
    score = 0
    scoring_details = {}
    
    # 1. Capability Match (40% weight) - Hard filter
    if required_caps:
        capability_match = sum(1 for cap in required_caps if facility["caps"].get(cap, 0)) / len(required_caps)
        # Apply hard filter - must meet minimum capability threshold
        if capability_match < 0.5:  # At least 50% of required capabilities
            return 0, {"capability_score": 0, "reason": "Insufficient capabilities"}
    else:
        capability_match = 1.0
    
    score += capability_match * 40
    scoring_details["capability_score"] = round(capability_match * 40, 1)
    
    # 2. Proximity Score (30% weight) - Based on estimated ETA
    if route_data.get('success'):
        # Use traffic-adjusted duration if available
        eta_minutes = route_data.get('estimated_duration_min', route_data.get('duration_min', 0))
        
        # Normalize ETA score (0-30 points)
        # Shorter ETA = higher score, max score for <30min, linear decay to 60min
        if eta_minutes <= 30:
            proximity_score = 30
        elif eta_minutes <= 60:
            proximity_score = 30 * (1 - (eta_minutes - 30) / 30)
        else:
            proximity_score = max(0, 30 * (1 - (eta_minutes - 60) / 60))
            
        # Apply traffic factor adjustment
        traffic_factor = route_data.get('traffic_multiplier', 1.0)
        proximity_score = proximity_score / traffic_factor
        
        score += proximity_score
        scoring_details["proximity_score"] = round(proximity_score, 1)
        scoring_details["eta_minutes"] = round(eta_minutes, 1)
        scoring_details["traffic_factor"] = round(traffic_factor, 2)
        scoring_details["estimated"] = route_data.get('estimated', False)
        scoring_details["peak_hour"] = route_data.get('peak_hour', False)
    else:
        # Fallback to straight-line distance if routing fails
        scoring_details["proximity_score"] = 0
        scoring_details["eta_minutes"] = "N/A"
        scoring_details["traffic_factor"] = 1.0
    
    # 3. ICU Availability (20% weight)
    icu_beds = facility.get("ICU_open", 0)
    icu_score = min(1.0, icu_beds / 5.0) * 20  # Max score at 5+ ICU beds
    score += icu_score
    scoring_details["icu_score"] = round(icu_score, 1)
    
    # 4. Acceptance Rate (10% weight)
    acceptance_rate = facility.get("acceptanceRate", 0.75)
    acceptance_score = acceptance_rate * 10
    score += acceptance_score
    scoring_details["acceptance_score"] = round(acceptance_score, 1)
    
    # 5. Specialization Bonuses
    specialization_bonus = 0
    
    # Case type specialization matching
    if case_type in facility.get("specialties", {}):
        if facility["specialties"][case_type]:
            specialization_bonus += 5
    
    # High-end equipment bonus for critical cases
    if triage_color == "RED":
        high_end_count = sum(1 for eq in facility.get("highend", {}).values() if eq)
        specialization_bonus += min(5, high_end_count)
    
    score += specialization_bonus
    scoring_details["specialization_bonus"] = specialization_bonus
    
    # Ensure score is within bounds
    final_score = min(100, max(0, score))
    scoring_details["total_score"] = round(final_score, 1)
    
    return final_score, scoring_details

def rank_facilities_with_free_routing(origin_coords, required_caps, case_type, triage_color, top_k=8):
    """
    Enhanced facility ranking with free routing data
    """
    ranked_facilities = []
    
    # Validate inputs
    if not origin_coords or len(origin_coords) != 2:
        st.error("Invalid origin coordinates")
        return []
    
    if not hasattr(st.session_state, 'facilities') or not st.session_state.facilities:
        st.error("No facilities data available")
        return []
    
    # Show progress for routing calculations
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_facilities = len(st.session_state.facilities)
    
    for i, facility in enumerate(st.session_state.facilities):
        try:
            # Update progress
            progress = (i + 1) / total_facilities
            progress_bar.progress(progress)
            status_text.text(f"Calculating routes... ({i + 1}/{total_facilities})")
            
            # Validate facility data
            if not facility or 'lat' not in facility or 'lon' not in facility:
                continue
            
            # Get free route information
            route_data = get_route_info_free(
                origin_coords[0], origin_coords[1],
                float(facility["lat"]), float(facility["lon"])
            )
            
            # Calculate enhanced score with routing data
            score, scoring_details = calculate_enhanced_facility_score_free(
                facility, required_caps, route_data, case_type, triage_color
            )
            
            # Skip facilities with insufficient capabilities
            if score == 0:
                continue
            
            # Generate route for visualization
            route_coords = interpolate_route(
                origin_coords[0], origin_coords[1],
                float(facility["lat"]), float(facility["lon"]), n=20
            )
            
            ranked_facilities.append({
                "name": facility.get("name", "Unknown Facility"),
                "type": facility.get("type", "Unknown"),
                "score": score,
                "scoring_details": scoring_details,
                "km": round(route_data.get('distance_km', 0), 1),
                "eta_min": scoring_details.get("eta_minutes", "N/A"),
                "traffic_factor": scoring_details.get("traffic_factor", 1.0),
                "estimated": scoring_details.get("estimated", False),
                "peak_hour": scoring_details.get("peak_hour", False),
                "ICU_open": facility.get("ICU_open", 0),
                "accept": int(facility.get("acceptanceRate", 0.75) * 100),
                "specialties": ", ".join([s for s, v in facility.get("specialties", {}).items() if v]) or "—",
                "highend": ", ".join([e for e, v in facility.get("highend", {}).items() if v]) or "—",
                "route": route_coords,
                "lat": float(facility["lat"]),
                "lon": float(facility["lon"]),
                "routing_success": route_data.get('success', False),
                "routing_provider": route_data.get('provider', 'Unknown')
            })
        except Exception as e:
            st.error(f"Error processing facility {facility.get('name', 'Unknown')}: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Sort by score (descending)
    ranked_facilities.sort(key=lambda x: (-x["score"], x["eta_min"] if isinstance(x["eta_min"], (int, float)) else 999))
    
    return ranked_facilities[:top_k]

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
        
        # Create realistic capability profiles based on facility type
        if "Tertiary" in n or "NEIGRIHMS" in n:
            # Tertiary facilities have most capabilities
            caps = {c: 1 for c in ["ICU","Ventilator","BloodBank","OR","CT","Thrombolysis","OBGYN_OT","CathLab","Dialysis","Neurosurgery"]}
            specs = {s: 1 for s in SPECIALTIES}
            hi = {i: 1 for i in INTERVENTIONS}
            icu_beds = rng.randint(4, 8)
            acceptance = round(0.8 + rng.random()*0.15, 2)
        elif "District" in n:
            # District hospitals have good capabilities
            caps = {c: int(rng.random()<0.8) for c in ["ICU","Ventilator","BloodBank","OR","CT","Thrombolysis","OBGYN_OT"]}
            caps["CathLab"] = 0; caps["Dialysis"] = 0; caps["Neurosurgery"] = 0
            specs = {s: int(rng.random()<0.7) for s in SPECIALTIES}
            hi = {i: int(rng.random()<0.6) for i in INTERVENTIONS}
            icu_beds = rng.randint(2, 5)
            acceptance = round(0.7 + rng.random()*0.2, 2)
        else:
            # CHC/PHC have basic capabilities
            caps = {c: int(rng.random()<0.4) for c in ["ICU","Ventilator","BloodBank","OR","CT"]}
            caps["Thrombolysis"] = 0; caps["OBGYN_OT"] = int("Maternal" in n)
            caps["CathLab"] = 0; caps["Dialysis"] = 0; caps["Neurosurgery"] = 0
            specs = {s: int(rng.random()<0.3) for s in SPECIALTIES}
            hi = {i: int(rng.random()<0.2) for i in INTERVENTIONS}
            icu_beds = rng.randint(0, 2)
            acceptance = round(0.6 + rng.random()*0.25, 2)
        
        fac.append(dict(
            name=f"{n} #{idx+1}" if names.count(n)>1 else n,
            lat=lat, lon=lon, ICU_open=icu_beds,
            acceptanceRate=acceptance,
            caps=caps, specialties=specs, highend=hi,
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

# === ENHANCED SYNTHETIC DATA SEEDING ===
RESUS = ["Airway positioning","Oxygen","IV fluids","Uterotonics","TXA","Bleeding control","Antibiotics","Nebulization","Immobilization","AED/CPR"]

def seed_referrals(n=500, rng_seed=42):
    """Enhanced synthetic data seeding with proper structure"""
    rng = random.Random(rng_seed)
    facs = st.session_state.facilities
    conds = ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"]
    
    # Clear existing referrals
    st.session_state.referrals = []
    
    # Define medically appropriate case profiles
    case_profiles = {
        "Maternal": {
            "icd_codes": ["O72.0", "O72.1", "O14.1"],
            "vitals_range": {"hr": (90, 140), "sbp": (80, 160), "rr": (18, 30), "spo2": (92, 99), "temp": (36.5, 38.5)},
            "required_caps": ["OBGYN_OT", "BloodBank", "ICU"],
            "typical_triage": ["RED", "YELLOW"],
            "transport_types": ["BLS", "ALS", "ALS + Vent", "Neonatal"]
        },
        "Trauma": {
            "icd_codes": ["S06.0", "S06.5"],
            "vitals_range": {"hr": (70, 150), "sbp": (70, 180), "rr": (16, 35), "spo2": (88, 98), "temp": (36.0, 38.0)},
            "required_caps": ["CT", "OR", "ICU", "Neurosurgery"],
            "typical_triage": ["RED", "YELLOW"],
            "transport_types": ["BLS", "ALS", "ALS + Vent"]
        },
        "Stroke": {
            "icd_codes": ["I63.9"],
            "vitals_range": {"hr": (60, 120), "sbp": (100, 200), "rr": (14, 25), "spo2": (90, 98), "temp": (36.0, 37.5)},
            "required_caps": ["CT", "Thrombolysis", "ICU"],
            "typical_triage": ["RED", "YELLOW"],
            "transport_types": ["BLS", "ALS"]
        },
        "Cardiac": {
            "icd_codes": ["I21.9"],
            "vitals_range": {"hr": (50, 130), "sbp": (80, 160), "rr": (16, 28), "spo2": (88, 96), "temp": (36.0, 37.8)},
            "required_caps": ["CathLab", "ICU"],
            "typical_triage": ["RED", "YELLOW"],
            "transport_types": ["BLS", "ALS"]
        },
        "Sepsis": {
            "icd_codes": ["A41.9"],
            "vitals_range": {"hr": (100, 160), "sbp": (70, 120), "rr": (20, 35), "spo2": (85, 94), "temp": (38.0, 40.0)},
            "required_caps": ["ICU"],
            "typical_triage": ["RED", "YELLOW"],
            "transport_types": ["BLS", "ALS"]
        },
        "Other": {
            "icd_codes": ["J96.0"],
            "vitals_range": {"hr": (80, 140), "sbp": (90, 150), "rr": (18, 32), "spo2": (86, 95), "temp": (36.5, 39.0)},
            "required_caps": ["Ventilator", "ICU"],
            "typical_triage": ["RED", "YELLOW", "GREEN"],
            "transport_types": ["BLS", "Other"]
        }
    }
    
    # Generate data over last 30 days
    base = time.time() - 30 * 24 * 3600

    for i in range(n):
        cond = rng.choices(conds, weights=[0.22, 0.23, 0.18, 0.18, 0.14, 0.05])[0]
        profile = case_profiles[cond]
        
        # Generate medically appropriate vitals
        hr = rng.randint(*profile["vitals_range"]["hr"])
        sbp = rng.randint(*profile["vitals_range"]["sbp"])
        rr = rng.randint(*profile["vitals_range"]["rr"])
        spo2 = rng.randint(*profile["vitals_range"]["spo2"])
        temp = round(rng.uniform(*profile["vitals_range"]["temp"]), 1)
        avpu = "A"
        
        vit = dict(hr=hr, rr=rr, sbp=sbp, temp=temp, spo2=spo2, avpu=avpu, complaint=cond)
        color = tri_color(vit)
        severity = {"RED": "Critical", "YELLOW": "Moderate", "GREEN": "Non-critical"}[color]

        lat, lon = rand_geo(rng)
        
        # Select appropriate destination
        suitable_facs = [
            f for f in facs 
            if all(f["caps"].get(cap, 0) for cap in profile["required_caps"])
        ]
        
        if not suitable_facs:
            dest = rng.choice(facs)
        else:
            dest = rng.choice(suitable_facs)
            
        dkm = dist_km(lat, lon, dest["lat"], dest["lon"])
        
        # More varied timestamps
        ts_first = base + rng.randint(0, 30 * 24 * 3600)
        hr_of_day = datetime.fromtimestamp(ts_first).hour
        traffic_mult = traffic_factor_for_hour(hr_of_day)
        speed_kmh = rng.choice([30, 36, 45, 50, 60])
        eta_min = max(5, int(dkm / speed_kmh * 60 * traffic_mult))
        route = interpolate_route(lat, lon, dest["lat"], dest["lon"], n=24)

        # Transport types
        transport_type = rng.choice(profile["transport_types"])
        amb_avail = (rng.random() > 0.25)
        
        # Time intervals
        t_dec = ts_first + rng.randint(60, 6 * 60)
        t_disp = t_dec + (rng.randint(2 * 60, 10 * 60) if amb_avail else rng.randint(15 * 60, 45 * 60))
        t_arr = t_disp + (eta_min * 60)
        t_hov = t_arr + rng.randint(5 * 60, 20 * 60)

        # ICD code
        icd_code = rng.choice(profile["icd_codes"])
        icd_label = next((item["label"] for item in ICD_LUT if item["icd_code"] == icd_code), f"{cond} Case")
        
        provisional_dx = dict(
            code=icd_code,
            label=icd_label,
            case_type=cond
        )

        # Generate interventions
        interventions = []
        for intervention in rng.sample(RESUS, rng.randint(0, 3)):
            interventions.append({
                "name": intervention,
                "type": "resuscitation",
                "timestamp": ts_first,
                "performed_by": "referrer",
                "status": "completed"
            })

        st.session_state.referrals.append(dict(
            id=f"S{i:04d}",
            patient=dict(
                name=f"Pt{i:04d}", 
                age=rng.randint(1, 85), 
                sex=("Female" if rng.random() < 0.5 else "Male"),
                id="", 
                location=dict(lat=lat, lon=lon)
            ),
            referrer=dict(
                name=rng.choice(["Dr. Rai", "Dr. Khonglah", "ANM Pynsuk", "Dr. Sharma", "Dr. Singh"]), 
                facility=rng.choice(["PHC Mawlai", "CHC Smit", "CHC Pynursla", "District Hospital Shillong",
                                   "Tertiary Shillong Hub", "PHC Nongpoh", "CHC Jowai"]),
                role=rng.choice(["Doctor/Physician", "ANM/ASHA/EMT"])
            ),
            provisionalDx=provisional_dx,
            interventions=interventions,
            triage=dict(
                complaint=cond, 
                decision=dict(color=color), 
                hr=hr, sbp=sbp, rr=rr, temp=temp, spo2=spo2, avpu=avpu
            ),
            clinical=dict(summary=f"Auto-seeded {cond.lower()} case"),
            severity=severity,
            reasons=dict(
                severity=True, 
                bedOrICUUnavailable=(rng.random() < 0.2), 
                specialTest=(rng.random() < 0.3), 
                requiredCapabilities=profile["required_caps"]
            ),
            dest=dest["name"],
            alternates=[],
            transport=dict(
                eta_min=eta_min, 
                traffic=traffic_mult, 
                speed_kmh=speed_kmh, 
                ambulance=transport_type, 
                priority=rng.choice(["Routine", "Urgent", "STAT"])
            ),
            route=route,
            times=dict(
                first_contact_ts=ts_first, 
                decision_ts=t_dec, 
                dispatch_ts=t_disp, 
                arrive_dest_ts=t_arr, 
                handover_ts=t_hov
            ),
            status=rng.choice(["HANDOVER", "ARRIVE_DEST", "DEPART_SCENE", "DISPATCHED", "PREALERT"]),
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

# === REAL-TIME SESSION STATE ===
if "system_activities" not in st.session_state:
    st.session_state.system_activities = []
    
if "notifications" not in st.session_state:
    st.session_state.notifications = []
    
if "live_cases" not in st.session_state:
    st.session_state.live_cases = {}

if "user_presence" not in st.session_state:
    st.session_state.user_presence = {}

if "published_referrals" not in st.session_state:
    st.session_state.published_referrals = set()

# Initialize facility matching session state
if "matched_primary" not in st.session_state:
    st.session_state.matched_primary = None
if "matched_alts" not in st.session_state:
    st.session_state.matched_alts = set()

# Normalize schema
st.session_state.facilities = [normalize_facility(x) for x in st.session_state.facilities]

# Auto-seed on first run (ensures ≥100)
if len(st.session_state.referrals) < 100:
    seed_referrals(n=500)

# === MAIN APP UI ===
st.title("AHECN – Streamlit MVP v1.9 (Enhanced Analytics Dashboard)")
tabs = st.tabs(["Referrer","Ambulance / EMT","Receiving Hospital","Government Analytics","Data / Admin","Facility Admin", "🚨 Real-time Dashboard"])

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
    
    # Store referrer name for event system
    st.session_state.referrer_name = r_name
    
    # Referrer Role Selector
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
    with v3:
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
    override_applied = st.session_state.get("triage_override_active", False)
    render_triage_banner(hr, rr, sbp, temp, spo2, avpu, complaint, override_applied)

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
    # ========== STEP 1: ICD-CODED DIAGNOSIS (ROLE-BASED) ==========
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
            
            # Display ICD details
            st.info(f"**Selected:** {row['label']} ({row['icd_code']}) • Age range: {row['age_min']}-{row['age_max']} years")
        else:
            st.warning("No ICD codes match your search/filters. Try different criteria or check 'Show all diagnoses'.")
            chosen_icd = None
            row = None

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
                else:
                    row = None
            else:
                st.info("No ICD suggestions for this age & case type")
                chosen_icd = None
                row = None
        
        # Additional notes
        dx_free = st.text_input("Additional notes (optional)", "")
        
        # Diagnosis payload for non-doctors
        dx_payload = dict(code=row["icd_code"] if chosen_icd else "", 
                         label=referral_reason or (row["label"] if chosen_icd else ""), 
                         case_type=str(complaint))
        
        if not referral_reason and not chosen_icd:
            st.error("Please provide a reason for referral or select an ICD diagnosis")

    # ========== INTERVENTIONS BY DIAGNOSIS ==========
    st.subheader("Interventions by Diagnosis")
    
    if referrer_role == "Doctor/Physician" and chosen_icd and row is not None:
        # Show default interventions from ICD as checkboxes
        default_iv = row.get("default_interventions", [])
        iv_selected = []  # Initialize here
        
        if default_iv:
            st.markdown("**Diagnosis-Specific Interventions**")
            st.caption(f"Default interventions for {row['label']}:")
            
            iv_cols = st.columns(2)
            for i, item in enumerate(default_iv):
                col_idx = i % 2
                if iv_cols[col_idx].checkbox(item, value=True, key=f"iv_{i}"):
                    iv_selected.append({
                        "name": item,
                        "type": "diagnosis_default",
                        "timestamp": now_ts(),
                        "performed_by": "referrer",
                        "status": "completed"
                    })
            
            # Additional custom interventions
            st.markdown("**Additional Interventions**")
            custom_iv = st.text_area("Add custom interventions (one per line)", 
                                   placeholder="Enter any additional interventions performed...",
                                   height=60)
            if custom_iv:
                for line in custom_iv.split('\n'):
                    line = line.strip()
                    if line:
                        iv_selected.append({
                            "name": line,
                            "type": "custom",
                            "timestamp": now_ts(),
                            "performed_by": "referrer",
                            "status": "completed"
                        })
        else:
            st.info("No default interventions for this diagnosis")
            iv_selected = []
    else:
        # For non-doctors or when no ICD selected
        st.markdown("**Interventions Performed**")
        custom_iv = st.text_area("Describe interventions performed", 
                               placeholder="List all interventions performed (one per line)...",
                               height=80)
        iv_selected = []
        if custom_iv:
            for line in custom_iv.split('\n'):
                line = line.strip()
                if line:
                    iv_selected.append({
                        "name": line,
                        "type": "custom", 
                        "timestamp": now_ts(),
                        "performed_by": "referrer",
                        "status": "completed"
                    })

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
    
    # Auto-suggest capabilities based on ICD selection
    auto_suggested_caps = []
    if chosen_icd and row is not None:
        auto_suggested_caps = row.get("default_caps", [])
        st.info(f"**Auto-suggested capabilities:** {', '.join(auto_suggested_caps) if auto_suggested_caps else 'None'}")
    
    need_caps = []
    if ref_tests:
        st.caption("Select required capabilities for this case")
        cap_cols = st.columns(5)
        CAP_LIST = ["ICU", "Ventilator", "BloodBank", "OR", "CT", "Thrombolysis", "OBGYN_OT", "CathLab", "Dialysis", "Neurosurgery"]
        for i, cap in enumerate(CAP_LIST):
            # Pre-select auto-suggested capabilities
            pre_select = cap in auto_suggested_caps
            if cap_cols[i % 5].checkbox(cap, value=pre_select, key=f"cap_{cap}"):
                need_caps.append(cap)

    # === ENHANCED FACILITY MATCHING WITH FREE ROUTING ===
    st.markdown("### 🎯 Enhanced Facility Matching (Free Routing)")

    # Show free routing configuration
    current_provider, enable_traffic = show_free_routing_configuration()

    if st.button("Find Best Matched Facilities with Free Routing", type="primary"):
        # Validate diagnosis before proceeding
        if referrer_role == "Doctor/Physician" and dx_payload is None:
            st.error("Please select an ICD diagnosis to find matching facilities")
        elif referrer_role == "ANM/ASHA/EMT" and not dx_payload.get("label"):
            st.error("Please provide a reason for referral to find matching facilities")
        else:
            # Calculate current triage color for scoring
            vitals = dict(hr=hr, rr=rr, sbp=sbp, temp=temp, spo2=spo2, avpu=avpu)
            context = dict(
                age=p_age,
                pregnant=(complaint == "Maternal"),
                infection=(complaint in ["Sepsis", "Other"]),
                o2_device=st.session_state.o2_device,
                spo2_scale=st.session_state.spo2_scale,
                behavior=st.session_state.pews_behavior
            )
            triage_color, _ = triage_decision(vitals, context)
            
            # Apply override if active
            if st.session_state.triage_override_active and st.session_state.triage_override_color:
                triage_color = st.session_state.triage_override_color

            # Get ranked facilities with free routing
            with st.spinner("Calculating optimal routes with free routing services..."):
                ranked_facilities = rank_facilities_with_free_routing(
                    origin_coords=(p_lat, p_lon),
                    required_caps=need_caps,
                    case_type=complaint,
                    triage_color=triage_color,
                    top_k=8
                )

            if not ranked_facilities:
                st.warning("No suitable facilities found. Try relaxing capability requirements.")
            else:
                # Display routing provider info
                provider_name = {
                    "osrm": "OSRM (Free Open Source)",
                    "graphhopper": "GraphHopper (Free Tier)", 
                    "openrouteservice": "OpenRouteService (Free)"
                }[current_provider]
                
                st.success(f"✓ Routing completed using {provider_name}")
                
                # Display ranked facilities
                st.markdown(f"#### 🏆 Top {len(ranked_facilities)} Matched Facilities")
                
                # Show traffic simulation status
                if enable_traffic:
                    current_hour = datetime.now().hour
                    if (7 <= current_hour <= 10) or (17 <= current_hour <= 20):
                        st.info("🚗 **Peak hours detected**: Estimated travel times include traffic delays")
                    else:
                        st.info("🛣️ **Off-peak hours**: Normal travel conditions")
                
                st.info(f"**Case Type:** {complaint} | **Triage:** {triage_color} | **Required Capabilities:** {', '.join(need_caps) if need_caps else 'None'}")
                
                # Reset selection state
                st.session_state.matched_primary = None
                st.session_state.matched_alts = set()

                # Display facilities with enhanced cards
                for i, facility in enumerate(ranked_facilities, 1):
                    is_primary = (st.session_state.matched_primary == facility["name"])
                    is_alternate = (facility["name"] in st.session_state.matched_alts)
                    
                    pick, alt = enhanced_facility_card(facility, i, is_primary, is_alternate)
                    
                    if pick:
                        st.session_state.matched_primary = facility["name"]
                        st.rerun()
                    if alt:
                        st.session_state.matched_alts.add(facility["name"])
                        st.rerun()

                # Set default primary if none selected
                if not st.session_state.matched_primary and ranked_facilities:
                    st.session_state.matched_primary = ranked_facilities[0]["name"]

                # Show selection summary
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.session_state.matched_primary:
                        st.success(f"**Primary:** {st.session_state.matched_primary}")
                    else:
                        st.warning("No primary facility selected")
                
                with col2:
                    if st.session_state.matched_alts:
                        st.info(f"**Alternates:** {', '.join(sorted(st.session_state.matched_alts))}")
                    else:
                        st.info("No alternate facilities selected")

                # Enhanced map visualization with actual routes
                show_map = st.checkbox("Show detailed routes to facilities", value=True)
                if show_map and st.session_state.matched_primary:
                    try:
                        primary_name = st.session_state.matched_primary
                        primary_fac = next((f for f in ranked_facilities if f["name"] == primary_name), None)
                        
                        if primary_fac and p_lat and p_lon:
                            # Create enhanced layers for visualization
                            layers = []
                            
                            # Origin layer
                            layers.append(pdk.Layer(
                                "ScatterplotLayer",
                                data=[{"lon": p_lon, "lat": p_lat}],
                                get_position="[lon, lat]",
                                get_radius=200,
                                get_fill_color=[66, 133, 244, 200],
                                get_line_color=[0, 0, 0, 255],
                                get_line_width=50,
                            ))
                            
                            # Facility layers with color coding by score
                            for i, fac in enumerate(ranked_facilities[:6]):  # Top 6 facilities
                                # Color based on score (green=high, yellow=medium, red=low)
                                if fac["score"] >= 80:
                                    color = [34, 197, 94, 200]  # Green
                                elif fac["score"] >= 60:
                                    color = [245, 158, 11, 200]  # Yellow
                                else:
                                    color = [239, 68, 68, 200]  # Red
                                    
                                # Highlight primary facility
                                if fac["name"] == primary_name:
                                    color = [139, 92, 246, 255]  # Purple for primary
                                
                                layers.append(pdk.Layer(
                                    "ScatterplotLayer",
                                    data=[{"lon": fac["lon"], "lat": fac["lat"]}],
                                    get_position="[lon, lat]",
                                    get_radius=180,
                                    get_fill_color=color,
                                    get_line_color=[255, 255, 255, 255],
                                    get_line_width=20,
                                ))
                                
                                # Route visualization for primary only (to reduce clutter)
                                if fac["name"] == primary_name and fac.get("route"):
                                    route_data = []
                                    for point in fac["route"]:
                                        route_data.append({"lon": point[1], "lat": point[0]})
                                    
                                    layers.append(pdk.Layer(
                                        "PathLayer",
                                        data=[{"path": route_data}],
                                        get_path="path",
                                        get_color=[16, 185, 129, 180],
                                        get_width=8,
                                        width_scale=8,
                                        width_min_pixels=4,
                                    ))
                            
                            st.pydeck_chart(pdk.Deck(
                                layers=layers,
                                initial_view_state=pdk.ViewState(latitude=p_lat, longitude=p_lon, zoom=10),
                                map_style="mapbox://styles/mapbox/dark-v10",
                            ))
                        else:
                            st.warning("Could not render map: missing location data")
                    except Exception as e:
                        st.error(f"Map rendering error: {str(e)}")
                        
    # Final referral details
    st.markdown("### Referral details")
    colA, colB, colC = st.columns(3)
    with colA:
        priority = st.selectbox("Transport priority", ["Routine", "Urgent", "STAT"], index=1)
    with colB:
        amb_type = st.selectbox("Ambulance type", ["BLS", "ALS", "ALS + Vent", "Neonatal", "Other"], index=1)
    with colC:
        consent = st.checkbox("Patient/family consent obtained", value=True)

    primary = st.session_state.get("matched_primary")
    alternates = sorted(list(st.session_state.get("matched_alts", [])))

    def _save_referral(dispatch=False):
        # Validate based on role
        if referrer_role == "Doctor/Physician" and dx_payload is None:
            st.error("Please select an ICD diagnosis to create referral")
            return None
        elif referrer_role == "ANM/ASHA/EMT" and not dx_payload.get("label"):
            st.error("Please provide a reason for referral")
            return None
            
        if not primary:
            st.error("Select a primary destination from 'Find Best Matched Facilities' above.")
            return None
            
        vit = dict(hr=hr, rr=rr, sbp=sbp, temp=temp, spo2=spo2, avpu=avpu, complaint=complaint)
        
        # Combine interventions properly
        all_interventions = iv_selected if 'iv_selected' in locals() else []
        
        # Add resuscitation interventions
        for resus in resus_done:
            all_interventions.append({
                "name": resus,
                "type": "resuscitation",
                "timestamp": now_ts(),
                "performed_by": "referrer",
                "status": "completed"
            })
        
        # Calculate triage color
        context = dict(
            age=p_age,
            pregnant=(complaint == "Maternal"),
            infection=(complaint in ["Sepsis", "Other"]),
            o2_device=st.session_state.o2_device,
            spo2_scale=st.session_state.spo2_scale,
            behavior=st.session_state.pews_behavior
        )
        triage_color, triage_details = triage_decision(vit, context)
        
        # Apply override if active
        if st.session_state.triage_override_active and st.session_state.triage_override_color:
            triage_color = st.session_state.triage_override_color
        
        # Create referral object
        ref = dict(
            id=f"R{int(time.time())}",
            patient=dict(
                name=p_name, age=p_age, sex=p_sex, id=p_id,
                location=dict(lat=p_lat, lon=p_lon)
            ),
            referrer=dict(
                name=r_name, facility=r_fac, role=referrer_role
            ),
            provisionalDx=dx_payload,
            interventions=all_interventions,
            triage=dict(
                complaint=complaint,
                decision=dict(color=triage_color),
                hr=hr, rr=rr, sbp=sbp, temp=temp, spo2=spo2, avpu=avpu
            ),
            clinical=dict(summary=ocr),
            severity=triage_color,
            reasons=dict(
                severity=ref_severity,
                bedOrICUUnavailable=ref_beds,
                specialTest=ref_tests,
                requiredCapabilities=need_caps
            ),
            dest=primary,
            alternates=alternates,
            transport=dict(
                eta_min=0,  # Will be calculated
                traffic=1.0,
                speed_kmh=40,
                ambulance=amb_type,
                priority=priority
            ),
            route=[],
            times=dict(
                first_contact_ts=now_ts(),
                decision_ts=now_ts(),
                dispatch_ts=None,
                arrive_dest_ts=None,
                handover_ts=None
            ),
            status="PREALERT",
            ambulance_available=True,
            audit_log=[]
        )
        
        # Add override info if applied
        if st.session_state.triage_override_active:
            ref["audit_log"].append({
                "timestamp": now_ts(),
                "user": r_name,
                "action": "TRIAGE_OVERRIDE",
                "details": {
                    "original_color": triage_details,
                    "override_color": triage_color,
                    "reason": st.session_state.triage_override_reason
                }
            })
        
        st.session_state.referrals.insert(0, ref)
        
        # === ADD REAL-TIME EVENT: Case Created ===
        event_system.publish_event(
            event_type="CASE_CREATED",
            data={
                "patient_name": p_name,
                "complaint": complaint,
                "triage_color": triage_color,
                "referrer": r_name,
                "facility": primary
            },
            user=r_name,
            facility=r_fac
        )
        
        return ref["id"]

    # Create referral button
    if st.button("Create referral", type="primary"):
        ref_id = _save_referral(dispatch=False)
        if ref_id:
            st.success(f"Referral {ref_id} created!")
            st.balloons()
            st.session_state.matched_primary = None
            st.session_state.matched_alts = set()
            st.rerun()

# ======== Real-time Dashboard Tab ========
with tabs[6]:
    st.header("🚨 Real-time Emergency Coordination Dashboard")
    
    # Dashboard overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Cases", len(st.session_state.live_cases))
    with col2:
        unread_count = len([n for n in st.session_state.notifications if not n.get('read', False)])
        st.metric("Unread Notifications", unread_count)
    with col3:
        active_count = len([a for a in st.session_state.system_activities if a.get('timestamp', 0) > time.time() - 3600])
        st.metric("Last Hour Activity", active_count)
    with col4:
        dispatched_count = len([c for c in st.session_state.live_cases.values() if c.get('status') in ['DISPATCHED', 'ENROUTE_SCENE']])
        st.metric("Ambulances Dispatched", dispatched_count)
    
    # Main real-time components
    tab1, tab2, tab3 = st.tabs(["Live Activity Feed", "Notification Center", "Live Case Tracker"])
    
    with tab1:
        show_realtime_activity_feed()
    
    with tab2:
        show_notification_center()
    
    with tab3:
        show_live_case_tracker()

# ======== Other Tabs (Placeholders) ========
with tabs[1]:
    st.header("Ambulance / EMT Dashboard")
    st.info("Ambulance coordination and dispatch interface coming soon...")
    
    # Quick dispatch demo
    if st.button("🚑 Simulate Ambulance Dispatch"):
        if st.session_state.referrals:
            latest_ref = st.session_state.referrals[0]
            publish_ambulance_dispatch_event(
                latest_ref['id'],
                "ALS",
                latest_ref.get('dest', 'Unknown Facility')
            )
            st.success("Ambulance dispatch simulated!")

with tabs[2]:
    st.header("Receiving Hospital Dashboard")
    st.info("Hospital acceptance and case management interface coming soon...")

with tabs[3]:
    st.header("Government Analytics Dashboard")
    st.info("Regional analytics and performance monitoring coming soon...")

with tabs[4]:
    st.header("Data / Admin Dashboard")
    st.info("System administration and data management coming soon...")

with tabs[5]:
    st.header("Facility Admin Dashboard")
    st.info("Facility management and capacity monitoring coming soon...")

# === Integrate real-time events with existing workflow ===
integrate_realtime_events()
