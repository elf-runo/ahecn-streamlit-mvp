import json
import streamlit as st
from storage import get_db, poll_events
from presence import heartbeat, online_users

st.session_state.setdefault("last_event_id", 0)

@st.fragment(run_every=3)
def realtime_bus(user, role):
    conn = get_db()

    heartbeat(user, role, st.session_state.get("editing_case"))

    st.caption("🟢 Online users")
    for u, r, editing in online_users(conn):
        st.write(f"{u} ({r})" + (f" – editing {editing}" if editing else ""))

    rows = poll_events(conn, st.session_state["last_event_id"], user=user)
    for row in rows:
        eid, ts, etype, case_id, actor, audience, payload = row
        payload = json.loads(payload or "{}")
        st.toast(f"{etype} — {actor}: {case_id}")
        st.session_state["last_event_id"] = max(st.session_state["last_event_id"], eid)
