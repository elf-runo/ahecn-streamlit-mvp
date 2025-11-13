import threading, time, random
from storage import get_db, publish_event
import streamlit as st

@st.cache_resource
def start_demo_feeder():
    stop = threading.Event()
    def run():
        conn = get_db()
        cases = ["R001","R002","R003","R004"]
        actors = ["Dr. Singh","EMT #3","CHC Mawryngkneng"]
        types = ["COMMENT_ADDED","CASE_STATUS_CHANGED","ROUTE_UPDATED","TRIAGE_OVERRIDE"]
        while not stop.is_set():
            time.sleep(random.randint(6,14))
            publish_event(
                conn,
                random.choice(types),
                random.choice(cases),
                random.choice(actors),
                payload={"status": random.choice(["ENROUTE","SCENE","TO_HOSP","HANDOVER"])}
            )
    t = threading.Thread(target=run, daemon=True); t.start()
    return stop
