import sqlite3, json, time
import streamlit as st

@st.cache_resource
def get_db():
    conn = sqlite3.connect("ahecn_realtime.db", check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS presence(
        user TEXT PRIMARY KEY,
        role TEXT,
        last_seen REAL,
        editing_case TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL,
        type TEXT,
        case_id TEXT,
        actor TEXT,
        audience TEXT,
        payload TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS locks(
        case_id TEXT PRIMARY KEY,
        holder TEXT,
        expires REAL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS notifications(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL,
        user TEXT,
        title TEXT,
        body TEXT,
        read INTEGER
    )""")
    conn.commit()
    return conn

def publish_event(conn, etype, case_id, actor, audience="all", payload=None):
    conn.execute(
        "INSERT INTO events(ts,type,case_id,actor,audience,payload) VALUES(?,?,?,?,?,?)",
        (time.time(), etype, case_id, actor, audience, json.dumps(payload or {}))
    )
    conn.commit()

def poll_events(conn, last_id=0, audience="all", user=None, limit=100):
    if user:
        q = """SELECT id,ts,type,case_id,actor,audience,payload FROM events
               WHERE id>? AND (audience=? OR audience=?)
               ORDER BY id ASC LIMIT ?"""
        rows = conn.execute(q, (last_id, "all", user, limit)).fetchall()
    else:
        rows = conn.execute("""SELECT id,ts,type,case_id,actor,audience,payload FROM events
                               WHERE id>? ORDER BY id ASC LIMIT ?""",
                             (last_id, limit)).fetchall()
    return rows
