# storage.py
import sqlite3
import json
import time
from typing import Optional, Dict, Any, List
import streamlit as st

DB_PATH = "ahecn_realtime.db"

@st.cache_resource
def get_db() -> sqlite3.Connection:
    """
    Returns a cached SQLite connection and ensures required tables exist.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS presence(
            user TEXT PRIMARY KEY,
            role TEXT,
            last_seen REAL,
            editing_case TEXT
        );

        CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            type TEXT,
            case_id TEXT,
            actor TEXT,
            audience TEXT,
            payload TEXT
        );

        CREATE TABLE IF NOT EXISTS locks(
            case_id TEXT PRIMARY KEY,
            holder TEXT,
            expires REAL
        );

        CREATE TABLE IF NOT EXISTS notifications(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            user TEXT,
            title TEXT,
            body TEXT,
            read INTEGER
        );
        """
    )
    conn.commit()
    return conn

# ---------- Low-level (DB-explicit) API ----------

def publish_event_db(
    conn: sqlite3.Connection,
    etype: str,
    case_id: str,
    actor: str,
    audience: str = "all",
    payload: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Insert an event using an explicit DB connection.
    Returns last inserted event id.
    """
    cur = conn.execute(
        "INSERT INTO events(ts,type,case_id,actor,audience,payload) VALUES(?,?,?,?,?,?)",
        (time.time(), etype, case_id, actor, audience, json.dumps(payload or {})),
    )
    conn.commit()
    return cur.lastrowid

def poll_events(
    conn: sqlite3.Connection,
    last_id: int = 0,
    audience: str = "all",
    user: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fetch events newer than last_id. If user is provided, returns
    events where audience is 'all' or the specific user.
    """
    if user:
        rows = conn.execute(
            """
            SELECT id,ts,type,case_id,actor,audience,payload
            FROM events
            WHERE id > ? AND (audience = ? OR audience = ?)
            ORDER BY id ASC
            LIMIT ?
            """,
            (last_id, "all", user, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id,ts,type,case_id,actor,audience,payload
            FROM events
            WHERE id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (last_id, limit),
        ).fetchall()

    return [
        {
            "id": r["id"],
            "ts": r["ts"],
            "type": r["type"],
            "case_id": r["case_id"],
            "actor": r["actor"],
            "audience": r["audience"],
            "payload": json.loads(r["payload"] or "{}"),
        }
        for r in rows
    ]

# ---------- App-friendly (connectionless) API ----------

def publish_event(event: Dict[str, Any]) -> int:
    """
    Convenience wrapper that matches the app’s expected signature:
    publish_event({...}) with keys: type, case_id, actor, audience?, payload?
    """
    conn = get_db()
    return publish_event_db(
        conn=conn,
        etype=event.get("type", "event"),
        case_id=event.get("case_id", ""),
        actor=event.get("actor", "system"),
        audience=event.get("audience", "all"),
        payload=event.get("payload") or {},
    )

def list_events(since_id: int = 0, limit: int = 200) -> List[Dict[str, Any]]:
    """
    Convenience method to read recent events without wiring a poller.
    """
    conn = get_db()
    rows = conn.execute(
        """
        SELECT id,ts,type,case_id,actor,audience,payload
        FROM events
        WHERE id > ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (since_id, limit),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "ts": r["ts"],
            "type": r["type"],
            "case_id": r["case_id"],
            "actor": r["actor"],
            "audience": r["audience"],
            "payload": json.loads(r["payload"] or "{}"),
        }
        for r in rows
    ]
