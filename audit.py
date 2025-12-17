# audit.py
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

DB_PATH = Path("data") / "mcecn_audit.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_audit_db() -> None:
    with _conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_epoch INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            actor_role TEXT,
            case_id TEXT,
            facility_id TEXT,
            status_from TEXT,
            status_to TEXT,
            decision_mode TEXT,
            payload_json TEXT
        )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_case_id ON audit_events(case_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
        con.commit()

def log_event(
    event_type: str,
    actor_role: str = "system",
    case_id: Optional[str] = None,
    facility_id: Optional[str] = None,
    status_from: Optional[str] = None,
    status_to: Optional[str] = None,
    decision_mode: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    payload = payload or {}
    now = int(time.time())
    with _conn() as con:
        con.execute("""
        INSERT INTO audit_events
        (ts_epoch, event_type, actor_role, case_id, facility_id, status_from, status_to, decision_mode, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now, event_type, actor_role, case_id, facility_id,
            status_from, status_to, decision_mode,
            json.dumps(payload, ensure_ascii=False)
        ))
        con.commit()

def read_events(case_id: Optional[str] = None) -> list[dict]:
    q = """
    SELECT ts_epoch, event_type, actor_role, case_id, facility_id,
           status_from, status_to, decision_mode, payload_json
    FROM audit_events
    """
    params = []
    if case_id:
        q += " WHERE case_id = ?"
        params.append(case_id)
    q += " ORDER BY ts_epoch ASC"

    with _conn() as con:
        rows = con.execute(q, params).fetchall()

    out = []
    for r in rows:
        out.append({
            "ts_epoch": r[0],
            "event_type": r[1],
            "actor_role": r[2],
            "case_id": r[3],
            "facility_id": r[4],
            "status_from": r[5],
            "status_to": r[6],
            "decision_mode": r[7],
            "payload": json.loads(r[8] or "{}"),
        })
    return out
