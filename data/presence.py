import time
from storage import get_db

def heartbeat(user, role, editing_case=None):
    conn = get_db()
    conn.execute("""
        INSERT INTO presence(user,role,last_seen,editing_case)
        VALUES(?,?,?,?)
        ON CONFLICT(user)
        DO UPDATE SET role=excluded.role, last_seen=excluded.last_seen, editing_case=excluded.editing_case
    """, (user, role, time.time(), editing_case))
    conn.commit()

def online_users(conn, window=12):
    cutoff = time.time() - window
    return conn.execute("SELECT user,role,editing_case FROM presence WHERE last_seen>?", (cutoff,)).fetchall()

def acquire_lock(conn, case_id, user, ttl=90):
    expires = time.time() + ttl
    row = conn.execute("SELECT holder,expires FROM locks WHERE case_id=?", (case_id,)).fetchone()
    if not row or row[1] < time.time():
        conn.execute("REPLACE INTO locks(case_id,holder,expires) VALUES(?,?,?)",
                     (case_id, user, expires))
        conn.commit()
        return True
    return row[0] == user

def release_lock(conn, case_id, user):
    conn = get_db()
    conn.execute("DELETE FROM locks WHERE case_id=? AND holder=?", (case_id, user))
    conn.commit()
