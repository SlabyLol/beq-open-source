"""
Beq Settings Store
===================
Why this exists: on Render (and most free hosts), the app's filesystem is
EPHEMERAL — every redeploy / restart wipes it and re-copies the repo from
git. To make settings (like the AI mode) actually stick, this module stores
them in the same database web/auth.py already uses:
  - Postgres, if you set DATABASE_URL (Render Postgres, Neon, Supabase, ...)
  - SQLite file locally otherwise (fine for local dev only).
"""
from __future__ import annotations
import os, sqlite3
from pathlib import Path

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "beq_users.db"

def _is_postgres() -> bool:
    return DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://")

def _pg_conn():
    import psycopg2
    url = DATABASE_URL
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return psycopg2.connect(url)

def _sqlite_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_settings_table():
    if _is_postgres():
        conn = _pg_conn(); cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.commit(); cur.close(); conn.close(); return
    with _sqlite_conn() as c:
        c.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        c.commit()

def get_setting(key: str, default: str | None = None) -> str | None:
    if _is_postgres():
        conn = _pg_conn(); cur = conn.cursor()
        cur.execute("SELECT value FROM settings WHERE key = %s", (key,))
        row = cur.fetchone(); cur.close(); conn.close()
        return row[0] if row else default
    with _sqlite_conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

def set_setting(key: str, value: str) -> None:
    if _is_postgres():
        conn = _pg_conn(); cur = conn.cursor()
        cur.execute(
            "INSERT INTO settings (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (key, value))
        conn.commit(); cur.close(); conn.close(); return
    with _sqlite_conn() as c:
        c.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
        c.commit()

def using_postgres() -> bool:
    return _is_postgres()
