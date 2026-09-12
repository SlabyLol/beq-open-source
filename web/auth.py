"""
Beq Auth + Token Quotas
Supports SQLite (default) or Postgres via DATABASE_URL (Neon / Supabase / Render).
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

WEEKLY_TOKEN_LIMIT = 5000
ADMIN_USERNAME = "admin"
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "beq_users.db"
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()


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
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(password: str) -> str:
    return hashlib.sha256(f"beq-v1:{password}".encode()).hexdigest()


def _week_start_iso() -> str:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=now.weekday())
    return start.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def init_db():
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                tokens_used INTEGER DEFAULT 0,
                week_start TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                expires_at TEXT NOT NULL
            )""")
        conn.commit()
        cur.execute("SELECT id FROM users WHERE username = %s", (ADMIN_USERNAME,))
        if cur.fetchone() is None:
            now = _now()
            cur.execute(
                "INSERT INTO users (username, email, password_hash, is_admin, tokens_used, week_start, created_at) VALUES (%s,%s,%s,1,0,%s,%s)",
                (ADMIN_USERNAME, "admin@beq.local", _hash("admin123"), now, now),
            )
            conn.commit()
            print("Created default admin: admin / admin123 — CHANGE IT!")
        cur.close()
        conn.close()
        print("Auth DB: Postgres (DATABASE_URL)")
        return

    with _sqlite_conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                tokens_used INTEGER DEFAULT 0,
                week_start TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )""")
        c.commit()
        row = c.execute("SELECT id FROM users WHERE username = ?", (ADMIN_USERNAME,)).fetchone()
        if not row:
            now = _now()
            c.execute(
                "INSERT INTO users (username, email, password_hash, is_admin, tokens_used, week_start, created_at) VALUES (?,?,?,1,0,?,?)",
                (ADMIN_USERNAME, "admin@beq.local", _hash("admin123"), now, now),
            )
            c.commit()
            print("Created default admin: admin / admin123 — CHANGE IT!")
    print("Auth DB: SQLite (set DATABASE_URL for Neon/Supabase)")


def register(username: str, email: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    email = email.strip().lower()
    if len(username) < 3:
        return False, "Username too short (min 3)"
    if "@" not in email or "." not in email:
        return False, "Invalid email"
    if len(password) < 6:
        return False, "Password too short (min 6)"
    try:
        if _is_postgres():
            conn = _pg_conn()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (username, email, password_hash, is_admin, tokens_used, week_start, created_at) VALUES (%s,%s,%s,0,0,%s,%s)",
                (username, email, _hash(password), _week_start_iso(), _now()),
            )
            conn.commit()
            cur.close()
            conn.close()
        else:
            with _sqlite_conn() as c:
                c.execute(
                    "INSERT INTO users (username, email, password_hash, is_admin, tokens_used, week_start, created_at) VALUES (?,?,?,0,0,?,?)",
                    (username, email, _hash(password), _week_start_iso(), _now()),
                )
                c.commit()
        return True, "Account created"
    except Exception as e:
        if "unique" in str(e).lower() or "integrity" in str(e).lower():
            return False, "Username or email already exists"
        return False, str(e)


def login(username_or_email: str, password: str) -> tuple[bool, str | None, str]:
    key = username_or_email.strip().lower()
    pw = _hash(password)
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, password_hash FROM users WHERE username = %s OR email = %s", (key, key))
        row = cur.fetchone()
        if not row or row[1] != pw:
            cur.close()
            conn.close()
            return False, None, "Invalid login"
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(days=14)).isoformat()
        cur.execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (%s,%s,%s)", (token, row[0], expires))
        conn.commit()
        cur.close()
        conn.close()
        return True, token, "OK"
    with _sqlite_conn() as c:
        row = c.execute("SELECT * FROM users WHERE username = ? OR email = ?", (key, key)).fetchone()
        if not row or row["password_hash"] != pw:
            return False, None, "Invalid login"
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(days=14)).isoformat()
        c.execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (?,?,?)", (token, row["id"], expires))
        c.commit()
        return True, token, "OK"


def logout(session_token: str | None):
    if not session_token:
        return
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM sessions WHERE token = %s", (session_token,))
        conn.commit()
        cur.close()
        conn.close()
        return
    with _sqlite_conn() as c:
        c.execute("DELETE FROM sessions WHERE token = ?", (session_token,))
        c.commit()


def get_user(session_token: str | None) -> dict | None:
    if not session_token:
        return None
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            """SELECT u.id, u.username, u.email, u.is_admin, u.tokens_used, u.week_start
               FROM users u JOIN sessions s ON s.user_id = u.id
               WHERE s.token = %s AND s.expires_at > %s""",
            (session_token, _now()),
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            return None
        user = {"id": row[0], "username": row[1], "email": row[2], "is_admin": row[3], "tokens_used": row[4], "week_start": row[5]}
        current_week = _week_start_iso()
        if user["week_start"][:10] != current_week[:10]:
            cur.execute("UPDATE users SET tokens_used = 0, week_start = %s WHERE id = %s", (current_week, user["id"]))
            conn.commit()
            user["tokens_used"] = 0
            user["week_start"] = current_week
        cur.close()
        conn.close()
        return user
    with _sqlite_conn() as c:
        row = c.execute(
            """SELECT u.* FROM users u JOIN sessions s ON s.user_id = u.id
               WHERE s.token = ? AND s.expires_at > ?""",
            (session_token, _now()),
        ).fetchone()
        if not row:
            return None
        user = dict(row)
        current_week = _week_start_iso()
        if user["week_start"][:10] != current_week[:10]:
            c.execute("UPDATE users SET tokens_used = 0, week_start = ? WHERE id = ?", (current_week, user["id"]))
            c.commit()
            user["tokens_used"] = 0
            user["week_start"] = current_week
        return user


def tokens_remaining(user: dict) -> int | None:
    if user.get("is_admin"):
        return None
    return max(0, WEEKLY_TOKEN_LIMIT - int(user.get("tokens_used", 0)))


def can_use_tokens(user: dict, amount: int) -> tuple[bool, str]:
    if user.get("is_admin"):
        return True, "admin"
    left = tokens_remaining(user)
    if left is not None and amount > left:
        return False, f"Weekly limit reached ({WEEKLY_TOKEN_LIMIT}). Remaining: {left}"
    return True, "ok"


def consume_tokens(user_id: int, amount: int):
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute("UPDATE users SET tokens_used = tokens_used + %s WHERE id = %s", (amount, user_id))
        conn.commit()
        cur.close()
        conn.close()
        return
    with _sqlite_conn() as c:
        c.execute("UPDATE users SET tokens_used = tokens_used + ? WHERE id = ?", (amount, user_id))
        c.commit()
