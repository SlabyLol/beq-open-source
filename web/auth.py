"""
Beq Auth + Token Quotas + API Keys + Device Trace
Postgres via DATABASE_URL (Neon) or SQLite fallback.

Limits:
  - Normal users: WEEKLY_TOKEN_LIMIT tokens/week, max MAX_API_KEYS API keys
  - Admin/owner: unlimited tokens, unlimited API keys
  - token_limit NULL = global default, -1 = inf, N = custom
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

WEEKLY_TOKEN_LIMIT = 5000
MAX_API_KEYS = 10
ADMIN_USERNAME = "admin"
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "beq_users.db"
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
UNLIMITED = -1


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
                token_limit INTEGER,
                week_start TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS token_limit INTEGER")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                expires_at TEXT NOT NULL
            )""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trace_links (
                trace_id TEXT NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id),
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                PRIMARY KEY (trace_id, user_id)
            )""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                key_hash TEXT UNIQUE NOT NULL,
                key_prefix TEXT NOT NULL,
                name TEXT DEFAULT 'default',
                created_at TEXT NOT NULL,
                last_used TEXT,
                revoked INTEGER DEFAULT 0
            )""")
        conn.commit()
        cur.execute("SELECT id FROM users WHERE username = %s", (ADMIN_USERNAME,))
        if cur.fetchone() is None:
            now = _now()
            cur.execute(
                "INSERT INTO users (username, email, password_hash, is_admin, tokens_used, token_limit, week_start, created_at) "
                "VALUES (%s,%s,%s,1,0,%s,%s,%s)",
                (ADMIN_USERNAME, "admin@beq.local", _hash("admin123"), UNLIMITED, now, now),
            )
            conn.commit()
            print("Created default admin: admin / admin123 (unlimited) — CHANGE IT!")
        else:
            cur.execute(
                "UPDATE users SET is_admin = 1, token_limit = %s WHERE username = %s",
                (UNLIMITED, ADMIN_USERNAME),
            )
            conn.commit()
        cur.close()
        conn.close()
        print("Auth DB: Postgres/Neon (DATABASE_URL)")
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
                token_limit INTEGER,
                week_start TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
        try:
            c.execute("ALTER TABLE users ADD COLUMN token_limit INTEGER")
        except sqlite3.OperationalError:
            pass
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS trace_links (
                trace_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                PRIMARY KEY (trace_id, user_id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                key_prefix TEXT NOT NULL,
                name TEXT DEFAULT 'default',
                created_at TEXT NOT NULL,
                last_used TEXT,
                revoked INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )""")
        c.commit()
        row = c.execute("SELECT id FROM users WHERE username = ?", (ADMIN_USERNAME,)).fetchone()
        if not row:
            now = _now()
            c.execute(
                "INSERT INTO users (username, email, password_hash, is_admin, tokens_used, token_limit, week_start, created_at) "
                "VALUES (?,?,?,1,0,?,?,?)",
                (ADMIN_USERNAME, "admin@beq.local", _hash("admin123"), UNLIMITED, now, now),
            )
            c.commit()
            print("Created default admin: admin / admin123 (unlimited) — CHANGE IT!")
        else:
            c.execute(
                "UPDATE users SET is_admin = 1, token_limit = ? WHERE username = ?",
                (UNLIMITED, ADMIN_USERNAME),
            )
            c.commit()
    print("Auth DB: SQLite (set DATABASE_URL for Neon)")


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
                "INSERT INTO users (username, email, password_hash, is_admin, tokens_used, token_limit, week_start, created_at) "
                "VALUES (%s,%s,%s,0,0,NULL,%s,%s)",
                (username, email, _hash(password), _week_start_iso(), _now()),
            )
            conn.commit()
            cur.close()
            conn.close()
        else:
            with _sqlite_conn() as c:
                c.execute(
                    "INSERT INTO users (username, email, password_hash, is_admin, tokens_used, token_limit, week_start, created_at) "
                    "VALUES (?,?,?,0,0,NULL,?,?)",
                    (username, email, _hash(password), _week_start_iso(), _now()),
                )
                c.commit()
        return True, "Account created"
    except Exception as e:
        if "unique" in str(e).lower() or "integrity" in str(e).lower():
            return False, "Username or email already exists"
        return False, str(e)


def link_trace(trace_id: str | None, user_id: int) -> None:
    if not trace_id:
        return
    now = _now()
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO trace_links (trace_id, user_id, first_seen, last_seen)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (trace_id, user_id) DO UPDATE SET last_seen = EXCLUDED.last_seen""",
            (trace_id, user_id, now, now),
        )
        conn.commit()
        cur.close()
        conn.close()
        return
    with _sqlite_conn() as c:
        c.execute(
            """INSERT INTO trace_links (trace_id, user_id, first_seen, last_seen)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(trace_id, user_id) DO UPDATE SET last_seen = excluded.last_seen""",
            (trace_id, user_id, now, now),
        )
        c.commit()


def accounts_for_trace(trace_id: str | None) -> list[str]:
    if not trace_id:
        return []
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            """SELECT u.username FROM trace_links t JOIN users u ON u.id = t.user_id
               WHERE t.trace_id = %s ORDER BY t.last_seen DESC""",
            (trace_id,),
        )
        rows = [r[0] for r in cur.fetchall()]
        cur.close()
        conn.close()
        return rows
    with _sqlite_conn() as c:
        rows = c.execute(
            """SELECT u.username FROM trace_links t JOIN users u ON u.id = t.user_id
               WHERE t.trace_id = ? ORDER BY t.last_seen DESC""",
            (trace_id,),
        ).fetchall()
        return [r["username"] for r in rows]


def login(username_or_email: str, password: str, trace_id: str | None = None) -> tuple[bool, str | None, str]:
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
        user_id = row[0]
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(days=14)).isoformat()
        cur.execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (%s,%s,%s)", (token, user_id, expires))
        conn.commit()
        cur.close()
        conn.close()
        link_trace(trace_id, user_id)
        return True, token, "OK"
    with _sqlite_conn() as c:
        row = c.execute("SELECT * FROM users WHERE username = ? OR email = ?", (key, key)).fetchone()
        if not row or row["password_hash"] != pw:
            return False, None, "Invalid login"
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(days=14)).isoformat()
        c.execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (?,?,?)", (token, row["id"], expires))
        c.commit()
        uid = row["id"]
    link_trace(trace_id, uid)
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
            """SELECT u.id, u.username, u.email, u.is_admin, u.tokens_used, u.token_limit, u.week_start
               FROM users u JOIN sessions s ON s.user_id = u.id
               WHERE s.token = %s AND s.expires_at > %s""",
            (session_token, _now()),
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            return None
        user = {"id": row[0], "username": row[1], "email": row[2], "is_admin": row[3],
                "tokens_used": row[4], "token_limit": row[5], "week_start": row[6]}
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


def get_user_by_id(user_id: int) -> dict | None:
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, email, is_admin, tokens_used, token_limit, week_start FROM users WHERE id = %s",
            (user_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return None
        return {"id": row[0], "username": row[1], "email": row[2], "is_admin": row[3],
                "tokens_used": row[4], "token_limit": row[5], "week_start": row[6]}
    with _sqlite_conn() as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def _effective_limit(user: dict) -> int | None:
    if user.get("is_admin"):
        return None
    limit = user.get("token_limit")
    if limit is None:
        return WEEKLY_TOKEN_LIMIT
    limit = int(limit)
    if limit == UNLIMITED:
        return None
    return limit


def tokens_remaining(user: dict) -> int | None:
    limit = _effective_limit(user)
    if limit is None:
        return None
    return max(0, limit - int(user.get("tokens_used", 0)))


def can_use_tokens(user: dict, amount: int) -> tuple[bool, str]:
    limit = _effective_limit(user)
    if limit is None:
        return True, "unlimited"
    left = tokens_remaining(user)
    if left is None:
        return True, "unlimited"
    if amount > left:
        return False, f"Weekly limit reached ({limit}). Remaining: {left}"
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


def list_users() -> list[dict]:
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            """SELECT id, username, email, is_admin, tokens_used, token_limit, created_at
               FROM users ORDER BY created_at ASC"""
        )
        cols = ["id", "username", "email", "is_admin", "tokens_used", "token_limit", "created_at"]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.close()
        conn.close()
        return rows
    with _sqlite_conn() as c:
        rows = c.execute(
            """SELECT id, username, email, is_admin, tokens_used, token_limit, created_at
               FROM users ORDER BY created_at ASC"""
        ).fetchall()
        return [dict(r) for r in rows]


def set_token_limit(user_id: int, limit_str: str) -> tuple[bool, str]:
    limit_str = limit_str.strip().lower()
    if limit_str in ("inf", "infinite", "unlimited", "∞"):
        value = UNLIMITED
    else:
        try:
            value = int(limit_str)
            if value < 0:
                return False, "Limit must be 0 or higher (or 'inf')."
        except ValueError:
            return False, "Enter a whole number or 'inf'."
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute("UPDATE users SET token_limit = %s WHERE id = %s", (value, user_id))
        conn.commit()
        cur.close()
        conn.close()
    else:
        with _sqlite_conn() as c:
            c.execute("UPDATE users SET token_limit = ? WHERE id = ?", (value, user_id))
            c.commit()
    label = "unlimited" if value == UNLIMITED else str(value)
    return True, f"Token limit set to {label}"


def reset_usage(user_id: int) -> None:
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute("UPDATE users SET tokens_used = 0 WHERE id = %s", (user_id,))
        conn.commit()
        cur.close()
        conn.close()
        return
    with _sqlite_conn() as c:
        c.execute("UPDATE users SET tokens_used = 0 WHERE id = ?", (user_id,))
        c.commit()


def set_admin(user_id: int, is_admin: bool) -> tuple[bool, str]:
    """Promote or demote a user. Built-in admin account cannot be demoted."""
    target = get_user_by_id(user_id)
    if not target:
        return False, "User not found"
    if target.get("username") == ADMIN_USERNAME and not is_admin:
        return False, "Refusing to demote the built-in admin account"
    flag = 1 if is_admin else 0
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute("UPDATE users SET is_admin = %s WHERE id = %s", (flag, user_id))
        if is_admin:
            cur.execute("UPDATE users SET token_limit = %s WHERE id = %s", (UNLIMITED, user_id))
        conn.commit()
        cur.close()
        conn.close()
    else:
        with _sqlite_conn() as c:
            c.execute("UPDATE users SET is_admin = ? WHERE id = ?", (flag, user_id))
            if is_admin:
                c.execute("UPDATE users SET token_limit = ? WHERE id = ?", (UNLIMITED, user_id))
            c.commit()
    return True, f"{'Admin enabled' if is_admin else 'Admin removed'} for {target.get('username')}"


def delete_user(user_id: int) -> tuple[bool, str]:
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            return False, "User not found"
        if row[0] == ADMIN_USERNAME:
            cur.close()
            conn.close()
            return False, "Refusing to delete the built-in admin account"
        cur.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM trace_links WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM api_keys WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        cur.close()
        conn.close()
        return True, "User deleted"
    with _sqlite_conn() as c:
        row = c.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return False, "User not found"
        if row["username"] == ADMIN_USERNAME:
            return False, "Refusing to delete the built-in admin account"
        c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM trace_links WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM api_keys WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        c.commit()
        return True, "User deleted"


def _hash_api_key(raw: str) -> str:
    return hashlib.sha256(f"beq-api:{raw}".encode()).hexdigest()


def create_api_key(user: dict, name: str = "default") -> tuple[bool, str, str | None]:
    is_admin = bool(user.get("is_admin"))
    if not is_admin:
        keys = list_api_keys(user["id"])
        active = [k for k in keys if not k.get("revoked")]
        if len(active) >= MAX_API_KEYS:
            return False, f"Max {MAX_API_KEYS} API keys. Revoke one first.", None
    raw = "beq_" + secrets.token_urlsafe(32)
    key_hash = _hash_api_key(raw)
    prefix = raw[:10]
    now = _now()
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO api_keys (user_id, key_hash, key_prefix, name, created_at, revoked) VALUES (%s,%s,%s,%s,%s,0)",
            (user["id"], key_hash, prefix, name[:64], now),
        )
        conn.commit()
        cur.close()
        conn.close()
    else:
        with _sqlite_conn() as c:
            c.execute(
                "INSERT INTO api_keys (user_id, key_hash, key_prefix, name, created_at, revoked) VALUES (?,?,?,?,?,0)",
                (user["id"], key_hash, prefix, name[:64], now),
            )
            c.commit()
    return True, "API key created — copy it now, it will not be shown again.", raw


def list_api_keys(user_id: int) -> list[dict]:
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, key_prefix, name, created_at, last_used, revoked FROM api_keys WHERE user_id = %s ORDER BY id",
            (user_id,),
        )
        cols = ["id", "key_prefix", "name", "created_at", "last_used", "revoked"]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.close()
        conn.close()
        return rows
    with _sqlite_conn() as c:
        rows = c.execute(
            "SELECT id, key_prefix, name, created_at, last_used, revoked FROM api_keys WHERE user_id = ? ORDER BY id",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def revoke_api_key(user_id: int, key_id: int) -> tuple[bool, str]:
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute("UPDATE api_keys SET revoked = 1 WHERE id = %s AND user_id = %s", (key_id, user_id))
        n = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return (True, "Revoked") if n else (False, "Key not found")
    with _sqlite_conn() as c:
        cur = c.execute("UPDATE api_keys SET revoked = 1 WHERE id = ? AND user_id = ?", (key_id, user_id))
        c.commit()
        return (True, "Revoked") if cur.rowcount else (False, "Key not found")


def user_from_api_key(raw_key: str | None) -> dict | None:
    if not raw_key:
        return None
    raw_key = raw_key.strip()
    if raw_key.lower().startswith("bearer "):
        raw_key = raw_key[7:].strip()
    h = _hash_api_key(raw_key)
    if _is_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            """SELECT u.id, u.username, u.email, u.is_admin, u.tokens_used, u.token_limit, u.week_start, k.id
               FROM api_keys k JOIN users u ON u.id = k.user_id
               WHERE k.key_hash = %s AND k.revoked = 0""",
            (h,),
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            return None
        cur.execute("UPDATE api_keys SET last_used = %s WHERE id = %s", (_now(), row[7]))
        conn.commit()
        user = {"id": row[0], "username": row[1], "email": row[2], "is_admin": row[3],
                "tokens_used": row[4], "token_limit": row[5], "week_start": row[6]}
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
            """SELECT u.*, k.id as key_id FROM api_keys k JOIN users u ON u.id = k.user_id
               WHERE k.key_hash = ? AND k.revoked = 0""",
            (h,),
        ).fetchone()
        if not row:
            return None
        c.execute("UPDATE api_keys SET last_used = ? WHERE id = ?", (_now(), row["key_id"]))
        c.commit()
        user = dict(row)
        user.pop("key_id", None)
        current_week = _week_start_iso()
        if user["week_start"][:10] != current_week[:10]:
            c.execute("UPDATE users SET tokens_used = 0, week_start = ? WHERE id = ?", (current_week, user["id"]))
            c.commit()
            user["tokens_used"] = 0
            user["week_start"] = current_week
        return user
