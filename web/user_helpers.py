"""Safe user lookup for Beq web app."""
from __future__ import annotations

from web import auth

try:
    from web import auth_compat  # noqa: F401
except Exception:
    pass


def session_token(request) -> str | None:
    try:
        return request.cookies.get("beq_session")
    except Exception:
        return None


def user_from_request(request):
    token = session_token(request)
    fn = getattr(auth, "user_from_session", None) or getattr(auth, "get_user", None)
    if not fn:
        return None
    try:
        return fn(token)
    except Exception as e:
        print(f"[user_helpers] {e}")
        return None
