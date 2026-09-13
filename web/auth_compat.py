"""Compatibility layer for web.auth used by app.py."""
from web import auth

# Session user
if not hasattr(auth, "user_from_session"):
    auth.user_from_session = getattr(auth, "get_user", lambda *_a, **_k: None)

if not hasattr(auth, "get_user") and hasattr(auth, "user_from_session"):
    auth.get_user = auth.user_from_session

# Token helpers
if not hasattr(auth, "check_and_consume"):

    def check_and_consume(user_id: int, amount: int):
        getter = getattr(auth, "get_user_by_id", None)
        user = getter(user_id) if getter else None
        if not user:
            return False, "User not found"
        can = getattr(auth, "can_use_tokens", None)
        if can:
            ok, msg = can(user, amount)
            if not ok:
                return False, msg
        consume = getattr(auth, "consume_tokens", None)
        if consume:
            consume(user_id, amount)
        return True, "ok"

    auth.check_and_consume = check_and_consume

if not hasattr(auth, "reset_tokens") and hasattr(auth, "reset_usage"):
    auth.reset_tokens = auth.reset_usage

if not hasattr(auth, "delete_api_key") and hasattr(auth, "revoke_api_key"):
    auth.delete_api_key = auth.revoke_api_key

_orig = getattr(auth, "create_api_key", None)
if _orig is not None:

    def create_api_key(user_or_id, name: str = "default"):
        if isinstance(user_or_id, int):
            getter = getattr(auth, "get_user_by_id", None)
            user = getter(user_or_id) if getter else None
            if not user:
                return False, "User not found", None
            return _orig(user, name)
        return _orig(user_or_id, name)

    auth.create_api_key = create_api_key
