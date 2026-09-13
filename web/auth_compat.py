"""Compatibility layer for web.auth used by app.py."""
from web import auth

if not hasattr(auth, "user_from_session"):
    auth.user_from_session = auth.get_user

if not hasattr(auth, "check_and_consume"):

    def check_and_consume(user_id: int, amount: int):
        user = auth.get_user_by_id(user_id)
        if not user:
            return False, "User not found"
        ok, msg = auth.can_use_tokens(user, amount)
        if not ok:
            return False, msg
        auth.consume_tokens(user_id, amount)
        return True, "ok"

    auth.check_and_consume = check_and_consume

if not hasattr(auth, "reset_tokens"):
    auth.reset_tokens = auth.reset_usage

if not hasattr(auth, "delete_api_key"):
    auth.delete_api_key = auth.revoke_api_key

_orig = auth.create_api_key


def create_api_key(user_or_id, name: str = "default"):
    if isinstance(user_or_id, int):
        user = auth.get_user_by_id(user_or_id)
        if not user:
            return False, "User not found", None
        return _orig(user, name)
    return _orig(user_or_id, name)


auth.create_api_key = create_api_key
