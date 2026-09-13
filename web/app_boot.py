"""Import side-effects before app: auth aliases."""
try:
    from web import auth_compat  # noqa: F401
except Exception as e:
    print(f"[app_boot] auth_compat: {e}")
