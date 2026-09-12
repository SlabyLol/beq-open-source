import os
from fastapi import Header, HTTPException
from web import ai_mode, auth, settings_store

BEQ_API_KEY = os.environ.get("BEQ_API_KEY", "").strip()

def _check_api_key(authorization: str | None) -> None:
    if not BEQ_API_KEY:
        raise HTTPException(status_code=503, detail="BEQ_API_KEY is not configured on this server.")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing 'Authorization: Bearer <key>' header.")
    if authorization.removeprefix("Bearer ").strip() != BEQ_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")

@app.get("/api/status")
async def api_status():
    mode_cfg = ai_mode.get_mode_config()
    return {
        "model_loaded": model is not None,
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "mode": mode_cfg["key"], "mode_label": mode_cfg["label"],
        "public_chat": mode_cfg["public_chat"],
        "persistence": "database" if settings_store.using_postgres() else "sqlite/file (not persistent on most free hosts)",
    }

@app.get("/api/config")
async def api_get_config(authorization: str | None = Header(default=None)):
    _check_api_key(authorization)
    mode_cfg = ai_mode.get_mode_config()
    return {
        "mode": mode_cfg["key"],
        "available_modes": {k: {"label": v["label"], "description": v["description"]} for k, v in ai_mode.MODES.items()},
        "model_loaded": model is not None,
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "weekly_token_limit": auth.WEEKLY_TOKEN_LIMIT,
    }

class ConfigUpdate(BaseModel):
    mode: str

@app.post("/api/config")
async def api_set_config(update: ConfigUpdate, authorization: str | None = Header(default=None)):
    _check_api_key(authorization)
    ok, msg = ai_mode.set_mode(update.mode)
    if not ok:
        return JSONResponse({"error": msg}, status_code=400)
    return {"ok": True, "mode": ai_mode.get_mode_key(), "message": msg}
