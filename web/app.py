"""
Beq Web – Auth, quotas, own model, AI.txt mode switching, user API keys
"""

import os
import secrets
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from fastapi import FastAPI, Request, Form, Header, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from model import BeqTransformer, CharTokenizer
from web import ai_mode, auth, settings_store
from web.mathtool import try_math_answer
from web.knowledge import try_knowledge_answer

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_LOG_PATH = REPO_ROOT / "checkpoints" / "train.log"
_training_state = {"process": None}

BEQ_API_KEY = os.environ.get("BEQ_API_KEY", "").strip()

CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "beq_best.pt"
TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "tokenizer.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LANGUAGE_PREFIXES = {
    "en": "",
    "de": "Antworte auf Deutsch.\n",
    "fr": "Réponds en français.\n",
    "es": "Responde en español.\n",
    "it": "Rispondi in italiano.\n",
    "pt": "Responda em português.\n",
    "nl": "Antwoord in het Nederlands.\n",
    "pl": "Odpowiedz po polsku.\n",
    "ru": "Ответь на русском.\n",
    "ja": "日本語で答えてください。\n",
    "zh": "请用中文回答。\n",
    "ko": "한국어로 답하세요.\n",
    "tr": "Türkçe cevap ver.\n",
    "ar": "أجب بالعربية.\n",
}

app = FastAPI(title="Beq")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

model = None
tokenizer = None


def load_model():
    global model, tokenizer
    if not CHECKPOINT_PATH.exists() or not TOKENIZER_PATH.exists():
        print("No checkpoint – train first")
        return False
    tokenizer = CharTokenizer.load(TOKENIZER_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    model = BeqTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Beq loaded | params={sum(p.numel() for p in model.parameters()):,}")
    return True


TRACE_COOKIE = "beq_trace"
TRACE_MAX_AGE = 400 * 24 * 3600


@app.middleware("http")
async def ensure_trace_cookie(request: Request, call_next):
    trace_id = request.cookies.get(TRACE_COOKIE)
    new_trace = trace_id is None
    if new_trace:
        trace_id = secrets.token_hex(16)
    request.state.trace_id = trace_id
    response = await call_next(request)
    if new_trace:
        response.set_cookie(
            TRACE_COOKIE, trace_id,
            max_age=TRACE_MAX_AGE, httponly=True, samesite="lax",
        )
    return response


@app.on_event("startup")
async def startup():
    auth.init_db()
    try:
        settings_store.init_settings_table()
    except Exception as e:
        print(f"[startup] settings DB not reachable yet ({e})")
    load_model()
    if not BEQ_API_KEY:
        print("[startup] WARNING: BEQ_API_KEY is not set — /api/config write disabled.")


def _session(request: Request):
    return request.cookies.get("beq_session")


def _user(request: Request):
    return auth.get_user(_session(request))


def _user_from_request(request: Request, authorization: str | None = None):
    u = _user(request)
    if u:
        return u
    auth_header = authorization or request.headers.get("authorization")
    return auth.user_from_api_key(auth_header)


def _ctx(request: Request, **extra):
    user = _user(request)
    remaining = auth.tokens_remaining(user) if user else None
    mode_cfg = ai_mode.get_mode_config()
    base = {
        "request": request,
        "user": user,
        "model_loaded": model is not None,
        "tokens_remaining": remaining,
        "weekly_limit": auth.WEEKLY_TOKEN_LIMIT,
        "ai_mode": mode_cfg,
    }
    base.update(extra)
    return base


def _is_admin(user: dict | None) -> bool:
    return bool(user and user.get("is_admin"))


def _api_dashboard_ctx(request: Request, **extra):
    user = _user(request)
    keys = auth.list_api_keys(user["id"]) if user else []
    active_count = sum(1 for k in keys if not k.get("revoked"))
    max_keys = None if (user and user.get("is_admin")) else auth.MAX_API_KEYS
    return _ctx(
        request,
        keys=keys,
        active_count=active_count,
        max_keys=max_keys,
        **extra,
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context=_ctx(request))


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(request=request, name="about.html", context=_ctx(request))


@app.get("/generate", response_class=HTMLResponse)
async def generate_page(request: Request):
    return templates.TemplateResponse(request=request, name="generate.html", context=_ctx(request))


@app.get("/api-dashboard", response_class=HTMLResponse)
async def api_dashboard(request: Request):
    user = _user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request=request, name="api.html", context=_api_dashboard_ctx(request),
    )


@app.post("/api-dashboard/create")
async def api_dashboard_create(request: Request, name: str = Form("default")):
    user = _user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    ok, msg, raw = auth.create_api_key(user, name=name)
    if not ok:
        return templates.TemplateResponse(
            request=request, name="api.html", context=_api_dashboard_ctx(request, flash_error=msg),
        )
    return templates.TemplateResponse(
        request=request, name="api.html", context=_api_dashboard_ctx(request, flash_success=msg, new_key=raw),
    )


@app.post("/api-dashboard/revoke/{key_id}")
async def api_dashboard_revoke(request: Request, key_id: int):
    user = _user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    ok, msg = auth.revoke_api_key(user["id"], key_id)
    return templates.TemplateResponse(
        request=request,
        name="api.html",
        context=_api_dashboard_ctx(
            request,
            flash_success=msg if ok else None,
            flash_error=None if ok else msg,
        ),
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="login.html", context=_ctx(request, error=None, success=None)
    )


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="register.html", context=_ctx(request, error=None, success=None)
    )


@app.post("/register")
async def register_post(
    request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...),
):
    ok, msg = auth.register(username, email, password)
    if not ok:
        return templates.TemplateResponse(
            request=request, name="register.html", context=_ctx(request, error=msg, success=None)
        )
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context=_ctx(request, error=None, success="Account created – please log in"),
    )


@app.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    trace_id = getattr(request.state, "trace_id", None) or request.cookies.get(TRACE_COOKIE)
    ok, token, msg = auth.login(username, password, trace_id=trace_id)
    if not ok:
        return templates.TemplateResponse(
            request=request, name="login.html", context=_ctx(request, error=msg, success=None)
        )
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie("beq_session", token, httponly=True, max_age=14 * 24 * 3600, samesite="lax")
    return resp


@app.get("/logout")
async def logout(request: Request):
    auth.logout(_session(request))
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie("beq_session")
    return resp


def _generate(prompt: str, max_tokens: int, temperature: float, preamble: str = "") -> tuple[str, str]:
    full_prompt = f"{preamble}{prompt}"
    ids = tokenizer.encode(full_prompt)
    if not ids:
        ids = tokenizer.encode("a")
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=min(max_tokens, 200),
            temperature=max(0.5, min(temperature, 1.1)),
            top_k=30,
            repetition_penalty=1.3,
        )
    return full_prompt, tokenizer.decode(out[0].tolist())


def _clean_completion(text: str) -> str:
    for marker in ("\nUser:", "\nuser:", "\n\nUser"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def _answer(prompt: str, max_tokens: int, temperature: float, preamble: str) -> tuple[str, str]:
    know = try_knowledge_answer(prompt)
    if know is not None:
        return prompt, know
    math_answer = try_math_answer(prompt)
    if math_answer is not None:
        return prompt, math_answer
    if model is None or tokenizer is None:
        return prompt, "[Model not loaded — train via Admin, or add facts in configs/*.sbe]"
    full_prompt, full = _generate(prompt, max_tokens, temperature, preamble)
    completion = full[len(full_prompt):] if full.startswith(full_prompt) else full
    completion = _clean_completion(completion)
    return full_prompt, completion


@app.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    prompt: str = Form(...),
    max_tokens: int = Form(80),
    temperature: float = Form(0.8),
    language: str = Form("en"),
):
    user = _user(request)
    mode_cfg = ai_mode.get_mode_config()
    error = None
    result = None
    lang = (language or "en").strip().lower()[:8]
    lang_prefix = LANGUAGE_PREFIXES.get(lang, "")

    if not mode_cfg["public_chat"] and not _is_admin(user):
        error = f"Beq is currently in '{mode_cfg['label']}' mode and not available to the public right now."
    elif model is None and try_math_answer(prompt) is None and try_knowledge_answer(prompt) is None:
        error = "Model not loaded. Train first or add knowledge in configs/*.sbe."
    elif user is None:
        error = "Please log in to generate."
    else:
        ok, msg = auth.can_use_tokens(user, max_tokens)
        if not ok:
            error = msg
        else:
            preamble = mode_cfg["preamble"] + lang_prefix
            full_prompt, completion = _answer(prompt, max_tokens, temperature, preamble)
            used = min(max_tokens, max(1, len(completion)))
            auth.consume_tokens(user["id"], used)
            user = auth.get_user(_session(request))
            result = {"prompt": prompt, "completion": completion, "full": full_prompt + completion}

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=_ctx(
            request,
            result=result,
            error=error,
            prompt=prompt,
            user=user,
            language=lang,
            max_tokens=max_tokens,
            temperature=temperature,
            tokens_remaining=auth.tokens_remaining(user) if user else None,
        ),
    )


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 80
    temperature: float = 0.8


@app.post("/api/generate")
async def api_generate(request: Request, req: GenerateRequest, authorization: str | None = Header(default=None)):
    user = _user_from_request(request, authorization)
    mode_cfg = ai_mode.get_mode_config()
    if not mode_cfg["public_chat"] and not _is_admin(user):
        return JSONResponse(
            {"error": f"Beq is in '{mode_cfg['label']}' mode and not public right now."},
            status_code=403,
        )
    if user is None:
        return JSONResponse(
            {"error": "Auth required. Login cookie or Authorization: Bearer beq_... API key."},
            status_code=401,
        )
    if (
        try_math_answer(req.prompt) is None
        and try_knowledge_answer(req.prompt) is None
        and (model is None or tokenizer is None)
    ):
        return JSONResponse({"error": "Model not loaded"}, status_code=503)
    ok, msg = auth.can_use_tokens(user, req.max_tokens)
    if not ok:
        return JSONResponse({"error": msg}, status_code=429)
    full_prompt, new_text = _answer(req.prompt, req.max_tokens, req.temperature, mode_cfg["preamble"])
    used = min(req.max_tokens, max(1, len(new_text)))
    auth.consume_tokens(user["id"], used)
    refreshed = auth.get_user_by_id(user["id"]) or user
    return {
        "prompt": req.prompt,
        "generated": full_prompt + new_text,
        "new_text": new_text,
        "tokens_used": used,
        "tokens_remaining": auth.tokens_remaining(refreshed),
        "mode": mode_cfg["key"],
    }


@app.get("/api/keys")
async def api_list_keys(request: Request):
    user = _user(request)
    if user is None:
        return JSONResponse({"error": "Login required"}, status_code=401)
    keys = auth.list_api_keys(user["id"])
    return {
        "keys": keys,
        "max_keys": None if user.get("is_admin") else auth.MAX_API_KEYS,
        "tokens_remaining": auth.tokens_remaining(user),
    }


@app.post("/api/keys")
async def api_create_key(request: Request, name: str = Form("default")):
    user = _user(request)
    if user is None:
        return JSONResponse({"error": "Login required"}, status_code=401)
    ok, msg, raw = auth.create_api_key(user, name=name)
    if not ok:
        return JSONResponse({"error": msg}, status_code=400)
    return {"ok": True, "message": msg, "api_key": raw, "name": name}


@app.post("/api/keys/{key_id}/revoke")
async def api_revoke_key(request: Request, key_id: int):
    user = _user(request)
    if user is None:
        return JSONResponse({"error": "Login required"}, status_code=401)
    ok, msg = auth.revoke_api_key(user["id"], key_id)
    if not ok:
        return JSONResponse({"error": msg}, status_code=404)
    return {"ok": True, "message": msg}


def _require_admin(request: Request):
    user = _user(request)
    if not _is_admin(user):
        return None
    return user


@app.get("/sbe-builder", response_class=HTMLResponse)
async def sbe_builder_page(request: Request, file: str | None = None):
    user = _require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    configs = REPO_ROOT / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in configs.glob("*.sbe"))
    content = "# Beq knowledge\n# Q: question?\n# A: exact answer\n\n"
    filename = file or "knowledge.sbe"
    if file:
        safe = Path(file).name
        path = configs / safe
        if path.exists() and path.suffix == ".sbe":
            content = path.read_text(encoding="utf-8")
            filename = safe
    return templates.TemplateResponse(
        request=request,
        name="sbe-builder.html",
        context=_ctx(request, files=files, content=content, filename=filename),
    )


@app.post("/sbe-builder/save")
async def sbe_builder_save(
    request: Request,
    filename: str = Form(...),
    content: str = Form(...),
):
    user = _require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    configs = REPO_ROOT / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    name = Path(filename).name
    if not name.endswith(".sbe") or "/" in filename or "\\" in filename:
        return templates.TemplateResponse(
            request=request,
            name="sbe-builder.html",
            context=_ctx(
                request,
                files=sorted(p.name for p in configs.glob("*.sbe")),
                content=content,
                filename=filename,
                flash_error="Filename must be a simple name ending in .sbe",
            ),
        )
    path = configs / name
    path.write_text(content, encoding="utf-8")
    return templates.TemplateResponse(
        request=request,
        name="sbe-builder.html",
        context=_ctx(
            request,
            files=sorted(p.name for p in configs.glob("*.sbe")),
            content=content,
            filename=name,
            flash_success=f"Saved configs/{name} — Beq will use exact A: answers.",
        ),
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    user = _require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    data_path = REPO_ROOT / "data" / "input.txt"
    data_stats = {
        "exists": data_path.exists(),
        "size_bytes": data_path.stat().st_size if data_path.exists() else 0,
    }
    proc = _training_state["process"]
    training_running = proc is not None and proc.poll() is None
    log_tail = ""
    if TRAIN_LOG_PATH.exists():
        log_tail = "\n".join(TRAIN_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:])
    users = auth.list_users()
    my_trace = getattr(request.state, "trace_id", None) or request.cookies.get(TRACE_COOKIE)
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context=_ctx(
            request,
            modes=ai_mode.MODES,
            data_stats=data_stats,
            training_running=training_running,
            log_tail=log_tail,
            checkpoint_exists=CHECKPOINT_PATH.exists(),
            users=users,
            unlimited_value=auth.UNLIMITED,
            weekly_default=auth.WEEKLY_TOKEN_LIMIT,
            my_trace=my_trace,
        ),
    )


@app.post("/admin/mode")
async def admin_set_mode(request: Request, mode: str = Form(...)):
    user = _require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    ai_mode.set_mode(mode)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/users/{user_id}/limit")
async def admin_set_user_limit(request: Request, user_id: int, limit: str = Form(...)):
    user = _require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    auth.set_token_limit(user_id, limit)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/users/{user_id}/admin")
async def admin_set_admin(request: Request, user_id: int, is_admin: str = Form("0")):
    user = _require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    flag = str(is_admin).strip().lower() in ("1", "true", "yes", "on")
    if hasattr(auth, "set_admin"):
        auth.set_admin(user_id, flag)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/users/{user_id}/reset")
async def admin_reset_user(request: Request, user_id: int):
    user = _require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    auth.reset_usage(user_id)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/users/{user_id}/delete")
async def admin_delete_user(request: Request, user_id: int):
    user = _require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    auth.delete_user(user_id)
    return RedirectResponse("/admin", status_code=303)


def _load_training_args() -> list[str]:
    config_path = REPO_ROOT / "configs" / "default.yaml"
    args = []
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"[admin_train] could not read config ({e})")
        return args
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    def add(flag, value):
        if value is not None:
            args.extend([flag, str(value)])
    add("--data", train_cfg.get("data_path"))
    add("--out_dir", train_cfg.get("out_dir"))
    add("--d_model", model_cfg.get("d_model"))
    add("--n_layers", model_cfg.get("n_layers"))
    add("--n_heads", model_cfg.get("n_heads"))
    add("--block_size", model_cfg.get("block_size"))
    add("--batch_size", train_cfg.get("batch_size"))
    add("--lr", train_cfg.get("learning_rate"))
    add("--max_steps", train_cfg.get("max_steps"))
    add("--eval_interval", train_cfg.get("eval_interval"))
    add("--save_interval", train_cfg.get("save_interval"))
    return args


@app.post("/admin/train")
async def admin_start_training(request: Request):
    user = _require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    proc = _training_state["process"]
    if proc is None or proc.poll() is not None:
        TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(TRAIN_LOG_PATH, "w", encoding="utf-8")
        cmd = [sys.executable, str(REPO_ROOT / "train" / "train.py")] + _load_training_args()
        _training_state["process"] = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=log_file, stderr=subprocess.STDOUT,
        )
    return RedirectResponse("/admin", status_code=303)


def _check_api_key(authorization: str | None) -> None:
    if not BEQ_API_KEY:
        raise HTTPException(status_code=503, detail="BEQ_API_KEY is not configured on this server.")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <key>")
    if authorization.removeprefix("Bearer ").strip() != BEQ_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")


@app.get("/api/status")
async def api_status():
    mode_cfg = ai_mode.get_mode_config()
    return {
        "model_loaded": model is not None,
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "mode": mode_cfg["key"],
        "mode_label": mode_cfg["label"],
        "public_chat": mode_cfg["public_chat"],
        "persistence": "database" if settings_store.using_postgres() else "sqlite/file",
    }


@app.get("/api/config")
async def api_get_config(authorization: str | None = Header(default=None)):
    _check_api_key(authorization)
    mode_cfg = ai_mode.get_mode_config()
    data_path = REPO_ROOT / "data" / "input.txt"
    return {
        "mode": mode_cfg["key"],
        "available_modes": {k: {"label": v["label"], "description": v["description"]} for k, v in ai_mode.MODES.items()},
        "model_loaded": model is not None,
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "data_bytes": data_path.stat().st_size if data_path.exists() else 0,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
