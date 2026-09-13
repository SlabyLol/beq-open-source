"""Beq Web - Auth, quotas, own model, AI mode, crawl/search"""

import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from fastapi import FastAPI, Form, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from model import BeqTransformer, CharTokenizer
from web import ai_mode, auth, crawler, settings_store
from web.knowledge import try_knowledge_answer
from web.math_tools import try_math_answer
from web.searcher import try_search_answer
from web import crawl_admin

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cpu"
CHECKPOINT = REPO_ROOT / "checkpoints" / "beq_latest.pt"
TRAIN_LOG = REPO_ROOT / "data" / "train.log"

LANGUAGE_PREFIXES = {
    "en": "",
    "de": "[Respond in German] ",
    "es": "[Respond in Spanish] ",
    "fr": "[Respond in French] ",
    "it": "[Respond in Italian] ",
    "pt": "[Respond in Portuguese] ",
}

app = FastAPI(title="Beq")
templates = Jinja2Templates(directory=str(REPO_ROOT / "web" / "templates"))
static_dir = REPO_ROOT / "web" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

model = None
tokenizer = None
training_state = {"process": None}


def load_model():
    global model, tokenizer
    try:
        if not CHECKPOINT.exists():
            print("[load_model] no checkpoint")
            model = tokenizer = None
            return
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        config = ckpt.get("config") or {}
        chars = ckpt.get("chars") or ckpt.get("vocab") or ""
        tokenizer = CharTokenizer(chars)
        model = BeqTransformer(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            block_size=config["block_size"],
            dropout=config.get("dropout", 0.1),
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Beq loaded | params={sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"[load_model] {e}")
        model = tokenizer = None


@app.on_event("startup")
def _startup():
    try:
        settings_store.init_settings_table()
    except Exception as e:
        print(f"[startup] settings: {e}")
    try:
        auth.init_db()
    except Exception as e:
        print(f"[startup] auth: {e}")
    load_model()


def _session(request: Request):
    return request.cookies.get("beq_session")


def _user(request: Request):
    return auth.user_from_session(_session(request))


def _is_admin(user) -> bool:
    return bool(user and user.get("is_admin"))


def _check_and_consume(user_id, tokens):
    return auth.check_and_consume(user_id, tokens)


def _ctx(request: Request, **extra):
    user = _user(request)
    mode_cfg = ai_mode.get_mode_config()
    base = {
        "request": request,
        "user": user,
        "is_admin": _is_admin(user),
        "ai_mode": mode_cfg,
        "model_loaded": model is not None,
        "languages": list(LANGUAGE_PREFIXES.keys()),
        "error": None,
        "prompt": None,
        "answer": None,
        "result": None,
    }
    base.update(extra)
    return base


def _user_from_request(request: Request, authorization: str | None):
    user = _user(request)
    if user:
        return user
    if authorization and authorization.lower().startswith("bearer "):
        key = authorization.split(" ", 1)[1].strip()
        return auth.user_from_api_key(key)
    return None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", _ctx(request))


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(request, "about.html", _ctx(request))


@app.get("/generate", response_class=HTMLResponse)
async def generate_page(request: Request):
    return templates.TemplateResponse(request, "generate.html", _ctx(request))


@app.get("/api-dashboard", response_class=HTMLResponse)
async def api_dashboard(request: Request):
    user = _user(request)
    keys = auth.list_api_keys(user["id"]) if user else []
    max_keys = 999 if _is_admin(user) else 10
    return templates.TemplateResponse(
        request, "api.html", _ctx(request, keys=keys, max_keys=max_keys)
    )


@app.post("/api-dashboard/create")
async def api_create_key(request: Request, name: str = Form("key")):
    user = _user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    ok, msg = auth.create_api_key(user["id"], name)
    keys = auth.list_api_keys(user["id"])
    return templates.TemplateResponse(
        request, "api.html", _ctx(request, keys=keys, max_keys=10, message=msg, ok=ok)
    )


@app.post("/api-dashboard/revoke/{key_id}")
async def api_revoke_key(request: Request, key_id: int):
    user = _user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    ok, msg = auth.delete_api_key(user["id"], key_id)
    keys = auth.list_api_keys(user["id"])
    return templates.TemplateResponse(
        request, "api.html", _ctx(request, keys=keys, max_keys=10, message=msg, ok=ok)
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", _ctx(request, error=None))


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(request, "register.html", _ctx(request, error=None))


@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
):
    ok, msg = auth.register(username, email, password)
    if not ok:
        return templates.TemplateResponse(request, "register.html", _ctx(request, error=msg))
    return templates.TemplateResponse(
        request, "login.html", _ctx(request, error="Registered - please log in")
    )


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    ok, msg, session = auth.login(username, password)
    if not ok:
        return templates.TemplateResponse(
            request, "login.html", _ctx(request, error=msg or "Invalid login")
        )
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(
        "beq_session", session, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 14
    )
    return resp


@app.get("/logout")
async def logout(request: Request):
    auth.logout(_session(request))
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie("beq_session")
    return resp


def _generate(prompt, max_tokens, temperature, preamble=""):
    full_prompt = (preamble or "") + prompt
    ids = tokenizer.encode(full_prompt) or tokenizer.encode("a")
    idx = torch.tensor([ids[-128:]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max(1, min(int(max_tokens), 200)),
            temperature=max(0.1, min(float(temperature), 1.5)),
            top_k=30,
            repetition_penalty=1.3,
        )
    return full_prompt, tokenizer.decode(out[0].tolist())


def _clean_completion(text):
    text = (text or "").strip()
    for stop in ("\nUser:", "\nHuman:", "\nAssistant:", "\nBeq:"):
        if stop in text:
            text = text.split(stop)[0].strip()
    return text


def _is_identity_fluff(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    markers = (
        "i am beq",
        "my name is beq",
        "helpful general-purpose assistant",
        "open-source model",
        "i run as your own",
    )
    hits = sum(1 for m in markers if m in t)
    if hits >= 1 and len(t) < 220:
        return True
    if hits >= 2:
        return True
    return False


def _answer(prompt, max_tokens, temperature, preamble):
    """Knowledge → math → web search → model (reject identity echo)."""
    know = try_knowledge_answer(prompt)
    if know and not (_is_identity_fluff(know) and len((prompt or "").split()) > 4):
        return prompt, know
    math_answer = try_math_answer(prompt)
    if math_answer:
        return prompt, math_answer
    search_answer = try_search_answer(prompt, use_web=True)
    if search_answer and not _is_identity_fluff(search_answer):
        return prompt, search_answer
    if model is None or tokenizer is None:
        if search_answer:
            return prompt, search_answer
        return prompt, (
            "I could not load the model and web search returned nothing. "
            "Try again, or ask the admin to crawl a page / check Crawler ON."
        )
    full_prompt, full = _generate(prompt, max_tokens, temperature, preamble)
    completion = full[len(full_prompt) :] if full.startswith(full_prompt) else full
    completion = _clean_completion(completion)
    if _is_identity_fluff(completion) or (
        prompt.strip().lower() in completion.lower() and len(completion) < len(prompt) + 30
    ):
        search_answer = try_search_answer(prompt, use_web=True)
        if search_answer:
            return prompt, search_answer
        if _is_identity_fluff(completion):
            return prompt, (
                "I am Beq. For factual questions I search the web — "
                "no result this time. Try e.g. What is the Sun? "
                "or turn Immer suchen ON in admin."
            )
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
    answer = None
    if not mode_cfg.get("public_chat", True) and not _is_admin(user):
        error = "Chat is restricted."
    elif (
        model is None
        and try_math_answer(prompt) is None
        and try_knowledge_answer(prompt) is None
        and try_search_answer(prompt) is None
    ):
        error = "Model not loaded. Train, add knowledge, or use search."
    else:
        if user and not _is_admin(user):
            allowed, msg = _check_and_consume(user["id"], max_tokens)
            if not allowed:
                error = msg
        if error is None:
            preamble = mode_cfg.get("preamble", "") + LANGUAGE_PREFIXES.get(language, "")
            _, answer = _answer(prompt, max_tokens, temperature, preamble)
    result = {"prompt": prompt, "completion": answer} if answer is not None else None
    return templates.TemplateResponse(
        request,
        "index.html",
        _ctx(request, prompt=prompt, answer=answer, error=error, result=result),
    )


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 80
    temperature: float = 0.8
    language: str = "en"


@app.post("/api/generate")
async def api_generate(
    request: Request,
    req: GenerateRequest,
    authorization: str | None = Header(default=None),
):
    user = _user_from_request(request, authorization)
    mode_cfg = ai_mode.get_mode_config()
    if not mode_cfg.get("public_chat", True) and not _is_admin(user):
        return JSONResponse({"error": "not public"}, status_code=403)
    if (
        model is None
        and try_math_answer(req.prompt) is None
        and try_knowledge_answer(req.prompt) is None
        and try_search_answer(req.prompt) is None
    ):
        return JSONResponse({"error": "Model not loaded"}, status_code=503)
    if user and not _is_admin(user):
        allowed, msg = _check_and_consume(user["id"], req.max_tokens)
        if not allowed:
            return JSONResponse({"error": msg}, status_code=429)
    preamble = mode_cfg.get("preamble", "") + LANGUAGE_PREFIXES.get(req.language, "")
    full_prompt, new_text = _answer(req.prompt, req.max_tokens, req.temperature, preamble)
    return {
        "prompt": req.prompt,
        "generated": (full_prompt + new_text)
        if not str(new_text).startswith(str(full_prompt))
        else new_text,
        "completion": new_text,
        "model": "beq",
    }


@app.get("/status")
async def status():
    data_path = REPO_ROOT / "data" / "input.txt"
    return {
        "ok": True,
        "model_loaded": model is not None,
        "mode": ai_mode.get_mode_config().get("key", "default"),
        "crawl": crawler.status(),
        "data_bytes": data_path.stat().st_size if data_path.exists() else 0,
    }


def require_admin(request: Request):
    user = _user(request)
    if not _is_admin(user):
        return None
    return user


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    user = require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    data_path = REPO_ROOT / "data" / "input.txt"
    data_stats = {
        "exists": data_path.exists(),
        "size_bytes": data_path.stat().st_size if data_path.exists() else 0,
    }
    proc = training_state["process"]
    training_running = proc is not None and proc.poll() is None
    log_tail = ""
    if TRAIN_LOG.exists():
        log_tail = "\n".join(
            TRAIN_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:]
        )
    users = auth.list_users()
    crawl_docs = crawler.list_docs(limit=20)
    crawl_status = crawler.status()
    return templates.TemplateResponse(
        request,
        "admin.html",
        _ctx(
            request,
            modes=ai_mode.MODES,
            data_stats=data_stats,
            training_running=training_running,
            log_tail=log_tail,
            checkpoint_exists=CHECKPOINT.exists(),
            users=users,
            unlimited_value=getattr(auth, "UNLIMITED", -1),
            weekly_default=getattr(auth, "WEEKLY_TOKEN_LIMIT", 5000),
            crawl_docs=crawl_docs,
            crawl_status=crawl_status,
            crawl_message=None,
        ),
    )


@app.post("/admin/mode")
async def admin_mode(request: Request, mode: str = Form(...)):
    user = require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    ai_mode.set_mode(mode)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/users/{user_id}/limit")
async def admin_user_limit(request: Request, user_id: int, limit: str = Form(...)):
    user = require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    auth.set_token_limit(user_id, limit)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/users/{user_id}/admin")
async def admin_user_admin(request: Request, user_id: int, is_admin: int = Form(...)):
    user = require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    auth.set_admin(user_id, bool(is_admin))
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/users/{user_id}/reset")
async def admin_user_reset(request: Request, user_id: int):
    user = require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    auth.reset_tokens(user_id)
    return RedirectResponse("/admin", status_code=303)


crawl_admin.register(
    app,
    require_admin=require_admin,
    templates=templates,
    ctx=_ctx,
    repo_root=REPO_ROOT,
    train_log_path=TRAIN_LOG,
    training_state=training_state,
    checkpoint_path=CHECKPOINT,
    auth=auth,
    ai_mode=ai_mode,
)
