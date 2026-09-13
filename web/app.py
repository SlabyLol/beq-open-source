"""
Beq Web – Auth, quotas, own model, AI mode, crawl/search tools, user API keys
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
from web.searcher import try_search_answer, search_all
from web import crawler

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
    "ru": "Ответь по-русски.\n",
    "ja": "日本語で答えてください。\n",
    "zh": "请用中文回答。\n",
    "ko": "한국어로 답하세요.\n",
    "ar": "أجب بالعربية.\n",
    "tr": "Türkçe cevap ver.\n",
}

app = FastAPI(title="Beq")
templates = Jinja2Templates(directory=str(REPO_ROOT / "web" / "templates"))
static_dir = REPO_ROOT / "web" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

model = None
tokenizer = None
TRACE_COOKIE = "beq_trace"


def load_model():
    global model, tokenizer
    if not CHECKPOINT_PATH.exists() or not TOKENIZER_PATH.exists():
        print("No checkpoint found — chat will use .sbe / math / search until you train.")
        return
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    tokenizer = CharTokenizer.load(TOKENIZER_PATH)
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


@app.middleware("http")
async def ensure_trace_cookie(request: Request, call_next):
    response = await call_next(request)
    if TRACE_COOKIE not in request.cookies:
        tid = secrets.token_hex(8)
        response.set_cookie(TRACE_COOKIE, tid, max_age=60 * 60 * 24 * 365)
        request.state.trace_id = tid
    else:
        request.state.trace_id = request.cookies.get(TRACE_COOKIE)
    return response


@app.on_event("startup")
async def startup():
    try:
        auth.init_db()
    except Exception as e:
        print(f"[auth] init_db: {e}")
    try:
        settings_store.init_settings_table()
    except Exception as e:
        print(f"[settings] {e}")
    load_model()


def _session(request: Request):
    return request.cookies.get("beq_session")


def _user(request: Request):
    return auth.user_from_session(_session(request))


def _user_from_request(request: Request, authorization: str | None = None):
    user = _user(request)
    if user:
        return user
    if authorization and authorization.lower().startswith("bearer "):
        key = authorization.split(" ", 1)[1].strip()
        return auth.user_from_api_key(key)
    return None


def _ctx(request: Request, **extra):
    user = _user(request)
    mode_cfg = ai_mode.get_mode_config()
    base = {
        "user": user,
        "model_loaded": model is not None,
        "ai_mode": mode_cfg,
        "languages": list(LANGUAGE_PREFIXES.keys()),
    }
    base.update(extra)
    return base


def _is_admin(user: dict | None) -> bool:
    return bool(user and user.get("is_admin"))


def _api_dashboard_ctx(request: Request, **extra):
    user = _user(request)
    keys = auth.list_api_keys(user["id"]) if user else []
    max_keys = None if (user and user.get("is_admin")) else auth.MAX_API_KEYS
    ctx = _ctx(request, keys=keys, max_keys=max_keys)
    ctx.update(extra)
    return ctx


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
    if not user:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request=request, name="api.html", context=_api_dashboard_ctx(request)
    )


@app.post("/api-dashboard/create")
async def api_dashboard_create(request: Request, name: str = Form("default")):
    user = _user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    ok, msg, _key = auth.create_api_key(user["id"], name)
    return templates.TemplateResponse(
        request=request,
        name="api.html",
        context=_api_dashboard_ctx(request, message=msg, ok=ok),
    )


@app.post("/api-dashboard/revoke/{key_id}")
async def api_dashboard_revoke(request: Request, key_id: int):
    user = _user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    ok, msg = auth.revoke_api_key(user["id"], key_id)
    return templates.TemplateResponse(
        request=request,
        name="api.html",
        context=_api_dashboard_ctx(request, message=msg, ok=ok),
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="login.html", context=_ctx(request, error=None)
    )


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="register.html", context=_ctx(request, error=None)
    )


@app.post("/register")
async def register_post(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
):
    ok, msg, session = auth.register(username, email, password)
    if not ok:
        return templates.TemplateResponse(
            request=request, name="register.html", context=_ctx(request, error=msg)
        )
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie("beq_session", session, httponly=True, max_age=60 * 60 * 24 * 30)
    return resp


@app.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    ok, msg, session = auth.login(username, password)
    if not ok:
        return templates.TemplateResponse(
            request=request, name="login.html", context=_ctx(request, error=msg)
        )
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie("beq_session", session, httponly=True, max_age=60 * 60 * 24 * 30)
    return resp


@app.get("/logout")
async def logout(request: Request):
    auth.logout(_session(request))
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie("beq_session")
    return resp


def _generate(prompt: str, max_tokens: int, temperature: float, preamble: str = "") -> tuple[str, str]:
    full_prompt = (preamble or "") + prompt
    ids = tokenizer.encode(full_prompt) or tokenizer.encode("a")
    idx = torch.tensor([ids[-tokenizer.max_seq_len if hasattr(tokenizer, "max_seq_len") else -128:]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max(1, min(int(max_tokens), 200)),
            temperature=max(0.1, min(float(temperature), 1.5)),
            top_k=30,
            repetition_penalty=1.3,
        )
    text = tokenizer.decode(out[0].tolist())
    return full_prompt, text


def _clean_completion(text: str) -> str:
    text = text.strip()
    for stop in ("\nUser:", "\nHuman:", "\nQ:"):
        if stop in text:
            text = text.split(stop)[0].strip()
    return text


def _answer(prompt: str, max_tokens: int, temperature: float, preamble: str) -> tuple[str, str]:
    # 1) exact .sbe  2) math  3) crawl store + web search  4) neural model
    know = try_knowledge_answer(prompt)
    if know is not None:
        return prompt, know
    math_answer = try_math_answer(prompt)
    if math_answer is not None:
        return prompt, math_answer
    search_answer = try_search_answer(prompt, use_web=True)
    if search_answer is not None:
        return prompt, search_answer
    if model is None or tokenizer is None:
        return prompt, "[Model not loaded — train via Admin, add configs/*.sbe, or crawl/search knowledge]"
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
    answer = None
    if not mode_cfg["public_chat"] and not _is_admin(user):
        error = f"Beq is currently in '{mode_cfg['label']}' mode and not available to the public right now."
    elif model is None and try_math_answer(prompt) is None and try_knowledge_answer(prompt) is None and try_search_answer(prompt) is None:
        error = "Model not loaded. Train first, add knowledge, or use search."
    elif not user and not mode_cfg.get("public_chat", True):
        error = "Please log in to generate."
    else:
        if user and not _is_admin(user):
            allowed, msg = auth.check_and_consume(user["id"], max_tokens)
            if not allowed:
                error = msg
        if error is None:
            lang_prefix = LANGUAGE_PREFIXES.get(language, "")
            preamble = mode_cfg["preamble"] + lang_prefix
            _, answer = _answer(prompt, max_tokens, temperature, preamble)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=_ctx(request, prompt=prompt, answer=answer, error=error),
    )


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 80
    temperature: float = 0.8
    language: str = "en"


@app.post("/api/generate")
async def api_generate(request: Request, req: GenerateRequest, authorization: str | None = Header(default=None)):
    user = _user_from_request(request, authorization)
    mode_cfg = ai_mode.get_mode_config()
    if not mode_cfg["public_chat"] and not _is_admin(user):
        return JSONResponse(
            {"error": f"Beq is in '{mode_cfg['label']}' mode and not public right now."},
            status_code=403,
        )
    if (
        model is None
        and try_math_answer(req.prompt) is None
        and try_knowledge_answer(req.prompt) is None
        and try_search_answer(req.prompt) is None
    ):
        return JSONResponse({"error": "Model not loaded"}, status_code=503)
    if user and not _is_admin(user):
        allowed, msg = auth.check_and_consume(user["id"], req.max_tokens)
        if not allowed:
            return JSONResponse({"error": msg}, status_code=429)
    lang_prefix = LANGUAGE_PREFIXES.get(req.language, "")
    preamble = mode_cfg["preamble"] + lang_prefix
    full_prompt, new_text = _answer(req.prompt, req.max_tokens, req.temperature, preamble)
    return {
        "prompt": req.prompt,
        "generated": full_prompt + new_text,
        "completion": new_text,
        "model": "beq",
    }


@app.get("/api/keys")
async def api_list_keys(request: Request):
    user = _user(request)
    if not user:
        return JSONResponse({"error": "Login required"}, status_code=401)
    return {
        "keys": auth.list_api_keys(user["id"]),
        "max_keys": None if user.get("is_admin") else auth.MAX_API_KEYS,
    }


@app.post("/api/keys")
async def api_create_key(request: Request, name: str = Form("default")):
    user = _user(request)
    if not user:
        return JSONResponse({"error": "Login required"}, status_code=401)
    ok, msg, key = auth.create_api_key(user["id"], name)
    if not ok:
        return JSONResponse({"error": msg}, status_code=400)
    return {"ok": True, "message": msg, "key": key}


@app.post("/api/keys/{key_id}/revoke")
async def api_revoke_key(request: Request, key_id: int):
    user = _user(request)
    if not user:
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
    path = configs / filename
    if path.exists():
        content = path.read_text(encoding="utf-8")
    return templates.TemplateResponse(
        request=request,
        name="sbe-builder.html",
        context=_ctx(request, files=files, filename=filename, content=content),
    )


@app.post("/sbe-builder/save")
async def sbe_builder_save(
    request: Request,
    filename: str = Form("knowledge.sbe"),
    content: str = Form(""),
):
    user = _require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    configs = REPO_ROOT / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    name = Path(filename).name
    if not name.endswith(".sbe"):
        name += ".sbe"
    (configs / name).write_text(content, encoding="utf-8")
    return RedirectResponse(f"/sbe-builder?file={name}", status_code=303)


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
    crawl_docs = crawler.list_docs(limit=20)
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
            crawl_docs=crawl_docs,
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


@app.get("/api/search")
async def api_search(q: str = "", web: str = "1"):
    """Search tool: crawl store + Wikipedia / DuckDuckGo."""
    if not q.strip():
        return {"error": "missing q"}
    return search_all(q.strip(), use_web=web not in ("0", "false", "no"))


@app.post("/admin/crawl")
async def admin_crawl(request: Request, url: str = Form(...)):
    user = _require_admin(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    result = crawler.crawl_url(url)
    accept = request.headers.get("accept") or ""
    if "application/json" in accept:
        return result
    return RedirectResponse("/admin", status_code=303)


@app.get("/api/crawl/docs")
async def api_crawl_docs(request: Request, limit: int = 30):
    user = _user(request)
    if not _is_admin(user):
        return JSONResponse({"error": "admin only"}, status_code=403)
    return {"docs": crawler.list_docs(limit=limit)}


@app.post("/api/crawl")
async def api_crawl(request: Request, url: str = Form(...)):
    user = _user(request)
    if not _is_admin(user):
        return JSONResponse({"error": "admin only"}, status_code=403)
    return crawler.crawl_url(url)


@app.get("/status")
async def status():
    data_path = REPO_ROOT / "data" / "input.txt"
    return {
        "ok": True,
        "model_loaded": model is not None,
        "mode": ai_mode.get_mode_key(),
        "crawl_docs": len(crawler.list_docs(limit=1000)),
        "data_bytes": data_path.stat().st_size if data_path.exists() else 0,
        "weekly_token_limit": auth.WEEKLY_TOKEN_LIMIT,
    }


class ConfigUpdate(BaseModel):
    mode: str


def _check_api_key(authorization: str | None):
    if not BEQ_API_KEY:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "API key required")
    key = authorization.split(" ", 1)[1].strip()
    if key != BEQ_API_KEY:
        # also allow user API keys via auth if present
        u = auth.user_from_api_key(key) if hasattr(auth, "user_from_api_key") else None
        if not u:
            raise HTTPException(401, "Invalid API key")


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
