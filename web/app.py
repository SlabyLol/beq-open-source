"""Beq Web - Auth, quotas, own model, AI mode, crawl/search"""

import os
import secrets
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
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "beq_best.pt"
TOKENIZER_PATH = REPO_ROOT / "checkpoints" / "tokenizer.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRACE_COOKIE = "beq_trace"
BEQ_API_KEY = os.environ.get("BEQ_API_KEY", "").strip()

LANGUAGE_PREFIXES = {
    "en": "", "de": "Antworte auf Deutsch.\n", "fr": "Reponds en francais.\n",
    "es": "Responde en espanol.\n", "it": "Rispondi in italiano.\n",
    "pt": "Responda em portugues.\n", "nl": "Antwoord in het Nederlands.\n",
    "pl": "Odpowiedz po polsku.\n", "ru": "", "ja": "", "zh": "", "ko": "",
    "ar": "", "tr": "",
}

app = FastAPI(title="Beq")
templates = Jinja2Templates(directory=str(REPO_ROOT / "web" / "templates"))
static_dir = REPO_ROOT / "web" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

model = None
tokenizer = None


def load_model():
    global model, tokenizer
    if not CHECKPOINT_PATH.exists() or not TOKENIZER_PATH.exists():
        print("No checkpoint - sbe/math/search only")
        return
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        tokenizer = CharTokenizer.load(TOKENIZER_PATH)
        config = ckpt["config"]
        model = BeqTransformer(
            vocab_size=config["vocab_size"], d_model=config["d_model"],
            n_layers=config["n_layers"], n_heads=config["n_heads"],
            max_seq_len=config["max_seq_len"],
        ).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Beq loaded | params={sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"[load_model] {e}")
        model = tokenizer = None


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
        print(f"[auth] {e}")
    try:
        settings_store.init_settings_table()
    except Exception as e:
        print(f"[settings] {e}")
    load_model()


def _session(request: Request):
    return request.cookies.get("beq_session")


def _user(request: Request):
    return auth.get_user(_session(request))


def _user_from_request(request: Request, authorization: str | None = None):
    user = _user(request)
    if user:
        return user
    if authorization and authorization.lower().startswith("bearer "):
        return auth.user_from_api_key(authorization.split(" ", 1)[1].strip())
    return None


def _ctx(request: Request, **extra):
    try:
        mode_cfg = ai_mode.get_mode_config()
    except Exception:
        mode_cfg = {"key": "default", "label": "Default", "badge": "Default",
                    "description": "", "preamble": "", "public_chat": True}
    base = {
        "request": request,
        "user": _user(request),
        "model_loaded": model is not None,
        "ai_mode": mode_cfg,
        "languages": list(LANGUAGE_PREFIXES.keys()),
        "error": None, "prompt": None, "answer": None, "result": None,
        "language": "en", "max_tokens": 100, "temperature": 0.8,
    }
    base.update(extra)
    return base


def _is_admin(user):
    return bool(user and user.get("is_admin"))


def _check_and_consume(user_id: int, amount: int):
    user = auth.get_user_by_id(user_id)
    if not user:
        return False, "User not found"
    ok, msg = auth.can_use_tokens(user, amount)
    if not ok:
        return False, msg
    auth.consume_tokens(user_id, amount)
    return True, "ok"


def _api_dashboard_ctx(request: Request, **extra):
    user = _user(request)
    keys = auth.list_api_keys(user["id"]) if user else []
    max_keys = None if (user and user.get("is_admin")) else auth.MAX_API_KEYS
    ctx = _ctx(request, keys=keys, max_keys=max_keys)
    ctx.update(extra)
    return ctx


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", _ctx(request))


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", _ctx(request))


@app.get("/generate", response_class=HTMLResponse)
async def generate_page(request: Request):
    return templates.TemplateResponse("generate.html", _ctx(request))


@app.get("/api-dashboard", response_class=HTMLResponse)
async def api_dashboard(request: Request):
    if not _user(request):
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("api.html", _api_dashboard_ctx(request))


@app.post("/api-dashboard/create")
async def api_dashboard_create(request: Request, name: str = Form("default")):
    user = _user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    ok, msg, _key = auth.create_api_key(user, name)
    return templates.TemplateResponse("api.html", _api_dashboard_ctx(request, message=msg, ok=ok))


@app.post("/api-dashboard/revoke/{key_id}")
async def api_dashboard_revoke(request: Request, key_id: int):
    user = _user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    ok, msg = auth.revoke_api_key(user["id"], key_id)
    return templates.TemplateResponse("api.html", _api_dashboard_ctx(request, message=msg, ok=ok))


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", _ctx(request, error=None))


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", _ctx(request, error=None))


@app.post("/register")
async def register_post(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    ok, msg = auth.register(username, email, password)
    if not ok:
        return templates.TemplateResponse("register.html", _ctx(request, error=msg))
    trace = getattr(request.state, "trace_id", None) or request.cookies.get(TRACE_COOKIE)
    ok2, session, msg2 = auth.login(username, password, trace_id=trace)
    if not ok2 or not session:
        return templates.TemplateResponse("login.html", _ctx(request, error="Registered - please log in"))
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie("beq_session", session, httponly=True, max_age=60 * 60 * 24 * 30)
    return resp


@app.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    trace = getattr(request.state, "trace_id", None) or request.cookies.get(TRACE_COOKIE)
    ok, session, msg = auth.login(username, password, trace_id=trace)
    if not ok or not session:
        return templates.TemplateResponse("login.html", _ctx(request, error=msg or "Invalid login"))
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie("beq_session", session, httponly=True, max_age=60 * 60 * 24 * 30)
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
        out = model.generate(idx, max_new_tokens=max(1, min(int(max_tokens), 200)),
                             temperature=max(0.1, min(float(temperature), 1.5)),
                             top_k=30, repetition_penalty=1.3)
    return full_prompt, tokenizer.decode(out[0].tolist())


def _clean_completion(text):
    text = text.strip()
    for stop in ("\nUser:", "\nHuman:", "\nQ:"):
        if stop in text:
            text = text.split(stop)[0].strip()
    return text


def _answer(prompt, max_tokens, temperature, preamble):
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
        return prompt, "[Model not loaded - train, .sbe, or crawl/search]"
    full_prompt, full = _generate(prompt, max_tokens, temperature, preamble)
    completion = full[len(full_prompt):] if full.startswith(full_prompt) else full
    return full_prompt, _clean_completion(completion)


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, prompt: str = Form(...), max_tokens: int = Form(80),
               temperature: float = Form(0.8), language: str = Form("en")):
    user = _user(request)
    mode_cfg = ai_mode.get_mode_config()
    error = answer = None
    if not mode_cfg.get("public_chat", True) and not _is_admin(user):
        error = f"Beq is in '{mode_cfg.get('label')}' mode - not public."
    elif model is None and try_math_answer(prompt) is None and try_knowledge_answer(prompt) is None and try_search_answer(prompt) is None:
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
    return templates.TemplateResponse("index.html", _ctx(request, prompt=prompt, answer=answer, error=error, result=result))


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 80
    temperature: float = 0.8
    language: str = "en"


@app.post("/api/generate")
async def api_generate(request: Request, req: GenerateRequest, authorization: str | None = Header(default=None)):
    user = _user_from_request(request, authorization)
    mode_cfg = ai_mode.get_mode_config()
    if not mode_cfg.get("public_chat", True) and not _is_admin(user):
        return JSONResponse({"error": "not public"}, status_code=403)
    if model is None and try_math_answer(req.prompt) is None and try_knowledge_answer(req.prompt) is None and try_search_answer(req.prompt) is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)
    if user and not _is_admin(user):
        allowed, msg = _check_and_consume(user["id"], req.max_tokens)
        if not allowed:
            return JSONResponse({"error": msg}, status_code=429)
    preamble = mode_cfg.get("preamble", "") + LANGUAGE_PREFIXES.get(req.language, "")
    full_prompt, new_text = _answer(req.prompt, req.max_tokens, req.temperature, preamble)
    return {"prompt": req.prompt, "generated": full_prompt + new_text, "completion": new_text, "model": "beq"}


@app.get("/api/keys")
async def api_list_keys(request: Request):
    user = _user(request)
    if not user:
        return JSONResponse({"error": "Login required"}, status_code=401)
    return {"keys": auth.list_api_keys(user["id"]), "max_keys": None if user.get("is_admin") else auth.MAX_API_KEYS}


@app.post("/api/keys")
async def api_create_key(request: Request, name: str = Form("default")):
    user = _user(request)
    if not user:
        return JSONResponse({"error": "Login required"}, status_code=401)
    ok, msg, key = auth.create_api_key(user, name)
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
    return user if _is_admin(user) else None


@app.get("/sbe-builder", response_class=HTMLResponse)
async def sbe_builder_page(request: Request, file: str | None = None):
    if _require_admin(request) is None:
        return RedirectResponse("/login", status_code=303)
    configs = REPO_ROOT / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in configs.glob("*.sbe"))
    filename = file or "knowledge.sbe"
    path = configs / filename
    content = path.read_text(encoding="utf-8") if path.exists() else "# Beq knowledge\n\n"
    return templates.TemplateResponse("sbe-builder.html", _ctx(request, files=files, filename=filename, content=content))


@app.post("/sbe-builder/save")
async def sbe_builder_save(request: Request, filename: str = Form("knowledge.sbe"), content: str = Form("")):
    if _require_admin(request) is None:
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
    if _require_admin(request) is None:
        return RedirectResponse("/login", status_code=303)
    data_path = REPO_ROOT / "data" / "input.txt"
    data_stats = {"exists": data_path.exists(), "size_bytes": data_path.stat().st_size if data_path.exists() else 0}
    users = auth.list_users()
    my_trace = getattr(request.state, "trace_id", None) or request.cookies.get(TRACE_COOKIE)
    crawl_docs = crawler.list_docs(limit=20)
    crawl_status = crawler.status()
    return templates.TemplateResponse("admin.html", _ctx(
        request, modes=ai_mode.MODES, data_stats=data_stats, training_running=False,
        log_tail="", checkpoint_exists=CHECKPOINT_PATH.exists(), users=users,
        unlimited_value=auth.UNLIMITED, weekly_default=auth.WEEKLY_TOKEN_LIMIT,
        my_trace=my_trace, crawl_docs=crawl_docs, crawl_status=crawl_status, crawl_message=None,
    ))


@app.post("/admin/mode")
async def admin_set_mode(request: Request, mode: str = Form(...)):
    if _require_admin(request) is None:
        return RedirectResponse("/login", status_code=303)
    ai_mode.set_mode(mode)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/users/{user_id}/limit")
async def admin_set_user_limit(request: Request, user_id: int, limit: str = Form(...)):
    if _require_admin(request) is None:
        return RedirectResponse("/login", status_code=303)
    auth.set_token_limit(user_id, limit)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/users/{user_id}/admin")
async def admin_set_admin(request: Request, user_id: int, is_admin: str = Form("0")):
    if _require_admin(request) is None:
        return RedirectResponse("/login", status_code=303)
    flag = str(is_admin).strip().lower() in ("1", "true", "yes", "on")
    if hasattr(auth, "set_admin"):
        auth.set_admin(user_id, flag)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/users/{user_id}/reset")
async def admin_reset_user(request: Request, user_id: int):
    if _require_admin(request) is None:
        return RedirectResponse("/login", status_code=303)
    auth.reset_usage(user_id)
    return RedirectResponse("/admin", status_code=303)


@app.get("/api/search")
async def api_search(q: str = "", web: str = "1"):
    if not q.strip():
        return {"error": "missing q"}
    return search_all(q.strip(), use_web=web not in ("0", "false", "no"))


@app.get("/api/crawl/docs")
async def api_crawl_docs(request: Request, limit: int = 30):
    if not _is_admin(_user(request)):
        return JSONResponse({"error": "admin only"}, status_code=403)
    return {"docs": crawler.list_docs(limit=limit)}


@app.post("/api/crawl")
async def api_crawl(request: Request, url: str = Form(...)):
    if not _is_admin(_user(request)):
        return JSONResponse({"error": "admin only"}, status_code=403)
    return crawler.crawl_url(url)


@app.get("/status")
async def status():
    data_path = REPO_ROOT / "data" / "input.txt"
    return {
        "ok": True,
        "model_loaded": model is not None,
        "mode": ai_mode.get_mode_key(),
        "crawl": crawler.status(),
        "data_bytes": data_path.stat().st_size if data_path.exists() else 0,
    }


try:
    from web import crawl_admin
    crawl_admin.register(
        app, require_admin=_require_admin, templates=templates, ctx=_ctx,
        repo_root=REPO_ROOT, train_log_path=REPO_ROOT / "checkpoints" / "train.log",
        training_state={"process": None}, checkpoint_path=CHECKPOINT_PATH,
        auth=auth, ai_mode=ai_mode,
    )
except Exception as e:
    print(f"[crawl_admin] skipped: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
