"""Admin crawl routes — import and call register(app, deps)."""

from __future__ import annotations

from pathlib import Path

from fastapi import Form, Request
from fastapi.responses import RedirectResponse

from web import crawler


def register(app, *, require_admin, templates, ctx, repo_root, train_log_path, training_state, checkpoint_path, auth, ai_mode):
    @app.post("/admin/crawl/settings")
    async def admin_crawl_settings(
        request: Request,
        enabled: str = Form(None),
        web_in_chat: str = Form(None),
        auto_search: str = Form(None),
    ):
        user = require_admin(request)
        if user is None:
            return RedirectResponse("/login", status_code=303)
        crawler.set_enabled(enabled is not None)
        crawler.set_web_in_chat(web_in_chat is not None)
        crawler.set_auto_search(auto_search is not None)
        return RedirectResponse("/admin", status_code=303)

    def _admin_page(request: Request, crawl_message=None):
        user = require_admin(request)
        if user is None:
            return RedirectResponse("/login", status_code=303)
        data_path = repo_root / "data" / "input.txt"
        data_stats = {
            "exists": data_path.exists(),
            "size_bytes": data_path.stat().st_size if data_path.exists() else 0,
        }
        proc = training_state["process"]
        training_running = proc is not None and proc.poll() is None
        log_tail = ""
        if train_log_path.exists():
            log_tail = "\n".join(train_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:])
        return templates.TemplateResponse(
            request=request,
            name="admin.html",
            context=ctx(
                request,
                modes=ai_mode.MODES,
                data_stats=data_stats,
                training_running=training_running,
                log_tail=log_tail,
                checkpoint_exists=checkpoint_path.exists(),
                users=auth.list_users(),
                unlimited_value=auth.UNLIMITED,
                weekly_default=auth.WEEKLY_TOKEN_LIMIT,
                my_trace=getattr(request.state, "trace_id", None),
                crawl_docs=crawler.list_docs(limit=20),
                crawl_status=crawler.status(),
                crawl_message=crawl_message,
            ),
        )

    @app.post("/admin/crawl/search")
    async def admin_crawl_search(request: Request, q: str = Form(...)):
        user = require_admin(request)
        if user is None:
            return RedirectResponse("/login", status_code=303)
        result = crawler.search_and_crawl(q, max_pages=3)
        msg = (
            f"Search '{q}': links={len(result.get('links_found') or [])}, "
            f"crawled_ok={result.get('crawled_ok')}"
        )
        return _admin_page(request, crawl_message=msg)

    @app.post("/admin/crawl/random")
    async def admin_crawl_random(request: Request):
        user = require_admin(request)
        if user is None:
            return RedirectResponse("/login", status_code=303)
        result = crawler.random_crawl(n=2)
        msg = f"Random crawl ok={result.get('crawled_ok')}"
        return _admin_page(request, crawl_message=msg)

    @app.post("/admin/crawl")
    async def admin_crawl_url(request: Request, url: str = Form(...)):
        user = require_admin(request)
        if user is None:
            return RedirectResponse("/login", status_code=303)
        result = crawler.crawl_url(url)
        msg = f"Crawl {url}: {result}"
        return _admin_page(request, crawl_message=msg)
