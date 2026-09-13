"""Beq Search — Wikipedia/DDG/store, simple and fast."""
from __future__ import annotations

import re
from urllib.parse import quote

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

from web import crawler

try:
    from web import crawl_always_patch  # noqa: F401
except Exception:
    pass

TIMEOUT = 2.5
USER_AGENT = "BeqSearchBot/1.7 (+https://beq.onrender.com)"


def _client():
    if httpx is None:
        return None
    return httpx.Client(
        timeout=TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )


def _clean_query(prompt: str) -> str:
    q = (prompt or "").strip()
    q = re.sub(
        r"^(please\s+)?(search\s+for|look\s+up|tell\s+me\s+about|define|explain|"
        r"what\s+is|what\s+are|what's|who\s+is|who's|where\s+is|when\s+is|"
        r"how\s+does|how\s+do|why\s+is|was\s+ist|wer\s+ist|erkläre|suche)\s+",
        "",
        q,
        flags=re.I,
    )
    return q.strip(" ?!.") or prompt.strip()


def search_wikipedia(query: str) -> dict | None:
    q = (query or "").strip()
    if not q or httpx is None:
        return None
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", q)
    candidates = [q]
    if words and words[-1].lower() != q.lower():
        candidates.append(words[-1].title())
    try:
        with _client() as client:
            for cand in candidates[:2]:
                try:
                    r = client.get(
                        f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(cand)}"
                    )
                    if r.status_code != 200:
                        continue
                    data = r.json()
                    if data.get("type") == "disambiguation":
                        continue
                    extract = (data.get("extract") or "").strip()
                    if len(extract) < 40:
                        continue
                    return {
                        "source": "wikipedia",
                        "title": data.get("title") or cand,
                        "url": (data.get("content_urls") or {})
                        .get("desktop", {})
                        .get("page")
                        or "",
                        "text": extract[:1600],
                    }
                except Exception:
                    continue
    except Exception as e:
        print(f"[searcher] wikipedia: {e}")
    return None


def search_duckduckgo(query: str) -> dict | None:
    q = (query or "").strip()
    if not q or httpx is None:
        return None
    try:
        with _client() as client:
            r = client.get(
                "https://api.duckduckgo.com/",
                params={"q": q, "format": "json", "no_html": 1, "skip_disambig": 1},
            )
            if r.status_code != 200:
                return None
            data = r.json()
            abstract = (data.get("AbstractText") or "").strip()
            heading = (data.get("Heading") or q).strip()
            url = (data.get("AbstractURL") or "").strip()
            if not abstract:
                for rel in data.get("RelatedTopics") or []:
                    if isinstance(rel, dict) and rel.get("Text"):
                        abstract = rel["Text"]
                        url = rel.get("FirstURL") or url
                        break
            if not abstract or len(abstract) < 30:
                return None
            return {
                "source": "duckduckgo",
                "title": heading,
                "url": url,
                "text": abstract[:1500],
            }
    except Exception as e:
        print(f"[searcher] ddg: {e}")
        return None


def try_search_answer(prompt: str, use_web: bool = True) -> str | None:
    try:
        if not crawler.is_enabled():
            return None
        if not crawler.auto_search_enabled() and not crawler.web_in_chat_enabled():
            return None

        q = _clean_query(prompt)
        parts: list[str] = []

        if use_web and crawler.web_in_chat_enabled():
            web = search_wikipedia(q) or search_duckduckgo(q)
            if web:
                parts.append(web["text"])
                if web.get("url"):
                    parts.append(f"(Source: {web.get('source', 'web')} — {web['url']})")

        if not parts:
            try:
                store = crawler.search_store(q, limit=3)
                if store and store[0]["score"] >= 0.3:
                    top = store[0]
                    parts.append(top["snippet"][:500])
                    if top.get("url"):
                        parts.append(f"(Crawled: {top['url']})")
            except Exception:
                pass

        if not parts:
            return None
        return "\n".join(parts)[:2000]
    except Exception as e:
        print(f"[searcher] try_search: {e}")
        return None
