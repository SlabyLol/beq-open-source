"""
Beq Search Tool — live crawl store + Wikipedia + DuckDuckGo + auto search&crawl.
"""

from __future__ import annotations

import re
from urllib.parse import quote

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

from web import crawler

TIMEOUT = 10.0
USER_AGENT = "BeqSearchBot/1.2 (+https://beq.onrender.com)"


def _client():
    if httpx is None:
        return None
    return httpx.Client(timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT})


def search_wikipedia(query: str) -> dict | None:
    q = (query or "").strip()
    if not q or httpx is None:
        return None
    try:
        with _client() as client:
            r = client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(q)}")
            if r.status_code == 404:
                s = client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={"action": "opensearch", "search": q, "limit": 1, "namespace": 0, "format": "json"},
                )
                s.raise_for_status()
                data = s.json()
                if not data or len(data) < 2 or not data[1]:
                    return None
                r = client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(data[1][0])}")
            r.raise_for_status()
            data = r.json()
            extract = (data.get("extract") or "").strip()
            if not extract:
                return None
            return {
                "source": "wikipedia",
                "title": data.get("title") or q,
                "url": (data.get("content_urls") or {}).get("desktop", {}).get("page") or "",
                "text": extract[:1500],
            }
    except Exception:
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
            r.raise_for_status()
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
            if not abstract:
                return None
            return {"source": "duckduckgo", "title": heading, "url": url, "text": abstract[:1500]}
    except Exception:
        return None


def search_all(query: str, use_web: bool = True) -> dict:
    q = (query or "").strip()
    result: dict = {"query": q, "store": [], "web": None, "crawler_enabled": crawler.is_enabled(), "live_crawl": None}
    if not q:
        return result
    result["store"] = crawler.search_store(q, limit=5)
    if use_web and crawler.is_enabled() and crawler.web_in_chat_enabled():
        result["web"] = search_wikipedia(q) or search_duckduckgo(q)
    return result


def _looks_like_search(prompt: str) -> bool:
    p = prompt.strip().lower()
    if len(p) < 3:
        return False
    triggers = (
        "what is ", "what's ", "who is ", "who's ", "where is ", "when is ",
        "what are ", "define ", "explain ", "search ", "look up ", "tell me about ",
        "how does ", "how do ", "why is ", "why are ",
        "was ist ", "wer ist ", "wo ist ", "erkläre ", "suche ",
    )
    if any(p.startswith(t) or f" {t}" in f" {p}" for t in triggers):
        return True
    if p.endswith("?") and len(p.split()) <= 14:
        return True
    return False


def try_search_answer(prompt: str, use_web: bool = True) -> str | None:
    """Answer from crawl store + Wikipedia + LIVE search&crawl when needed."""
    if not crawler.is_enabled():
        return None
    if not crawler.auto_search_enabled() and not crawler.web_in_chat_enabled():
        return None

    store_only = not (use_web and crawler.web_in_chat_enabled())

    q = prompt.strip()
    q = re.sub(
        r"^(please\s+)?(search\s+for|look\s+up|tell\s+me\s+about|define|explain|what\s+is|what's|who\s+is|who's|how\s+does|why\s+is)\s+",
        "",
        q,
        flags=re.I,
    ).strip(" ?")

    store_hits = crawler.search_store(prompt if not q else q, limit=3)

    if not _looks_like_search(prompt):
        if store_hits and store_hits[0]["score"] >= 0.55:
            top = store_hits[0]
            title = top.get("title") or ""
            body = top["snippet"][:500]
            src = top.get("url") or ""
            prefix = f"{title}: " if title else ""
            return f"{prefix}{body}\n\n(Source: crawled {src})"
        return None

    found = search_all(q or prompt, use_web=not store_only)
    parts: list[str] = []

    if found.get("web"):
        w = found["web"]
        parts.append(w["text"])
        if w.get("url"):
            parts.append(f"(Source: {w.get('source', 'web')} — {w['url']})")

    if found.get("store"):
        top = found["store"][0]
        if top["score"] >= 0.3:
            t = top.get("title") or ""
            parts.append(f"From crawl store{' — ' + t if t else ''}: {top['snippet'][:400]}")
            if top.get("url"):
                parts.append(f"(Crawled: {top['url']})")

    # Live crawl only when we have nothing useful yet (avoid long request timeouts)
    need_live = not parts
    if need_live and crawler.auto_search_enabled() and not store_only:
        try:
            live = crawler.search_and_crawl(q or prompt, max_pages=2)
            found["live_crawl"] = {
                "crawled_ok": live.get("crawled_ok"),
                "links": live.get("links_found", [])[:5],
            }
            again = crawler.search_store(q or prompt, limit=2)
            if again:
                top = again[0]
                t = top.get("title") or ""
                parts.append(f"{t + ': ' if t else ''}{top['snippet'][:500]}")
                parts.append(f"(Live crawled for: {q or prompt})")
            for r in (live.get("results") or [])[:2]:
                if r.get("ok") and r.get("preview") and not again:
                    parts.append(r["preview"][:400])
                    parts.append(f"(Crawled: {r.get('url')})")
        except Exception as e:
            print(f"[searcher] live crawl: {e}")

    if not parts:
        return None
    return "\n".join(parts)[:1800]
