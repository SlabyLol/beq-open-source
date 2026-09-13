"""
Beq Search Tool — answer from crawl store + public web (Wikipedia, DuckDuckGo).

Used when .sbe knowledge and math tool do not match.
Does not require API keys.
"""

from __future__ import annotations

import re
from urllib.parse import quote

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

from web import crawler

TIMEOUT = 8.0
USER_AGENT = "BeqSearchBot/1.0 (+https://beq.onrender.com)"


def _client():
    if httpx is None:
        return None
    return httpx.Client(timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT})


def search_wikipedia(query: str) -> dict | None:
    q = (query or "").strip()
    if not q or httpx is None:
        return None
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(q)}"
    try:
        with _client() as client:
            r = client.get(url)
            if r.status_code == 404:
                # try search API for best title
                s = client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "opensearch",
                        "search": q,
                        "limit": 1,
                        "namespace": 0,
                        "format": "json",
                    },
                )
                s.raise_for_status()
                data = s.json()
                if not data or len(data) < 2 or not data[1]:
                    return None
                title = data[1][0]
                r = client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}")
            r.raise_for_status()
            data = r.json()
            extract = (data.get("extract") or "").strip()
            if not extract:
                return None
            return {
                "source": "wikipedia",
                "title": data.get("title") or q,
                "url": (data.get("content_urls") or {}).get("desktop", {}).get("page")
                or data.get("url")
                or "",
                "text": extract[:1200],
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
                # related topics fallback
                for rel in data.get("RelatedTopics") or []:
                    if isinstance(rel, dict) and rel.get("Text"):
                        abstract = rel["Text"]
                        url = rel.get("FirstURL") or url
                        break
            if not abstract:
                return None
            return {
                "source": "duckduckgo",
                "title": heading,
                "url": url,
                "text": abstract[:1200],
            }
    except Exception:
        return None


def search_all(query: str, use_web: bool = True) -> dict:
    """Combined search: local crawl store first, then Wikipedia, then DDG."""
    q = (query or "").strip()
    result: dict = {"query": q, "store": [], "web": None}
    if not q:
        return result

    result["store"] = crawler.search_store(q, limit=5)

    if use_web:
        web = search_wikipedia(q) or search_duckduckgo(q)
        result["web"] = web

    return result


def _looks_like_search(prompt: str) -> bool:
    p = prompt.strip().lower()
    if len(p) < 3:
        return False
    triggers = (
        "what is ", "what's ", "who is ", "who's ", "where is ", "when is ",
        "what are ", "define ", "explain ", "search ", "look up ", "tell me about ",
        "was ist ", "wer ist ", "wo ist ", "erkläre ", "suche ",
    )
    if any(p.startswith(t) or f" {t}" in f" {p}" for t in triggers):
        return True
    # short factual questions
    if p.endswith("?") and len(p.split()) <= 12:
        return True
    return False


def try_search_answer(prompt: str, use_web: bool = True) -> str | None:
    """
    Return a short answer string from crawl store / web search, or None.
    Intended for the Beq answer pipeline after .sbe and math.
    """
    if not _looks_like_search(prompt):
        # still try store-only match for any prompt
        store = crawler.search_store(prompt, limit=1)
        if store and store[0]["score"] >= 0.6:
            sn = store[0]["snippet"]
            return f"{sn[:500]}\n\n(Source: crawled {store[0].get('url', '')})"
        return None

    # strip leading search verbs
    q = prompt.strip()
    q = re.sub(
        r"^(please\s+)?(search\s+for|look\s+up|tell\s+me\s+about|define|explain|what\s+is|what's|who\s+is|who's)\s+",
        "",
        q,
        flags=re.I,
    ).strip(" ?")

    found = search_all(q, use_web=use_web)

    parts: list[str] = []
    if found.get("web"):
        w = found["web"]
        parts.append(w["text"])
        if w.get("url"):
            parts.append(f"(Source: {w.get('source', 'web')} — {w['url']})")
    if found.get("store"):
        top = found["store"][0]
        if top["score"] >= 0.35:
            parts.append(f"From crawl store: {top['snippet'][:350]}")
            if top.get("url"):
                parts.append(f"(Crawled: {top['url']})")

    if not parts:
        return None
    return "\n".join(parts)[:1500]
