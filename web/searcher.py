"""Beq Search — fast Wikipedia (few tries) + DDG + store. Chat must stay under ~10s."""
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

TIMEOUT = 3.0
USER_AGENT = "BeqSearchBot/1.5 (+https://beq.onrender.com)"


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
    # Max 2 title attempts + optional opensearch — keep total under ~9s
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", q)
    candidates = [q]
    if words:
        last = words[-1]
        if last.lower() != q.lower():
            candidates.append(last.title() if last.islower() else last)
    seen = set()
    try:
        with _client() as client:
            for cand in candidates[:2]:
                cand = cand.strip()
                if not cand or cand.lower() in seen:
                    continue
                seen.add(cand.lower())
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
            try:
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
                if s.status_code != 200:
                    return None
                data = s.json()
                if not (data and len(data) > 1 and data[1]):
                    return None
                title = data[1][0]
                r = client.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
                )
                if r.status_code != 200:
                    return None
                d = r.json()
                extract = (d.get("extract") or "").strip()
                if len(extract) < 40:
                    return None
                return {
                    "source": "wikipedia",
                    "title": d.get("title") or title,
                    "url": (d.get("content_urls") or {})
                    .get("desktop", {})
                    .get("page")
                    or "",
                    "text": extract[:1600],
                }
            except Exception:
                return None
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


def search_all(query: str, use_web: bool = True) -> dict:
    q = (query or "").strip()
    result = {"query": q, "store": [], "web": None, "crawler_enabled": crawler.is_enabled()}
    if not q:
        return result
    try:
        result["store"] = crawler.search_store(q, limit=5)
    except Exception:
        result["store"] = []
    if use_web and crawler.is_enabled() and crawler.web_in_chat_enabled():
        result["web"] = search_wikipedia(q) or search_duckduckgo(q)
    return result


def _looks_like_search(prompt: str) -> bool:
    p = prompt.strip().lower()
    if len(p) < 2:
        return False
    triggers = (
        "what is ", "what's ", "who is ", "who's ", "where is ", "when is ",
        "what are ", "define ", "explain ", "search ", "look up ", "tell me about ",
        "how does ", "how do ", "why is ", "why are ",
        "was ist ", "wer ist ", "wo ist ", "erkläre ", "suche ",
        "capital of", "meaning of",
    )
    if any(p.startswith(t) or f" {t}" in f" {p}" for t in triggers):
        return True
    if p.endswith("?") and len(p.split()) <= 16:
        return True
    if 1 <= len(p.split()) <= 8 and not p.startswith(("hi", "hello", "hey", "thanks")):
        return True
    return False


def try_search_answer(prompt: str, use_web: bool = True) -> str | None:
    if not crawler.is_enabled():
        return None
    if not crawler.auto_search_enabled() and not crawler.web_in_chat_enabled():
        return None

    always = True
    try:
        always = crawler.always_search_enabled()
    except Exception:
        always = True

    q = _clean_query(prompt)

    if not always and not _looks_like_search(prompt):
        try:
            store_hits = crawler.search_store(q, limit=2)
            if store_hits and store_hits[0]["score"] >= 0.6:
                top = store_hits[0]
                return f"{top['snippet'][:500]}\n\n(Source: crawled {top.get('url') or ''})"
        except Exception:
            pass
        return None

    store_only = not (use_web and crawler.web_in_chat_enabled())
    parts: list[str] = []

    if not store_only:
        web = search_wikipedia(q)
        if not web:
            web = search_duckduckgo(q)
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
        except Exception as e:
            print(f"[searcher] store: {e}")

    if not parts:
        return None
    return "\n".join(parts)[:2000]
